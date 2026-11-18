// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Pure-Rust Web Mercator (`EPSG:3857`) slippy-tile + vector map engine for `egui`.
//! Supports continuous fractional zoom, cursor-anchored pan/zoom, offline SQLite raster
//! tile caching, and multi-provider basemaps (Offline Vector Dark, Carto Dark, USGS Topo, OSM).

use crate::db::FlywayStore;
use eframe::egui::{ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, Vec2};
use reqwest::blocking::Client;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

const TILE_SIZE: f64 = 256.0;
const EARTH_RADIUS_M: f64 = 6_378_137.0;
const MAX_MERCATOR_LAT: f64 = 85.051_128_78;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BasemapProvider {
    VectorDarkOffline,
    CartoDarkMatter,
    UsgsNationalTopo,
    OpenStreetMap,
}

impl BasemapProvider {
    pub fn label(self) -> &'static str {
        match self {
            Self::VectorDarkOffline => "Vector Dark (Offline)",
            Self::CartoDarkMatter => "Carto Dark Matter",
            Self::UsgsNationalTopo => "USGS National Topo",
            Self::OpenStreetMap => "OpenStreetMap",
        }
    }

    pub fn cache_key(self) -> &'static str {
        match self {
            Self::VectorDarkOffline => "offline",
            Self::CartoDarkMatter => "carto_dark",
            Self::UsgsNationalTopo => "usgs_topo",
            Self::OpenStreetMap => "osm_std",
        }
    }

    pub fn tile_url(self, z: u32, x: u32, y: u32) -> Option<String> {
        match self {
            Self::VectorDarkOffline => None,
            Self::CartoDarkMatter => Some(format!(
                "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
            )),
            // ArcGIS MapServer tile order is /tile/{z}/{y}/{x}
            Self::UsgsNationalTopo => Some(format!(
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
            )),
            Self::OpenStreetMap => Some(format!("https://tile.openstreetmap.org/{z}/{x}/{y}.png")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub provider: BasemapProvider,
    pub z: u32,
    pub x: u32,
    pub y: u32,
}

struct DecodedTile {
    key: TileKey,
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

pub struct ProjectedTile {
    pub min_wx: f64,
    pub min_wy: f64,
    pub max_wx: f64,
    pub max_wy: f64,
    pub texture: TextureHandle,
}

pub struct MapViewport {
    pub center_lat: f64,
    pub center_lon: f64,
    pub zoom: f64,
    pub pitch_deg: f64,
    pub bearing_deg: f64,
    pub alt_exaggeration: f64,
    pub provider: BasemapProvider,
    textures: HashMap<TileKey, TextureHandle>,
    in_flight: HashSet<TileKey>,
    failed: HashSet<TileKey>,
    req_tx: Sender<(TileKey, Context)>,
    resp_rx: Receiver<Option<(TileKey, DecodedTile)>>,
}

impl MapViewport {
    pub fn new(db_path: PathBuf) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<(TileKey, Context)>();
        let (resp_tx, resp_rx) = mpsc::channel::<Option<(TileKey, DecodedTile)>>();

        spawn_tile_worker(db_path, req_rx, resp_tx);

        Self {
            // Centered on Willamette Valley / Corvallis & Finley NWR within Pacific Flyway context
            center_lat: 44.48,
            center_lon: -122.95,
            zoom: 6.4,
            pitch_deg: 42.0,
            bearing_deg: -18.0,
            alt_exaggeration: 1.0,
            provider: BasemapProvider::VectorDarkOffline,
            textures: HashMap::new(),
            in_flight: HashSet::new(),
            failed: HashSet::new(),
            req_tx,
            resp_rx,
        }
    }

    pub fn is_25d(&self) -> bool {
        self.pitch_deg > 1.0 || self.bearing_deg.abs() > 1.0
    }

    pub fn toggle_25d(&mut self) {
        if self.is_25d() {
            self.pitch_deg = 0.0;
            self.bearing_deg = 0.0;
        } else {
            self.pitch_deg = 42.0;
            self.bearing_deg = -18.0;
        }
    }

    pub fn wrap_bearing(deg: f64) -> f64 {
        let mut d = deg;
        while d > 180.0 {
            d -= 360.0;
        }
        while d < -180.0 {
            d += 360.0;
        }
        d
    }

    pub fn jump_to(&mut self, lat: f64, lon: f64, zoom: f64) {
        self.center_lat = lat.clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
        self.center_lon = wrap_lon(lon);
        self.zoom = zoom.clamp(3.5, 14.5);
    }

    /// Projects geographic `(lat, lon)` to global Web Mercator pixel coordinates at `zoom`.
    pub fn lat_lon_to_world_px(lat: f64, lon: f64, zoom: f64) -> (f64, f64) {
        let clamped_lat = lat.clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
        let scale = TILE_SIZE * 2.0_f64.powf(zoom);
        let x = (wrap_lon(lon) + 180.0) / 360.0 * scale;
        let lat_rad = clamped_lat.to_radians();
        let merc_n = ((PI / 4.0) + (lat_rad / 2.0)).tan().ln();
        let y = (0.5 - merc_n / (2.0 * PI)) * scale;
        (x, y)
    }

    /// Inverse Web Mercator from global pixel coordinates `(wx, wy)` at `zoom` to `(lat, lon)`.
    pub fn world_px_to_lat_lon(wx: f64, wy: f64, zoom: f64) -> (f64, f64) {
        let scale = TILE_SIZE * 2.0_f64.powf(zoom);
        let lon = wrap_lon((wx / scale) * 360.0 - 180.0);
        let n = PI - 2.0 * PI * (wy / scale);
        let lat = n.sinh().atan().to_degrees().clamp(-MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
        (lat, lon)
    }

    #[inline]
    fn camera_depth_px(canvas_rect: Rect) -> f64 {
        (canvas_rect.height() as f64 * 1.15).max(680.0)
    }

    #[inline]
    fn project_delta_px(&self, dx: f64, dy: f64, alt_agl_m: f64, canvas_rect: Rect) -> (Pos2, f32) {
        let b = self.bearing_deg.to_radians();
        let p = self.pitch_deg.clamp(0.0, 62.0).to_radians();
        let ge = dx;
        let gn = -dy;
        let rx = ge * b.cos() - gn * b.sin();
        let fy = ge * b.sin() + gn * b.cos();
        let elev_z = (alt_agl_m / 36.0) * self.alt_exaggeration.max(0.0);
        let cam_d = Self::camera_depth_px(canvas_rect);
        let depth = fy * p.sin() - elev_z * p.cos();
        let denom = (cam_d + depth).max(cam_d * 0.22);
        let persp = cam_d / denom;
        let center_screen = canvas_rect.center();
        let sx = center_screen.x + (rx * persp) as f32;
        let sy = center_screen.y - ((fy * p.cos() + elev_z * p.sin()) * persp) as f32;
        (Pos2::new(sx, sy), persp as f32)
    }

    /// Projects geographic `(lat, lon, alt_agl_m)` into 2.5D perspective screen coordinates and foreshortening scale.
    pub fn project_ground_with_scale(
        &self,
        lat: f64,
        lon: f64,
        alt_agl_m: f64,
        canvas_rect: Rect,
    ) -> (Pos2, f32) {
        let scale = TILE_SIZE * 2.0_f64.powf(self.zoom);
        let dlon = wrap_lon(lon - self.center_lon);
        let dx = (dlon / 360.0) * scale;
        let (_, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let (_, wy) = Self::lat_lon_to_world_px(lat, lon, self.zoom);
        let dy = wy - cy;
        self.project_delta_px(dx, dy, alt_agl_m, canvas_rect)
    }

    /// Projects raw Web Mercator world pixel coordinates `(wx, wy)` + `alt_agl_m` into 2.5D screen space.
    pub fn project_world_px(
        &self,
        wx: f64,
        wy: f64,
        alt_agl_m: f64,
        canvas_rect: Rect,
    ) -> (Pos2, f32) {
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        self.project_delta_px(wx - cx, wy - cy, alt_agl_m, canvas_rect)
    }

    /// Converts `(lat, lon)` on the ground plane (`alt_agl_m = 0`) into screen `Pos2` inside `canvas_rect`.
    pub fn lat_lon_to_screen(&self, lat: f64, lon: f64, canvas_rect: Rect) -> Pos2 {
        self.project_ground_with_scale(lat, lon, 0.0, canvas_rect).0
    }

    #[inline]
    fn screen_to_world_delta(&self, pos: Pos2, canvas_rect: Rect) -> (f64, f64) {
        let b = self.bearing_deg.to_radians();
        let p = self.pitch_deg.clamp(0.0, 62.0).to_radians();
        let cam_d = Self::camera_depth_px(canvas_rect);
        let center_screen = canvas_rect.center();
        let ux = (pos.x - center_screen.x) as f64;
        let uy_up = -(pos.y - center_screen.y) as f64;
        let fy = (uy_up * cam_d) / (p.cos() * cam_d - uy_up * p.sin()).max(cam_d * 0.15);
        let denom = (cam_d + fy * p.sin()).max(cam_d * 0.22);
        let persp = cam_d / denom;
        let rx = ux / persp;
        let dx = rx * b.cos() + fy * b.sin();
        let gn = -rx * b.sin() + fy * b.cos();
        (dx, -gn)
    }

    /// Converts screen `Pos2` inside `canvas_rect` to geographic `(lat, lon)` on the ground plane.
    pub fn screen_to_lat_lon(&self, pos: Pos2, canvas_rect: Rect) -> (f64, f64) {
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let (dx, dy) = self.screen_to_world_delta(pos, canvas_rect);
        let wx = cx + dx;
        let wy = cy + dy;
        Self::world_px_to_lat_lon(wx, wy, self.zoom)
    }

    /// Pans map by screen drag delta `delta_px`, compensating for current bearing and pitch tilt.
    pub fn pan_by_pixels(&mut self, delta_px: Vec2) {
        let b = self.bearing_deg.to_radians();
        let p = self.pitch_deg.clamp(0.0, 62.0).to_radians();
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let max_world = TILE_SIZE * 2.0_f64.powf(self.zoom);
        let d_rx = delta_px.x as f64;
        let d_fy = -(delta_px.y as f64) / p.cos().max(0.32);
        let d_dx = d_rx * b.cos() + d_fy * b.sin();
        let d_dy = -(-d_rx * b.sin() + d_fy * b.cos());
        let nx = cx - d_dx;
        let ny = (cy - d_dy).clamp(0.0, max_world);
        let (lat, lon) = Self::world_px_to_lat_lon(nx, ny, self.zoom);
        self.center_lat = lat;
        self.center_lon = lon;
    }

    /// Zooms smoothly around `anchor_screen` so the geographic point under the cursor stays fixed.
    pub fn zoom_around(&mut self, zoom_delta: f64, anchor_screen: Pos2, canvas_rect: Rect) {
        let (anchor_lat, anchor_lon) = self.screen_to_lat_lon(anchor_screen, canvas_rect);
        let new_zoom = (self.zoom + zoom_delta).clamp(3.5, 14.5);
        if (new_zoom - self.zoom).abs() < 1e-6 {
            return;
        }
        self.zoom = new_zoom;

        let (ax, ay) = Self::lat_lon_to_world_px(anchor_lat, anchor_lon, self.zoom);
        let (dx_off, dy_off) = self.screen_to_world_delta(anchor_screen, canvas_rect);
        let new_cx = ax - dx_off;
        let max_world = TILE_SIZE * 2.0_f64.powf(self.zoom);
        let new_cy = (ay - dy_off).clamp(0.0, max_world);
        let (lat, lon) = Self::world_px_to_lat_lon(new_cx, new_cy, self.zoom);
        self.center_lat = lat;
        self.center_lon = lon;
    }

    /// Ground resolution in meters per screen pixel at current center latitude.
    pub fn meters_per_pixel(&self) -> f64 {
        let cos_lat = self.center_lat.to_radians().cos().abs().max(0.05);
        (2.0 * PI * EARTH_RADIUS_M * cos_lat) / (TILE_SIZE * 2.0_f64.powf(self.zoom))
    }

    /// Polls decoded tiles from worker thread and returns visible projected world-space tiles for 2.5D warping.
    pub fn visible_world_tiles(
        &mut self,
        ctx: &Context,
        canvas_rect: Rect,
    ) -> Vec<ProjectedTile> {
        while let Ok(msg) = self.resp_rx.try_recv() {
            if let Some((key, decoded)) = msg {
                self.in_flight.remove(&key);
                if decoded.rgba.is_empty() {
                    self.failed.insert(key);
                } else {
                    if self.textures.len() > 320 {
                        if let Some(first_key) = self.textures.keys().next().copied() {
                            self.textures.remove(&first_key);
                        }
                    }
                    let img = ColorImage::from_rgba_unmultiplied(
                        [decoded.width, decoded.height],
                        &decoded.rgba,
                    );
                    let tex_name = format!(
                        "tile_{}_{}_{}_{}",
                        key.provider.cache_key(),
                        key.z,
                        key.x,
                        key.y
                    );
                    let handle = ctx.load_texture(tex_name, img, TextureOptions::LINEAR);
                    self.textures.insert(key, handle);
                }
            }
        }

        if self.provider == BasemapProvider::VectorDarkOffline {
            return Vec::new();
        }

        let tile_z = (self.zoom.floor() as u32).clamp(3, 14);
        let n_tiles = 1_u32 << tile_z;
        let sub_scale = 2.0_f64.powf(self.zoom - tile_z as f64);
        let denom = TILE_SIZE * sub_scale;

        let (cx_z, cy_z) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let mut world_left = cx_z - (canvas_rect.width() as f64) * 0.6;
        let mut world_right = cx_z + (canvas_rect.width() as f64) * 0.6;
        let mut world_top = cy_z - (canvas_rect.height() as f64) * 0.6;
        let mut world_bottom = cy_z + (canvas_rect.height() as f64) * 0.6;

        for sy_frac in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for sx_frac in [0.0_f32, 0.5, 1.0] {
                let p = Pos2::new(
                    canvas_rect.left() + canvas_rect.width() * sx_frac,
                    canvas_rect.top() + canvas_rect.height() * sy_frac,
                );
                let (dx, dy) = self.screen_to_world_delta(p, canvas_rect);
                world_left = world_left.min(cx_z + dx);
                world_right = world_right.max(cx_z + dx);
                world_top = world_top.min(cy_z + dy);
                world_bottom = world_bottom.max(cy_z + dy);
            }
        }

        let center_tx = (cx_z / denom).floor() as i32;
        let center_ty = (cy_z / denom).floor() as i32;
        let min_tx = ((world_left / denom).floor() as i32).max(center_tx - 5);
        let max_tx = ((world_right / denom).floor() as i32).min(center_tx + 5);
        let min_ty = ((world_top / denom).floor() as i32).clamp(0, (n_tiles as i32) - 1).max(center_ty - 5);
        let max_ty = ((world_bottom / denom).floor() as i32).clamp(0, (n_tiles as i32) - 1).min(center_ty + 5);

        let mut visible = Vec::new();
        for ty in min_ty..=max_ty {
            if ty < 0 || ty >= n_tiles as i32 {
                continue;
            }
            for tx in min_tx..=max_tx {
                let wrapped_tx = tx.rem_euclid(n_tiles as i32) as u32;
                let key = TileKey {
                    provider: self.provider,
                    z: tile_z,
                    x: wrapped_tx,
                    y: ty as u32,
                };
                if let Some(tex) = self.textures.get(&key) {
                    visible.push(ProjectedTile {
                        min_wx: (tx as f64) * denom,
                        min_wy: (ty as f64) * denom,
                        max_wx: ((tx + 1) as f64) * denom,
                        max_wy: ((ty + 1) as f64) * denom,
                        texture: tex.clone(),
                    });
                } else if !self.in_flight.contains(&key)
                    && !self.failed.contains(&key)
                    && self.in_flight.len() < 24
                {
                    self.in_flight.insert(key);
                    let _ = self.req_tx.send((key, ctx.clone()));
                }
            }
        }
        visible
    }

    /// Flat 2D compatibility wrapper returning unwarped screen rectangles.
    pub fn visible_raster_tiles(
        &mut self,
        ctx: &Context,
        canvas_rect: Rect,
    ) -> Vec<(Rect, TextureHandle)> {
        self.visible_world_tiles(ctx, canvas_rect)
            .into_iter()
            .map(|t| {
                let p0 = self.project_world_px(t.min_wx, t.min_wy, 0.0, canvas_rect).0;
                let p1 = self.project_world_px(t.max_wx, t.max_wy, 0.0, canvas_rect).0;
                (Rect::from_two_pos(p0, p1), t.texture)
            })
            .collect()
    }

    pub fn retry_failed_tiles(&mut self) {
        self.failed.clear();
    }
}

fn wrap_lon(lon: f64) -> f64 {
    let mut l = lon;
    while l > 180.0 {
        l -= 360.0;
    }
    while l < -180.0 {
        l += 360.0;
    }
    l
}

/// Great-circle Haversine distance in kilometers.
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6371.0_f64;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat * 0.5).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon * 0.5).sin().powi(2);
    2.0 * r * a.clamp(0.0, 1.0).sqrt().asin()
}

fn spawn_tile_worker(
    db_path: PathBuf,
    req_rx: Receiver<(TileKey, Context)>,
    resp_tx: Sender<Option<(TileKey, DecodedTile)>>,
) {
    std::thread::spawn(move || {
        let client = Client::builder()
            .user_agent("AeroFlyway-Desktop-GIS/0.1 (+https://crvscience.com)")
            .timeout(Duration::from_secs(8))
            .build()
            .ok();

        let store = open_tile_store(&db_path);

        while let Ok((key, ctx)) = req_rx.recv() {
            let mut png_bytes = store
                .as_ref()
                .and_then(|s| s.get_cached_tile(key.provider.cache_key(), key.z as u8, key.x, key.y));

            if png_bytes.is_none() {
                if let (Some(http), Some(url)) = (&client, key.provider.tile_url(key.z, key.x, key.y))
                {
                    if let Ok(resp) = http.get(&url).send() {
                        if resp.status().is_success() {
                            if let Ok(bytes) = resp.bytes() {
                                let vec_bytes = bytes.to_vec();
                                if let Some(s) = &store {
                                    s.save_cached_tile(
                                        key.provider.cache_key(),
                                        key.z as u8,
                                        key.x,
                                        key.y,
                                        &vec_bytes,
                                    );
                                }
                                png_bytes = Some(vec_bytes);
                            }
                        }
                    }
                }
            }

            let decoded = match png_bytes.and_then(|b| image::load_from_memory(&b).ok()) {
                Some(dyn_img) => {
                    let rgba8 = dyn_img.to_rgba8();
                    let (w, h) = rgba8.dimensions();
                    DecodedTile {
                        key,
                        width: w as usize,
                        height: h as usize,
                        rgba: rgba8.into_raw(),
                    }
                }
                None => DecodedTile {
                    key,
                    width: 0,
                    height: 0,
                    rgba: Vec::new(),
                },
            };

            let _ = resp_tx.send(Some((key, decoded)));
            ctx.request_repaint();
        }
    });
}

fn open_tile_store(path: &Path) -> Option<FlywayStore> {
    FlywayStore::open(path).ok()
}

/// Tests candidate screen offsets around `anchor` and returns the first bounding rectangle
/// completely inside `canvas_rect.shrink(4.0)` that does not intersect any `placed` rectangle
/// (with a 3 px spacing margin). Returns `None` if all candidate slots collide.
pub fn place_non_overlapping_label_rect(
    anchor: Pos2,
    marker_radius_px: f32,
    box_size: Vec2,
    canvas_rect: Rect,
    placed: &[Rect],
) -> Option<Rect> {
    let bounds = canvas_rect.shrink(4.0);
    if bounds.width() < box_size.x || bounds.height() < box_size.y {
        return None;
    }
    let r = marker_radius_px.max(2.0);
    let candidates = [
        Pos2::new(anchor.x + r + 8.0, anchor.y - box_size.y * 0.5),
        Pos2::new(anchor.x - box_size.x * 0.5, anchor.y - r - 6.0 - box_size.y),
        Pos2::new(anchor.x - box_size.x * 0.5, anchor.y + r + 6.0),
        Pos2::new(anchor.x - r - 8.0 - box_size.x, anchor.y - box_size.y * 0.5),
        Pos2::new(anchor.x + r + 7.0, anchor.y - r - 4.0 - box_size.y),
        Pos2::new(anchor.x + r + 7.0, anchor.y + r + 4.0),
        Pos2::new(anchor.x - r - 7.0 - box_size.x, anchor.y - r - 4.0 - box_size.y),
        Pos2::new(anchor.x - r - 7.0 - box_size.x, anchor.y + r + 4.0),
        Pos2::new(anchor.x + r + 22.0, anchor.y - r - 22.0 - box_size.y),
        Pos2::new(anchor.x - r - 22.0 - box_size.x, anchor.y + r + 22.0),
        Pos2::new(anchor.x + r + 24.0, anchor.y + 14.0),
        Pos2::new(anchor.x - r - 24.0 - box_size.x, anchor.y - 14.0 - box_size.y),
    ];

    for min_pos in candidates {
        let rect = Rect::from_min_size(min_pos, box_size);
        let inside = rect.min.x >= bounds.min.x
            && rect.max.x <= bounds.max.x
            && rect.min.y >= bounds.min.y
            && rect.max.y <= bounds.max.y;
        if !inside {
            continue;
        }
        if !placed.iter().any(|p| p.expand(3.0).intersects(rect)) {
            return Some(rect);
        }
    }
    None
}

/// Screen-space collision-avoiding label placer shared across map overlays and validation tests.
pub struct MapLabelPlacer {
    pub canvas_rect: Rect,
    pub placed: Vec<Rect>,
}

impl MapLabelPlacer {
    pub fn new(canvas_rect: Rect) -> Self {
        Self {
            canvas_rect,
            placed: Vec::new(),
        }
    }

    pub fn reserve_rect(&mut self, rect: Rect) {
        self.placed.push(rect);
    }

    pub fn try_place(
        &mut self,
        anchor: Pos2,
        marker_radius_px: f32,
        box_size: Vec2,
    ) -> Option<Rect> {
        let rect = place_non_overlapping_label_rect(
            anchor,
            marker_radius_px,
            box_size,
            self.canvas_rect,
            &self.placed,
        )?;
        self.placed.push(rect);
        Some(rect)
    }
}
