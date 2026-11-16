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

pub struct MapViewport {
    pub center_lat: f64,
    pub center_lon: f64,
    pub zoom: f64,
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
            provider: BasemapProvider::VectorDarkOffline,
            textures: HashMap::new(),
            in_flight: HashSet::new(),
            failed: HashSet::new(),
            req_tx,
            resp_rx,
        }
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

    /// Converts `(lat, lon)` into screen `Pos2` inside `canvas_rect`.
    pub fn lat_lon_to_screen(&self, lat: f64, lon: f64, canvas_rect: Rect) -> Pos2 {
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let (wx, wy) = Self::lat_lon_to_world_px(lat, lon, self.zoom);
        let center_screen = canvas_rect.center();
        Pos2::new(
            center_screen.x + (wx - cx) as f32,
            center_screen.y + (wy - cy) as f32,
        )
    }

    /// Converts screen `Pos2` inside `canvas_rect` to geographic `(lat, lon)`.
    pub fn screen_to_lat_lon(&self, pos: Pos2, canvas_rect: Rect) -> (f64, f64) {
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let center_screen = canvas_rect.center();
        let wx = cx + (pos.x - center_screen.x) as f64;
        let wy = cy + (pos.y - center_screen.y) as f64;
        Self::world_px_to_lat_lon(wx, wy, self.zoom)
    }

    /// Pans map by screen drag delta `delta_px`.
    pub fn pan_by_pixels(&mut self, delta_px: Vec2) {
        let (cx, cy) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let max_world = TILE_SIZE * 2.0_f64.powf(self.zoom);
        let nx = cx - delta_px.x as f64;
        let ny = (cy - delta_px.y as f64).clamp(0.0, max_world);
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
        let center_screen = canvas_rect.center();
        let offset_x = (anchor_screen.x - center_screen.x) as f64;
        let offset_y = (anchor_screen.y - center_screen.y) as f64;
        let new_cx = ax - offset_x;
        let max_world = TILE_SIZE * 2.0_f64.powf(self.zoom);
        let new_cy = (ay - offset_y).clamp(0.0, max_world);
        let (lat, lon) = Self::world_px_to_lat_lon(new_cx, new_cy, self.zoom);
        self.center_lat = lat;
        self.center_lon = lon;
    }

    /// Ground resolution in meters per screen pixel at current center latitude.
    pub fn meters_per_pixel(&self) -> f64 {
        let cos_lat = self.center_lat.to_radians().cos().abs().max(0.05);
        (2.0 * PI * EARTH_RADIUS_M * cos_lat) / (TILE_SIZE * 2.0_f64.powf(self.zoom))
    }

    /// Polls decoded tiles from worker thread and returns visible `(Rect, TextureHandle)` pairs.
    pub fn visible_raster_tiles(
        &mut self,
        ctx: &Context,
        canvas_rect: Rect,
    ) -> Vec<(Rect, TextureHandle)> {
        while let Ok(msg) = self.resp_rx.try_recv() {
            if let Some((key, decoded)) = msg {
                self.in_flight.remove(&key);
                if decoded.rgba.is_empty() {
                    self.failed.insert(key);
                } else {
                    if self.textures.len() > 320 {
                        // Simple eviction to bound GPU texture cache
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
        let scaled_tile_px = (TILE_SIZE * sub_scale) as f32;

        let (cx_z, cy_z) = Self::lat_lon_to_world_px(self.center_lat, self.center_lon, self.zoom);
        let center_screen = canvas_rect.center();
        let world_left = cx_z - (canvas_rect.width() as f64) * 0.5;
        let world_right = cx_z + (canvas_rect.width() as f64) * 0.5;
        let world_top = cy_z - (canvas_rect.height() as f64) * 0.5;
        let world_bottom = cy_z + (canvas_rect.height() as f64) * 0.5;

        let denom = TILE_SIZE * sub_scale;
        let min_tx = (world_left / denom).floor() as i32;
        let max_tx = (world_right / denom).floor() as i32;
        let min_ty = (world_top / denom).floor().max(0.0) as i32;
        let max_ty = (world_bottom / denom).floor().min((n_tiles as f64) - 1.0) as i32;

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

                let screen_x = center_screen.x + ((tx as f64) * denom - cx_z) as f32;
                let screen_y = center_screen.y + ((ty as f64) * denom - cy_z) as f32;
                let rect = Rect::from_min_size(
                    Pos2::new(screen_x, screen_y),
                    Vec2::new(scaled_tile_px + 0.5, scaled_tile_px + 0.5),
                );

                if let Some(tex) = self.textures.get(&key) {
                    visible.push((rect, tex.clone()));
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
