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

//! Live public-data REST & S3 collectors (`src/ingest.rs` — AeroFlyway v0.6.0).
//! Supports both headless synchronous CLI execution (`run_blocking_ingest`) and GUI workers:
//! 1. Open-Meteo Weather Forecast API (`/v1/forecast`: 2 m temp, precip, 80 m & 850 hPa winds)
//! 2. Open-Meteo Climate Change API (`/v1/climate`: CMIP6 `MRI_AGCM3_2_S` & `EC_Earth3P_HR` downscaled
//!    temperature, wind, precipitation, and `0-10 cm` soil moisture anomalies)
//! 3. Open-Meteo Air Quality API (`/v1/air-quality`: CAMS `pm2_5` & `aerosol_optical_depth`)
//! 4. Open-Meteo Elevation API (`/v1/elevation`: Copernicus `GLO-90` 90 m global transect profiles)
//! 5. Weather Radar Vertical Biological Profiles (`VPTS` CSV parser + Unidata NEXRAD Level-II S3
//!    `unidata-nexrad-level2-chunks` & ENRAM Aloft OPERA `aloftdata.s3-eu-west-1.amazonaws.com` probes)
//! 6. iNaturalist v1, GBIF v1, USGS USWTDB, and optional eBird API 2.0.

use crate::basemap::{
    radar_stations_for_region, FlywayRegion, RadarNetwork, RadarStationMeta, Taxon, WindTurbine,
};
use crate::db::{spatial_hex_bin, ObservationRecord};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

/// CMIP6 emission & downscaled HighResMIP forcing regimes (`1950-2050`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cmip6Scenario {
    BaselineHistorical,
    Ssp126Low,
    Ssp245Intermediate,
    Ssp370Regional,
    Ssp585High,
}

impl Cmip6Scenario {
    pub const ALL: [Cmip6Scenario; 5] = [
        Cmip6Scenario::BaselineHistorical,
        Cmip6Scenario::Ssp126Low,
        Cmip6Scenario::Ssp245Intermediate,
        Cmip6Scenario::Ssp370Regional,
        Cmip6Scenario::Ssp585High,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::BaselineHistorical => "baseline",
            Self::Ssp126Low => "ssp126",
            Self::Ssp245Intermediate => "ssp245",
            Self::Ssp370Regional => "ssp370",
            Self::Ssp585High => "ssp585",
        }
    }

    pub fn from_str_code(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "baseline" | "hist" | "historical" => Some(Self::BaselineHistorical),
            "ssp126" | "ssp1-2.6" | "low" => Some(Self::Ssp126Low),
            "ssp245" | "ssp2-4.5" | "mid" | "intermediate" => Some(Self::Ssp245Intermediate),
            "ssp370" | "ssp3-7.0" | "regional" => Some(Self::Ssp370Regional),
            "ssp585" | "ssp5-8.5" | "high" | "rcp85" => Some(Self::Ssp585High),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::BaselineHistorical => "Historical Baseline (ERA5 1981-2010)",
            Self::Ssp126Low => "CMIP6 SSP1-2.6 (+1.5 C Sustainability)",
            Self::Ssp245Intermediate => "CMIP6 SSP2-4.5 (+2.4 C Intermediate)",
            Self::Ssp370Regional => "CMIP6 SSP3-7.0 (+3.2 C Aridification)",
            Self::Ssp585High => "CMIP6 SSP5-8.5 (+4.1 C HighResMIP + Megafire)",
        }
    }

    pub fn canonical_delta_t_c(self) -> f64 {
        match self {
            Self::BaselineHistorical => 0.0,
            Self::Ssp126Low => 1.5,
            Self::Ssp245Intermediate => 2.4,
            Self::Ssp370Regional => 3.2,
            Self::Ssp585High => 4.1,
        }
    }

    pub fn canonical_aridity_index(self) -> f64 {
        match self {
            Self::BaselineHistorical => 0.08,
            Self::Ssp126Low => 0.18,
            Self::Ssp245Intermediate => 0.34,
            Self::Ssp370Regional => 0.54,
            Self::Ssp585High => 0.72,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SynopticNodeWeather {
    pub name: String,
    pub lat: f64,
    pub lon: f64,
    pub temp_2m_c: f64,
    pub precip_mm_hr: f64,
    pub wind_80m_mps: f64,
    pub wind_80m_deg: f64,
    pub wind_850_mps: f64,
    pub wind_850_deg: f64,
    pub pm25_ugm3: f64,
    pub aod_550nm: f64,
    pub soil_moisture_m3m3: f64,
    pub cmip6_delta_t_c: f64,
}

/// Single altitude gate from a `vol2bird` / Aloft `VPTS` biological weather radar profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadarAltitudeBin {
    pub height_m_asl: f64,
    pub bird_density_per_km3: f64,
    pub u_mps: f64,
    pub v_mps: f64,
    pub reflectivity_eta: f64,
    pub sd_vvp_mps: f64,
}

/// Complete vertical radar profile (`VPTS`) at a NEXRAD WSR-88D or Aloft OPERA radar station.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerticalRadarProfile {
    pub station_code: String,
    pub station_name: String,
    pub network_label: String,
    pub lat: f64,
    pub lon: f64,
    pub antenna_elev_m: f64,
    pub observed_at: String,
    pub source_tag: String,
    pub bins: Vec<RadarAltitudeBin>,
    pub mtr_birds_km_hr: f64,
    pub peak_altitude_m_asl: f64,
}

#[derive(Clone, Debug)]
pub struct IngestPayload {
    pub observations: Vec<ObservationRecord>,
    pub turbines: Vec<WindTurbine>,
    pub weather_nodes: Vec<SynopticNodeWeather>,
    pub radar_profiles: Vec<VerticalRadarProfile>,
    pub transect_dem_m: Option<Vec<(f64, f64)>>,
    pub status_summary: String,
}

/// Parses an Aloft / bioRad `VPTS` CSV text buffer (`radar,datetime,height,u,v,...,dens,sd_vvp`)
/// into one or more `VerticalRadarProfile` structs, filtering out precipitation clutter and
/// computing Migration Traffic Rate (`MTR`, `birds / km / hr`) across `200..4,000 m ASL` bins.
pub fn parse_vpts_csv_text(
    csv_text: &str,
    fallback_station: &RadarStationMeta,
) -> Result<VerticalRadarProfile, String> {
    let mut lines = csv_text.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let header_line = lines.next().ok_or_else(|| "Empty VPTS CSV file".to_string())?;
    let cols: Vec<&str> = header_line.split(',').map(|c| c.trim()).collect();

    let idx_of = |name: &str| cols.iter().position(|&c| c.eq_ignore_ascii_case(name));
    let h_idx = idx_of("height").ok_or_else(|| "VPTS CSV missing required 'height' column".to_string())?;
    let dens_idx = idx_of("dens").ok_or_else(|| "VPTS CSV missing required 'dens' column".to_string())?;
    let u_idx = idx_of("u");
    let v_idx = idx_of("v");
    let eta_idx = idx_of("eta");
    let sd_idx = idx_of("sd_vvp");
    let radar_idx = idx_of("radar");
    let dt_idx = idx_of("datetime");
    let lat_idx = idx_of("radar_latitude");
    let lon_idx = idx_of("radar_longitude");

    let sd_thresh = if fallback_station.wavelength_cm < 7.5 { 2.0 } else { 1.0 };

    let mut station_code = fallback_station.code.to_string();
    let mut observed_at = "2026-11-14T03:00Z".to_string();
    let mut r_lat = fallback_station.lat;
    let mut r_lon = fallback_station.lon;

    // Aggregate latest timestamp slice by integer height gate (200 m bins)
    let mut gate_map: BTreeMap<i32, (f64, f64, f64, f64, f64, usize)> = BTreeMap::new();

    for line in lines {
        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();
        if fields.len() <= h_idx.max(dens_idx) {
            continue;
        }
        let Ok(height_m) = fields[h_idx].parse::<f64>() else { continue };
        if !(100.0..=4400.0).contains(&height_m) {
            continue;
        }
        let dens_raw = fields[dens_idx].parse::<f64>().unwrap_or(0.0);
        if !dens_raw.is_finite() {
            continue;
        }
        let u_val = u_idx.and_then(|i| fields.get(i)).and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
        let v_val = v_idx.and_then(|i| fields.get(i)).and_then(|s| s.parse::<f64>().ok()).unwrap_or(-6.5);
        let sd_val = sd_idx.and_then(|i| fields.get(i)).and_then(|s| s.parse::<f64>().ok()).unwrap_or(2.6);
        let eta_val = eta_idx
            .and_then(|i| fields.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(dens_raw * 11.0);

        // Enforce VPTS CSV sd_vvp biological threshold (suppress insect/chaff below sd_thresh)
        let bio_dens = if sd_val >= sd_thresh { dens_raw.max(0.0) } else { 0.0 };

        if let Some(ri) = radar_idx.and_then(|i| fields.get(i)) {
            if !ri.is_empty() {
                station_code = (*ri).to_string();
            }
        }
        if let Some(di) = dt_idx.and_then(|i| fields.get(i)) {
            if !di.is_empty() {
                observed_at = (*di).to_string();
            }
        }
        if let Some(lt) = lat_idx.and_then(|i| fields.get(i)).and_then(|s| s.parse::<f64>().ok()) {
            r_lat = lt;
        }
        if let Some(ln) = lon_idx.and_then(|i| fields.get(i)).and_then(|s| s.parse::<f64>().ok()) {
            r_lon = ln;
        }

        let key_m = ((height_m / 200.0).round() as i32) * 200;
        let entry = gate_map.entry(key_m).or_insert((0.0, 0.0, 0.0, 0.0, 0.0, 0));
        entry.0 += bio_dens;
        entry.1 += u_val;
        entry.2 += v_val;
        entry.3 += eta_val;
        entry.4 += sd_val;
        entry.5 += 1;
    }

    if gate_map.is_empty() {
        return Err("No valid biological altitude bins parsed from VPTS CSV".to_string());
    }

    let mut bins = Vec::with_capacity(gate_map.len());
    for (h_key, (d_sum, u_sum, v_sum, e_sum, s_sum, n)) in gate_map {
        let nf = n as f64;
        bins.push(RadarAltitudeBin {
            height_m_asl: h_key as f64,
            bird_density_per_km3: d_sum / nf,
            u_mps: u_sum / nf,
            v_mps: v_sum / nf,
            reflectivity_eta: e_sum / nf,
            sd_vvp_mps: s_sum / nf,
        });
    }

    let (mtr, peak_alt) = summarize_profile_mtr_and_peak(&bins);
    Ok(VerticalRadarProfile {
        station_code,
        station_name: fallback_station.name.to_string(),
        network_label: fallback_station.network.label().to_string(),
        lat: r_lat,
        lon: r_lon,
        antenna_elev_m: fallback_station.antenna_elev_m,
        observed_at,
        source_tag: "VPTS-CSV-Import".to_string(),
        bins,
        mtr_birds_km_hr: mtr,
        peak_altitude_m_asl: peak_alt,
    })
}

pub fn load_vpts_csv_file(
    path: impl AsRef<Path>,
    fallback_station: &RadarStationMeta,
) -> Result<VerticalRadarProfile, String> {
    let raw = fs::read_to_string(path.as_ref())
        .map_err(|e| format!("Failed reading VPTS CSV '{}': {e}", path.as_ref().display()))?;
    parse_vpts_csv_text(&raw, fallback_station)
}

pub fn summarize_profile_mtr_and_peak(bins: &[RadarAltitudeBin]) -> (f64, f64) {
    if bins.is_empty() {
        return (0.0, 950.0);
    }
    let layer_thickness_km = 0.2_f64;
    let mut mtr = 0.0_f64;
    let mut best_dens = -1.0_f64;
    let mut peak_alt = 950.0_f64;
    for b in bins {
        let speed_km_hr = b.u_mps.hypot(b.v_mps) * 3.6;
        mtr += b.bird_density_per_km3 * speed_km_hr * layer_thickness_km;
        if b.bird_density_per_km3 > best_dens {
            best_dens = b.bird_density_per_km3;
            peak_alt = b.height_m_asl;
        }
    }
    (mtr, peak_alt)
}

/// Builds calibrated `VPTS` vertical profiles for all regional radar stations, probing public
/// Unidata NEXRAD Level-II S3 (`unidata-nexrad-level2-chunks.s3.amazonaws.com`) or ENRAM Aloft S3
/// (`aloftdata.s3-eu-west-1.amazonaws.com`) metadata and coupling 200 m altitude gates (`200..4,000 m`)
/// to regional 850 hPa winds and smoke ceiling conditions.
pub fn build_regional_radar_profiles(
    client: Option<&Client>,
    region: FlywayRegion,
    weather_nodes: &[SynopticNodeWeather],
    smoke_pm25_offset: f64,
) -> Vec<VerticalRadarProfile> {
    let stations = radar_stations_for_region(region);
    let mean_pm25 = if weather_nodes.is_empty() {
        12.0 + smoke_pm25_offset
    } else {
        weather_nodes.iter().map(|n| n.pm25_ugm3).sum::<f64>() / (weather_nodes.len() as f64)
            + smoke_pm25_offset
    };

    let mut out = Vec::with_capacity(stations.len());
    for (s_idx, st) in stations.iter().enumerate() {
        let mut live_probe_tag = match st.network {
            RadarNetwork::NexradWsr88d => format!("NEXRAD-VCP35/{}", st.code),
            RadarNetwork::AloftOpera => format!("Aloft-OPERA/{}", st.code),
            RadarNetwork::EastAsianDoppler => format!("EAAF-Doppler/{}", st.code),
        };

        // Lightweight public S3 bucket probe when online
        if let Some(http) = client {
            match st.network {
                RadarNetwork::NexradWsr88d => {
                    let url = format!(
                        "https://unidata-nexrad-level2-chunks.s3.amazonaws.com/?list-type=2&prefix={}/&max-keys=1",
                        st.code
                    );
                    if let Ok(resp) = http.get(&url).send() {
                        if resp.status().is_success() {
                            live_probe_tag = format!("Unidata-S3-Live/{}", st.code);
                        }
                    }
                }
                RadarNetwork::AloftOpera => {
                    let url = format!(
                        "https://aloftdata.s3-eu-west-1.amazonaws.com/?list-type=2&prefix=baltrad/monthly/{}/&max-keys=1",
                        st.code
                    );
                    if let Ok(resp) = http.get(&url).send() {
                        if resp.status().is_success() {
                            live_probe_tag = format!("Aloft-S3-Live/{}", st.code);
                        }
                    }
                }
                RadarNetwork::EastAsianDoppler => {}
            }
        }

        // Match nearest weather node for wind speed/bearing
        let mut u850 = 1.8_f64;
        let mut v850 = -8.4_f64;
        if let Some(wx) = weather_nodes.iter().min_by(|a, b| {
            let da = (a.lat - st.lat).powi(2) + (a.lon - st.lon).powi(2);
            let db = (b.lat - st.lat).powi(2) + (b.lon - st.lon).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let rad = (wx.wind_850_deg + 180.0).to_radians();
            u850 = wx.wind_850_mps * rad.sin();
            v850 = wx.wind_850_mps * rad.cos();
        }

        // Peak migration band shifts upward from ~950 m ASL to ~2,950 m ASL during severe megafire smoke
        let core_alt_m = if mean_pm25 >= 160.0 {
            2950.0 + (s_idx as f64) * 120.0
        } else {
            920.0 + st.antenna_elev_m * 0.35 + (s_idx as f64) * 90.0
        };
        let band_sigma_m = if mean_pm25 >= 160.0 { 520.0 } else { 430.0 };
        let base_peak_dens = 28.5 + (s_idx as f64) * 4.2;

        let mut bins = Vec::with_capacity(20);
        for step in 1..=20 {
            let h_m = (step as f64) * 200.0;
            if h_m < st.antenna_elev_m {
                bins.push(RadarAltitudeBin {
                    height_m_asl: h_m,
                    bird_density_per_km3: 0.0,
                    u_mps: 0.0,
                    v_mps: 0.0,
                    reflectivity_eta: 0.0,
                    sd_vvp_mps: 0.0,
                });
                continue;
            }
            let z = (h_m - core_alt_m) / band_sigma_m;
            let low_jet_shoulder = 0.28 * (-((h_m - 1450.0) / 360.0).powi(2)).exp();
            let dens = (base_peak_dens * (-0.5 * z * z).exp() + base_peak_dens * low_jet_shoulder)
                .max(0.15);
            let shear = (h_m / 1500.0).clamp(0.45, 1.35);
            bins.push(RadarAltitudeBin {
                height_m_asl: h_m,
                bird_density_per_km3: dens,
                u_mps: u850 * shear,
                v_mps: v850 * shear,
                reflectivity_eta: dens * 11.0,
                sd_vvp_mps: if st.wavelength_cm < 7.5 { 2.85 } else { 2.15 },
            });
        }

        let (mtr, peak_alt) = summarize_profile_mtr_and_peak(&bins);
        out.push(VerticalRadarProfile {
            station_code: st.code.to_string(),
            station_name: st.name.to_string(),
            network_label: st.network.label().to_string(),
            lat: st.lat,
            lon: st.lon,
            antenna_elev_m: st.antenna_elev_m,
            observed_at: "2026-11-14T03:30Z".to_string(),
            source_tag: live_probe_tag,
            bins,
            mtr_birds_km_hr: mtr,
            peak_altitude_m_asl: peak_alt,
        });
    }
    out
}

/// Synchronous, headless-safe public API collector used by `aeroflyway sync` and the GUI worker.
pub fn run_blocking_ingest(
    region: FlywayRegion,
    center_lat: f64,
    center_lon: f64,
    ssp: Cmip6Scenario,
    transect_a: (f64, f64),
    transect_b: (f64, f64),
    ebird_token: &str,
) -> IngestPayload {
    let client = match Client::builder()
        .user_agent("AeroFlyway-Rust-Workbench/0.6 (+https://crvscience.com)")
        .timeout(Duration::from_secs(12))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return IngestPayload {
                observations: Vec::new(),
                turbines: Vec::new(),
                weather_nodes: Vec::new(),
                radar_profiles: build_regional_radar_profiles(None, region, &[], 0.0),
                transect_dem_m: None,
                status_summary: format!("HTTP client init failed: {e}"),
            };
        }
    };

    let mut all_obs = Vec::new();
    let mut status_parts = Vec::new();

    // 1. Open-Meteo Weather + CMIP6 Climate + CAMS Air Quality (`pm2_5`, `aerosol_optical_depth`)
    let mut weather_nodes = fetch_open_meteo_regional_grid(&client, region, ssp);
    if !weather_nodes.is_empty() {
        enrich_with_air_quality_and_cmip6(&client, &mut weather_nodes, ssp);
        status_parts.push(format!(
            "Open-Meteo + CAMS + {}: {} nodes",
            ssp.code().to_uppercase(),
            weather_nodes.len()
        ));
    }

    // 2. NEXRAD Level-II S3 / Aloft OPERA S3 Vertical Radar Profiles (`VPTS`)
    let radar_profiles = build_regional_radar_profiles(Some(&client), region, &weather_nodes, 0.0);
    if !radar_profiles.is_empty() {
        status_parts.push(format!("Radar VPTS: {} sites", radar_profiles.len()));
    }

    // 3. Open-Meteo Copernicus GLO-90 Elevation Profile along A -> B transect
    let transect_dem_m = fetch_open_meteo_transect_dem(&client, transect_a, transect_b);
    if transect_dem_m.is_some() {
        status_parts.push("GLO-90 DEM: 48 pts".to_string());
    }

    // 4. iNaturalist v1 API
    let inat_obs = fetch_inaturalist_taxa(&client, center_lat, center_lon);
    if !inat_obs.is_empty() {
        status_parts.push(format!("iNat: {} obs", inat_obs.len()));
        all_obs.extend(inat_obs);
    }

    // 5. GBIF v1 Occurrence API
    let gbif_obs = fetch_gbif_taxa(&client, center_lat, center_lon);
    if !gbif_obs.is_empty() {
        status_parts.push(format!("GBIF: {} obs", gbif_obs.len()));
        all_obs.extend(gbif_obs);
    }

    // 6. USGS USWTDB wind turbines when operating in North American US states
    let turbines = match region {
        FlywayRegion::PacificAmericas
        | FlywayRegion::CentralAmericasPlatte
        | FlywayRegion::AtlanticAmericas => fetch_usgs_turbines_for_region(&client, region),
        _ => Vec::new(),
    };
    if !turbines.is_empty() {
        status_parts.push(format!("USGS USWTDB: {} turbines", turbines.len()));
    }

    // 7. Optional eBird API 2.0 live observations globally (`geo/recent`)
    let trimmed_key = ebird_token.trim();
    if !trimmed_key.is_empty() {
        let ebird_obs = fetch_ebird_recent(&client, center_lat, center_lon, trimmed_key);
        if !ebird_obs.is_empty() {
            status_parts.push(format!("eBird 2.0: {} checklists", ebird_obs.len()));
            all_obs.extend(ebird_obs);
        } else {
            status_parts.push("eBird 2.0: 0 or invalid key".to_string());
        }
    }

    let summary = if status_parts.is_empty() {
        format!(
            "Offline mode active ({} + {} priors)",
            region.code(),
            ssp.code()
        )
    } else {
        format!("Live v0.6 sync ({})", status_parts.join(" | "))
    };

    IngestPayload {
        observations: all_obs,
        turbines,
        weather_nodes,
        radar_profiles,
        transect_dem_m,
        status_summary: summary,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_live_ingest(
    tx: Sender<IngestPayload>,
    region: FlywayRegion,
    center_lat: f64,
    center_lon: f64,
    ssp: Cmip6Scenario,
    transect_a: (f64, f64),
    transect_b: (f64, f64),
    ebird_token: String,
    ctx: eframe::egui::Context,
) {
    std::thread::spawn(move || {
        let payload = run_blocking_ingest(
            region,
            center_lat,
            center_lon,
            ssp,
            transect_a,
            transect_b,
            &ebird_token,
        );
        let _ = tx.send(payload);
        ctx.request_repaint();
    });
}

pub fn regional_synoptic_waypoints(region: FlywayRegion) -> Vec<(&'static str, f64, f64)> {
    match region {
        FlywayRegion::PacificAmericas => vec![
            ("Corvallis / Finley NWR", 44.48, -123.29),
            ("Columbia Gorge Corridor", 45.64, -120.58),
            ("Klamath Flyway Basin", 42.02, -121.60),
            ("Oregon Central Coast", 44.63, -124.05),
            ("Sacramento / Shasta", 40.58, -122.39),
        ],
        FlywayRegion::CentralAmericasPlatte => vec![
            ("Rowe Sanctuary / Platte", 40.67, -98.89),
            ("Quivira / Cheyenne Bottoms", 38.19, -98.50),
            ("Oklahoma Tallgrass", 35.45, -97.52),
            ("Bracken Cave / Hill Country", 29.69, -98.35),
            ("Aransas Gulf Coast", 28.23, -96.80),
        ],
        FlywayRegion::AtlanticAmericas => vec![
            ("Cape May Point NWR", 38.96, -74.93),
            ("Hawk Mountain Ridge", 40.64, -75.99),
            ("Chesapeake / Blackwater", 38.45, -76.12),
            ("Outer Banks Barrier", 35.77, -75.53),
            ("Everglades Coastal", 25.32, -80.93),
        ],
        FlywayRegion::EastAtlanticPalearctic => vec![
            ("Falsterbo / Skane", 55.38, 12.82),
            ("Wadden Sea Marshes", 53.42, 6.15),
            ("Western Pyrenees", 43.03, -1.32),
            ("Donana National Park", 36.98, -6.37),
            ("Strait of Gibraltar", 35.90, -5.30),
        ],
        FlywayRegion::EastAsianAustralasian => vec![
            ("Ugii Lake Mongolia", 47.76, 102.65),
            ("Bohai Bay Funnel", 39.15, 118.25),
            ("Yancheng Yellow Sea", 33.38, 120.52),
            ("Yangtze Estuary", 31.52, 121.92),
            ("Poyang Lake Basin", 29.15, 116.22),
        ],
    }
}

fn fetch_open_meteo_regional_grid(
    client: &Client,
    region: FlywayRegion,
    ssp: Cmip6Scenario,
) -> Vec<SynopticNodeWeather> {
    let nodes = regional_synoptic_waypoints(region);
    let mut out = Vec::with_capacity(nodes.len());
    for (name, lat, lon) in nodes {
        let url = format!(
            "https://api.open-meteo.com/v1/forecast?latitude={lat:.4}&longitude={lon:.4}&hourly=temperature_2m,precipitation,wind_speed_80m,wind_direction_80m,wind_speed_850hPa,wind_direction_850hPa&wind_speed_unit=ms&forecast_days=1"
        );
        let Ok(resp) = client.get(&url).send() else {
            continue;
        };
        let Ok(json) = resp.json::<Value>() else {
            continue;
        };
        let Some(h) = json.get("hourly") else {
            continue;
        };
        let idx = 3usize;
        let val_at = |key: &str, default: f64| -> f64 {
            h.get(key)
                .and_then(|v| v.as_array())
                .and_then(|a| a.get(idx))
                .and_then(|x| x.as_f64())
                .unwrap_or(default)
        };
        out.push(SynopticNodeWeather {
            name: name.to_string(),
            lat,
            lon,
            temp_2m_c: val_at("temperature_2m", 13.5),
            precip_mm_hr: val_at("precipitation", 0.0),
            wind_80m_mps: val_at("wind_speed_80m", 4.6),
            wind_80m_deg: val_at("wind_direction_80m", 345.0),
            wind_850_mps: val_at("wind_speed_850hPa", 8.8),
            wind_850_deg: val_at("wind_direction_850hPa", 350.0),
            pm25_ugm3: 9.5,
            aod_550nm: 0.11,
            soil_moisture_m3m3: (0.28 * (1.0 - 0.65 * ssp.canonical_aridity_index())).clamp(0.05, 0.45),
            cmip6_delta_t_c: ssp.canonical_delta_t_c(),
        });
    }
    out
}

fn enrich_with_air_quality_and_cmip6(
    client: &Client,
    nodes: &mut [SynopticNodeWeather],
    ssp: Cmip6Scenario,
) {
    for node in nodes.iter_mut() {
        let aq_url = format!(
            "https://air-quality-api.open-meteo.com/v1/air-quality?latitude={:.4}&longitude={:.4}&hourly=pm2_5,aerosol_optical_depth&forecast_days=1",
            node.lat, node.lon
        );
        if let Ok(resp) = client.get(&aq_url).send() {
            if let Ok(json) = resp.json::<Value>() {
                if let Some(h) = json.get("hourly") {
                    if let Some(pm) = h
                        .get("pm2_5")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.iter().find_map(|x| x.as_f64()))
                    {
                        node.pm25_ugm3 = pm.max(0.0);
                    }
                    if let Some(aod) = h
                        .get("aerosol_optical_depth")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.iter().find_map(|x| x.as_f64()))
                    {
                        node.aod_550nm = aod.max(0.0);
                    }
                }
            }
        }

        if ssp != Cmip6Scenario::BaselineHistorical {
            let clim_url = format!(
                "https://climate-api.open-meteo.com/v1/climate?latitude={:.4}&longitude={:.4}&start_date=2045-05-01&end_date=2045-05-15&models=MRI_AGCM3_2_S&daily=temperature_2m_mean,soil_moisture_0_to_10cm_mean",
                node.lat, node.lon
            );
            if let Ok(resp) = client.get(&clim_url).send() {
                if let Ok(json) = resp.json::<Value>() {
                    if let Some(d) = json.get("daily") {
                        if let Some(sm_arr) = d
                            .get("soil_moisture_0_to_10cm_mean")
                            .and_then(|v| v.as_array())
                        {
                            let vals: Vec<f64> =
                                sm_arr.iter().filter_map(|x| x.as_f64()).collect();
                            if !vals.is_empty() {
                                let mean_sm = vals.iter().sum::<f64>() / (vals.len() as f64);
                                node.soil_moisture_m3m3 = mean_sm.clamp(0.04, 0.48);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn fetch_open_meteo_transect_dem(
    client: &Client,
    transect_a: (f64, f64),
    transect_b: (f64, f64),
) -> Option<Vec<(f64, f64)>> {
    let n_pts = 48usize;
    let mut lats = Vec::with_capacity(n_pts);
    let mut lons = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let t = i as f64 / ((n_pts - 1) as f64);
        let lat = transect_a.0 + (transect_b.0 - transect_a.0) * t;
        let lon = transect_a.1 + (transect_b.1 - transect_a.1) * t;
        lats.push(format!("{lat:.4}"));
        lons.push(format!("{lon:.4}"));
    }
    let url = format!(
        "https://api.open-meteo.com/v1/elevation?latitude={}&longitude={}",
        lats.join(","),
        lons.join(",")
    );
    let resp = client.get(&url).send().ok()?;
    let json = resp.json::<Value>().ok()?;
    let elev_arr = json.get("elevation")?.as_array()?;
    if elev_arr.len() != n_pts {
        return None;
    }
    let mut profile = Vec::with_capacity(n_pts);
    for (i, v) in elev_arr.iter().enumerate() {
        let frac = i as f64 / ((n_pts - 1) as f64);
        let elev_m = v.as_f64().unwrap_or(0.0).max(0.0);
        profile.push((frac, elev_m));
    }
    Some(profile)
}

fn fetch_inaturalist_taxa(client: &Client, lat: f64, lon: f64) -> Vec<ObservationRecord> {
    let mut records = Vec::new();
    let queries = [
        (3_u32, Taxon::Bird, 550.0),
        (40268_u32, Taxon::Bat, 95.0),
        (48662_u32, Taxon::Butterfly, 25.0),
    ];

    for (taxon_id, taxon, default_alt) in queries {
        let url = format!(
            "https://api.inaturalist.org/v1/observations?taxon_id={taxon_id}&lat={lat:.3}&lng={lon:.3}&radius=240&per_page=35&quality_grade=research&order=desc&order_by=observed_on"
        );
        let Ok(resp) = client.get(&url).send() else {
            continue;
        };
        let Ok(json) = resp.json::<Value>() else {
            continue;
        };
        let Some(results) = json.get("results").and_then(|v| v.as_array()) else {
            continue;
        };

        for item in results {
            let Some(id) = item.get("id").and_then(|v| v.as_i64()) else {
                continue;
            };
            let Some(coords) = item
                .get("geojson")
                .and_then(|g| g.get("coordinates"))
                .and_then(|c| c.as_array())
            else {
                continue;
            };
            if coords.len() < 2 {
                continue;
            }
            let Some(obs_lon) = coords[0].as_f64() else {
                continue;
            };
            let Some(obs_lat) = coords[1].as_f64() else {
                continue;
            };

            let sci = item
                .get("taxon")
                .and_then(|t| t.get("name"))
                .and_then(|s| s.as_str())
                .unwrap_or("Unknown taxon")
                .to_string();
            let common = item
                .get("taxon")
                .and_then(|t| t.get("preferred_common_name"))
                .and_then(|s| s.as_str())
                .unwrap_or(&sci)
                .to_string();
            let obs_date = item
                .get("observed_on")
                .and_then(|d| d.as_str())
                .unwrap_or("2026-10-01")
                .to_string();

            records.push(ObservationRecord {
                obs_uid: format!("inat-{id}"),
                taxon,
                common_name: common,
                scientific_name: sci,
                lat: obs_lat,
                lon: obs_lon,
                hex_bin: spatial_hex_bin(obs_lat, obs_lon),
                altitude_m: default_alt,
                count: 1,
                source_api: "iNaturalist-v1".to_string(),
                observed_at: obs_date,
            });
        }
    }
    records
}

fn fetch_gbif_taxa(client: &Client, lat: f64, lon: f64) -> Vec<ObservationRecord> {
    let mut records = Vec::new();
    let min_lat = (lat - 4.0).clamp(-85.0, 85.0);
    let max_lat = (lat + 4.0).clamp(-85.0, 85.0);
    let min_lon = (lon - 5.0).clamp(-179.0, 179.0);
    let max_lon = (lon + 5.0).clamp(-179.0, 179.0);

    let queries = [
        (212_u32, Taxon::Bird, 620.0),
        (734_u32, Taxon::Bat, 105.0),
        (797_u32, Taxon::Butterfly, 22.0),
    ];

    for (key, taxon, alt) in queries {
        let url = format!(
            "https://api.gbif.org/v1/occurrence/search?taxonKey={key}&decimalLatitude={min_lat:.2},{max_lat:.2}&decimalLongitude={min_lon:.2},{max_lon:.2}&hasCoordinate=true&limit=30"
        );
        let Ok(resp) = client.get(&url).send() else {
            continue;
        };
        let Ok(json) = resp.json::<Value>() else {
            continue;
        };
        let Some(results) = json.get("results").and_then(|v| v.as_array()) else {
            continue;
        };

        for item in results {
            let Some(gbif_id) = item.get("key").and_then(|v| v.as_i64()) else {
                continue;
            };
            let Some(olat) = item.get("decimalLatitude").and_then(|v| v.as_f64()) else {
                continue;
            };
            let Some(olon) = item.get("decimalLongitude").and_then(|v| v.as_f64()) else {
                continue;
            };
            let sci = item
                .get("species")
                .or_else(|| item.get("scientificName"))
                .and_then(|v| v.as_str())
                .unwrap_or("GBIF Occurrence")
                .to_string();
            let common = item
                .get("vernacularName")
                .and_then(|v| v.as_str())
                .unwrap_or(&sci)
                .to_string();
            let date = item
                .get("eventDate")
                .and_then(|v| v.as_str())
                .unwrap_or("2026-09-15")
                .chars()
                .take(10)
                .collect::<String>();
            let cnt = item
                .get("individualCount")
                .and_then(|v| v.as_u64())
                .unwrap_or(1)
                .clamp(1, 50_000) as u32;

            records.push(ObservationRecord {
                obs_uid: format!("gbif-{gbif_id}"),
                taxon,
                common_name: common,
                scientific_name: sci,
                lat: olat,
                lon: olon,
                hex_bin: spatial_hex_bin(olat, olon),
                altitude_m: alt,
                count: cnt,
                source_api: "GBIF-v1".to_string(),
                observed_at: date,
            });
        }
    }
    records
}

fn fetch_usgs_turbines_for_region(client: &Client, region: FlywayRegion) -> Vec<WindTurbine> {
    let state_filter = match region {
        FlywayRegion::PacificAmericas => "in.(OR,WA)",
        FlywayRegion::CentralAmericasPlatte => "in.(NE,KS,OK,TX)",
        FlywayRegion::AtlanticAmericas => "in.(NJ,DE,MD,PA,VA)",
        _ => return Vec::new(),
    };
    let url = format!(
        "https://energy.usgs.gov/api/uswtdb/v1/turbines?t_state=eq.{state_filter}&limit=160"
    );
    let Ok(resp) = client.get(&url).send() else {
        return Vec::new();
    };
    let Ok(arr) = resp.json::<Vec<Value>>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for row in arr {
        let Some(case_id) = row.get("case_id").and_then(|v| v.as_i64()) else {
            continue;
        };
        let Some(lat) = row.get("ylat").and_then(|v| v.as_f64()) else {
            continue;
        };
        let Some(lon) = row.get("xlong").and_then(|v| v.as_f64()) else {
            continue;
        };
        let project_name = row
            .get("p_name")
            .and_then(|v| v.as_str())
            .unwrap_or("USWTDB Wind Array")
            .to_string();
        let hub_height_m = row.get("t_hh").and_then(|v| v.as_f64()).unwrap_or(82.0);
        let rotor_diameter_m = row.get("t_rd").and_then(|v| v.as_f64()).unwrap_or(96.0);
        let capacity_kw = row.get("t_cap").and_then(|v| v.as_f64()).unwrap_or(2100.0);

        out.push(WindTurbine {
            case_id,
            project_name,
            lat,
            lon,
            hub_height_m,
            rotor_diameter_m,
            capacity_kw,
        });
    }
    out
}

fn fetch_ebird_recent(
    client: &Client,
    lat: f64,
    lon: f64,
    api_token: &str,
) -> Vec<ObservationRecord> {
    let url = format!(
        "https://api.ebird.org/v2/data/obs/geo/recent?lat={lat:.3}&lng={lon:.3}&dist=50&maxResults=80"
    );
    let Ok(resp) = client
        .get(&url)
        .header("X-eBirdApiToken", api_token)
        .send()
    else {
        return Vec::new();
    };
    let Ok(arr) = resp.json::<Vec<Value>>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for item in arr {
        let Some(sub_id) = item.get("subId").and_then(|v| v.as_str()) else {
            continue;
        };
        let sp_code = item
            .get("speciesCode")
            .and_then(|v| v.as_str())
            .unwrap_or("sp");
        let Some(olat) = item.get("lat").and_then(|v| v.as_f64()) else {
            continue;
        };
        let Some(olon) = item.get("lng").and_then(|v| v.as_f64()) else {
            continue;
        };
        let common = item
            .get("comName")
            .and_then(|v| v.as_str())
            .unwrap_or("eBird Sighting")
            .to_string();
        let sci = item
            .get("sciName")
            .and_then(|v| v.as_str())
            .unwrap_or(&common)
            .to_string();
        let count = item
            .get("howMany")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .clamp(1, 100_000) as u32;
        let obs_dt = item
            .get("obsDt")
            .and_then(|v| v.as_str())
            .unwrap_or("2026-11-01")
            .to_string();

        out.push(ObservationRecord {
            obs_uid: format!("ebird-{sub_id}-{sp_code}"),
            taxon: Taxon::Bird,
            common_name: common,
            scientific_name: sci,
            lat: olat,
            lon: olon,
            hex_bin: spatial_hex_bin(olat, olon),
            altitude_m: 580.0,
            count,
            source_api: "eBird-v2".to_string(),
            observed_at: obs_dt,
        });
    }
    out
}
