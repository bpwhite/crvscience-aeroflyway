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

//! RFC 7946 GeoJSON FeatureCollection & Analytical CSV Corridor Telemetry Exporters (`export.rs` — v0.6.0).
//! Exports multi-taxa spatial H25 hexagonal corridor densities, 3D A->B airspace transect profiles,
//! VIIRS nighttime skyglow domes (`Lights-Out` radiance), NEXRAD/Aloft `VPTS` radar stations,
//! wind-turbine rotor envelopes, host-plant patches, sightings, and agent streamlines.

use crate::basemap::{
    sample_terrain_elevation_m, seed_milkweed_patches_for_region, seed_wind_turbines_for_region,
    skyglow_domes_for_region, FlywayRegion, MilkweedPatch, Taxon, WindTurbine,
};
use crate::db::{baseline_observations_for_region, hex_bin_center, spatial_hex_bin, ObservationRecord};
use crate::ingest::Cmip6Scenario;
use crate::map_engine::haversine_km;
use crate::physics::{
    default_synoptic_nodes_for_region, effective_skyglow_radiance_nw, AeroPhysicsEngine,
    SeasonPreset, SimulationParams,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
struct HexAgg {
    birds: usize,
    bats: usize,
    butterflies: usize,
    fallout: usize,
    skyglow_birds: usize,
    alt_sum_m: f64,
    gdd_sum: f64,
    max_gen: u8,
}

/// Writes an [RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) compliant GeoJSON
/// `FeatureCollection` (`WGS84` `[lon, lat, elev_m]`, counter-clockwise polygon rings, root `bbox`)
/// into `export_dir/aeroflyway_corridor.geojson`.
pub fn export_corridor_geojson(
    export_dir: impl AsRef<Path>,
    engine: &AeroPhysicsEngine,
    params: &SimulationParams,
    observations: &[ObservationRecord],
    turbines: &[WindTurbine],
    milkweed: &[MilkweedPatch],
    gpu_scale_factor: f64,
) -> Result<(PathBuf, usize), String> {
    let dir = export_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| format!("Create export dir failed: {e}"))?;
    let out_path = dir.join("aeroflyway_corridor.geojson");
    export_corridor_geojson_to_path(
        &out_path,
        engine,
        params,
        observations,
        turbines,
        milkweed,
        gpu_scale_factor,
    )
}

/// Writes an [RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) compliant GeoJSON
/// `FeatureCollection` directly to `out_path`.
pub fn export_corridor_geojson_to_path(
    out_path: impl AsRef<Path>,
    engine: &AeroPhysicsEngine,
    params: &SimulationParams,
    observations: &[ObservationRecord],
    turbines: &[WindTurbine],
    milkweed: &[MilkweedPatch],
    gpu_scale_factor: f64,
) -> Result<(PathBuf, usize), String> {
    let out_path = out_path.as_ref().to_path_buf();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("Create export dir failed: {e}"))?;
        }
    }

    let mut features: Vec<Value> = Vec::new();
    let mut min_lon = 180.0_f64;
    let mut max_lon = -180.0_f64;
    let mut min_lat = 90.0_f64;
    let mut max_lat = -90.0_f64;

    let mut track_bbox = |lon: f64, lat: f64| {
        min_lon = min_lon.min(lon);
        max_lon = max_lon.max(lon);
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
    };

    // 1. Active A -> B Vertical Airspace Cross-Section Transect (`LineString` with 3D terrain elevation)
    let (lat_a, lon_a) = params.transect_start;
    let (lat_b, lon_b) = params.transect_end;
    let span_km = haversine_km(lat_a, lon_a, lat_b, lon_b);
    let mut transect_coords = Vec::with_capacity(33);
    for i in 0..=32 {
        let t = i as f64 / 32.0;
        let lat = lat_a + (lat_b - lat_a) * t;
        let lon = lon_a + (lon_b - lon_a) * t;
        let elev = sample_terrain_elevation_m(lat, lon);
        track_bbox(lon, lat);
        transect_coords.push(json!([round6(lon), round6(lat), round1(elev)]));
    }
    features.push(json!({
        "type": "Feature",
        "id": "transect-ab",
        "geometry": {
            "type": "LineString",
            "coordinates": transect_coords,
        },
        "properties": {
            "feature_class": "airspace_transect",
            "region": params.region.code(),
            "doy": round1(engine.metrics.day_of_year),
            "cmip6": params.cmip6_scenario.code(),
            "lights_out_curfew_pct": round1(params.lights_out_curfew_pct),
            "span_km": round1(span_km),
            "swath_half_width_km": round1(params.transect_half_width_km),
        }
    }));

    // 2. Spatial H25 Hexagonal Corridor Polygons (counter-clockwise 7-point closed rings per RFC 7946)
    let hex_map = aggregate_hex_corridors(engine);
    let mut hex_keys: Vec<&String> = hex_map.keys().collect();
    hex_keys.sort();
    for hex_key in hex_keys {
        let agg = &hex_map[hex_key];
        let total_twin = agg.birds + agg.bats + agg.butterflies;
        if total_twin < 2 {
            continue;
        }
        let Some((clat, clon)) = hex_bin_center(hex_key) else {
            continue;
        };
        let r_lat = 0.112_f64;
        let r_lon = r_lat / clat.to_radians().cos().abs().max(0.25);

        let mut ring = Vec::with_capacity(7);
        for k in 0..6 {
            let ang = (k as f64) * TAU / 6.0;
            let vx = clon + r_lon * ang.cos();
            let vy = clat + r_lat * ang.sin();
            track_bbox(vx, vy);
            ring.push(json!([round6(vx), round6(vy)]));
        }
        ring.push(ring[0].clone());

        let mean_alt = agg.alt_sum_m / (total_twin as f64);
        let mean_gdd = if agg.butterflies > 0 {
            agg.gdd_sum / (agg.butterflies as f64)
        } else {
            0.0
        };
        features.push(json!({
            "type": "Feature",
            "id": hex_key,
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "feature_class": "corridor_hex_bin",
                "hex_bin": hex_key,
                "estimated_birds": ((agg.birds as f64) * gpu_scale_factor).round() as u64,
                "estimated_bats": ((agg.bats as f64) * gpu_scale_factor).round() as u64,
                "estimated_butterflies": ((agg.butterflies as f64) * gpu_scale_factor).round() as u64,
                "fallout_birds": ((agg.fallout as f64) * gpu_scale_factor).round() as u64,
                "skyglow_entrained_birds": ((agg.skyglow_birds as f64) * gpu_scale_factor).round() as u64,
                "mean_alt_agl_m": round1(mean_alt),
                "monarch_mean_gdd": round1(mean_gdd),
                "monarch_peak_gen": agg.max_gen,
            }
        }));
    }

    // 3. VIIRS Nighttime Skyglow Domes & Active Lights-Out Curfew Radiance
    for (idx, dome) in skyglow_domes_for_region(params.region).iter().enumerate() {
        track_bbox(dome.lon, dome.lat);
        let eff_nw = effective_skyglow_radiance_nw(
            dome.peak_radiance_nw_cm2_sr,
            engine.metrics.hour_of_day,
            engine.metrics.is_nocturnal,
            params.lights_out_curfew_pct,
            params.skyglow_intensity_mult,
        );
        features.push(json!({
            "type": "Feature",
            "id": format!("skyglow-{}", idx + 1),
            "geometry": {
                "type": "Point",
                "coordinates": [round6(dome.lon), round6(dome.lat)],
            },
            "properties": {
                "feature_class": "urban_skyglow_dome",
                "city_name": dome.city_name,
                "radius_km": round1(dome.radius_km),
                "peak_radiance_nw_cm2_sr": round1(dome.peak_radiance_nw_cm2_sr),
                "effective_radiance_nw_cm2_sr": round1(eff_nw),
                "lights_out_curfew_pct": round1(params.lights_out_curfew_pct),
                "glass_hazard_index": round1(dome.glass_hazard_index),
            }
        }));
    }

    // 4. NEXRAD WSR-88D & Aloft OPERA Vertical Radar Stations (`VPTS`)
    for prof in &engine.radar_profiles {
        track_bbox(prof.lon, prof.lat);
        features.push(json!({
            "type": "Feature",
            "id": format!("radar-{}", prof.station_code),
            "geometry": {
                "type": "Point",
                "coordinates": [round6(prof.lon), round6(prof.lat), round1(prof.antenna_elev_m)],
            },
            "properties": {
                "feature_class": "weather_radar_vpts",
                "station_code": prof.station_code,
                "station_name": prof.station_name,
                "network": prof.network_label,
                "mtr_birds_km_hr": round1(prof.mtr_birds_km_hr),
                "peak_altitude_m_asl": round1(prof.peak_altitude_m_asl),
                "source_tag": prof.source_tag,
            }
        }));
    }

    // 5. Milkweed Stepping-Stone Patches
    for patch in milkweed {
        track_bbox(patch.lon, patch.lat);
        features.push(json!({
            "type": "Feature",
            "id": format!("milkweed-{}", patch.id),
            "geometry": {
                "type": "Point",
                "coordinates": [round6(patch.lon), round6(patch.lat)],
            },
            "properties": {
                "feature_class": "milkweed_patch",
                "label": patch.label,
                "area_ha": round1(patch.area_ha),
                "is_virtual_counterfactual": patch.is_virtual,
                "gdd_threshold": round1(patch.gdd_threshold),
            }
        }));
    }

    // 6. Wind Turbines & Smart Curtailment Parameters
    for turb in turbines {
        track_bbox(turb.lon, turb.lat);
        features.push(json!({
            "type": "Feature",
            "id": format!("turbine-{}", turb.case_id),
            "geometry": {
                "type": "Point",
                "coordinates": [round6(turb.lon), round6(turb.lat), round1(turb.hub_height_m)],
            },
            "properties": {
                "feature_class": "wind_turbine",
                "project_name": turb.project_name,
                "hub_height_m": round1(turb.hub_height_m),
                "rotor_diameter_m": round1(turb.rotor_diameter_m),
                "capacity_kw": round1(turb.capacity_kw),
                "active_cut_in_mps": round1(params.turbine_cut_in_mps),
            }
        }));
    }

    // 7. Public API + Pinned Bioacoustic Observations
    for obs in observations {
        track_bbox(obs.lon, obs.lat);
        features.push(json!({
            "type": "Feature",
            "id": obs.obs_uid,
            "geometry": {
                "type": "Point",
                "coordinates": [round6(obs.lon), round6(obs.lat), round1(obs.altitude_m)],
            },
            "properties": {
                "feature_class": "taxa_observation",
                "taxon": obs.taxon.code(),
                "common_name": obs.common_name,
                "scientific_name": obs.scientific_name,
                "count": obs.count,
                "source_api": obs.source_api,
                "observed_at": obs.observed_at,
                "hex_bin": obs.hex_bin,
            }
        }));
    }

    // 8. Representative Multi-Taxa Agent Streamline Trails (up to 180 LineStrings)
    for ag in engine.agents.iter().step_by(12).take(180) {
        if ag.trail.len() >= 2 {
            let coords: Vec<Value> = ag
                .trail
                .iter()
                .map(|&(lt, ln)| {
                    track_bbox(ln, lt);
                    json!([round6(ln), round6(lt), round1(ag.alt_agl_m)])
                })
                .collect();
            features.push(json!({
                "type": "Feature",
                "id": format!("trajectory-{}", ag.id),
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "feature_class": "agent_trajectory",
                    "taxon": ag.taxon.code(),
                    "alt_agl_m": round1(ag.alt_agl_m),
                    "generation": ag.generation,
                    "gdd_accum": round1(ag.gdd_accum),
                    "in_fallout": ag.in_fallout_stopover,
                    "skyglow_entrained": ag.skyglow_entrained,
                }
            }));
        }
    }

    let feature_count = features.len();
    let bbox = if min_lon <= max_lon && min_lat <= max_lat {
        json!([round6(min_lon), round6(min_lat), round6(max_lon), round6(max_lat)])
    } else {
        json!([-124.8, 34.2, -116.8, 49.0])
    };

    let root = json!({
        "type": "FeatureCollection",
        "bbox": bbox,
        "features": features,
    });

    let serialized =
        serde_json::to_string_pretty(&root).map_err(|e| format!("GeoJSON serialize error: {e}"))?;
    fs::write(&out_path, serialized).map_err(|e| format!("Write GeoJSON failed: {e}"))?;

    Ok((out_path, feature_count))
}

/// Writes analytical CSV tables (`aeroflyway_corridor_telemetry.csv` and
/// `aeroflyway_observations_and_acoustics.csv`) into `export_dir`.
pub fn export_corridor_csv_bundle(
    export_dir: impl AsRef<Path>,
    engine: &AeroPhysicsEngine,
    params: &SimulationParams,
    observations: &[ObservationRecord],
    gpu_scale_factor: f64,
) -> Result<(PathBuf, PathBuf, usize), String> {
    let dir = export_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| format!("Create export dir failed: {e}"))?;

    let hex_csv_path = dir.join("aeroflyway_corridor_telemetry.csv");
    let obs_csv_path = dir.join("aeroflyway_observations_and_acoustics.csv");

    let hex_map = aggregate_hex_corridors(engine);
    let mut hex_keys: Vec<&String> = hex_map.keys().collect();
    hex_keys.sort();

    let mut hex_out = String::from(
        "hex_bin,center_lat,center_lon,region,doy,est_birds,est_bats,est_butterflies,fallout_birds,skyglow_entrained_birds,mean_alt_agl_m,monarch_mean_gdd,monarch_peak_gen,turbine_cut_in_mps,lights_out_curfew_pct,radar_mtr_birds_km_hr,enkf_gain,enkf_rmse\n",
    );
    let mut row_count = 0usize;

    for key in hex_keys {
        let agg = &hex_map[key];
        let total = agg.birds + agg.bats + agg.butterflies;
        if total == 0 {
            continue;
        }
        let (clat, clon) = hex_bin_center(key).unwrap_or((0.0, 0.0));
        let mean_alt = agg.alt_sum_m / (total as f64);
        let mean_gdd = if agg.butterflies > 0 {
            agg.gdd_sum / (agg.butterflies as f64)
        } else {
            0.0
        };
        hex_out.push_str(&format!(
            "{},{:.5},{:.5},{},{:.0},{},{},{},{},{},{:.1},{:.1},{},{:.2},{:.1},{:.1},{:.3},{:.4}\n",
            csv_escape(key),
            clat,
            clon,
            params.region.code(),
            engine.metrics.day_of_year,
            ((agg.birds as f64) * gpu_scale_factor).round() as u64,
            ((agg.bats as f64) * gpu_scale_factor).round() as u64,
            ((agg.butterflies as f64) * gpu_scale_factor).round() as u64,
            ((agg.fallout as f64) * gpu_scale_factor).round() as u64,
            ((agg.skyglow_birds as f64) * gpu_scale_factor).round() as u64,
            mean_alt,
            mean_gdd,
            agg.max_gen,
            params.turbine_cut_in_mps,
            params.lights_out_curfew_pct,
            engine.metrics.radar_mtr_birds_km_hr,
            engine.metrics.enkf_gain,
            engine.metrics.enkf_rmse,
        ));
        row_count += 1;
    }
    fs::write(&hex_csv_path, hex_out).map_err(|e| format!("Write corridor CSV failed: {e}"))?;

    let mut obs_out = String::from(
        "obs_uid,taxon,common_name,scientific_name,lat,lon,hex_bin,altitude_m,count,source_api,observed_at\n",
    );
    for o in observations {
        obs_out.push_str(&format!(
            "{},{},{},{},{:.5},{:.5},{},{:.1},{},{},{}\n",
            csv_escape(&o.obs_uid),
            o.taxon.code(),
            csv_escape(&o.common_name),
            csv_escape(&o.scientific_name),
            o.lat,
            o.lon,
            csv_escape(&o.hex_bin),
            o.altitude_m,
            o.count,
            csv_escape(&o.source_api),
            csv_escape(&o.observed_at),
        ));
        row_count += 1;
    }
    fs::write(&obs_csv_path, obs_out).map_err(|e| format!("Write observations CSV failed: {e}"))?;

    Ok((hex_csv_path, obs_csv_path, row_count))
}

fn aggregate_hex_corridors(engine: &AeroPhysicsEngine) -> HashMap<String, HexAgg> {
    let mut map: HashMap<String, HexAgg> = HashMap::new();
    for ag in &engine.agents {
        let key = spatial_hex_bin(ag.lat, ag.lon);
        let entry = map.entry(key).or_default();
        match ag.taxon {
            Taxon::Bird => {
                entry.birds += 1;
                if ag.in_fallout_stopover {
                    entry.fallout += 1;
                }
                if ag.skyglow_entrained {
                    entry.skyglow_birds += 1;
                }
            }
            Taxon::Bat => entry.bats += 1,
            Taxon::Butterfly => {
                entry.butterflies += 1;
                entry.gdd_sum += ag.gdd_accum;
                entry.max_gen = entry.max_gen.max(ag.generation);
            }
        }
        entry.alt_sum_m += ag.alt_agl_m;
    }
    map
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorridorLightsOutInterventionRow {
    pub region_code: String,
    pub region_label: String,
    pub doy: f64,
    pub sim_hours: f64,
    pub lights_out_curfew_pct: f64,
    pub peak_viirs_nw_cm2_sr: f64,
    pub effective_viirs_nw_cm2_sr: f64,
    pub baseline_urban_strikes: u32,
    pub intervened_urban_strikes: u32,
    pub avoided_urban_strikes: u32,
    pub avoided_urban_strikes_pct: f64,
    pub baseline_bat_contacts: u32,
    pub intervened_bat_contacts: u32,
    pub avoided_bat_contacts: u32,
    pub avoided_bat_contacts_pct: f64,
    pub skyglow_entrained_baseline: usize,
    pub skyglow_entrained_intervened: usize,
    pub fallout_stopover_birds: usize,
    pub radar_mtr_birds_km_hr: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiCorridorLightsOutSummary {
    pub schema_version: String,
    pub doy: f64,
    pub sim_hours: f64,
    pub lights_out_curfew_pct: f64,
    pub corridor_count: usize,
    pub total_baseline_urban_strikes: u64,
    pub total_intervened_urban_strikes: u64,
    pub total_avoided_urban_strikes: u64,
    pub aggregate_avoided_urban_strikes_pct: f64,
    pub total_baseline_bat_contacts: u64,
    pub total_intervened_bat_contacts: u64,
    pub total_avoided_bat_contacts: u64,
    pub aggregate_avoided_bat_contacts_pct: f64,
    pub corridors: Vec<CorridorLightsOutInterventionRow>,
}

/// Evaluates paired unmitigated (`0%` Lights-Out) vs. intervention (`lights_out_curfew_pct`)
/// migration mortality across all 5 global flyway corridors under identical meteorological forcing.
pub fn evaluate_multi_corridor_lights_out_summary(
    hours: f64,
    doy: f64,
    lights_out_curfew_pct: f64,
) -> MultiCorridorLightsOutSummary {
    let clamped_hours = hours.clamp(1.0, 240.0);
    let clamped_doy = doy.clamp(1.0, 365.0);
    let clamped_curfew = lights_out_curfew_pct.clamp(0.0, 100.0);
    let season = SeasonPreset::from_doy(clamped_doy);
    let dt_hr = 0.5_f64;
    let steps = ((clamped_hours / dt_hr).round() as usize).max(1);

    let mut corridors = Vec::with_capacity(FlywayRegion::ALL.len());
    let mut total_base_urban = 0u64;
    let mut total_int_urban = 0u64;
    let mut total_base_bat = 0u64;
    let mut total_int_bat = 0u64;

    for region in FlywayRegion::ALL {
        let turbines = seed_wind_turbines_for_region(region);
        let milkweed = seed_milkweed_patches_for_region(region);
        let observations = baseline_observations_for_region(region);
        let weather = default_synoptic_nodes_for_region(region, Cmip6Scenario::BaselineHistorical);
        let (ta, tb) = region.default_transect();

        let mut base_engine = AeroPhysicsEngine::new();
        base_engine.weather_nodes = weather.clone();
        base_engine.reseed_agents(region, season);
        let base_params = SimulationParams {
            region,
            season,
            day_of_year: clamped_doy,
            advance_annual_calendar: true,
            cmip6_scenario: Cmip6Scenario::BaselineHistorical,
            lights_out_curfew_pct: 0.0,
            turbine_cut_in_mps: 3.5,
            transect_start: ta,
            transect_end: tb,
            ..SimulationParams::default()
        };

        let mut int_engine = AeroPhysicsEngine::new();
        int_engine.weather_nodes = weather;
        int_engine.reseed_agents(region, season);
        let int_params = SimulationParams {
            lights_out_curfew_pct: clamped_curfew,
            turbine_cut_in_mps: 5.5,
            ..base_params.clone()
        };

        for _ in 0..steps {
            base_engine.step(dt_hr, &base_params, &turbines, &milkweed, &observations);
            int_engine.step(dt_hr, &int_params, &turbines, &milkweed, &observations);
        }

        let peak_viirs = skyglow_domes_for_region(region)
            .iter()
            .map(|d| d.peak_radiance_nw_cm2_sr)
            .fold(0.0_f64, f64::max);
        let eff_viirs = effective_skyglow_radiance_nw(
            peak_viirs,
            1.0,
            true,
            clamped_curfew,
            int_params.skyglow_intensity_mult,
        );

        let base_urban = base_engine
            .metrics
            .cumulative_urban_strikes
            .max(int_engine.metrics.baseline_unmitigated_strikes);
        let int_urban = int_engine.metrics.cumulative_urban_strikes;
        let avoided_urban = base_urban.saturating_sub(int_urban);
        let avoided_urban_pct = if base_urban > 0 {
            (avoided_urban as f64 / base_urban as f64) * 100.0
        } else {
            int_engine.metrics.avoided_urban_strikes_pct
        };

        let base_bat = base_engine
            .metrics
            .cumulative_bat_rotor_contacts
            .max(int_engine.metrics.baseline_uncurtailed_contacts);
        let int_bat = int_engine.metrics.cumulative_bat_rotor_contacts;
        let avoided_bat = base_bat.saturating_sub(int_bat);
        let avoided_bat_pct = if base_bat > 0 {
            (avoided_bat as f64 / base_bat as f64) * 100.0
        } else {
            int_engine.metrics.avoided_bat_contacts_pct
        };

        total_base_urban += base_urban as u64;
        total_int_urban += int_urban as u64;
        total_base_bat += base_bat as u64;
        total_int_bat += int_bat as u64;

        corridors.push(CorridorLightsOutInterventionRow {
            region_code: region.code().to_string(),
            region_label: region.label().to_string(),
            doy: round1(clamped_doy),
            sim_hours: round1(clamped_hours),
            lights_out_curfew_pct: round1(clamped_curfew),
            peak_viirs_nw_cm2_sr: round1(peak_viirs),
            effective_viirs_nw_cm2_sr: round1(eff_viirs),
            baseline_urban_strikes: base_urban,
            intervened_urban_strikes: int_urban,
            avoided_urban_strikes: avoided_urban,
            avoided_urban_strikes_pct: round1(avoided_urban_pct),
            baseline_bat_contacts: base_bat,
            intervened_bat_contacts: int_bat,
            avoided_bat_contacts: avoided_bat,
            avoided_bat_contacts_pct: round1(avoided_bat_pct),
            skyglow_entrained_baseline: base_engine.metrics.skyglow_entrained_birds,
            skyglow_entrained_intervened: int_engine.metrics.skyglow_entrained_birds,
            fallout_stopover_birds: int_engine.metrics.fallout_stopover_birds,
            radar_mtr_birds_km_hr: round1(int_engine.metrics.radar_mtr_birds_km_hr),
        });
    }

    let total_avoided_urban = total_base_urban.saturating_sub(total_int_urban);
    let agg_urban_pct = if total_base_urban > 0 {
        (total_avoided_urban as f64 / total_base_urban as f64) * 100.0
    } else {
        0.0
    };
    let total_avoided_bat = total_base_bat.saturating_sub(total_int_bat);
    let agg_bat_pct = if total_base_bat > 0 {
        (total_avoided_bat as f64 / total_base_bat as f64) * 100.0
    } else {
        0.0
    };

    MultiCorridorLightsOutSummary {
        schema_version: "aeroflyway_multi_corridor_lights_out_v1".to_string(),
        doy: round1(clamped_doy),
        sim_hours: round1(clamped_hours),
        lights_out_curfew_pct: round1(clamped_curfew),
        corridor_count: corridors.len(),
        total_baseline_urban_strikes: total_base_urban,
        total_intervened_urban_strikes: total_int_urban,
        total_avoided_urban_strikes: total_avoided_urban,
        aggregate_avoided_urban_strikes_pct: round1(agg_urban_pct),
        total_baseline_bat_contacts: total_base_bat,
        total_intervened_bat_contacts: total_int_bat,
        total_avoided_bat_contacts: total_avoided_bat,
        aggregate_avoided_bat_contacts_pct: round1(agg_bat_pct),
        corridors,
    }
}

/// Formats a [`MultiCorridorLightsOutSummary`] as an RFC 4180 CSV table (5 corridor rows + aggregate row).
pub fn format_multi_corridor_lights_out_csv(summary: &MultiCorridorLightsOutSummary) -> String {
    let mut out = String::from(
        "region_code,region_label,doy,sim_hours,lights_out_curfew_pct,peak_viirs_nw_cm2_sr,effective_viirs_nw_cm2_sr,baseline_urban_strikes,intervened_urban_strikes,avoided_urban_strikes,avoided_urban_strikes_pct,baseline_bat_contacts,intervened_bat_contacts,avoided_bat_contacts,avoided_bat_contacts_pct,skyglow_entrained_baseline,skyglow_entrained_intervened,fallout_stopover_birds,radar_mtr_birds_km_hr\n",
    );
    for r in &summary.corridors {
        out.push_str(&format!(
            "{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{:.1},{},{},{},{:.1},{},{},{},{:.1}\n",
            csv_escape(&r.region_code),
            csv_escape(&r.region_label),
            r.doy,
            r.sim_hours,
            r.lights_out_curfew_pct,
            r.peak_viirs_nw_cm2_sr,
            r.effective_viirs_nw_cm2_sr,
            r.baseline_urban_strikes,
            r.intervened_urban_strikes,
            r.avoided_urban_strikes,
            r.avoided_urban_strikes_pct,
            r.baseline_bat_contacts,
            r.intervened_bat_contacts,
            r.avoided_bat_contacts,
            r.avoided_bat_contacts_pct,
            r.skyglow_entrained_baseline,
            r.skyglow_entrained_intervened,
            r.fallout_stopover_birds,
            r.radar_mtr_birds_km_hr,
        ));
    }
    out.push_str(&format!(
        "ALL_CORRIDORS,Global Five-Corridor Aggregate,{:.1},{:.1},{:.1},0.0,0.0,{},{},{},{:.1},{},{},{},{:.1},0,0,0,0.0\n",
        summary.doy,
        summary.sim_hours,
        summary.lights_out_curfew_pct,
        summary.total_baseline_urban_strikes,
        summary.total_intervened_urban_strikes,
        summary.total_avoided_urban_strikes,
        summary.aggregate_avoided_urban_strikes_pct,
        summary.total_baseline_bat_contacts,
        summary.total_intervened_bat_contacts,
        summary.total_avoided_bat_contacts,
        summary.aggregate_avoided_bat_contacts_pct,
    ));
    out
}

/// Evaluates and writes `aeroflyway_multi_corridor_lights_out.json` and `.csv` into `out_dir`.
pub fn export_multi_corridor_lights_out_summary(
    out_dir: impl AsRef<Path>,
    hours: f64,
    doy: f64,
    lights_out_curfew_pct: f64,
) -> Result<(PathBuf, PathBuf, MultiCorridorLightsOutSummary), String> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| format!("Create export dir failed: {e}"))?;
    let summary = evaluate_multi_corridor_lights_out_summary(hours, doy, lights_out_curfew_pct);
    let json_path = dir.join("aeroflyway_multi_corridor_lights_out.json");
    let csv_path = dir.join("aeroflyway_multi_corridor_lights_out.csv");

    let json_txt = serde_json::to_string_pretty(&summary)
        .map_err(|e| format!("Multi-corridor JSON serialize error: {e}"))?;
    fs::write(&json_path, json_txt).map_err(|e| format!("Write multi-corridor JSON failed: {e}"))?;
    let csv_txt = format_multi_corridor_lights_out_csv(&summary);
    fs::write(&csv_path, csv_txt).map_err(|e| format!("Write multi-corridor CSV failed: {e}"))?;

    Ok((json_path, csv_path, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_corridor_lights_out_summary_covers_all_five_flyways_and_reduces_skyglow() {
        let summary = evaluate_multi_corridor_lights_out_summary(12.0, 272.0, 85.0);
        assert_eq!(summary.schema_version, "aeroflyway_multi_corridor_lights_out_v1");
        assert_eq!(summary.corridor_count, 5);
        assert_eq!(summary.corridors.len(), 5);
        for row in &summary.corridors {
            assert!(row.peak_viirs_nw_cm2_sr > 0.0);
            assert!(row.effective_viirs_nw_cm2_sr < row.peak_viirs_nw_cm2_sr);
            assert!(row.intervened_urban_strikes <= row.baseline_urban_strikes);
            assert!(row.intervened_bat_contacts <= row.baseline_bat_contacts);
        }
        assert!(summary.total_intervened_urban_strikes <= summary.total_baseline_urban_strikes);
    }

    #[test]
    fn multi_corridor_csv_and_json_export_roundtrip_is_well_formed() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "aeroflyway_multi_corridor_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        let (json_path, csv_path, summary) =
            export_multi_corridor_lights_out_summary(&tmp_dir, 8.0, 272.0, 80.0)
                .expect("multi-corridor export succeeds");
        assert!(json_path.exists());
        assert!(csv_path.exists());

        let json_txt = fs::read_to_string(&json_path).expect("read json");
        let parsed: MultiCorridorLightsOutSummary =
            serde_json::from_str(&json_txt).expect("parse summary json");
        assert_eq!(parsed.corridor_count, 5);
        assert_eq!(parsed.corridors.len(), summary.corridors.len());

        let csv_txt = fs::read_to_string(&csv_path).expect("read csv");
        let lines: Vec<&str> = csv_txt.lines().collect();
        assert_eq!(lines.len(), 7, "header + 5 corridors + ALL_CORRIDORS aggregate");
        assert!(lines[0].starts_with("region_code,region_label,doy,sim_hours"));
        assert!(lines[6].starts_with("ALL_CORRIDORS,"));

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn multi_corridor_higher_curfew_monotonically_lowers_effective_skyglow() {
        let s0 = evaluate_multi_corridor_lights_out_summary(6.0, 272.0, 0.0);
        let s85 = evaluate_multi_corridor_lights_out_summary(6.0, 272.0, 85.0);
        for (r0, r85) in s0.corridors.iter().zip(s85.corridors.iter()) {
            assert_eq!(r0.region_code, r85.region_code);
            assert!(r85.effective_viirs_nw_cm2_sr < r0.effective_viirs_nw_cm2_sr);
        }
        assert!(s85.total_intervened_urban_strikes <= s0.total_intervened_urban_strikes);
    }
}
