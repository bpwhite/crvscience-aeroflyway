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
    sample_terrain_elevation_m, skyglow_domes_for_region, MilkweedPatch, Taxon, WindTurbine,
};
use crate::db::{hex_bin_center, spatial_hex_bin, ObservationRecord};
use crate::map_engine::haversine_km;
use crate::physics::{effective_skyglow_radiance_nw, AeroPhysicsEngine, SimulationParams};
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
/// `FeatureCollection` (`WGS84` `[lon, lat, elev_m]`, counter-clockwise polygon rings, root `bbox`).
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
    for (hex_key, agg) in &hex_map {
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
