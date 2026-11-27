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

//! Headless CLI Subcommand Suite & Multi-Scenario Rayon Batch Runner (`src/cli.rs` — v0.6.0).
//! Supports zero-windowing execution on headless Linux/SSH/CI servers:
//! - `aeroflyway [gui] [--skip-hub]` : Launches native `eframe` + `wgpu` 100k-agent GIS workstation
//! - `aeroflyway regions`            : Lists all 5 global flyway region presets, skyglow domes & radars
//! - `aeroflyway radar`              : Fetches or parses NEXRAD/Aloft `VPTS` vertical radar profiles
//! - `aeroflyway batch`              : Rayon-parallel parameter sweep across global regions, DOY, CMIP6,
//!                                     smoke PM2.5, Lights-Out curfews, turbine cut-in & milkweed patches
//! - `aeroflyway inspect-wav`        : Recursive Rayon batch WAV STFT processor + PNG & SQLite pinning
//! - `aeroflyway sync`               : Headless Open-Meteo + CMIP6 + CAMS + Radar S3 + GLO-90 + biodiversity sync

use crate::acoustics::{compute_stft_spectrogram, load_wav_file};
use crate::basemap::{
    landmarks_for_region, radar_stations_for_region, seed_milkweed_patches_for_region,
    seed_wind_turbines_for_region, skyglow_domes_for_region, FlywayRegion, MilkweedPatch, Taxon,
    WindTurbine,
};
use crate::db::{baseline_observations_for_region, spatial_hex_bin, FlywayStore, ObservationRecord};
use crate::export::{
    export_corridor_csv_bundle, export_corridor_geojson, export_multi_corridor_lights_out_summary,
};
use crate::ingest::{
    build_regional_radar_profiles, load_vpts_csv_file, run_blocking_ingest, Cmip6Scenario,
    SynopticNodeWeather,
};
use crate::physics::{
    default_synoptic_nodes_for_region, AeroPhysicsEngine, SeasonPreset, SimulationParams,
};
use image::RgbaImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CliDispatchOutcome {
    HandledExit,
    LaunchGui { skip_startup_hub: bool },
    Screenshot { out_path: PathBuf, view: u8 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchScenarioConfig {
    pub scenario_id: String,
    pub region_code: String,
    pub doy: f64,
    pub cmip6_code: String,
    pub hours: f64,
    pub turbine_cut_in_mps: f64,
    pub lights_out_pct: f64,
    pub virtual_milkweed_added: usize,
    pub temp_offset_c: f64,
    pub smoke_pm25_ugm3: f64,
    pub wind_850_multiplier: f64,
    pub precip_front_mm_hr: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchScenarioOutcome {
    pub scenario_id: String,
    pub region_code: String,
    pub doy: f64,
    pub cmip6_code: String,
    pub hours: f64,
    pub turbine_cut_in_mps: f64,
    pub lights_out_pct: f64,
    pub virtual_milkweed_added: usize,
    pub temp_offset_c: f64,
    pub smoke_pm25_ugm3: f64,
    pub baseline_bat_contacts: u32,
    pub curtailed_bat_contacts: u32,
    pub avoided_bat_pct: f64,
    pub curtailed_energy_pct: f64,
    pub pareto_protection_per_mwh_pct: f64,
    pub avoided_urban_strikes_pct: f64,
    pub skyglow_entrained_birds: usize,
    pub trophic_mismatch_days: f64,
    pub trophic_overlap_pct: f64,
    pub smoke_disrupted_birds: usize,
    pub radar_mtr_birds_km_hr: f64,
    pub monarch_mean_gdd: f64,
    pub monarch_peak_gen: u8,
    pub monarch_north_lat: f64,
    pub fallout_stopover_birds: usize,
    pub final_enkf_rmse: f64,
}

#[derive(Clone, Debug)]
struct WavBatchRow {
    file_path: String,
    sample_rate_khz: f64,
    duration_ms: f64,
    pulses: usize,
    fc_khz: f64,
    fmin_khz: f64,
    fmax_khz: f64,
    mean_dur_ms: f64,
    aci_index: f64,
    lat: f64,
    lon: f64,
    has_guano_gps: bool,
    taxon: Taxon,
    guild_label: String,
    scientific_hint: String,
    png_saved: Option<String>,
}

pub fn try_dispatch_cli() -> Result<CliDispatchOutcome, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Ok(CliDispatchOutcome::LaunchGui {
            skip_startup_hub: false,
        });
    }

    if let Some(pos) = args.iter().position(|a| a == "--screenshot") {
        let out = args
            .get(pos + 1)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "Missing output PNG path after `--screenshot`.".to_string())?;
        let view = if let Some(vpos) = args
            .iter()
            .position(|a| a == "--screenshot-view" || a == "--view")
        {
            let raw = args
                .get(vpos + 1)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "Missing view index after `--screenshot-view`.".to_string())?;
            let parsed: u8 = raw
                .parse()
                .map_err(|_| format!("Invalid screenshot view '{raw}' (expected 1, 2, or 3)."))?;
            if !(1..=3).contains(&parsed) {
                return Err(format!(
                    "Unsupported screenshot view {parsed} (expected 1, 2, or 3)."
                ));
            }
            parsed
        } else {
            1
        };
        return Ok(CliDispatchOutcome::Screenshot {
            out_path: PathBuf::from(out),
            view,
        });
    }

    match args[0].as_str() {
        "--skip-hub" => Ok(CliDispatchOutcome::LaunchGui {
            skip_startup_hub: true,
        }),
        "gui" => {
            let skip = args.iter().any(|a| a == "--skip-hub");
            Ok(CliDispatchOutcome::LaunchGui {
                skip_startup_hub: skip,
            })
        }
        "-h" | "--help" | "help" => {
            print_cli_help();
            Ok(CliDispatchOutcome::HandledExit)
        }
        "-V" | "--version" | "version" => {
            println!("aeroflyway 0.6.0 (Startup Hub + VIIRS Lights-Out + NEXRAD/Aloft VPTS + CMIP6 Suite)");
            Ok(CliDispatchOutcome::HandledExit)
        }
        "regions" => {
            print_regions_catalog();
            Ok(CliDispatchOutcome::HandledExit)
        }
        "radar" => {
            run_cli_radar(&args[1..])?;
            Ok(CliDispatchOutcome::HandledExit)
        }
        "batch" => {
            run_cli_batch(&args[1..])?;
            Ok(CliDispatchOutcome::HandledExit)
        }
        "inspect-wav" => {
            run_cli_inspect_wav(&args[1..])?;
            Ok(CliDispatchOutcome::HandledExit)
        }
        "sync" => {
            run_cli_sync(&args[1..])?;
            Ok(CliDispatchOutcome::HandledExit)
        }
        "compare-corridors" => {
            run_cli_compare_corridors(&args[1..])?;
            Ok(CliDispatchOutcome::HandledExit)
        }
        other => Err(format!(
            "Unknown subcommand '{other}'. Run `aeroflyway --help` for usage."
        )),
    }
}

pub fn print_regions_catalog() {
    println!("AeroFlyway v0.6.0 — Supported Global Flyway Corridors (`--region <code>`):");
    for r in FlywayRegion::ALL {
        let (clat, clon, _) = r.default_viewport();
        let (min_lat, max_lat, min_lon, max_lon) = r.corridor_bounds();
        let lms: Vec<&str> = landmarks_for_region(r).iter().take(3).map(|l| l.name).collect();
        let radars: Vec<&str> = radar_stations_for_region(r).iter().map(|s| s.code).collect();
        let domes: Vec<&str> = skyglow_domes_for_region(r)
            .iter()
            .take(2)
            .map(|d| d.city_name)
            .collect();
        println!(
            "  - {:<12} | {} | Center ({:.2}, {:.2}) | Bounds [{:.1}..{:.1} N, {:.1}..{:.1} E]\n      Refuges: {} | Radars: {} | VIIRS Skyglow: {}",
            r.code(),
            r.label(),
            clat,
            clon,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            lms.join(", "),
            radars.join(", "),
            domes.join(", ")
        );
    }
}

pub fn print_cli_help() {
    println!(
        "\
AeroFlyway v0.6.0 — Startup Hub, VIIRS Skyglow Lights-Out, NEXRAD/Aloft Radar & STFT Suite

USAGE:
  aeroflyway [gui] [--skip-hub]              Launch native eframe + wgpu 100k-agent GIS workstation
  aeroflyway regions                         List all 5 global flyway corridors, radars & VIIRS domes
  aeroflyway radar [OPTIONS]                 Inspect or parse NEXRAD/Aloft VPTS vertical radar profiles
  aeroflyway batch [OPTIONS]                 Run Rayon-parallel multi-scenario parameter sweeps
  aeroflyway inspect-wav <PATH> [OPTIONS]    Batch-process a .wav file or recursive folder via STFT
  aeroflyway sync [OPTIONS]                  Headless sync of Open-Meteo, CMIP6, Radar S3, GLO-90 & biodiversity
  aeroflyway compare-corridors [OPTIONS]     Compare migration mortality & Lights-Out curfews across all 5 corridors

RADAR OPTIONS (`aeroflyway radar`):
  --region <code>            pacific | central | atlantic | palearctic | east_asia [default: pacific]
  --vpts-csv <path>          Parse an Aloft/bioRad VPTS CSV file (`radar,datetime,height,u,v,...,dens,sd_vvp`)
  --smoke-pm25 <f64>         Evaluate smoke-plume ceiling shift on vertical profile [default: 0.0]
  --out-dir <dir>            Save `radar_vpts_profiles.csv` table [default: exports]

BATCH OPTIONS (`aeroflyway batch`):
  --region <code>            pacific | central | atlantic | palearctic | east_asia [default: pacific]
  --doy <f64>                Starting Day of Year (1..365; overrides --season) [default: 272.0]
  --season <name>            spring | summer | autumn | overwinter [default: autumn]
  --ssp <code>               baseline | ssp126 | ssp245 | ssp370 | ssp585 [default: baseline]
  --lights-out <csv>         Municipal Lights-Out curfew dimming % (23:00-06:00) [default: 35.0]
  --smoke-pm25 <csv>         Wildfire smoke PM2.5 offsets in ug/m^3 [default: 0.0]
  --hours <f64>              Simulated duration per scenario in hours [default: 72.0]
  --cut-in <csv>             Turbine cut-in wind speeds in m/s [default: 3.0,4.0,5.0,5.5,6.5]
  --milkweed-add <csv>       Virtual host-plant patches added in corridor gaps [default: 0,4,8]
  --temp-offset <csv>        Additional thermal anomaly offsets in C [default: 0.0,2.0]
  --wind-mult <f64>          850 hPa jet speed multiplier [default: 1.0]
  --precip <f64>             Precipitation fallout front in mm/hr [default: 0.4]
  --db <path>                SQLite WAL database path [default: aeroflyway.db]
  --out-dir <dir>            Output folder for CSV/JSON/GeoJSON [default: exports]
  --export-best-geojson      Also write RFC 7946 GeoJSON for top Pareto-optimal scenario

INSPECT-WAV OPTIONS (`aeroflyway inspect-wav <file_or_dir>`):
  --fft <usize>              Power-of-two STFT window size (256..4096) [default: 1024]
  --pin-db                   Upsert classified bird/bat call detections into SQLite observations
  --export-pngs              Render Magma spectrogram PNGs into <out-dir>/spectrograms/
  --default-lat <f64>        Fallback latitude when WAV has no GUANO Loc Position [default: 44.4128]
  --default-lon <f64>        Fallback longitude when WAV has no GUANO Loc Position [default: -123.3285]
  --db <path>                SQLite WAL database path [default: aeroflyway.db]
  --out-dir <dir>            Output folder for batch_wav_acoustics.csv [default: exports]

SYNC OPTIONS (`aeroflyway sync`):
  --region <code>            pacific | central | atlantic | palearctic | east_asia [default: pacific]
  --ssp <code>               baseline | ssp126 | ssp245 | ssp370 | ssp585 [default: baseline]
  --lat <f64>                Override query center latitude [default: region center]
  --lon <f64>                Override query center longitude [default: region center]
  --ebird-key <token>        Optional eBird API 2.0 key (or EBIRD_API_KEY env var)
  --warmup-hours <f64>       EnKF assimilation spin-up hours before export [default: 12.0]
  --db <path>                SQLite WAL database path [default: aeroflyway.db]
  --export-dir <dir>         Optional folder to immediately write fresh GeoJSON & CSV snapshots

COMPARE-CORRIDORS OPTIONS (`aeroflyway compare-corridors`):
  --doy <f64>                Day of Year (1..365) [default: 272.0]
  --hours <f64>              Simulated duration per corridor in hours [default: 24.0]
  --lights-out <f64>         Lights-Out curfew dimming % [default: 85.0]
  --out-dir <dir>            Output folder for JSON & CSV summary [default: exports]
"
    );
}

fn run_cli_radar(args: &[String]) -> Result<(), String> {
    let mut region = FlywayRegion::PacificAmericas;
    let mut vpts_csv: Option<PathBuf> = None;
    let mut smoke_pm25 = 0.0_f64;
    let mut out_dir = PathBuf::from("exports");

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--region" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --region")?;
                region = FlywayRegion::from_str_code(s).ok_or_else(|| {
                    format!("Unknown --region '{s}' (try pacific, central, atlantic, palearctic, east_asia)")
                })?;
            }
            "--vpts-csv" => {
                i += 1;
                vpts_csv = Some(PathBuf::from(args.get(i).ok_or("Missing value for --vpts-csv")?));
            }
            "--smoke-pm25" => {
                i += 1;
                smoke_pm25 = parse_arg_val(args.get(i), "--smoke-pm25")?;
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(args.get(i).ok_or("Missing value for --out-dir")?);
            }
            "-h" | "--help" => {
                print_cli_help();
                return Ok(());
            }
            flag => return Err(format!("Unrecognized `radar` flag '{flag}'")),
        }
        i += 1;
    }

    let stations = radar_stations_for_region(region);
    let profiles = if let Some(csv_path) = vpts_csv {
        let fallback = stations.first().ok_or("No radar station defined for region")?;
        vec![load_vpts_csv_file(csv_path, fallback)?]
    } else {
        let wx = default_synoptic_nodes_for_region(region, Cmip6Scenario::BaselineHistorical);
        build_regional_radar_profiles(None, region, &wx, smoke_pm25)
    };

    fs::create_dir_all(&out_dir).map_err(|e| format!("Create out-dir failed: {e}"))?;
    let csv_out = out_dir.join("radar_vpts_profiles.csv");
    let mut buf = String::from(
        "station_code,station_name,lat,lon,height_m_asl,bird_density_per_km3,u_mps,v_mps,reflectivity_eta,sd_vvp_mps,mtr_birds_km_hr\n",
    );

    for p in &profiles {
        println!(
            "[AeroFlyway v0.6 Radar] {} — {} ({}) | MTR: {:.0} birds/km/hr | Peak Layer: {:.0} m ASL",
            p.station_code, p.station_name, p.source_tag, p.mtr_birds_km_hr, p.peak_altitude_m_asl
        );
        for b in &p.bins {
            buf.push_str(&format!(
                "{},\"{}\",{:.4},{:.4},{:.0},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1}\n",
                p.station_code,
                p.station_name,
                p.lat,
                p.lon,
                b.height_m_asl,
                b.bird_density_per_km3,
                b.u_mps,
                b.v_mps,
                b.reflectivity_eta,
                b.sd_vvp_mps,
                p.mtr_birds_km_hr
            ));
        }
    }
    fs::write(&csv_out, buf).map_err(|e| format!("Write radar CSV error: {e}"))?;
    println!("[AeroFlyway v0.6 Radar] Saved VPTS altitude table -> {}", csv_out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn execute_batch_sweep_grid(
    region: FlywayRegion,
    doy: f64,
    cmip6_scenario: Cmip6Scenario,
    hours: f64,
    cut_in_speeds: &[f64],
    lights_out_pcts: &[f64],
    milkweed_additions: &[usize],
    temp_offsets: &[f64],
    smoke_pm25_offsets: &[f64],
    wind_mult: f64,
    precip_mm_hr: f64,
    base_turbines: &[WindTurbine],
    base_milkweed: &[MilkweedPatch],
    observations: &[ObservationRecord],
    weather_nodes: &[SynopticNodeWeather],
) -> Vec<BatchScenarioOutcome> {
    let season = SeasonPreset::from_doy(doy);
    let smokes = if smoke_pm25_offsets.is_empty() {
        vec![0.0]
    } else {
        smoke_pm25_offsets.to_vec()
    };
    let lights = if lights_out_pcts.is_empty() {
        vec![35.0]
    } else {
        lights_out_pcts.to_vec()
    };

    let mut configs = Vec::new();
    let mut idx = 1usize;
    for &cut_in in cut_in_speeds {
        for &lo_pct in &lights {
            for &mw_add in milkweed_additions {
                for &t_off in temp_offsets {
                    for &smk in &smokes {
                        configs.push((
                            idx,
                            BatchScenarioConfig {
                                scenario_id: format!("S{idx:03}"),
                                region_code: region.code().to_string(),
                                doy,
                                cmip6_code: cmip6_scenario.code().to_string(),
                                hours,
                                turbine_cut_in_mps: cut_in,
                                lights_out_pct: lo_pct,
                                virtual_milkweed_added: mw_add,
                                temp_offset_c: t_off,
                                smoke_pm25_ugm3: smk,
                                wind_850_multiplier: wind_mult,
                                precip_front_mm_hr: precip_mm_hr,
                            },
                        ));
                        idx += 1;
                    }
                }
            }
        }
    }

    let (ta, tb) = region.default_transect();

    configs
        .par_iter()
        .map(|(_, cfg)| {
            let mut engine = AeroPhysicsEngine::new();
            engine.weather_nodes = if weather_nodes.is_empty() {
                default_synoptic_nodes_for_region(region, cmip6_scenario)
            } else {
                weather_nodes.to_vec()
            };
            engine.radar_profiles = build_regional_radar_profiles(
                None,
                region,
                &engine.weather_nodes,
                cfg.smoke_pm25_ugm3,
            );
            engine.reseed_agents(region, season);

            let mut patches = base_milkweed.to_vec();
            patches.extend(generate_corridor_gap_milkweed(
                region,
                cfg.virtual_milkweed_added,
            ));

            let params = SimulationParams {
                region,
                season,
                day_of_year: cfg.doy,
                advance_annual_calendar: true,
                cmip6_scenario,
                wind_850_multiplier: cfg.wind_850_multiplier,
                precip_front_mm_hr: cfg.precip_front_mm_hr,
                turbine_cut_in_mps: cfg.turbine_cut_in_mps,
                temp_offset_c: cfg.temp_offset_c,
                smoke_pm25_offset_ugm3: cfg.smoke_pm25_ugm3,
                lights_out_curfew_pct: cfg.lights_out_pct,
                skyglow_intensity_mult: 1.0,
                radar_nudge_weight: 0.65,
                transect_start: ta,
                transect_end: tb,
                transect_half_width_km: 38.0,
            };

            let dt_hr = 0.5_f64;
            let steps = ((cfg.hours / dt_hr).round() as usize).max(1);
            for _ in 0..steps {
                engine.step(dt_hr, &params, base_turbines, &patches, observations);
            }

            let m = &engine.metrics;
            let pareto_ratio = if m.curtailed_energy_pct > 0.05 {
                m.avoided_bat_contacts_pct / m.curtailed_energy_pct
            } else {
                0.0
            };

            BatchScenarioOutcome {
                scenario_id: cfg.scenario_id.clone(),
                region_code: cfg.region_code.clone(),
                doy: cfg.doy,
                cmip6_code: cfg.cmip6_code.clone(),
                hours: cfg.hours,
                turbine_cut_in_mps: cfg.turbine_cut_in_mps,
                lights_out_pct: cfg.lights_out_pct,
                virtual_milkweed_added: cfg.virtual_milkweed_added,
                temp_offset_c: cfg.temp_offset_c,
                smoke_pm25_ugm3: cfg.smoke_pm25_ugm3,
                baseline_bat_contacts: m.baseline_uncurtailed_contacts,
                curtailed_bat_contacts: m.cumulative_bat_rotor_contacts,
                avoided_bat_pct: m.avoided_bat_contacts_pct,
                curtailed_energy_pct: m.curtailed_energy_pct,
                pareto_protection_per_mwh_pct: pareto_ratio,
                avoided_urban_strikes_pct: m.avoided_urban_strikes_pct,
                skyglow_entrained_birds: m.skyglow_entrained_birds,
                trophic_mismatch_days: m.trophic_mismatch_days,
                trophic_overlap_pct: m.trophic_overlap_pct,
                smoke_disrupted_birds: m.smoke_disrupted_birds,
                radar_mtr_birds_km_hr: m.radar_mtr_birds_km_hr,
                monarch_mean_gdd: m.monarch_mean_gdd,
                monarch_peak_gen: m.monarch_peak_generation,
                monarch_north_lat: m.monarch_northern_front_lat,
                fallout_stopover_birds: m.fallout_stopover_birds,
                final_enkf_rmse: m.enkf_rmse,
            }
        })
        .collect()
}

pub fn write_batch_sweep_reports(
    out_dir: impl AsRef<Path>,
    outcomes: &[BatchScenarioOutcome],
) -> Result<(PathBuf, PathBuf), String> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir).map_err(|e| format!("Failed creating {}: {e}", dir.display()))?;

    let csv_path = dir.join("batch_parameter_sweep.csv");
    let json_path = dir.join("batch_parameter_sweep.json");

    let mut csv_buf = String::from(
        "scenario_id,region,doy,cmip6,hours,turbine_cut_in_mps,lights_out_pct,virtual_milkweed_added,temp_offset_c,smoke_pm25_ugm3,baseline_bat_contacts,curtailed_bat_contacts,avoided_bat_pct,curtailed_energy_pct,pareto_protection_per_mwh_pct,avoided_urban_strikes_pct,skyglow_entrained_birds,trophic_mismatch_days,trophic_overlap_pct,smoke_disrupted_birds,radar_mtr_birds_km_hr,monarch_mean_gdd,monarch_peak_gen,monarch_north_lat,fallout_stopover_birds,final_enkf_rmse\n",
    );
    for r in outcomes {
        csv_buf.push_str(&format!(
            "{},{},{:.1},{},{:.1},{:.2},{:.1},{},{:.2},{:.1},{},{},{:.2},{:.2},{:.2},{:.2},{},{:.2},{:.1},{},{:.1},{:.1},{},{:.3},{},{:.4}\n",
            r.scenario_id,
            r.region_code,
            r.doy,
            r.cmip6_code,
            r.hours,
            r.turbine_cut_in_mps,
            r.lights_out_pct,
            r.virtual_milkweed_added,
            r.temp_offset_c,
            r.smoke_pm25_ugm3,
            r.baseline_bat_contacts,
            r.curtailed_bat_contacts,
            r.avoided_bat_pct,
            r.curtailed_energy_pct,
            r.pareto_protection_per_mwh_pct,
            r.avoided_urban_strikes_pct,
            r.skyglow_entrained_birds,
            r.trophic_mismatch_days,
            r.trophic_overlap_pct,
            r.smoke_disrupted_birds,
            r.radar_mtr_birds_km_hr,
            r.monarch_mean_gdd,
            r.monarch_peak_gen,
            r.monarch_north_lat,
            r.fallout_stopover_birds,
            r.final_enkf_rmse,
        ));
    }
    fs::write(&csv_path, csv_buf).map_err(|e| format!("Write sweep CSV error: {e}"))?;

    let json_txt = serde_json::to_string_pretty(outcomes)
        .map_err(|e| format!("Serialize sweep JSON error: {e}"))?;
    fs::write(&json_path, json_txt).map_err(|e| format!("Write sweep JSON error: {e}"))?;

    Ok((csv_path, json_path))
}

fn run_cli_batch(args: &[String]) -> Result<(), String> {
    let mut region = FlywayRegion::PacificAmericas;
    let mut season = SeasonPreset::AutumnFalloutAndBatPulse;
    let mut explicit_doy: Option<f64> = None;
    let mut ssp = Cmip6Scenario::BaselineHistorical;
    let mut hours = 72.0_f64;
    let mut cut_ins = vec![3.0, 4.0, 5.0, 5.5, 6.5];
    let mut lights_out = vec![35.0];
    let mut mw_adds = vec![0usize, 4, 8];
    let mut temp_offsets = vec![0.0, 2.0];
    let mut smoke_pm25 = vec![0.0];
    let mut wind_mult = 1.0_f64;
    let mut precip = 0.4_f64;
    let mut db_path = PathBuf::from("aeroflyway.db");
    let mut out_dir = PathBuf::from("exports");
    let mut export_best_geojson = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--region" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --region")?;
                region = FlywayRegion::from_str_code(s).ok_or_else(|| {
                    format!("Unknown --region '{s}' (try pacific, central, atlantic, palearctic, east_asia)")
                })?;
            }
            "--doy" => {
                i += 1;
                explicit_doy = Some(parse_arg_val::<f64>(args.get(i), "--doy")?.clamp(1.0, 365.0));
            }
            "--season" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --season")?;
                season = parse_season_flag(s)?;
            }
            "--ssp" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --ssp")?;
                ssp = Cmip6Scenario::from_str_code(s).ok_or_else(|| {
                    format!("Unknown --ssp '{s}' (expected baseline, ssp126, ssp245, ssp370, ssp585)")
                })?;
            }
            "--lights-out" => {
                i += 1;
                lights_out = parse_csv_list::<f64>(args.get(i), "--lights-out")?;
            }
            "--smoke-pm25" => {
                i += 1;
                smoke_pm25 = parse_csv_list::<f64>(args.get(i), "--smoke-pm25")?;
            }
            "--hours" => {
                i += 1;
                hours = parse_arg_val(args.get(i), "--hours")?;
            }
            "--cut-in" => {
                i += 1;
                cut_ins = parse_csv_list::<f64>(args.get(i), "--cut-in")?;
            }
            "--milkweed-add" => {
                i += 1;
                mw_adds = parse_csv_list::<usize>(args.get(i), "--milkweed-add")?;
            }
            "--temp-offset" => {
                i += 1;
                temp_offsets = parse_csv_list::<f64>(args.get(i), "--temp-offset")?;
            }
            "--wind-mult" => {
                i += 1;
                wind_mult = parse_arg_val(args.get(i), "--wind-mult")?;
            }
            "--precip" => {
                i += 1;
                precip = parse_arg_val(args.get(i), "--precip")?;
            }
            "--db" => {
                i += 1;
                db_path = PathBuf::from(args.get(i).ok_or("Missing value for --db")?);
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(args.get(i).ok_or("Missing value for --out-dir")?);
            }
            "--export-best-geojson" => {
                export_best_geojson = true;
            }
            "-h" | "--help" => {
                print_cli_help();
                return Ok(());
            }
            flag => return Err(format!("Unrecognized `batch` flag '{flag}'")),
        }
        i += 1;
    }

    let doy = explicit_doy.unwrap_or_else(|| season.canonical_doy());
    let _store = FlywayStore::open(&db_path)?;
    let turbines = seed_wind_turbines_for_region(region);
    let milkweed = seed_milkweed_patches_for_region(region);
    let observations = baseline_observations_for_region(region);
    let wx = default_synoptic_nodes_for_region(region, ssp);

    let total_runs =
        cut_ins.len() * lights_out.len() * mw_adds.len() * temp_offsets.len() * smoke_pm25.len();
    println!(
        "[AeroFlyway v0.6 Batch] Running {total_runs} scenarios via Rayon ({hours:.1} h | region={} | DOY={doy:.0} | {})...",
        region.code(),
        ssp.code().to_uppercase()
    );

    let outcomes = execute_batch_sweep_grid(
        region,
        doy,
        ssp,
        hours,
        &cut_ins,
        &lights_out,
        &mw_adds,
        &temp_offsets,
        &smoke_pm25,
        wind_mult,
        precip,
        &turbines,
        &milkweed,
        &observations,
        &wx,
    );

    println!(
        "{:<6} {:>6} {:>6} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "ID", "CutIn", "LtsOut", "+MW", "AvoidBat", "AvoidUrb", "Pareto", "Mismtch"
    );
    for r in &outcomes {
        println!(
            "{:<6} {:>5.1}m {:>5.0}% {:>5} {:>7.1}% {:>7.1}% {:>8.1} {:>+6.1}d",
            r.scenario_id,
            r.turbine_cut_in_mps,
            r.lights_out_pct,
            r.virtual_milkweed_added,
            r.avoided_bat_pct,
            r.avoided_urban_strikes_pct,
            r.pareto_protection_per_mwh_pct,
            r.trophic_mismatch_days,
        );
    }

    let (csv_path, json_path) = write_batch_sweep_reports(&out_dir, &outcomes)?;
    println!(
        "[AeroFlyway v0.6 Batch] Saved CSV -> {} | JSON -> {}",
        csv_path.display(),
        json_path.display()
    );

    if let Some(best) = outcomes.iter().max_by(|a, b| {
        a.pareto_protection_per_mwh_pct
            .partial_cmp(&b.pareto_protection_per_mwh_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "[AeroFlyway v0.6 Batch] Peak Pareto knee: {} (cut-in {:.1} m/s -> {:.1}% bat risk avoided | Lights-Out {:.0}% -> {:.1}% urban strikes avoided)",
            best.scenario_id,
            best.turbine_cut_in_mps,
            best.avoided_bat_pct,
            best.lights_out_pct,
            best.avoided_urban_strikes_pct
        );

        if export_best_geojson {
            let mut best_engine = AeroPhysicsEngine::new();
            best_engine.weather_nodes = wx;
            best_engine.radar_profiles = build_regional_radar_profiles(
                None,
                region,
                &best_engine.weather_nodes,
                best.smoke_pm25_ugm3,
            );
            best_engine.reseed_agents(region, SeasonPreset::from_doy(doy));
            let mut best_mw = milkweed.clone();
            best_mw.extend(generate_corridor_gap_milkweed(
                region,
                best.virtual_milkweed_added,
            ));
            let (ta, tb) = region.default_transect();
            let best_params = SimulationParams {
                region,
                season: SeasonPreset::from_doy(doy),
                day_of_year: doy,
                advance_annual_calendar: true,
                cmip6_scenario: ssp,
                wind_850_multiplier: wind_mult,
                precip_front_mm_hr: precip,
                turbine_cut_in_mps: best.turbine_cut_in_mps,
                temp_offset_c: best.temp_offset_c,
                smoke_pm25_offset_ugm3: best.smoke_pm25_ugm3,
                lights_out_curfew_pct: best.lights_out_pct,
                skyglow_intensity_mult: 1.0,
                radar_nudge_weight: 0.65,
                transect_start: ta,
                transect_end: tb,
                transect_half_width_km: 38.0,
            };
            let steps = ((hours / 0.5).round() as usize).max(1);
            for _ in 0..steps {
                best_engine.step(0.5, &best_params, &turbines, &best_mw, &observations);
            }
            let (gj_path, n_feat) = export_corridor_geojson(
                &out_dir,
                &best_engine,
                &best_params,
                &observations,
                &turbines,
                &best_mw,
                1.0,
            )?;
            println!(
                "[AeroFlyway v0.6 Batch] Exported Pareto-optimal GeoJSON ({n_feat} features) -> {}",
                gj_path.display()
            );
        }
    }

    Ok(())
}

fn run_cli_inspect_wav(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_cli_help();
        return Ok(());
    }

    let target_path = PathBuf::from(&args[0]);
    let mut fft_size = 1024usize;
    let mut pin_db = false;
    let mut export_pngs = false;
    let mut default_lat = 44.4128_f64;
    let mut default_lon = -123.3285_f64;
    let mut db_path = PathBuf::from("aeroflyway.db");
    let mut out_dir = PathBuf::from("exports");

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--fft" => {
                i += 1;
                fft_size = parse_arg_val::<usize>(args.get(i), "--fft")?
                    .next_power_of_two()
                    .clamp(256, 4096);
            }
            "--pin-db" => pin_db = true,
            "--export-pngs" => export_pngs = true,
            "--default-lat" => {
                i += 1;
                default_lat = parse_arg_val(args.get(i), "--default-lat")?;
            }
            "--default-lon" => {
                i += 1;
                default_lon = parse_arg_val(args.get(i), "--default-lon")?;
            }
            "--db" => {
                i += 1;
                db_path = PathBuf::from(args.get(i).ok_or("Missing value for --db")?);
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(args.get(i).ok_or("Missing value for --out-dir")?);
            }
            flag => return Err(format!("Unrecognized `inspect-wav` flag '{flag}'")),
        }
        i += 1;
    }

    let mut wav_files = Vec::new();
    collect_wav_paths(&target_path, &mut wav_files)?;
    if wav_files.is_empty() {
        return Err(format!("No .wav files found at {}", target_path.display()));
    }
    wav_files.sort();

    let png_dir = out_dir.join("spectrograms");
    if export_pngs {
        fs::create_dir_all(&png_dir)
            .map_err(|e| format!("Failed creating {}: {e}", png_dir.display()))?;
    }

    println!(
        "[AeroFlyway STFT] Processing {} WAV file(s) in parallel via Rayon (FFT={fft_size})...",
        wav_files.len()
    );

    let results: Vec<Result<WavBatchRow, String>> = wav_files
        .par_iter()
        .map(|wav_path| {
            let rec = load_wav_file(wav_path)?;
            let spec = compute_stft_spectrogram(&rec, fft_size);
            let (lat, lon) = rec.guano_lat_lon.unwrap_or((default_lat, default_lon));

            let png_saved = if export_pngs {
                let top_khz = if rec.nyquist_khz() > 40.0 {
                    rec.nyquist_khz().min(95.0)
                } else {
                    rec.nyquist_khz().min(14.0)
                };
                let cimg = spec.to_color_image(-76.0, top_khz);
                let [w, h] = cimg.size;
                let stem = wav_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spectrogram");
                let out_png = png_dir.join(format!("{stem}_stft.png"));
                if let Some(rgba_img) =
                    RgbaImage::from_raw(w as u32, h as u32, cimg.as_raw().to_vec())
                {
                    rgba_img
                        .save(&out_png)
                        .map_err(|e| format!("Failed saving {}: {e}", out_png.display()))?;
                    Some(out_png.display().to_string())
                } else {
                    return Err(format!(
                        "Failed packing RGBA buffer for {}",
                        out_png.display()
                    ));
                }
            } else {
                None
            };

            Ok(WavBatchRow {
                file_path: wav_path.display().to_string(),
                sample_rate_khz: rec.sample_rate_hz as f64 / 1000.0,
                duration_ms: spec.duration_ms,
                pulses: spec.pulses.len(),
                fc_khz: spec.characteristic_freq_khz,
                fmin_khz: spec.f_min_khz,
                fmax_khz: spec.f_max_khz,
                mean_dur_ms: spec.mean_duration_ms,
                aci_index: spec.aci_index,
                lat,
                lon,
                has_guano_gps: rec.guano_lat_lon.is_some(),
                taxon: spec.classified_taxon,
                guild_label: spec.guild_label,
                scientific_hint: spec.scientific_hint,
                png_saved,
            })
        })
        .collect();

    fs::create_dir_all(&out_dir).map_err(|e| format!("Create out-dir error: {e}"))?;
    let csv_out_path = out_dir.join("batch_wav_acoustics.csv");
    let mut csv_txt = String::from(
        "file_path,sample_rate_khz,duration_ms,pulses,fc_khz,fmin_khz,fmax_khz,mean_dur_ms,aci_index,lat,lon,has_guano_gps,taxon,guild_label,scientific_hint\n",
    );

    let mut db_obs = Vec::new();
    println!(
        "{:<34} {:>7} {:>6} {:>8} {:>8} {:<30}",
        "File", "SR(kHz)", "Pulses", "Fc(kHz)", "ACI", "Classified Guild"
    );

    for (idx, item) in results.into_iter().enumerate() {
        match item {
            Ok(row) => {
                let short_name = Path::new(&row.file_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&row.file_path);
                println!(
                    "{:<34} {:>7.1} {:>6} {:>8.2} {:>8.1} {:<30}",
                    short_name,
                    row.sample_rate_khz,
                    row.pulses,
                    row.fc_khz,
                    row.aci_index,
                    row.guild_label,
                );

                csv_txt.push_str(&format!(
                    "\"{}\",{:.1},{:.1},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.5},{:.5},{},{},\"{}\",\"{}\"\n",
                    row.file_path.replace('"', "\"\""),
                    row.sample_rate_khz,
                    row.duration_ms,
                    row.pulses,
                    row.fc_khz,
                    row.fmin_khz,
                    row.fmax_khz,
                    row.mean_dur_ms,
                    row.aci_index,
                    row.lat,
                    row.lon,
                    row.has_guano_gps,
                    row.taxon.code(),
                    row.guild_label.replace('"', "\"\""),
                    row.scientific_hint.replace('"', "\"\""),
                ));

                if let Some(png) = &row.png_saved {
                    println!("  -> Saved spectrogram PNG: {png}");
                }

                if pin_db {
                    let alt_m = match row.taxon {
                        Taxon::Bat => 115.0,
                        Taxon::Bird => 420.0,
                        Taxon::Butterfly => 18.0,
                    };
                    db_obs.push(ObservationRecord {
                        obs_uid: format!("cli-wav-{idx:05}-{short_name}"),
                        taxon: row.taxon,
                        common_name: format!("{} (Fc {:.1} kHz)", row.guild_label, row.fc_khz),
                        scientific_name: row.scientific_hint,
                        lat: row.lat,
                        lon: row.lon,
                        hex_bin: spatial_hex_bin(row.lat, row.lon),
                        altitude_m: alt_m,
                        count: row.pulses.max(1) as u32,
                        source_api: format!("CLI-WAV/{short_name}"),
                        observed_at: "2026-11-14T04:15Z".to_string(),
                    });
                }
            }
            Err(err) => eprintln!("  [WARN] {err}"),
        }
    }

    fs::write(&csv_out_path, csv_txt).map_err(|e| format!("Write WAV CSV error: {e}"))?;
    println!(
        "[AeroFlyway STFT] Wrote acoustic feature table -> {}",
        csv_out_path.display()
    );

    if pin_db && !db_obs.is_empty() {
        let store = FlywayStore::open(&db_path)?;
        let n_up = store.upsert_observations(&db_obs)?;
        println!(
            "[AeroFlyway STFT] Pinned {n_up} acoustic detection(s) into {}",
            db_path.display()
        );
    }

    Ok(())
}

fn run_cli_sync(args: &[String]) -> Result<(), String> {
    let mut region = FlywayRegion::PacificAmericas;
    let mut ssp = Cmip6Scenario::BaselineHistorical;
    let mut explicit_lat: Option<f64> = None;
    let mut explicit_lon: Option<f64> = None;
    let mut ebird_key = std::env::var("EBIRD_API_KEY").unwrap_or_default();
    let mut warmup_hours = 12.0_f64;
    let mut db_path = PathBuf::from("aeroflyway.db");
    let mut export_dir: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--region" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --region")?;
                region = FlywayRegion::from_str_code(s).ok_or_else(|| {
                    format!("Unknown --region '{s}' (try pacific, central, atlantic, palearctic, east_asia)")
                })?;
            }
            "--ssp" => {
                i += 1;
                let s = args.get(i).ok_or("Missing value for --ssp")?;
                ssp = Cmip6Scenario::from_str_code(s).ok_or_else(|| {
                    format!("Unknown --ssp '{s}' (expected baseline, ssp126, ssp245, ssp370, ssp585)")
                })?;
            }
            "--lat" => {
                i += 1;
                explicit_lat = Some(parse_arg_val(args.get(i), "--lat")?);
            }
            "--lon" => {
                i += 1;
                explicit_lon = Some(parse_arg_val(args.get(i), "--lon")?);
            }
            "--ebird-key" => {
                i += 1;
                ebird_key = args
                    .get(i)
                    .ok_or("Missing value for --ebird-key")?
                    .to_string();
            }
            "--warmup-hours" => {
                i += 1;
                warmup_hours = parse_arg_val(args.get(i), "--warmup-hours")?;
            }
            "--db" => {
                i += 1;
                db_path = PathBuf::from(args.get(i).ok_or("Missing value for --db")?);
            }
            "--export-dir" => {
                i += 1;
                export_dir = Some(PathBuf::from(
                    args.get(i).ok_or("Missing value for --export-dir")?,
                ));
            }
            "-h" | "--help" => {
                print_cli_help();
                return Ok(());
            }
            flag => return Err(format!("Unrecognized `sync` flag '{flag}'")),
        }
        i += 1;
    }

    let (def_lat, def_lon, _) = region.default_viewport();
    let lat = explicit_lat.unwrap_or(def_lat);
    let lon = explicit_lon.unwrap_or(def_lon);
    let (ta, tb) = region.default_transect();

    println!(
        "[AeroFlyway v0.6 Sync] Polling Open-Meteo, CMIP6 ({}), Radar S3, GLO-90 & biodiversity around ({lat:.4}, {lon:.4}) [{}]...",
        ssp.code().to_uppercase(),
        region.code()
    );
    let payload = run_blocking_ingest(region, lat, lon, ssp, ta, tb, &ebird_key);
    println!("[AeroFlyway v0.6 Sync] {}", payload.status_summary);

    let store = FlywayStore::open(&db_path)?;
    store.switch_region_seed(region)?;
    if !payload.observations.is_empty() {
        let n = store.upsert_observations(&payload.observations)?;
        println!("[AeroFlyway v0.6 Sync] Upserted {n} public observation record(s).");
    }
    if !payload.turbines.is_empty() {
        let n = store.upsert_turbines(&payload.turbines)?;
        println!("[AeroFlyway v0.6 Sync] Upserted {n} wind turbine record(s).");
    }
    for prof in &payload.radar_profiles {
        let rows: Vec<(f64, f64, f64, f64, f64)> = prof
            .bins
            .iter()
            .map(|b| {
                (
                    b.height_m_asl,
                    b.bird_density_per_km3,
                    b.u_mps,
                    b.v_mps,
                    b.sd_vvp_mps,
                )
            })
            .collect();
        store.upsert_radar_bins(&prof.station_code, &prof.observed_at, &rows)?;
    }

    if let Some(dir) = export_dir {
        let observations = store.load_observations();
        let turbines = store.load_turbines();
        let milkweed = store.load_milkweed_patches();
        let mut engine = AeroPhysicsEngine::new();
        engine.reseed_agents(region, SeasonPreset::AutumnFalloutAndBatPulse);
        if !payload.weather_nodes.is_empty() {
            engine.weather_nodes = payload.weather_nodes;
        }
        if !payload.radar_profiles.is_empty() {
            engine.radar_profiles = payload.radar_profiles;
        }
        engine.cached_transect_dem = payload.transect_dem_m;
        let params = SimulationParams {
            region,
            cmip6_scenario: ssp,
            transect_start: ta,
            transect_end: tb,
            ..SimulationParams::default()
        };
        let steps = ((warmup_hours / 0.5).round() as usize).max(1);
        for _ in 0..steps {
            engine.step(0.5, &params, &turbines, &milkweed, &observations);
        }

        let (gj_path, n_feat) = export_corridor_geojson(
            &dir,
            &engine,
            &params,
            &observations,
            &turbines,
            &milkweed,
            1.0,
        )?;
        let (hex_csv, obs_csv, n_rows) =
            export_corridor_csv_bundle(&dir, &engine, &params, &observations, 1.0)?;
        println!(
            "[AeroFlyway v0.6 Sync] Exported {n_feat} GeoJSON features ({}) and {n_rows} CSV rows ({}, {}).",
            gj_path.display(),
            hex_csv.display(),
            obs_csv.display()
        );
    }

    Ok(())
}

pub fn generate_corridor_gap_milkweed(region: FlywayRegion, count: usize) -> Vec<MilkweedPatch> {
    let (min_lat, max_lat, min_lon, max_lon) = region.corridor_bounds();
    let mid_lon = 0.5 * (min_lon + max_lon);
    let span_lat = (max_lat - min_lat).max(2.0);
    let n_sites = count.min(16);

    (0..n_sites)
        .map(|i| {
            let frac = 0.16 + ((i as f64) / 16.0) * 0.68;
            let lat = min_lat + frac * span_lat;
            let lon_jiggle = if i % 2 == 0 { -0.06 } else { 0.06 } * (max_lon - min_lon);
            MilkweedPatch {
                id: 10_000 + i as i64,
                label: format!("{} Gap Stepping-Stone #{}", region.code(), i + 1),
                lat,
                lon: (mid_lon + lon_jiggle).clamp(min_lon + 0.1, max_lon - 0.1),
                area_ha: 28.0,
                is_virtual: true,
                gdd_threshold: 160.0,
            }
        })
        .collect()
}

fn collect_wav_paths(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("Path does not exist: {}", path.display()));
    }
    if path.is_file() {
        if is_wav_ext(path) {
            out.push(path.to_path_buf());
        }
        return Ok(());
    }
    let entries =
        fs::read_dir(path).map_err(|e| format!("Failed reading dir {}: {e}", path.display()))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_wav_paths(&p, out)?;
        } else if is_wav_ext(&p) {
            out.push(p);
        }
    }
    Ok(())
}

fn is_wav_ext(p: &Path) -> bool {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}

fn parse_season_flag(val: &str) -> Result<SeasonPreset, String> {
    match val.to_ascii_lowercase().as_str() {
        "spring" => Ok(SeasonPreset::SpringNorthbound),
        "summer" => Ok(SeasonPreset::SummerBreeding),
        "autumn" | "fall" => Ok(SeasonPreset::AutumnFalloutAndBatPulse),
        "overwinter" | "winter" => Ok(SeasonPreset::OverwinterRoost),
        _ => Err(format!(
            "Invalid --season '{val}' (expected spring, summer, autumn, or overwinter)"
        )),
    }
}

fn parse_arg_val<T: std::str::FromStr>(opt: Option<&String>, flag: &str) -> Result<T, String> {
    let raw = opt.ok_or_else(|| format!("Missing value after {flag}"))?;
    raw.parse::<T>()
        .map_err(|_| format!("Invalid value '{raw}' for {flag}"))
}

fn parse_csv_list<T: std::str::FromStr>(
    opt: Option<&String>,
    flag: &str,
) -> Result<Vec<T>, String> {
    let raw = opt.ok_or_else(|| format!("Missing comma-separated list after {flag}"))?;
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val = trimmed
            .parse::<T>()
            .map_err(|_| format!("Invalid item '{trimmed}' in {flag}"))?;
        out.push(val);
    }
    if out.is_empty() {
        return Err(format!("{flag} list cannot be empty"));
    }
    Ok(out)
}

pub fn run_cli_compare_corridors(args: &[String]) -> Result<(), String> {
    let mut doy = 272.0_f64;
    let mut hours = 24.0_f64;
    let mut lights_out = 85.0_f64;
    let mut out_dir = PathBuf::from("exports");

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--doy" => {
                i += 1;
                doy = parse_arg_val::<f64>(args.get(i), "--doy")?.clamp(1.0, 365.0);
            }
            "--hours" => {
                i += 1;
                hours = parse_arg_val::<f64>(args.get(i), "--hours")?.clamp(1.0, 240.0);
            }
            "--lights-out" => {
                i += 1;
                lights_out = parse_arg_val::<f64>(args.get(i), "--lights-out")?.clamp(0.0, 100.0);
            }
            "--out-dir" => {
                i += 1;
                out_dir = PathBuf::from(args.get(i).ok_or("Missing value for --out-dir")?);
            }
            "-h" | "--help" => {
                print_cli_help();
                return Ok(());
            }
            flag => return Err(format!("Unrecognized `compare-corridors` flag '{flag}'")),
        }
        i += 1;
    }

    let (json_path, csv_path, summary) =
        export_multi_corridor_lights_out_summary(&out_dir, hours, doy, lights_out)?;
    println!(
        "[AeroFlyway v0.6 Compare] {} corridors | DOY {:.0} | {:.1}h | Lights-Out {:.0}% | Urban strikes {} -> {} (-{:.1}%) | Bat contacts {} -> {} (-{:.1}%)",
        summary.corridor_count,
        summary.doy,
        summary.sim_hours,
        summary.lights_out_curfew_pct,
        summary.total_baseline_urban_strikes,
        summary.total_intervened_urban_strikes,
        summary.aggregate_avoided_urban_strikes_pct,
        summary.total_baseline_bat_contacts,
        summary.total_intervened_bat_contacts,
        summary.aggregate_avoided_bat_contacts_pct,
    );
    println!(
        "  Wrote JSON -> {} and CSV -> {}",
        json_path.display(),
        csv_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_compare_corridors_writes_five_corridor_summary_files() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "aeroflyway_cli_compare_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        let args = vec![
            "--doy".to_string(),
            "272".to_string(),
            "--hours".to_string(),
            "6".to_string(),
            "--lights-out".to_string(),
            "85".to_string(),
            "--out-dir".to_string(),
            tmp_dir.display().to_string(),
        ];
        run_cli_compare_corridors(&args).expect("compare-corridors CLI executes cleanly");
        assert!(tmp_dir.join("aeroflyway_multi_corridor_lights_out.json").exists());
        assert!(tmp_dir.join("aeroflyway_multi_corridor_lights_out.csv").exists());
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn cli_compare_corridors_rejects_unknown_flag() {
        let err = run_cli_compare_corridors(&["--unknown-flag".to_string()])
            .expect_err("unknown flag must fail");
        assert!(err.contains("Unrecognized `compare-corridors` flag"));
    }
}
