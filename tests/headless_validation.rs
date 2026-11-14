use aeroflyway::acoustics::{
    compute_stft_spectrogram, load_wav_file, synthesize_preset_recording, write_wav_16bit,
    SpectrogramResult, SynthPreset, WavRecording,
};
use aeroflyway::basemap::{
    radar_stations_for_region, seed_milkweed_patches_for_region, seed_wind_turbines_for_region,
    FlywayRegion, Taxon,
};
use aeroflyway::db::{
    baseline_observations_for_region, spatial_hex_bin, FlywayStore, ModelSnapshot,
    ObservationRecord,
};
use aeroflyway::ingest::{
    build_regional_radar_profiles, parse_vpts_csv_text, summarize_profile_mtr_and_peak,
    Cmip6Scenario,
};
use aeroflyway::physics::{
    compute_lights_out_collision_reduction_pct, default_synoptic_nodes_for_region,
    effective_skyglow_radiance_nw, AeroPhysicsEngine, SeasonPreset, SimulationParams,
};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

static SUMMARY_LOCK: Mutex<()> = Mutex::new(());

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(prefix: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp validation workspace");
        Self { path: dir }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn validation_dir() -> PathBuf {
    let dir = repo_root().join("validation");
    fs::create_dir_all(&dir).expect("create validation directory");
    dir
}

fn write_summary_section(section_key: &str, section_value: Value) {
    let _guard = SUMMARY_LOCK.lock().expect("lock summary mutex");
    let val_dir = validation_dir();
    let local_path = val_dir.join("aeroflyway_validation_summary.json");

    let mut root = if local_path.exists() {
        fs::read_to_string(&local_path)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };

    if !root.is_object() {
        root = json!({});
    }
    let obj = root.as_object_mut().expect("root JSON object");
    obj.insert("schema_version".into(), json!("1.0"));
    obj.insert("repo".into(), json!("crvscience-aeroflyway"));
    obj.insert("version".into(), json!("0.6.0"));
    obj.insert("preprint_id".into(), json!("APP-PREPRINT-01"));
    obj.insert("concept_doi".into(), json!("10.5281/zenodo.24222366"));
    obj.insert(
        "preprint_url".into(),
        json!("https://crvscience.com/apps/preprints/aeroflyway-avian-migration-radar"),
    );

    let tests_entry = obj
        .entry("tests".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !tests_entry.is_object() {
        *tests_entry = Value::Object(Map::new());
    }
    tests_entry
        .as_object_mut()
        .expect("tests JSON map")
        .insert(section_key.to_string(), section_value);

    let serialized = serde_json::to_string_pretty(&root).expect("serialize validation summary");
    fs::write(&local_path, format!("{serialized}\n")).expect("write local summary JSON");

    if let Some(hub_root) = repo_root()
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
    {
        let hub_releases = hub_root.join("app-preprints").join("releases");
        if hub_releases.is_dir() {
            let hub_path =
                hub_releases.join("aeroflyway-avian-migration-radar-validation-telemetry.json");
            fs::write(&hub_path, format!("{serialized}\n"))
                .expect("write hub release telemetry copy");
        }
    }
}

fn band_peak_and_mean_db(spec: &SpectrogramResult, min_khz: f64, max_khz: f64) -> (f64, f64) {
    let bin_hz = (spec.sample_rate_hz as f64) / (spec.fft_size as f64);
    let mut max_db = f64::NEG_INFINITY;
    let mut sum_db = 0.0_f64;
    let mut count = 0usize;

    for frame in 0..spec.num_frames {
        for bin in 0..spec.num_bins {
            let freq_khz = (bin as f64) * bin_hz / 1000.0;
            if freq_khz >= min_khz && freq_khz <= max_khz {
                let db = spec.power_dbfs[frame * spec.num_bins + bin] as f64;
                if db > max_db {
                    max_db = db;
                }
                sum_db += db;
                count += 1;
            }
        }
    }

    let mean_db = if count > 0 {
        sum_db / count as f64
    } else {
        -120.0
    };
    (max_db, mean_db)
}

#[derive(Default)]
struct NocturnalStratification {
    lat_25_32_entrained: usize,
    lat_32_38_entrained: usize,
    lat_38_45_entrained: usize,
    alt_0_300_birds: usize,
    alt_300_800_birds: usize,
    alt_800_plus_birds: usize,
    mean_alt_agl_m: f64,
}

fn stratify_airborne_birds(engine: &AeroPhysicsEngine) -> NocturnalStratification {
    let mut out = NocturnalStratification::default();
    let mut alt_sum = 0.0_f64;
    let mut bird_count = 0usize;

    for a in &engine.agents {
        if a.taxon != Taxon::Bird || a.alt_agl_m <= 15.0 {
            continue;
        }
        bird_count += 1;
        alt_sum += a.alt_agl_m;

        if a.alt_agl_m < 300.0 {
            out.alt_0_300_birds += 1;
        } else if a.alt_agl_m < 800.0 {
            out.alt_300_800_birds += 1;
        } else {
            out.alt_800_plus_birds += 1;
        }

        if a.skyglow_entrained {
            if a.lat < 32.0 {
                out.lat_25_32_entrained += 1;
            } else if a.lat < 38.0 {
                out.lat_32_38_entrained += 1;
            } else {
                out.lat_38_45_entrained += 1;
            }
        }
    }

    if bird_count > 0 {
        out.mean_alt_agl_m = alt_sum / bird_count as f64;
    }
    out
}

#[test]
fn t1_lights_out_collision_mortality_curve() {
    let region = FlywayRegion::AtlanticAmericas;
    let season = SeasonPreset::AutumnFalloutAndBatPulse;
    let ssp = Cmip6Scenario::BaselineHistorical;
    let weather_nodes = default_synoptic_nodes_for_region(region, ssp);
    let radar_profiles = build_regional_radar_profiles(None, region, &weather_nodes, 0.0);
    let turbines = seed_wind_turbines_for_region(region);
    let patches = seed_milkweed_patches_for_region(region);
    let obs = baseline_observations_for_region(region);

    let curfew_levels = [0.0_f64, 20.0, 40.0, 60.0, 80.0, 100.0];
    let dt_hr = 0.25_f64;
    let steps = 144usize; // 36 simulated hours across nocturnal migration window

    let mut csv = String::from(
        "lights_out_pct,sim_hours,airborne_birds,skyglow_entrained_birds,cumulative_urban_strikes,baseline_unmitigated_strikes,avoided_urban_strikes_pct,theoretical_reduction_pct,effective_radiance_nw,lat_25_32_entrained,lat_32_38_entrained,lat_38_45_entrained,alt_0_300m_birds,alt_300_800m_birds,alt_800m_plus_birds,mean_alt_agl_m\n",
    );

    let mut baseline_strikes = 0u32;
    let mut curfew_80_strikes = 0u32;
    let mut curfew_80_avoided_pct = 0.0_f64;
    let mut baseline_entrained = 0usize;
    let mut curfew_80_entrained = 0usize;
    let mut sweep_records = Vec::new();

    for &curfew_pct in &curfew_levels {
        let mut engine = AeroPhysicsEngine::new();
        engine.reseed_agents(region, season);
        engine.weather_nodes = weather_nodes.clone();
        engine.radar_profiles = radar_profiles.clone();

        let mut params = SimulationParams::default();
        params.region = region;
        params.season = season;
        params.day_of_year = 268.0;
        params.cmip6_scenario = ssp;
        params.lights_out_curfew_pct = curfew_pct;

        let mut peak_nocturnal_strat = NocturnalStratification::default();
        let mut peak_nocturnal_airborne = 0usize;
        let mut peak_nocturnal_entrained = 0usize;

        for _ in 0..steps {
            engine.step(dt_hr, &params, &turbines, &patches, &obs);
            if engine.metrics.is_nocturnal
                && engine.metrics.skyglow_entrained_birds >= peak_nocturnal_entrained
            {
                peak_nocturnal_entrained = engine.metrics.skyglow_entrained_birds;
                peak_nocturnal_airborne = engine.metrics.avian_flux_agents_airborne;
                peak_nocturnal_strat = stratify_airborne_birds(&engine);
            }
        }

        let effective_radiance =
            effective_skyglow_radiance_nw(85.0, 2.0, true, curfew_pct, params.skyglow_intensity_mult);
        let theoretical_reduction = compute_lights_out_collision_reduction_pct(curfew_pct);

        if (curfew_pct - 0.0).abs() < 1e-9 {
            baseline_strikes = engine.metrics.cumulative_urban_strikes;
            baseline_entrained = peak_nocturnal_entrained;
        }
        if (curfew_pct - 80.0).abs() < 1e-9 {
            curfew_80_strikes = engine.metrics.cumulative_urban_strikes;
            curfew_80_avoided_pct = engine.metrics.avoided_urban_strikes_pct;
            curfew_80_entrained = peak_nocturnal_entrained;
        }

        assert!(
            engine.metrics.baseline_unmitigated_strikes > 0,
            "36-hr Atlantic nocturnal simulation should accumulate baseline skyglow strike exposure"
        );
        assert!(
            engine.metrics.cumulative_urban_strikes <= engine.metrics.baseline_unmitigated_strikes,
            "Curtailed strikes must never exceed baseline unmitigated strikes"
        );

        csv.push_str(&format!(
            "{:.1},{:.2},{},{},{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{},{:.2}\n",
            curfew_pct,
            engine.metrics.sim_hours,
            peak_nocturnal_airborne,
            peak_nocturnal_entrained,
            engine.metrics.cumulative_urban_strikes,
            engine.metrics.baseline_unmitigated_strikes,
            engine.metrics.avoided_urban_strikes_pct,
            theoretical_reduction,
            effective_radiance,
            peak_nocturnal_strat.lat_25_32_entrained,
            peak_nocturnal_strat.lat_32_38_entrained,
            peak_nocturnal_strat.lat_38_45_entrained,
            peak_nocturnal_strat.alt_0_300_birds,
            peak_nocturnal_strat.alt_300_800_birds,
            peak_nocturnal_strat.alt_800_plus_birds,
            peak_nocturnal_strat.mean_alt_agl_m,
        ));

        sweep_records.push(json!({
            "lights_out_pct": curfew_pct,
            "sim_hours": engine.metrics.sim_hours,
            "peak_nocturnal_airborne_birds": peak_nocturnal_airborne,
            "peak_nocturnal_entrained_birds": peak_nocturnal_entrained,
            "cumulative_urban_strikes": engine.metrics.cumulative_urban_strikes,
            "baseline_unmitigated_strikes": engine.metrics.baseline_unmitigated_strikes,
            "avoided_urban_strikes_pct": engine.metrics.avoided_urban_strikes_pct,
            "theoretical_collision_reduction_pct": theoretical_reduction,
            "effective_skyglow_radiance_nw": effective_radiance,
            "mean_alt_agl_m": peak_nocturnal_strat.mean_alt_agl_m
        }));
    }

    assert!(
        curfew_80_strikes < baseline_strikes && curfew_80_avoided_pct > 45.0,
        "80% Lights-Out intervention must substantially reduce urban collision mortality (baseline={baseline_strikes}, curfew_80={curfew_80_strikes}, avoided={curfew_80_avoided_pct:.2}%)"
    );
    assert!(
        curfew_80_entrained <= baseline_entrained,
        "80% Lights-Out intervention should not increase peak skyglow entrainment"
    );

    let csv_path = validation_dir().join("lights_out_mortality_curve.csv");
    let mut f = fs::File::create(&csv_path).expect("create lights_out_mortality_curve.csv");
    f.write_all(csv.as_bytes()).expect("write mortality CSV");

    write_summary_section(
        "t1_lights_out_collision_mortality",
        json!({
            "region": region.code(),
            "season": "AutumnFalloutAndBatPulse",
            "sim_hours": dt_hr * steps as f64,
            "baseline_strikes_0pct": baseline_strikes,
            "intervention_strikes_80pct": curfew_80_strikes,
            "avoided_urban_strikes_80pct": curfew_80_avoided_pct,
            "baseline_peak_entrained_birds": baseline_entrained,
            "intervention_peak_entrained_birds_80pct": curfew_80_entrained,
            "csv_artifact": "validation/lights_out_mortality_curve.csv",
            "sweep_levels": sweep_records
        }),
    );
}

#[test]
fn t2_nexrad_vpts_altitude_profile_integration() {
    let region = FlywayRegion::AtlanticAmericas;
    let stations = radar_stations_for_region(region);
    let fallback_station = &stations[0];

    // Construct a full NEXRAD S-band VPTS table with a 0 m ground/insect clutter gate (sd_vvp < 2.0)
    // followed by nocturnal passerine flight layers across 200..2600 m ASL.
    let vpts_csv = "\
radar,datetime,radar_latitude,radar_longitude,height,dens,u,v,eta,sd_vvp
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,0,5.2,0.8,1.1,64.0,1.10
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,200,14.8,2.1,6.4,510.0,2.85
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,400,31.6,3.4,9.8,1320.0,3.60
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,600,44.2,4.1,11.9,1910.0,4.25
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,800,36.5,4.6,12.2,1580.0,4.05
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,1000,24.1,4.8,11.5,1010.0,3.70
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,1200,15.3,4.2,10.1,640.0,3.35
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,1400,9.4,3.5,8.6,390.0,3.10
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,1600,5.6,2.9,7.4,225.0,2.85
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,1800,3.1,2.4,6.3,128.0,2.60
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,2000,1.7,2.0,5.2,72.0,2.40
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,2200,0.8,1.5,4.1,34.0,2.25
KDOX,2026-10-04T04:30Z,38.8256,-75.4400,2400,0.3,1.1,3.2,12.0,2.10
";

    let parsed = parse_vpts_csv_text(vpts_csv, fallback_station).expect("parse VPTS CSV text");
    assert_eq!(parsed.station_code, "KDOX");
    assert_eq!(
        parsed.bins.len(),
        12,
        "S-band sd_vvp < 2.0 m/s clutter gate at 0 m must be filtered out leaving 12 avian altitude bins"
    );
    assert!(
        (parsed.peak_altitude_m_asl - 600.0).abs() < 1e-6,
        "Peak migration altitude should resolve to 600.0 m ASL"
    );

    let (recomputed_mtr, recomputed_peak) = summarize_profile_mtr_and_peak(&parsed.bins);
    assert!((parsed.mtr_birds_km_hr - recomputed_mtr).abs() < 1e-6);
    assert!((parsed.peak_altitude_m_asl - recomputed_peak).abs() < 1e-6);

    let manual_mtr: f64 = parsed
        .bins
        .iter()
        .map(|b| b.bird_density_per_km3 * b.u_mps.hypot(b.v_mps) * 3.6 * 0.2)
        .sum();
    assert!(
        (parsed.mtr_birds_km_hr - manual_mtr).abs() < 1e-6,
        "Integrated MTR must exactly equal sum(dens * speed_km_hr * 0.2 km)"
    );

    let weather_nodes = default_synoptic_nodes_for_region(region, Cmip6Scenario::BaselineHistorical);
    let regional_profiles = build_regional_radar_profiles(None, region, &weather_nodes, 0.0);
    assert!(
        !regional_profiles.is_empty(),
        "Regional VPTS synthesis must produce profiles for flyway radar stations"
    );

    let mut csv = String::from(
        "profile_source,station_code,height_m_asl,bird_density_per_km3,u_mps,v_mps,ground_speed_mps,reflectivity_eta,sd_vvp_mps,bin_mtr_contrib_birds_km_hr,cumulative_mtr_birds_km_hr\n",
    );

    let mut cum_parsed = 0.0_f64;
    for b in &parsed.bins {
        let speed = b.u_mps.hypot(b.v_mps);
        let contrib = b.bird_density_per_km3 * speed * 3.6 * 0.2;
        cum_parsed += contrib;
        csv.push_str(&format!(
            "VPTS-CSV,{},{:.1},{:.4},{:.3},{:.3},{:.3},{:.2},{:.2},{:.4},{:.4}\n",
            parsed.station_code,
            b.height_m_asl,
            b.bird_density_per_km3,
            b.u_mps,
            b.v_mps,
            speed,
            b.reflectivity_eta,
            b.sd_vvp_mps,
            contrib,
            cum_parsed
        ));
    }

    let mut regional_summaries = Vec::new();
    for rp in &regional_profiles {
        let (rp_mtr, rp_peak) = summarize_profile_mtr_and_peak(&rp.bins);
        assert!((rp.mtr_birds_km_hr - rp_mtr).abs() < 1e-6);
        assert!((rp.peak_altitude_m_asl - rp_peak).abs() < 1e-6);

        let mut cum_rp = 0.0_f64;
        for b in &rp.bins {
            let speed = b.u_mps.hypot(b.v_mps);
            let contrib = b.bird_density_per_km3 * speed * 3.6 * 0.2;
            cum_rp += contrib;
            csv.push_str(&format!(
                "Regional-Model,{},{:.1},{:.4},{:.3},{:.3},{:.3},{:.2},{:.2},{:.4},{:.4}\n",
                rp.station_code,
                b.height_m_asl,
                b.bird_density_per_km3,
                b.u_mps,
                b.v_mps,
                speed,
                b.reflectivity_eta,
                b.sd_vvp_mps,
                contrib,
                cum_rp
            ));
        }

        regional_summaries.push(json!({
            "station_code": rp.station_code,
            "station_name": rp.station_name,
            "bin_count": rp.bins.len(),
            "mtr_birds_km_hr": rp.mtr_birds_km_hr,
            "peak_altitude_m_asl": rp.peak_altitude_m_asl
        }));
    }

    let csv_path = validation_dir().join("vpts_altitude_profile.csv");
    let mut f = fs::File::create(&csv_path).expect("create vpts_altitude_profile.csv");
    f.write_all(csv.as_bytes()).expect("write VPTS CSV");

    write_summary_section(
        "t2_nexrad_vpts_profile_integration",
        json!({
            "parsed_station": parsed.station_code,
            "clutter_rows_filtered": 1,
            "retained_avian_bins": parsed.bins.len(),
            "parsed_mtr_birds_km_hr": parsed.mtr_birds_km_hr,
            "parsed_peak_altitude_m_asl": parsed.peak_altitude_m_asl,
            "peak_bin_density_per_km3": parsed.bins[2].bird_density_per_km3,
            "regional_station_count": regional_profiles.len(),
            "regional_profiles": regional_summaries,
            "csv_artifact": "validation/vpts_altitude_profile.csv"
        }),
    );
}

#[test]
fn t3_bioacoustic_nfc_stft_classification_and_sqlite_roundtrip() {
    let temp = TempDirGuard::new("aeroflyway_t3_acoustics");
    let wav_path = temp.path.join("passerine_nfc_duet.wav");
    let db_path = temp.path.join("aeroflyway_validation.db");

    let duet = synthesize_preset_recording(SynthPreset::PasserineNfcDuet48k);
    let coords = (40.7128_f64, -74.0060_f64);
    write_wav_16bit(&wav_path, &duet, Some(coords)).expect("write 16-bit GUANO WAV");
    let reloaded = load_wav_file(&wav_path).expect("reload 16-bit WAV");

    assert_eq!(reloaded.sample_rate_hz, 48_000);
    assert_eq!(reloaded.guano_lat_lon, Some(coords));
    assert_eq!(reloaded.samples_mono.len(), duet.samples_mono.len());

    let duet_spec = compute_stft_spectrogram(&reloaded, 1024);
    assert!(
        duet_spec.pulses.len() >= 2,
        "Duet spectrogram must resolve >= 2 distinct nocturnal flight calls (found {})",
        duet_spec.pulses.len()
    );
    assert!(duet_spec.aci_index > 0.0, "ACI complexity index must be positive");

    // Isolate the Thrush call window (0..300 ms = 0..14400 samples at 48 kHz)
    let split_idx = ((reloaded.sample_rate_hz as f64) * 0.300) as usize;
    let thrush_rec = WavRecording {
        source_label: "Thrush-Window".to_string(),
        sample_rate_hz: reloaded.sample_rate_hz,
        channels: 1,
        bits_per_sample: 16,
        samples_mono: reloaded.samples_mono[..split_idx].to_vec(),
        guano_lat_lon: reloaded.guano_lat_lon,
    };
    let warbler_rec = WavRecording {
        source_label: "Warbler-Window".to_string(),
        sample_rate_hz: reloaded.sample_rate_hz,
        channels: 1,
        bits_per_sample: 16,
        samples_mono: reloaded.samples_mono[split_idx..].to_vec(),
        guano_lat_lon: reloaded.guano_lat_lon,
    };

    let thrush_spec = compute_stft_spectrogram(&thrush_rec, 1024);
    let warbler_spec = compute_stft_spectrogram(&warbler_rec, 1024);

    let (thrush_low_peak_db, thrush_low_mean_db) = band_peak_and_mean_db(&thrush_spec, 2.3, 4.2);
    let (thrush_high_peak_db, thrush_high_mean_db) = band_peak_and_mean_db(&thrush_spec, 6.2, 8.2);
    let (warbler_low_peak_db, warbler_low_mean_db) = band_peak_and_mean_db(&warbler_spec, 2.3, 4.2);
    let (warbler_high_peak_db, warbler_high_mean_db) =
        band_peak_and_mean_db(&warbler_spec, 6.2, 8.2);

    assert!(
        (2.3..=4.3).contains(&thrush_spec.characteristic_freq_khz),
        "Thrush window characteristic freq should be ~3.2 kHz, got {:.3}",
        thrush_spec.characteristic_freq_khz
    );
    assert!(
        thrush_spec.classified_taxon == Taxon::Bird
            && thrush_spec.guild_label.to_lowercase().contains("thrush"),
        "Expected Thrush guild classification, got '{}'",
        thrush_spec.guild_label
    );
    assert!(
        thrush_low_peak_db > thrush_high_peak_db + 12.0,
        "Thrush low-band peak ({thrush_low_peak_db:.2} dBFS) must dominate high band ({thrush_high_peak_db:.2} dBFS)"
    );

    assert!(
        (6.2..=8.2).contains(&warbler_spec.characteristic_freq_khz),
        "Warbler window characteristic freq should be ~7.0-7.4 kHz, got {:.3}",
        warbler_spec.characteristic_freq_khz
    );
    assert!(
        warbler_spec.classified_taxon == Taxon::Bird
            && warbler_spec.guild_label.to_lowercase().contains("warbler"),
        "Expected Warbler guild classification, got '{}'",
        warbler_spec.guild_label
    );
    assert!(
        warbler_high_peak_db > warbler_low_peak_db + 12.0,
        "Warbler high-band peak ({warbler_high_peak_db:.2} dBFS) must dominate low band ({warbler_low_peak_db:.2} dBFS)"
    );

    // SQLite campaign, radar bin, snapshot, and bioacoustic detection persistence roundtrip
    let store = FlywayStore::open(&db_path).expect("open SQLite FlywayStore");
    store
        .save_campaign_metadata(
            "Atlantic Headless NFC Validation",
            FlywayRegion::AtlanticAmericas,
            268.0,
            "historical",
            80.0,
        )
        .expect("save campaign metadata");

    let new_obs = vec![
        ObservationRecord {
            obs_uid: "val-nfc-thrush-01".to_string(),
            taxon: Taxon::Bird,
            common_name: "Swainson's Thrush (NFC)".to_string(),
            scientific_name: "Catharus ustulatus".to_string(),
            lat: coords.0,
            lon: coords.1,
            hex_bin: spatial_hex_bin(coords.0, coords.1),
            altitude_m: 600.0,
            count: 1,
            source_api: "AeroFlyway-STFT".to_string(),
            observed_at: "2026-11-14T05:10Z".to_string(),
        },
        ObservationRecord {
            obs_uid: "val-nfc-warbler-01".to_string(),
            taxon: Taxon::Bird,
            common_name: "Blackpoll Warbler (NFC)".to_string(),
            scientific_name: "Setophaga striata".to_string(),
            lat: coords.0,
            lon: coords.1,
            hex_bin: spatial_hex_bin(coords.0, coords.1),
            altitude_m: 540.0,
            count: 1,
            source_api: "AeroFlyway-STFT".to_string(),
            observed_at: "2026-11-14T05:12Z".to_string(),
        },
    ];
    let upserted_obs = store
        .upsert_observations(&new_obs)
        .expect("upsert bioacoustic NFC detections");
    assert_eq!(upserted_obs, 2);

    let radar_bins_inserted = store
        .upsert_radar_bins(
            "KDOX",
            "2026-10-04T04:30Z",
            &[
                (400.0, 31.6, 3.4, 9.8, 1320.0),
                (600.0, 44.2, 4.1, 11.9, 1910.0),
            ],
        )
        .expect("upsert radar VPTS bins");
    assert_eq!(radar_bins_inserted, 2);

    store.record_snapshot(&ModelSnapshot {
        tick_hour: 12.0,
        enkf_rmse: 0.142,
        avian_flux_index: 38.5,
        bat_rotor_contacts: 0,
        monarch_front_lat: 36.2,
    });

    drop(store);
    let reopened = FlywayStore::open(&db_path).expect("reopen SQLite FlywayStore");
    let summary = reopened.load_campaign_summary("aeroflyway_validation.db");
    let loaded_obs = reopened.load_observations();

    assert_eq!(summary.campaign_name, "Atlantic Headless NFC Validation");
    assert_eq!(summary.region_code, FlywayRegion::AtlanticAmericas.code());
    assert!((summary.lights_out_pct - 80.0).abs() < 1e-6);
    assert!(loaded_obs.iter().any(|o| o.obs_uid == "val-nfc-thrush-01"));
    assert!(loaded_obs.iter().any(|o| o.obs_uid == "val-nfc-warbler-01"));

    write_summary_section(
        "t3_bioacoustic_stft_sqlite_roundtrip",
        json!({
            "wav_sample_rate_hz": reloaded.sample_rate_hz,
            "wav_duration_ms": reloaded.duration_ms(),
            "guano_lat": coords.0,
            "guano_lon": coords.1,
            "duet_pulses_detected": duet_spec.pulses.len(),
            "duet_aci_index": duet_spec.aci_index,
            "thrush_characteristic_freq_khz": thrush_spec.characteristic_freq_khz,
            "thrush_guild": thrush_spec.guild_label,
            "thrush_band_2_3_4_2_peak_dbfs": thrush_low_peak_db,
            "thrush_band_2_3_4_2_mean_dbfs": thrush_low_mean_db,
            "thrush_band_6_2_8_2_peak_dbfs": thrush_high_peak_db,
            "thrush_band_6_2_8_2_mean_dbfs": thrush_high_mean_db,
            "warbler_characteristic_freq_khz": warbler_spec.characteristic_freq_khz,
            "warbler_guild": warbler_spec.guild_label,
            "warbler_band_6_2_8_2_peak_dbfs": warbler_high_peak_db,
            "warbler_band_6_2_8_2_mean_dbfs": warbler_high_mean_db,
            "warbler_band_2_3_4_2_peak_dbfs": warbler_low_peak_db,
            "warbler_band_2_3_4_2_mean_dbfs": warbler_low_mean_db,
            "sqlite_total_observations": loaded_obs.len(),
            "sqlite_upserted_nfc_records": upserted_obs,
            "sqlite_upserted_radar_bins": radar_bins_inserted,
            "sqlite_campaign_name": summary.campaign_name,
            "sqlite_campaign_lights_out_pct": summary.lights_out_pct
        }),
    );
}
