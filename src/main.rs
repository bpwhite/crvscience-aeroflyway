//! AeroFlyway v0.6.0 — Native Rust `wgpu` + WGSL 100k-Agent Multi-Flyway GIS Workstation:
//! - Startup Campaign Hub (`AppStage::StartupHub`) + Native Persistent Menu Bar (`egui::menu::bar`)
//! - VIIRS Nighttime Skyglow Urban Collision Modeling & BirdCast `Lights-Out` Curfew Sliders
//! - NEXRAD WSR-88D & ENRAM Aloft Vertical Radar Profile (`VPTS`) Ingestion & Altitude Explorer
//! - Continuous 365-Day DOY Phenology, Open-Meteo CMIP6 + CAMS PM2.5 Smoke Forcing & WAV STFT Lab.

mod acoustics;
mod basemap;
mod cli;
mod db;
mod export;
mod gpu_pipeline;
mod ingest;
mod map_engine;
mod physics;

use acoustics::{
    compute_stft_spectrogram, load_wav_file, parse_wav_bytes, synthesize_preset_recording,
    write_wav_16bit, SpectrogramResult, SynthPreset, WavRecording,
};
use basemap::{
    landmarks_for_region, radar_stations_for_region, skyglow_domes_for_region, FlywayRegion,
    LandmarkKind, MilkweedPatch, Taxon, WindTurbine, COLUMBIA_RIVER, KLAMATH_SACRAMENTO_CORRIDOR,
    PACIFIC_COASTLINE, PUGET_SOUND_RING, STATE_BOUNDARIES, WILLAMETTE_RIVER,
};
use cli::{
    execute_batch_sweep_grid, try_dispatch_cli, write_batch_sweep_reports, BatchScenarioOutcome,
    CliDispatchOutcome,
};
use db::{
    hex_bin_center, spatial_hex_bin, CampaignSummary, FlywayStore, ModelSnapshot, ObservationRecord,
};
use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2,
};
use eframe::egui_wgpu;
use export::{export_corridor_csv_bundle, export_corridor_geojson};
use gpu_pipeline::{
    pack_gpu_scene_buffers, pack_gpu_uniforms, FlywayGpuResources, GpuReadbackSummary,
    MapGpuSimAndPaintCallback, SharedReadbackHandle, TransectGpuPaintCallback, DEFAULT_GPU_AGENTS,
    MAX_GPU_AGENTS,
};
use ingest::{
    build_regional_radar_profiles, load_vpts_csv_file, spawn_live_ingest, Cmip6Scenario,
    IngestPayload,
};
use map_engine::{haversine_km, BasemapProvider, MapViewport};
use physics::{
    compute_lights_out_collision_reduction_pct, default_synoptic_nodes_for_region,
    effective_skyglow_radiance_nw, AeroPhysicsEngine, SeasonPreset, SimulationParams,
};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AppStage {
    StartupHub,
    Workbench,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapToolMode {
    PanAndInspect,
    DrawTransect,
    PlantVirtualMilkweed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BottomDockTab {
    AirspaceTransect,
    RadarVerticalProfile,
    AcousticSpectrogram,
    BatchSweepLab,
}

struct LayerToggles {
    birds: bool,
    bats: bool,
    butterflies: bool,
    wake_trails: bool,
    skyglow_domes: bool,
    radar_rings: bool,
    observations: bool,
    turbines: bool,
    milkweed: bool,
    hex_heatmap: bool,
    wind_barbs: bool,
    landmarks: bool,
}

impl Default for LayerToggles {
    fn default() -> Self {
        Self {
            birds: true,
            bats: true,
            butterflies: true,
            wake_trails: true,
            skyglow_domes: true,
            radar_rings: true,
            observations: true,
            turbines: true,
            milkweed: true,
            hex_heatmap: true,
            wind_barbs: true,
            landmarks: true,
        }
    }
}

struct AcousticInspectorState {
    recording: WavRecording,
    spectrogram: SpectrogramResult,
    texture: Option<TextureHandle>,
    texture_dirty: bool,
    fft_size: usize,
    db_floor: f32,
    view_max_khz: f64,
    show_ridge_overlay: bool,
    wav_path_input: String,
    pinned_counter: usize,
}

impl AcousticInspectorState {
    fn new_with_default_preset() -> Self {
        let rec = synthesize_preset_recording(SynthPreset::HoaryBatUltrasonic250k);
        let max_khz = rec.nyquist_khz().min(60.0);
        let spec = compute_stft_spectrogram(&rec, 1024);
        Self {
            recording: rec,
            spectrogram: spec,
            texture: None,
            texture_dirty: true,
            fft_size: 1024,
            db_floor: -76.0,
            view_max_khz: max_khz,
            show_ridge_overlay: true,
            wav_path_input: String::new(),
            pinned_counter: 1,
        }
    }

    fn load_new_recording(&mut self, rec: WavRecording) {
        let default_top = if rec.nyquist_khz() > 40.0 {
            rec.nyquist_khz().min(95.0)
        } else {
            rec.nyquist_khz().min(14.0)
        };
        self.view_max_khz = default_top.max(4.0);
        self.spectrogram = compute_stft_spectrogram(&rec, self.fft_size);
        self.recording = rec;
        self.texture_dirty = true;
    }

    fn recompute_stft(&mut self) {
        self.spectrogram = compute_stft_spectrogram(&self.recording, self.fft_size);
        self.texture_dirty = true;
    }

    fn ensure_texture(&mut self, ctx: &egui::Context) -> &TextureHandle {
        if self.texture.is_none() || self.texture_dirty {
            let img = self
                .spectrogram
                .to_color_image(self.db_floor, self.view_max_khz);
            self.texture = Some(ctx.load_texture(
                "bioacoustic_stft_spectrogram",
                img,
                TextureOptions::LINEAR,
            ));
            self.texture_dirty = false;
        }
        self.texture.as_ref().expect("spectrogram texture initialized")
    }
}

struct BatchSweepUiState {
    sweep_hours: f64,
    cut_in_csv: String,
    lights_out_csv: String,
    milkweed_csv: String,
    temp_csv: String,
    smoke_csv: String,
    running_sweep: bool,
    outcomes: Vec<BatchScenarioOutcome>,
    sweep_tx: Sender<Vec<BatchScenarioOutcome>>,
    sweep_rx: Receiver<Vec<BatchScenarioOutcome>>,
}

impl BatchSweepUiState {
    fn new() -> Self {
        let (sweep_tx, sweep_rx) = mpsc::channel();
        Self {
            sweep_hours: 48.0,
            cut_in_csv: "3.5, 5.0, 6.0".to_string(),
            lights_out_csv: "0.0, 50.0, 85.0".to_string(),
            milkweed_csv: "0, 6".to_string(),
            temp_csv: "0.0, 2.0".to_string(),
            smoke_csv: "0.0, 180.0".to_string(),
            running_sweep: false,
            outcomes: Vec::new(),
            sweep_tx,
            sweep_rx,
        }
    }
}

struct AeroFlywayApp {
    stage: AppStage,
    campaign_name_input: String,
    campaign_db_path_input: String,
    active_campaign: CampaignSummary,
    show_help_modal: bool,
    selected_radar_idx: usize,
    vpts_csv_path_input: String,

    store: FlywayStore,
    map: MapViewport,
    engine: AeroPhysicsEngine,
    params: SimulationParams,
    layers: LayerToggles,
    observations: Vec<ObservationRecord>,
    turbines: Vec<WindTurbine>,
    milkweed: Vec<MilkweedPatch>,

    gpu_available: bool,
    use_gpu_compute: bool,
    gpu_agent_count: u32,
    gpu_tick_seed: u32,
    last_frame_sim_dt_hr: f64,
    gpu_readback: Arc<SharedReadbackHandle>,
    gpu_summary: Option<GpuReadbackSummary>,
    gpu_cum_rotor_hits: u64,
    gpu_cum_baseline_hits: u64,
    gpu_cum_urban_strikes: u64,
    gpu_cum_urban_baseline: u64,

    bottom_tab: BottomDockTab,
    acoustics: AcousticInspectorState,
    batch_ui: BatchSweepUiState,
    export_dir_input: String,

    running: bool,
    sim_speed_hr_per_sec: f64,
    tool_mode: MapToolMode,
    transect_drag_anchor: Option<(f64, f64)>,
    virtual_patch_counter: usize,
    ebird_api_token: String,
    ingest_in_progress: bool,
    status_line: String,
    ingest_tx: Sender<IngestPayload>,
    ingest_rx: Receiver<IngestPayload>,
    snapshot_accum_hr: f64,
}

impl AeroFlywayApp {
    fn new(cc: &eframe::CreationContext<'_>, skip_startup_hub: bool) -> Self {
        configure_dark_telemetry_visuals(&cc.egui_ctx);

        let db_path = PathBuf::from("aeroflyway.db");
        let store = FlywayStore::open(&db_path).expect("Failed to initialize aeroflyway.db");
        let active_campaign = store.load_campaign_summary("aeroflyway.db");
        let observations = store.load_observations();
        let turbines = store.load_turbines();
        let milkweed = store.load_milkweed_patches();

        let params = SimulationParams::default();
        let gpu_readback = SharedReadbackHandle::new();
        let mut gpu_available = false;

        if let Some(wgpu_render_state) = cc.wgpu_render_state.as_ref() {
            let gpu_res = FlywayGpuResources::initialize(
                wgpu_render_state,
                params.region,
                params.season,
                Arc::clone(&gpu_readback),
            );
            wgpu_render_state
                .renderer
                .write()
                .callback_resources
                .insert(gpu_res);
            gpu_available = true;
        }

        let (ingest_tx, ingest_rx) = mpsc::channel();
        let ebird_env = std::env::var("EBIRD_API_KEY").unwrap_or_default();

        let init_status = if gpu_available {
            format!(
                "v0.6 ready — Startup Hub, VIIRS Lights-Out, NEXRAD/Aloft VPTS & wgpu ({DEFAULT_GPU_AGENTS} agents)"
            )
        } else {
            "v0.6 ready — Startup Hub, VIIRS Lights-Out, NEXRAD/Aloft VPTS & CPU Rayon".to_string()
        };

        Self {
            stage: if skip_startup_hub {
                AppStage::Workbench
            } else {
                AppStage::StartupHub
            },
            campaign_name_input: active_campaign.campaign_name.clone(),
            campaign_db_path_input: "aeroflyway.db".to_string(),
            active_campaign,
            show_help_modal: false,
            selected_radar_idx: 0,
            vpts_csv_path_input: String::new(),
            store,
            map: MapViewport::new(db_path),
            engine: AeroPhysicsEngine::new(),
            params,
            layers: LayerToggles::default(),
            observations,
            turbines,
            milkweed,
            gpu_available,
            use_gpu_compute: gpu_available,
            gpu_agent_count: DEFAULT_GPU_AGENTS,
            gpu_tick_seed: 1,
            last_frame_sim_dt_hr: 0.0,
            gpu_readback,
            gpu_summary: None,
            gpu_cum_rotor_hits: 0,
            gpu_cum_baseline_hits: 0,
            gpu_cum_urban_strikes: 0,
            gpu_cum_urban_baseline: 0,
            bottom_tab: BottomDockTab::AirspaceTransect,
            acoustics: AcousticInspectorState::new_with_default_preset(),
            batch_ui: BatchSweepUiState::new(),
            export_dir_input: "exports".to_string(),
            running: true,
            sim_speed_hr_per_sec: 2.0,
            tool_mode: MapToolMode::PanAndInspect,
            transect_drag_anchor: None,
            virtual_patch_counter: 1,
            ebird_api_token: ebird_env,
            ingest_in_progress: false,
            status_line: init_status,
            ingest_tx,
            ingest_rx,
            snapshot_accum_hr: 0.0,
        }
    }

    fn gpu_mode_active(&self) -> bool {
        self.gpu_available && self.use_gpu_compute
    }

    fn current_scale_factor(&self) -> f64 {
        if self.gpu_mode_active() {
            (self.gpu_agent_count as f64) / 2400.0
        } else {
            1.0
        }
    }

    fn open_or_switch_campaign_db(&mut self, db_file: &str, title: &str, region: FlywayRegion) {
        let p = PathBuf::from(db_file.trim());
        if let Ok(new_store) = FlywayStore::open(&p) {
            self.store = new_store;
            self.campaign_db_path_input = db_file.trim().to_string();
            self.campaign_name_input = title.to_string();
            self.apply_region_switch(region);
            let _ = self.store.save_campaign_metadata(
                title,
                self.params.region,
                self.params.day_of_year,
                self.params.cmip6_scenario.code(),
                self.params.lights_out_curfew_pct,
            );
            self.active_campaign = self.store.load_campaign_summary(db_file.trim());
            self.status_line = format!("Opened campaign '{}' ({})", title, db_file.trim());
        }
    }

    fn reset_all_counters(&mut self) {
        self.engine.metrics.cumulative_bat_rotor_contacts = 0;
        self.engine.metrics.baseline_uncurtailed_contacts = 0;
        self.engine.metrics.cumulative_urban_strikes = 0;
        self.engine.metrics.baseline_unmitigated_strikes = 0;
        self.gpu_cum_rotor_hits = 0;
        self.gpu_cum_baseline_hits = 0;
        self.gpu_cum_urban_strikes = 0;
        self.gpu_cum_urban_baseline = 0;
    }

    fn apply_region_switch(&mut self, new_region: FlywayRegion) {
        self.params.region = new_region;
        self.selected_radar_idx = 0;
        let (clat, clon, zoom) = new_region.default_viewport();
        let (ta, tb) = new_region.default_transect();
        self.params.transect_start = ta;
        self.params.transect_end = tb;
        self.map.jump_to(clat, clon, zoom);

        let _ = self.store.switch_region_seed(new_region);
        self.observations = self.store.load_observations();
        self.turbines = self.store.load_turbines();
        self.milkweed = self.store.load_milkweed_patches();

        self.engine.weather_nodes =
            default_synoptic_nodes_for_region(new_region, self.params.cmip6_scenario);
        self.engine.radar_profiles = build_regional_radar_profiles(
            None,
            new_region,
            &self.engine.weather_nodes,
            self.params.smoke_pm25_offset_ugm3,
        );
        self.engine.cached_transect_dem = None;
        self.engine.reseed_agents(new_region, self.params.season);
        self.reset_all_counters();
        self.status_line = format!(
            "Switched flyway corridor -> {} (DOY {:.0}, Lights-Out {:.0}%)",
            new_region.label(),
            self.params.day_of_year,
            self.params.lights_out_curfew_pct
        );
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.running = !self.running;
            }
            if i.key_pressed(egui::Key::F1) {
                self.show_help_modal = !self.show_help_modal;
            }
            if i.key_pressed(egui::Key::H) {
                self.stage = match self.stage {
                    AppStage::StartupHub => AppStage::Workbench,
                    AppStage::Workbench => AppStage::StartupHub,
                };
            }
            if i.key_pressed(egui::Key::R) {
                self.stage = AppStage::Workbench;
                self.bottom_tab = BottomDockTab::RadarVerticalProfile;
            }
            if i.key_pressed(egui::Key::L) {
                self.params.lights_out_curfew_pct = if self.params.lights_out_curfew_pct < 70.0 {
                    85.0
                } else {
                    0.0
                };
            }
        });
    }

    fn handle_dropped_wav_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        for file in dropped {
            if let Some(path) = &file.path {
                let ext = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if ext == "csv" {
                    let stations = radar_stations_for_region(self.params.region);
                    if let Some(st) = stations.first() {
                        if let Ok(prof) = load_vpts_csv_file(path, st) {
                            self.status_line = format!(
                                "Imported VPTS radar profile '{}' ({:.0} birds/km/hr MTR)",
                                prof.station_code, prof.mtr_birds_km_hr
                            );
                            if self.engine.radar_profiles.is_empty() {
                                self.engine.radar_profiles.push(prof);
                            } else {
                                self.engine.radar_profiles[0] = prof;
                            }
                            self.selected_radar_idx = 0;
                            self.stage = AppStage::Workbench;
                            self.bottom_tab = BottomDockTab::RadarVerticalProfile;
                            return;
                        }
                    }
                }
                match load_wav_file(path) {
                    Ok(rec) => {
                        let summary = format!(
                            "Loaded WAV '{}' ({:.1} kHz, {:.0} ms)",
                            rec.source_label,
                            rec.sample_rate_hz as f64 / 1000.0,
                            rec.duration_ms()
                        );
                        self.acoustics.load_new_recording(rec);
                        self.stage = AppStage::Workbench;
                        self.bottom_tab = BottomDockTab::AcousticSpectrogram;
                        self.status_line = summary;
                        return;
                    }
                    Err(err) => {
                        self.status_line = format!("WAV load error: {err}");
                    }
                }
            } else if let Some(bytes) = &file.bytes {
                let name = if file.name.is_empty() {
                    "dropped_audio.wav".to_string()
                } else {
                    file.name.clone()
                };
                match parse_wav_bytes(bytes, name) {
                    Ok(rec) => {
                        self.acoustics.load_new_recording(rec);
                        self.stage = AppStage::Workbench;
                        self.bottom_tab = BottomDockTab::AcousticSpectrogram;
                        self.status_line =
                            "Loaded dropped WAV byte buffer into STFT Inspector".to_string();
                        return;
                    }
                    Err(err) => {
                        self.status_line = format!("WAV parse error: {err}");
                    }
                }
            }
        }
    }

    fn pin_current_acoustic_detection(&mut self) {
        let (lat, lon) = self
            .acoustics
            .recording
            .guano_lat_lon
            .unwrap_or((self.map.center_lat, self.map.center_lon));
        let spec = &self.acoustics.spectrogram;
        let alt_m = match spec.classified_taxon {
            Taxon::Bat => 115.0,
            Taxon::Bird => 420.0,
            Taxon::Butterfly => 18.0,
        };
        let uid = format!(
            "wav-stft-{:04}-{:.0}khz",
            self.acoustics.pinned_counter, spec.characteristic_freq_khz
        );
        self.acoustics.pinned_counter += 1;

        let rec = ObservationRecord {
            obs_uid: uid,
            taxon: spec.classified_taxon,
            common_name: format!(
                "{} (Fc {:.1} kHz, ACI {:.1})",
                spec.guild_label, spec.characteristic_freq_khz, spec.aci_index
            ),
            scientific_name: spec.scientific_hint.clone(),
            lat,
            lon,
            hex_bin: spatial_hex_bin(lat, lon),
            altitude_m: alt_m,
            count: (spec.pulses.len().max(1)) as u32,
            source_api: format!("WAV-STFT/{}", self.acoustics.recording.source_label),
            observed_at: "2026-11-14T04:15Z".to_string(),
        };

        if self.store.upsert_observations(&[rec]).is_ok() {
            self.observations = self.store.load_observations();
            self.map.jump_to(lat, lon, self.map.zoom.max(7.8));
            self.status_line = format!(
                "Pinned '{}' at ({lat:.4}, {lon:.4}) into SQLite & GIS map",
                spec.guild_label
            );
        }
    }

    fn trigger_live_sync(&mut self, ctx: &egui::Context) {
        if self.ingest_in_progress {
            return;
        }
        self.ingest_in_progress = true;
        self.status_line = format!(
            "Polling Open-Meteo, CMIP6 ({}), CAMS PM2.5, Radar S3, GLO-90 DEM & biodiversity...",
            self.params.cmip6_scenario.code().to_uppercase()
        );
        spawn_live_ingest(
            self.ingest_tx.clone(),
            self.params.region,
            self.map.center_lat,
            self.map.center_lon,
            self.params.cmip6_scenario,
            self.params.transect_start,
            self.params.transect_end,
            self.ebird_api_token.clone(),
            ctx.clone(),
        );
    }

    fn poll_background_channels(&mut self) {
        while let Ok(payload) = self.ingest_rx.try_recv() {
            self.ingest_in_progress = false;
            if !payload.observations.is_empty() {
                let _ = self.store.upsert_observations(&payload.observations);
                self.observations = self.store.load_observations();
            }
            if !payload.turbines.is_empty() {
                let _ = self.store.upsert_turbines(&payload.turbines);
                self.turbines = self.store.load_turbines();
            }
            if !payload.weather_nodes.is_empty() {
                self.engine.weather_nodes = payload.weather_nodes;
            }
            if !payload.radar_profiles.is_empty() {
                self.engine.radar_profiles = payload.radar_profiles;
            }
            if payload.transect_dem_m.is_some() {
                self.engine.cached_transect_dem = payload.transect_dem_m;
            }
            self.status_line = payload.status_summary;
        }

        while let Ok(sweep_res) = self.batch_ui.sweep_rx.try_recv() {
            self.batch_ui.running_sweep = false;
            let out_dir = self.export_dir_input.trim().to_string();
            if let Ok((csv_p, _)) = write_batch_sweep_reports(&out_dir, &sweep_res) {
                self.status_line = format!(
                    "Completed {} Rayon v0.6 sweep scenarios -> {}",
                    sweep_res.len(),
                    csv_p.display()
                );
            }
            self.batch_ui.outcomes = sweep_res;
        }

        if let Ok(mut guard) = self.gpu_readback.latest.lock() {
            if let Some(summary) = guard.take() {
                self.gpu_cum_rotor_hits = self
                    .gpu_cum_rotor_hits
                    .saturating_add(summary.bat_rotor_hits_step as u64);
                self.gpu_cum_baseline_hits = self
                    .gpu_cum_baseline_hits
                    .saturating_add(summary.bat_baseline_hits_step as u64);
                self.gpu_cum_urban_strikes = self
                    .gpu_cum_urban_strikes
                    .saturating_add(summary.urban_strikes_step as u64);
                self.gpu_cum_urban_baseline = self
                    .gpu_cum_urban_baseline
                    .saturating_add(summary.urban_baseline_strikes_step as u64);
                self.gpu_summary = Some(summary);
            }
        }
    }

    fn render_native_menu_bar(&mut self, ctx: &egui::Context) {
        let mut pending_region: Option<FlywayRegion> = None;

        egui::TopBottomPanel::top("native_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Startup Campaign Hub (H)").clicked() {
                        self.stage = AppStage::StartupHub;
                        ui.close_menu();
                    }
                    if ui.button("Save Campaign Metadata to SQLite").clicked() {
                        let _ = self.store.save_campaign_metadata(
                            &self.campaign_name_input,
                            self.params.region,
                            self.params.day_of_year,
                            self.params.cmip6_scenario.code(),
                            self.params.lights_out_curfew_pct,
                        );
                        self.active_campaign = self
                            .store
                            .load_campaign_summary(&self.campaign_db_path_input);
                        self.status_line = format!(
                            "Saved campaign '{}' into {}",
                            self.campaign_name_input, self.campaign_db_path_input
                        );
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Export RFC 7946 GeoJSON").clicked() {
                        if let Ok((p, n)) = export_corridor_geojson(
                            self.export_dir_input.trim(),
                            &self.engine,
                            &self.params,
                            &self.observations,
                            &self.turbines,
                            &self.milkweed,
                            self.current_scale_factor(),
                        ) {
                            self.status_line = format!("Exported {n} GeoJSON features -> {}", p.display());
                        }
                        ui.close_menu();
                    }
                    if ui.button("Export Corridor CSV Bundle").clicked() {
                        if let Ok((hp, _, n)) = export_corridor_csv_bundle(
                            self.export_dir_input.trim(),
                            &self.engine,
                            &self.params,
                            &self.observations,
                            self.current_scale_factor(),
                        ) {
                            self.status_line = format!("Exported {n} CSV rows -> {}", hp.display());
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Reset Collision & Strike Counters").clicked() {
                        self.reset_all_counters();
                        ui.close_menu();
                    }
                });

                ui.menu_button("Corridor", |ui| {
                    for r in FlywayRegion::ALL {
                        if ui
                            .selectable_label(self.params.region == r, r.label())
                            .clicked()
                        {
                            pending_region = Some(r);
                            self.stage = AppStage::Workbench;
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    for (lbl, doy) in [
                        ("Spring Surge (DOY 118)", 118.0),
                        ("Summer Breeding (DOY 192)", 192.0),
                        ("Autumn Fallout & Bats (DOY 272)", 272.0),
                        ("Overwinter Sanctuary (DOY 345)", 345.0),
                    ] {
                        if ui.button(lbl).clicked() {
                            self.params.day_of_year = doy;
                            self.params.season = SeasonPreset::from_doy(doy);
                            ui.close_menu();
                        }
                    }
                });

                ui.menu_button("Layers", |ui| {
                    ui.checkbox(&mut self.layers.birds, "Birds (850 hPa / Skyglow / Smoke)");
                    ui.checkbox(&mut self.layers.bats, "Bats (Chiroptera Rotor Zone)");
                    ui.checkbox(&mut self.layers.butterflies, "Butterflies (Lepidoptera GDD)");
                    ui.separator();
                    ui.checkbox(&mut self.layers.skyglow_domes, "VIIRS Nighttime Skyglow Domes");
                    ui.checkbox(&mut self.layers.radar_rings, "NEXRAD / Aloft Radar Rings");
                    ui.checkbox(&mut self.layers.turbines, "Wind Turbines & Rotors");
                    ui.checkbox(&mut self.layers.milkweed, "Host-Plant Stepping-Stones");
                    ui.checkbox(&mut self.layers.hex_heatmap, "Spatial H25 Hex Choropleth");
                    ui.checkbox(&mut self.layers.wind_barbs, "Synoptic 850 hPa Wind Vectors");
                    ui.checkbox(&mut self.layers.landmarks, "Refuges, Motus & Summits");
                });

                ui.menu_button("Tools", |ui| {
                    if ui.button("Sync Live APIs + Radar S3").clicked() {
                        self.trigger_live_sync(ctx);
                        ui.close_menu();
                    }
                    if ui.button("Apply BirdCast 85% Lights-Out Curfew (L)").clicked() {
                        self.params.lights_out_curfew_pct = 85.0;
                        ui.close_menu();
                    }
                    if ui.button("Unmitigated Urban Skyglow (0% Curfew)").clicked() {
                        self.params.lights_out_curfew_pct = 0.0;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Simulate Megafire Smoke Plume (195 ug/m3)").clicked() {
                        self.params.smoke_pm25_offset_ugm3 = 195.0;
                        self.engine.radar_profiles = build_regional_radar_profiles(
                            None,
                            self.params.region,
                            &self.engine.weather_nodes,
                            195.0,
                        );
                        ui.close_menu();
                    }
                    if ui.button("Clear Wildfire Smoke Offset").clicked() {
                        self.params.smoke_pm25_offset_ugm3 = 0.0;
                        self.engine.radar_profiles = build_regional_radar_profiles(
                            None,
                            self.params.region,
                            &self.engine.weather_nodes,
                            0.0,
                        );
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Open Radar Vertical Profile Tab (R)").clicked() {
                        self.stage = AppStage::Workbench;
                        self.bottom_tab = BottomDockTab::RadarVerticalProfile;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    if ui.button("Shortcuts & Scientific Citations (F1)").clicked() {
                        self.show_help_modal = true;
                        ui.close_menu();
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let stage_lbl = match self.stage {
                        AppStage::StartupHub => "Enter 3D GIS Map ->",
                        AppStage::Workbench => "Startup Hub",
                    };
                    if ui.button(stage_lbl).clicked() {
                        self.stage = match self.stage {
                            AppStage::StartupHub => AppStage::Workbench,
                            AppStage::Workbench => AppStage::StartupHub,
                        };
                    }
                    ui.small(
                        RichText::new(format!(
                            "Campaign: {} ({})",
                            self.campaign_name_input, self.campaign_db_path_input
                        ))
                        .monospace()
                        .color(Color32::from_rgb(125, 211, 252)),
                    );
                });
            });
        });

        if let Some(r) = pending_region {
            if r != self.params.region {
                self.apply_region_switch(r);
            }
        }
    }

    fn render_startup_hub(&mut self, ctx: &egui::Context) {
        let mut chosen_region: Option<FlywayRegion> = None;
        let mut preset_campaign: Option<(&'static str, &'static str, FlywayRegion)> = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.heading(
                        RichText::new("AEROFLYWAY v0.6.0 — Global Aeroecology, VIIRS Lights-Out & Radar Workstation")
                            .size(20.0)
                            .color(Color32::from_rgb(56, 189, 248)),
                    );
                });
                ui.label(
                    RichText::new(
                        "Select a global flyway corridor, resume or create a SQLite WAL campaign workspace, and launch into 100,000-agent WGSL simulation, NEXRAD/Aloft vertical radar profiling, or WAV STFT bioacoustics.",
                    )
                    .color(Color32::from_rgb(148, 163, 184)),
                );
                ui.add_space(12.0);

                ui.columns(2, |cols| {
                    // LEFT COLUMN: Campaign Workspace Manager + Scenario Pre-Flight
                    cols[0].group(|ui| {
                        ui.heading("1. SQLite Campaign Workspace & Pre-Flight");
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label("Campaign Title:");
                            ui.text_edit_singleline(&mut self.campaign_name_input);
                        });
                        ui.horizontal(|ui| {
                            ui.label("SQLite `.db` File:");
                            ui.text_edit_singleline(&mut self.campaign_db_path_input);
                            if ui.button("Open / Create .db").clicked() {
                                let db_f = self.campaign_db_path_input.clone();
                                let title = self.campaign_name_input.clone();
                                let reg = self.params.region;
                                self.open_or_switch_campaign_db(&db_f, &title, reg);
                            }
                        });

                        ui.add_space(6.0);
                        ui.label(RichText::new("Quick Campaign Templates:").strong());
                        for (db_f, title, reg) in [
                            ("aeroflyway.db", "Pacific Willamette & Gorge Baseline", FlywayRegion::PacificAmericas),
                            ("platte_nexrad_spring.db", "Platte Crane & Bracken NEXRAD", FlywayRegion::CentralAmericasPlatte),
                            ("atlantic_lights_out.db", "NYC-Philly-Cape May Lights-Out", FlywayRegion::AtlanticAmericas),
                            ("palearctic_aloft.db", "Falsterbo-Gibraltar Aloft VPTS", FlywayRegion::EastAtlanticPalearctic),
                        ] {
                            if ui.button(format!("{title} ({db_f})")).clicked() {
                                preset_campaign = Some((db_f, title, reg));
                            }
                        }

                        ui.add_space(10.0);
                        ui.separator();
                        ui.heading("Pre-Flight Climate, Skyglow & Compute");
                        ui.add(
                            egui::Slider::new(&mut self.params.day_of_year, 1.0..=365.0)
                                .prefix("Day of Year (DOY): ")
                                .fixed_decimals(0),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.params.lights_out_curfew_pct, 0.0..=100.0)
                                .text("Lights-Out Curfew (23:00-06:00)")
                                .suffix("%"),
                        );
                        egui::ComboBox::from_id_salt("hub_cmip6_combo")
                            .selected_text(self.params.cmip6_scenario.label())
                            .show_ui(ui, |ui| {
                                for ssp in Cmip6Scenario::ALL {
                                    ui.selectable_value(
                                        &mut self.params.cmip6_scenario,
                                        ssp,
                                        ssp.label(),
                                    );
                                }
                            });

                        ui.horizontal(|ui| {
                            ui.label("Optional eBird 2.0 Key:");
                            ui.text_edit_singleline(&mut self.ebird_api_token);
                        });
                        if self.gpu_available {
                            ui.checkbox(
                                &mut self.use_gpu_compute,
                                format!("Enable wgpu WGSL Compute & Instanced Shaders ({} agents)", self.gpu_agent_count),
                            );
                        }
                    });

                    // RIGHT COLUMN: 5 Global Corridor Cards + Mode Launchers
                    cols[1].group(|ui| {
                        ui.heading("2. Global Flyway Corridor Presets");
                        ui.add_space(4.0);
                        for reg in FlywayRegion::ALL {
                            let selected = self.params.region == reg;
                            let border_c = if selected {
                                Color32::from_rgb(56, 189, 248)
                            } else {
                                Color32::from_rgb(42, 56, 78)
                            };
                            egui::Frame::NONE
                                .fill(Color32::from_rgb(17, 26, 41))
                                .stroke(Stroke::new(1.2, border_c))
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(8.0)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        if ui.selectable_label(selected, RichText::new(reg.label()).strong()).clicked() {
                                            chosen_region = Some(reg);
                                        }
                                    });
                                    ui.small(
                                        RichText::new(reg.short_card_summary())
                                            .monospace()
                                            .color(Color32::from_rgb(148, 163, 184)),
                                    );
                                });
                            ui.add_space(4.0);
                        }

                        ui.add_space(8.0);
                        ui.separator();
                        ui.heading("3. Launch Workbench Mode");
                        ui.horizontal_wrapped(|ui| {
                            if ui
                                .button(
                                    RichText::new("Launch 3D Flyway GIS Workbench")
                                        .strong()
                                        .color(Color32::from_rgb(125, 211, 252)),
                                )
                                .clicked()
                            {
                                self.bottom_tab = BottomDockTab::AirspaceTransect;
                                self.stage = AppStage::Workbench;
                            }
                            if ui.button("Open NEXRAD / Aloft Radar Lab").clicked() {
                                self.bottom_tab = BottomDockTab::RadarVerticalProfile;
                                self.stage = AppStage::Workbench;
                            }
                            if ui.button("Open WAV Bioacoustic STFT Lab").clicked() {
                                self.bottom_tab = BottomDockTab::AcousticSpectrogram;
                                self.stage = AppStage::Workbench;
                            }
                            if ui.button("Open Multi-Scenario Batch Sweep").clicked() {
                                self.bottom_tab = BottomDockTab::BatchSweepLab;
                                self.stage = AppStage::Workbench;
                            }
                        });
                    });
                });
            });
        });

        if let Some((db_f, title, reg)) = preset_campaign {
            self.open_or_switch_campaign_db(db_f, title, reg);
        } else if let Some(r) = chosen_region {
            if r != self.params.region {
                self.apply_region_switch(r);
            }
        }
    }

    fn render_help_modal(&mut self, ctx: &egui::Context) {
        if !self.show_help_modal {
            return;
        }
        let mut open = self.show_help_modal;
        egui::Window::new("AeroFlyway v0.6.0 — Shortcuts & Scientific Grounding")
            .open(&mut open)
            .resizable(false)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.label(RichText::new("Keyboard Shortcuts").strong());
                ui.monospace("  Space : Play / Pause multi-taxa simulation");
                ui.monospace("  H     : Toggle Startup Campaign Hub <-> 3D GIS Map");
                ui.monospace("  R     : Jump to NEXRAD / Aloft Vertical Radar Profile (VPTS) tab");
                ui.monospace("  L     : Toggle 85% BirdCast Lights-Out curfew vs 0% unmitigated");
                ui.monospace("  F1    : Show / hide this reference window");
                ui.separator();
                ui.label(RichText::new("Empirical Data & Model Grounding").strong());
                ui.small("- VIIRS Nighttime Lights & Building Collisions: EOG VIIRS DNB radiance (nW/cm^2/sr) coupled to Van Doren et al. (2021, PNAS) McCormick Place window-lighting collision model & BirdCast/Audubon Lights-Out (23:00-06:00).");
                ui.small("- Weather Radar Vertical Profiles (VPTS): Unidata NEXRAD Level-II S3 (KRTX, KMAX, KUEX, KDDC, KEWX, KDIX) & ENRAM Aloft OPERA S3 (nlhrw, bejab, behel) vol2bird VPTS CSV format.");
                ui.small("- Climate & Megafire Smoke: Open-Meteo Forecast, CMIP6 Climate (/v1/climate), CAMS Air Quality (/v1/air-quality), Copernicus GLO-90 DEM (/v1/elevation), and Overton et al. (2022, Ecology) PM2.5 >= 160 ug/m^3 ceiling climbs.");
            });
        self.show_help_modal = open;
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        let mut pending_region: Option<FlywayRegion> = None;

        egui::TopBottomPanel::top("top_command_bar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("AEROFLYWAY v0.6")
                        .strong()
                        .size(15.0)
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.separator();

                let mut sel_reg = self.params.region;
                egui::ComboBox::from_id_salt("global_region_combo")
                    .selected_text(sel_reg.label())
                    .show_ui(ui, |ui| {
                        for r in FlywayRegion::ALL {
                            if ui.selectable_value(&mut sel_reg, r, r.label()).clicked() {
                                pending_region = Some(r);
                            }
                        }
                    });

                ui.separator();
                let play_label = if self.running { "Pause" } else { "Play" };
                if ui.button(play_label).clicked() {
                    self.running = !self.running;
                }
                if ui.button("Step +1h").clicked() {
                    self.engine.step(
                        1.0,
                        &self.params,
                        &self.turbines,
                        &self.milkweed,
                        &self.observations,
                    );
                    self.last_frame_sim_dt_hr = 1.0;
                    self.gpu_tick_seed = self.gpu_tick_seed.wrapping_add(1);
                }

                for (spd, lbl) in [(0.8, "1x"), (3.0, "4x"), (12.0, "16x")] {
                    let selected = (self.sim_speed_hr_per_sec - spd).abs() < 0.1;
                    if ui.selectable_label(selected, lbl).clicked() {
                        self.sim_speed_hr_per_sec = spd;
                    }
                }

                if self.gpu_available {
                    ui.separator();
                    let badge = if self.use_gpu_compute {
                        format!("WGSL GPU ({}k)", self.gpu_agent_count / 1000)
                    } else {
                        "CPU Rayon (2.4k)".to_string()
                    };
                    if ui.selectable_label(self.use_gpu_compute, badge).clicked() {
                        self.use_gpu_compute = !self.use_gpu_compute;
                    }
                }

                ui.separator();
                egui::ComboBox::from_id_salt("basemap_combo")
                    .selected_text(self.map.provider.label())
                    .show_ui(ui, |ui| {
                        for p in [
                            BasemapProvider::VectorDarkOffline,
                            BasemapProvider::CartoDarkMatter,
                            BasemapProvider::UsgsNationalTopo,
                            BasemapProvider::OpenStreetMap,
                        ] {
                            if ui
                                .selectable_value(&mut self.map.provider, p, p.label())
                                .clicked()
                            {
                                self.map.retry_failed_tiles();
                            }
                        }
                    });

                ui.separator();
                for (mode, txt) in [
                    (MapToolMode::PanAndInspect, "Pan / Inspect"),
                    (MapToolMode::DrawTransect, "Draw A-B Transect"),
                    (MapToolMode::PlantVirtualMilkweed, "Plant Host Patch"),
                ] {
                    if ui.selectable_label(self.tool_mode == mode, txt).clicked() {
                        self.tool_mode = mode;
                    }
                }

                ui.separator();
                let sync_txt = if self.ingest_in_progress {
                    "Syncing..."
                } else {
                    "Sync Live APIs"
                };
                if ui
                    .add_enabled(!self.ingest_in_progress, egui::Button::new(sync_txt))
                    .clicked()
                {
                    self.trigger_live_sync(ctx);
                }
            });
        });

        if let Some(r) = pending_region {
            if r != self.params.region {
                self.apply_region_switch(r);
            }
        }
    }

    fn render_left_dock(&mut self, ctx: &egui::Context) {
        let mut jump_region: Option<FlywayRegion> = None;

        egui::SidePanel::left("left_analytics_dock")
            .resizable(true)
            .default_width(348.0)
            .min_width(296.0)
            .max_width(455.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Live Aeroecology, Skyglow & Radar");
                    ui.add_space(4.0);

                    let m = &self.engine.metrics;
                    let scale_ratio = self.current_scale_factor();

                    let (birds_air, birds_fall, birds_glow, bats_act, rotor_hits, urb_strikes, mean_gdd, north_lat) =
                        if self.gpu_mode_active() {
                            if let Some(gs) = self.gpu_summary {
                                (
                                    gs.birds_airborne as usize,
                                    gs.birds_fallout as usize,
                                    gs.birds_skyglow_entrained as usize,
                                    gs.bats_flying as usize,
                                    self.gpu_cum_rotor_hits,
                                    self.gpu_cum_urban_strikes,
                                    if gs.mean_gdd > 1.0 {
                                        gs.mean_gdd
                                    } else {
                                        m.monarch_mean_gdd
                                    },
                                    if gs.max_north_lat > 18.0 {
                                        gs.max_north_lat
                                    } else {
                                        m.monarch_northern_front_lat
                                    },
                                )
                            } else {
                                (
                                    (m.avian_flux_agents_airborne as f64 * scale_ratio) as usize,
                                    (m.fallout_stopover_birds as f64 * scale_ratio) as usize,
                                    (m.skyglow_entrained_birds as f64 * scale_ratio) as usize,
                                    (m.active_migrating_bats as f64 * scale_ratio) as usize,
                                    self.gpu_cum_rotor_hits,
                                    self.gpu_cum_urban_strikes,
                                    m.monarch_mean_gdd,
                                    m.monarch_northern_front_lat,
                                )
                            }
                        } else {
                            (
                                m.avian_flux_agents_airborne,
                                m.fallout_stopover_birds,
                                m.skyglow_entrained_birds,
                                m.active_migrating_bats,
                                m.cumulative_bat_rotor_contacts as u64,
                                m.cumulative_urban_strikes as u64,
                                m.monarch_mean_gdd,
                                m.monarch_northern_front_lat,
                            )
                        };

                    let avoided_bat_pct =
                        if self.gpu_mode_active() && self.gpu_cum_baseline_hits > 0 {
                            let saved = self
                                .gpu_cum_baseline_hits
                                .saturating_sub(self.gpu_cum_rotor_hits)
                                as f64;
                            (100.0 * saved / self.gpu_cum_baseline_hits as f64).clamp(0.0, 100.0)
                        } else {
                            m.avoided_bat_contacts_pct
                        };

                    let avoided_urb_pct =
                        if self.gpu_mode_active() && self.gpu_cum_urban_baseline > 12 {
                            let saved = self
                                .gpu_cum_urban_baseline
                                .saturating_sub(self.gpu_cum_urban_strikes)
                                as f64;
                            (100.0 * saved / self.gpu_cum_urban_baseline as f64).clamp(0.0, 100.0)
                        } else {
                            compute_lights_out_collision_reduction_pct(self.params.lights_out_curfew_pct)
                        };

                    egui::Grid::new("kpi_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.5])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new("Clock / DOY:")
                                    .color(Color32::from_rgb(148, 163, 184)),
                            );
                            let hr = m.hour_of_day;
                            let phase = if m.is_nocturnal { "Night" } else { "Day" };
                            ui.label(
                                RichText::new(format!(
                                    "{:02.0}:{:02.0} ({phase}) | DOY {:.0} ({:.1}h)",
                                    hr.floor(),
                                    (hr.fract() * 60.0).floor(),
                                    m.day_of_year,
                                    m.photoperiod_hr
                                ))
                                .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Airborne Birds:")
                                    .color(Color32::from_rgb(56, 189, 248)),
                            );
                            ui.label(
                                RichText::new(format!("{birds_air} ({birds_fall} halt)"))
                                    .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Skyglow / Strikes:")
                                    .color(Color32::from_rgb(244, 114, 182)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{birds_glow} trapped | {urb_strikes} strikes (-{avoided_urb_pct:.0}%)"
                                ))
                                .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Radar VPTS MTR:")
                                    .color(Color32::from_rgb(34, 211, 238)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{:.0} birds/km/h | peak {:.0}m",
                                    m.radar_mtr_birds_km_hr, m.radar_peak_alt_m
                                ))
                                .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Bats / Curtail:")
                                    .color(Color32::from_rgb(192, 132, 252)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{bats_act} | -{avoided_bat_pct:.0}% risk ({rotor_hits} hits)"
                                ))
                                .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Lepidoptera:")
                                    .color(Color32::from_rgb(251, 191, 36)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "Gen {} | {mean_gdd:.0} GDD | {north_lat:.2} N",
                                    m.monarch_peak_generation
                                ))
                                .monospace(),
                            );
                            ui.end_row();

                            ui.label(
                                RichText::new("Mismatch / Smoke:")
                                    .color(Color32::from_rgb(251, 146, 60)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{:+.1}d ({:.0}% ovlp) | {:.0} ug/m3",
                                    m.trophic_mismatch_days,
                                    m.trophic_overlap_pct,
                                    m.mean_corridor_pm25_ugm3
                                ))
                                .monospace(),
                            );
                            ui.end_row();
                        });

                    if self.gpu_available {
                        ui.add_space(5.0);
                        ui.label("GPU Instanced Swarm Size (VRAM Zero-Copy)");
                        ui.add(
                            egui::Slider::new(&mut self.gpu_agent_count, 10_000..=MAX_GPU_AGENTS)
                                .step_by(2048.0)
                                .suffix(" agents"),
                        );
                    }

                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("VIIRS Skyglow, Lights-Out & Radar");
                    ui.label("Municipal Lights-Out Curfew (23:00-06:00)");
                    ui.add(
                        egui::Slider::new(&mut self.params.lights_out_curfew_pct, 0.0..=100.0)
                            .suffix("% dimmed"),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Unmitigated (0%)").clicked() {
                            self.params.lights_out_curfew_pct = 0.0;
                        }
                        if ui.button("50% Window Dim").clicked() {
                            self.params.lights_out_curfew_pct = 50.0;
                        }
                        if ui.button("BirdCast Alert (85%)").clicked() {
                            self.params.lights_out_curfew_pct = 85.0;
                        }
                    });
                    ui.label("VIIRS Anthropogenic Skyglow Multiplier");
                    ui.add(
                        egui::Slider::new(&mut self.params.skyglow_intensity_mult, 0.0..=2.5)
                            .suffix("x"),
                    );
                    ui.label("NEXRAD / Aloft VPTS Altitude Assimilation");
                    ui.add(
                        egui::Slider::new(&mut self.params.radar_nudge_weight, 0.0..=1.0)
                            .fixed_decimals(2),
                    );

                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("365-Day Phenology & CMIP6 Climate");

                    let prev_season = self.params.season;
                    egui::ComboBox::from_id_salt("season_preset")
                        .selected_text(self.params.season.label())
                        .width(290.0)
                        .show_ui(ui, |ui| {
                            for s in [
                                SeasonPreset::SpringNorthbound,
                                SeasonPreset::SummerBreeding,
                                SeasonPreset::AutumnFalloutAndBatPulse,
                                SeasonPreset::OverwinterRoost,
                            ] {
                                ui.selectable_value(&mut self.params.season, s, s.label());
                            }
                        });
                    if self.params.season != prev_season {
                        self.params.day_of_year = self.params.season.canonical_doy();
                        self.engine
                            .reseed_agents(self.params.region, self.params.season);
                        self.reset_all_counters();
                    }

                    if ui
                        .add(
                            egui::Slider::new(&mut self.params.day_of_year, 1.0..=365.0)
                                .prefix("DOY ")
                                .fixed_decimals(0),
                        )
                        .changed()
                    {
                        self.params.season = SeasonPreset::from_doy(self.params.day_of_year);
                    }
                    ui.checkbox(
                        &mut self.params.advance_annual_calendar,
                        "Auto-advance DOY with simulation clock",
                    );

                    let prev_ssp = self.params.cmip6_scenario;
                    egui::ComboBox::from_id_salt("cmip6_ssp_combo")
                        .selected_text(self.params.cmip6_scenario.label())
                        .width(290.0)
                        .show_ui(ui, |ui| {
                            for ssp in Cmip6Scenario::ALL {
                                ui.selectable_value(
                                    &mut self.params.cmip6_scenario,
                                    ssp,
                                    ssp.label(),
                                );
                            }
                        });
                    if self.params.cmip6_scenario != prev_ssp {
                        self.engine.weather_nodes = default_synoptic_nodes_for_region(
                            self.params.region,
                            self.params.cmip6_scenario,
                        );
                    }

                    ui.label("Megafire Smoke Plume PM2.5 Offset (Ceiling >160)");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.params.smoke_pm25_offset_ugm3, 0.0..=280.0)
                                .suffix(" ug/m3"),
                        )
                        .changed()
                    {
                        self.engine.radar_profiles = build_regional_radar_profiles(
                            None,
                            self.params.region,
                            &self.engine.weather_nodes,
                            self.params.smoke_pm25_offset_ugm3,
                        );
                    }

                    ui.label("Smart Turbine Cut-In Wind Speed (Bat Curtailment)");
                    ui.add(
                        egui::Slider::new(&mut self.params.turbine_cut_in_mps, 3.0..=7.5)
                            .suffix(" m/s"),
                    );

                    ui.label("850 hPa Low-Level Jet Multiplier");
                    ui.add(
                        egui::Slider::new(&mut self.params.wind_850_multiplier, 0.2..=2.5)
                            .suffix("x"),
                    );

                    ui.label("Precipitation Fallout Front (mm/hr)");
                    ui.add(
                        egui::Slider::new(&mut self.params.precip_front_mm_hr, 0.0..=5.0)
                            .suffix(" mm/h"),
                    );

                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("Global Flyway Quick-Switch");
                    ui.horizontal_wrapped(|ui| {
                        for (reg, short_lbl) in [
                            (FlywayRegion::PacificAmericas, "Pacific"),
                            (FlywayRegion::CentralAmericasPlatte, "Platte/TX"),
                            (FlywayRegion::AtlanticAmericas, "Cape May"),
                            (FlywayRegion::EastAtlanticPalearctic, "Palearctic"),
                            (FlywayRegion::EastAsianAustralasian, "East Asia"),
                        ] {
                            let active = self.params.region == reg;
                            if ui.selectable_label(active, short_lbl).clicked() {
                                jump_region = Some(reg);
                            }
                        }
                    });

                    ui.add_space(4.0);
                    ui.small(
                        RichText::new(&self.status_line).color(Color32::from_rgb(148, 163, 184)),
                    );
                });
            });

        if let Some(reg) = jump_region {
            if reg != self.params.region {
                self.apply_region_switch(reg);
            }
        }
    }

    fn render_bottom_dock(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("bottom_workbench_dock")
            .resizable(true)
            .default_height(255.0)
            .min_height(185.0)
            .max_height(440.0)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::AirspaceTransect,
                        "Airspace Cross-Section (0-4,200 m ASL)",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::RadarVerticalProfile,
                        "NEXRAD / Aloft Vertical Profile (VPTS)",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::AcousticSpectrogram,
                        "WAV STFT Bioacoustic Inspector",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::BatchSweepLab,
                        "Multi-Scenario Batch Sweep Lab",
                    );
                });
                ui.separator();

                match self.bottom_tab {
                    BottomDockTab::AirspaceTransect => self.paint_transect_tab(ui),
                    BottomDockTab::RadarVerticalProfile => self.paint_radar_vpts_tab(ui),
                    BottomDockTab::AcousticSpectrogram => self.paint_acoustic_tab(ctx, ui),
                    BottomDockTab::BatchSweepLab => self.paint_batch_sweep_tab(ctx, ui),
                }
            });
    }

    fn paint_transect_tab(&self, ui: &mut egui::Ui) {
        let (total_km, terrain, cpu_points) = self.engine.compute_transect_slice(&self.params);
        let dem_src = if self.engine.cached_transect_dem.is_some() {
            "Copernicus GLO-90 Live DEM"
        } else {
            "Regional Analytical DEM"
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(format!(
                    "Transect A -> B ({total_km:.0} km, +/-{:.0} km swath | {dem_src})",
                    self.params.transect_half_width_km
                ))
                .strong(),
            );
            ui.separator();
            ui.colored_label(Color32::from_rgb(244, 114, 182), "Skyglow (<300 m)");
            ui.colored_label(Color32::from_rgb(192, 132, 252), "Chiroptera (40-340 m)");
            ui.colored_label(Color32::from_rgb(56, 189, 248), "Aves (380-3,850 m)");
            ui.colored_label(Color32::from_rgb(34, 211, 238), "Radar VPTS Peak");
        });

        let avail = ui.available_size();
        let (plot_rect, _) = ui.allocate_exact_size(avail, Sense::hover());
        let painter = ui.painter_at(plot_rect);

        painter.rect_filled(plot_rect, CornerRadius::same(4), Color32::from_rgb(11, 17, 27));
        painter.rect_stroke(
            plot_rect,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(38, 52, 73)),
            StrokeKind::Inside,
        );

        let inner = Rect::from_min_max(
            Pos2::new(plot_rect.min.x + 48.0, plot_rect.min.y + 10.0),
            Pos2::new(plot_rect.max.x - 16.0, plot_rect.max.y - 20.0),
        );
        if inner.width() < 40.0 || inner.height() < 30.0 {
            return;
        }

        let max_elev_m = 4200.0_f64;
        let map_xz = |dist_km: f64, alt_asl_m: f64| -> Pos2 {
            let fx = (dist_km / total_km).clamp(0.0, 1.0) as f32;
            let fy = (alt_asl_m / max_elev_m).clamp(0.0, 1.0) as f32;
            Pos2::new(
                inner.left() + fx * inner.width(),
                inner.bottom() - fy * inner.height(),
            )
        };

        if self.engine.metrics.mean_corridor_pm25_ugm3 >= 160.0 {
            let y_plume_top = map_xz(0.0, 3600.0).y;
            let smoke_rect = Rect::from_min_max(
                Pos2::new(inner.left(), y_plume_top),
                Pos2::new(inner.right(), inner.bottom()),
            );
            painter.rect_filled(
                smoke_rect,
                CornerRadius::same(0),
                Color32::from_rgba_unmultiplied(249, 115, 22, 34),
            );
        }

        // Radar VPTS peak migration altitude reference line
        if self.layers.radar_rings && self.engine.metrics.radar_peak_alt_m > 100.0 {
            let y_rad = map_xz(0.0, self.engine.metrics.radar_peak_alt_m).y;
            painter.line_segment(
                [Pos2::new(inner.left(), y_rad), Pos2::new(inner.right(), y_rad)],
                Stroke::new(1.1, Color32::from_rgba_unmultiplied(34, 211, 238, 150)),
            );
        }

        for elev in [1000.0, 2000.0, 3000.0, 4000.0] {
            let y = map_xz(0.0, elev).y;
            painter.line_segment(
                [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
                Stroke::new(0.8, Color32::from_rgba_unmultiplied(71, 85, 105, 90)),
            );
            painter.text(
                Pos2::new(inner.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                format!("{elev:.0}m"),
                FontId::monospace(10.0),
                Color32::from_rgb(148, 163, 184),
            );
        }

        if self.layers.turbines {
            let rotor_pts: Vec<Pos2> = terrain
                .iter()
                .map(|(d, terr)| map_xz(*d, *terr + 135.0))
                .collect();
            painter.add(egui::Shape::line(
                rotor_pts,
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(244, 63, 94, 120)),
            ));
        }

        let mut terr_poly = Vec::with_capacity(terrain.len());
        for (d, elev) in &terrain {
            let top_p = map_xz(*d, *elev);
            let bot_p = Pos2::new(top_p.x, inner.bottom());
            painter.line_segment(
                [bot_p, top_p],
                Stroke::new(3.0, Color32::from_rgba_unmultiplied(30, 58, 47, 155)),
            );
            terr_poly.push(top_p);
        }
        painter.add(egui::Shape::line(
            terr_poly,
            Stroke::new(1.8, Color32::from_rgb(74, 222, 128)),
        ));

        if self.gpu_mode_active() {
            painter.add(egui_wgpu::Callback::new_paint_callback(
                inner,
                TransectGpuPaintCallback {
                    agent_count: self.gpu_agent_count,
                },
            ));
        } else {
            for pt in &cpu_points {
                let show = match pt.taxon {
                    Taxon::Bird => self.layers.birds,
                    Taxon::Bat => self.layers.bats,
                    Taxon::Butterfly => self.layers.butterflies,
                };
                if !show {
                    continue;
                }
                let screen_p = map_xz(pt.dist_km, pt.terrain_m + pt.alt_agl_m);
                let col = if pt.taxon == Taxon::Bird && pt.skyglow_entrained {
                    Color32::from_rgba_unmultiplied(244, 114, 182, 235)
                } else if pt.taxon == Taxon::Bird && pt.smoke_disrupted {
                    Color32::from_rgba_unmultiplied(249, 115, 22, 235)
                } else {
                    taxon_color(pt.taxon, 220)
                };
                painter.circle_filled(screen_p, 2.3, col);
            }
        }
    }

    fn paint_radar_vpts_tab(&mut self, ui: &mut egui::Ui) {
        let mut jump_coords: Option<(f64, f64)> = None;
        let mut import_path: Option<String> = None;

        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Radar Profiler:").strong());
            for (idx, prof) in self.engine.radar_profiles.iter().enumerate() {
                let sel = self.selected_radar_idx == idx;
                if ui
                    .selectable_label(sel, format!("{} ({})", prof.station_code, prof.station_name))
                    .clicked()
                {
                    self.selected_radar_idx = idx;
                    jump_coords = Some((prof.lat, prof.lon));
                }
            }

            ui.separator();
            ui.label("Import VPTS .csv:");
            ui.add(egui::TextEdit::singleline(&mut self.vpts_csv_path_input).desired_width(150.0));
            if ui.button("Load VPTS").clicked() {
                let p = self.vpts_csv_path_input.trim().to_string();
                if !p.is_empty() {
                    import_path = Some(p);
                }
            }
        });

        if let Some((lt, ln)) = jump_coords {
            self.map.jump_to(lt, ln, self.map.zoom.max(6.8));
        }

        if let Some(path_str) = import_path {
            let stations = radar_stations_for_region(self.params.region);
            if let Some(st) = stations.get(self.selected_radar_idx).or_else(|| stations.first()) {
                match load_vpts_csv_file(&path_str, st) {
                    Ok(prof) => {
                        self.status_line = format!(
                            "Loaded VPTS CSV '{}' -> MTR {:.0} birds/km/hr, peak {:.0} m ASL",
                            prof.station_code, prof.mtr_birds_km_hr, prof.peak_altitude_m_asl
                        );
                        if self.selected_radar_idx < self.engine.radar_profiles.len() {
                            self.engine.radar_profiles[self.selected_radar_idx] = prof;
                        } else {
                            self.engine.radar_profiles.push(prof);
                        }
                    }
                    Err(e) => self.status_line = e,
                }
            }
        }

        let Some(active_prof) = self
            .engine
            .radar_profiles
            .get(self.selected_radar_idx)
            .or_else(|| self.engine.radar_profiles.first())
        else {
            ui.label("No radar profiles loaded for current region.");
            return;
        };

        ui.small(
            RichText::new(format!(
                "{} | Source: {} | Antenna: {:.0} m ASL | Migration Traffic Rate (MTR): {:.0} birds/km/hr | Peak Layer: {:.0} m ASL",
                active_prof.network_label,
                active_prof.source_tag,
                active_prof.antenna_elev_m,
                active_prof.mtr_birds_km_hr,
                active_prof.peak_altitude_m_asl
            ))
            .monospace()
            .color(Color32::from_rgb(165, 243, 252)),
        );

        let avail = ui.available_size();
        let (outer, _) = ui.allocate_exact_size(avail, Sense::hover());
        let painter = ui.painter_at(outer);

        painter.rect_filled(outer, CornerRadius::same(4), Color32::from_rgb(8, 13, 22));
        painter.rect_stroke(
            outer,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(38, 52, 73)),
            StrokeKind::Inside,
        );

        let inner = Rect::from_min_max(
            Pos2::new(outer.min.x + 62.0, outer.min.y + 10.0),
            Pos2::new(outer.max.x - 140.0, outer.max.y - 20.0),
        );
        if inner.width() < 60.0 || inner.height() < 30.0 {
            return;
        }

        let max_dens = active_prof
            .bins
            .iter()
            .map(|b| b.bird_density_per_km3)
            .fold(10.0_f64, f64::max)
            * 1.15;
        let max_h = 4200.0_f64;

        for elev in [1000.0, 2000.0, 3000.0, 4000.0] {
            let fy = (elev / max_h) as f32;
            let y = inner.bottom() - fy * inner.height();
            painter.line_segment(
                [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
                Stroke::new(0.7, Color32::from_rgba_unmultiplied(51, 65, 85, 110)),
            );
            painter.text(
                Pos2::new(inner.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                format!("{elev:.0}m"),
                FontId::monospace(9.5),
                Color32::from_rgb(148, 163, 184),
            );
        }

        let bar_h = (inner.height() / (active_prof.bins.len().max(1) as f32) - 2.0).clamp(2.5, 14.0);
        for b in &active_prof.bins {
            let fy = (b.height_m_asl / max_h).clamp(0.0, 1.0) as f32;
            let cy = inner.bottom() - fy * inner.height();
            let w_frac = (b.bird_density_per_km3 / max_dens).clamp(0.0, 1.0) as f32;
            let bar_rect = Rect::from_min_max(
                Pos2::new(inner.left(), cy - bar_h * 0.5),
                Pos2::new(inner.left() + w_frac * inner.width(), cy + bar_h * 0.5),
            );
            let is_peak = (b.height_m_asl - active_prof.peak_altitude_m_asl).abs() < 110.0;
            let fill = if is_peak {
                Color32::from_rgb(34, 211, 238)
            } else {
                Color32::from_rgba_unmultiplied(56, 189, 248, 185)
            };
            painter.rect_filled(bar_rect, CornerRadius::same(2), fill);

            // Wind vector barb on right column
            let barb_origin = Pos2::new(inner.right() + 48.0, cy);
            let vec_px = Vec2::new((b.u_mps as f32) * 2.2, -(b.v_mps as f32) * 2.2);
            painter.arrow(
                barb_origin,
                vec_px,
                Stroke::new(1.2, Color32::from_rgba_unmultiplied(167, 243, 208, 195)),
            );
        }

        painter.text(
            Pos2::new(inner.left(), outer.max.y - 8.0),
            Align2::LEFT_CENTER,
            format!("0 -> {max_dens:.1} birds/km^3 (vol2bird / VPTS reflectivity)"),
            FontId::monospace(9.5),
            Color32::from_rgb(148, 163, 184),
        );
        painter.text(
            Pos2::new(inner.right() + 18.0, outer.min.y + 10.0),
            Align2::LEFT_TOP,
            "VPTS Wind (u,v)",
            FontId::monospace(9.5),
            Color32::from_rgb(167, 243, 208),
        );
    }

    fn paint_acoustic_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let mut trigger_pin = false;
        ui.horizontal_wrapped(|ui| {
            for preset in [
                SynthPreset::HoaryBatUltrasonic250k,
                SynthPreset::MyotisBuzzUltrasonic250k,
                SynthPreset::PasserineNfcDuet48k,
            ] {
                if ui.button(preset.label()).clicked() {
                    let rec = synthesize_preset_recording(preset);
                    self.acoustics.load_new_recording(rec);
                }
            }

            ui.separator();
            ui.label("WAV Path:");
            ui.add(
                egui::TextEdit::singleline(&mut self.acoustics.wav_path_input).desired_width(145.0),
            );
            if ui.button("Load .wav").clicked() {
                let p = self.acoustics.wav_path_input.trim().to_string();
                if !p.is_empty() {
                    match load_wav_file(&p) {
                        Ok(rec) => {
                            self.status_line = format!("Loaded WAV '{}'", rec.source_label);
                            self.acoustics.load_new_recording(rec);
                        }
                        Err(e) => self.status_line = e,
                    }
                }
            }

            if ui.button("Save Sample .wav").clicked() {
                let out_dir = PathBuf::from(self.export_dir_input.trim());
                let _ = std::fs::create_dir_all(&out_dir);
                let wav_out = out_dir.join(&self.acoustics.recording.source_label);
                let coords = self
                    .acoustics
                    .recording
                    .guano_lat_lon
                    .or(Some((self.map.center_lat, self.map.center_lon)));
                match write_wav_16bit(&wav_out, &self.acoustics.recording, coords) {
                    Ok(()) => {
                        self.status_line =
                            format!("Saved 16-bit GUANO WAV -> {}", wav_out.display());
                    }
                    Err(e) => self.status_line = e,
                }
            }

            ui.separator();
            for n_fft in [256_usize, 512, 1024, 2048] {
                if ui
                    .selectable_label(self.acoustics.fft_size == n_fft, format!("FFT {n_fft}"))
                    .clicked()
                {
                    self.acoustics.fft_size = n_fft;
                    self.acoustics.recompute_stft();
                }
            }

            let prev_floor = self.acoustics.db_floor;
            ui.add(
                egui::Slider::new(&mut self.acoustics.db_floor, -95.0..=-35.0)
                    .text("dB floor")
                    .fixed_decimals(0),
            );
            if (self.acoustics.db_floor - prev_floor).abs() > 0.2 {
                self.acoustics.texture_dirty = true;
            }

            ui.checkbox(&mut self.acoustics.show_ridge_overlay, "Ridge");
            if ui.button("Pin Call to Map").clicked() {
                trigger_pin = true;
            }
        });

        if trigger_pin {
            self.pin_current_acoustic_detection();
        }

        let spec_meta = format!(
            "File: {} ({:.1} kHz, {:.0} ms, {} pulses)  |  Guild: {} ({})  |  Fc: {:.1} kHz  |  ACI: {:.1}",
            self.acoustics.recording.source_label,
            self.acoustics.recording.sample_rate_hz as f64 / 1000.0,
            self.acoustics.spectrogram.duration_ms,
            self.acoustics.spectrogram.pulses.len(),
            self.acoustics.spectrogram.guild_label,
            self.acoustics.spectrogram.scientific_hint,
            self.acoustics.spectrogram.characteristic_freq_khz,
            self.acoustics.spectrogram.aci_index,
        );
        ui.small(
            RichText::new(spec_meta)
                .monospace()
                .color(Color32::from_rgb(165, 243, 252)),
        );

        let tex_id = self.acoustics.ensure_texture(ctx).id();
        let avail = ui.available_size();
        let (outer, resp) = ui.allocate_exact_size(avail, Sense::hover());
        let painter = ui.painter_at(outer);

        painter.rect_filled(outer, CornerRadius::same(4), Color32::from_rgb(8, 11, 24));
        painter.rect_stroke(
            outer,
            CornerRadius::same(4),
            Stroke::new(1.0, Color32::from_rgb(45, 58, 82)),
            StrokeKind::Inside,
        );

        let inner = Rect::from_min_max(
            Pos2::new(outer.min.x + 52.0, outer.min.y + 8.0),
            Pos2::new(outer.max.x - 14.0, outer.max.y - 20.0),
        );
        if inner.width() < 40.0 || inner.height() < 24.0 {
            return;
        }

        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        painter.image(tex_id, inner, uv, Color32::WHITE);

        let top_khz = self
            .acoustics
            .view_max_khz
            .clamp(2.0, self.acoustics.spectrogram.max_freq_khz.max(2.0));

        for step in 0..=4 {
            let frac = step as f32 / 4.0;
            let y = inner.bottom() - frac * inner.height();
            let khz = (frac as f64) * top_khz;
            painter.text(
                Pos2::new(inner.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                format!("{khz:.1}k"),
                FontId::monospace(9.5),
                Color32::from_rgb(203, 213, 225),
            );
        }

        if self.acoustics.show_ridge_overlay && self.acoustics.spectrogram.num_frames > 1 {
            let n_f = self.acoustics.spectrogram.num_frames;
            for (f_idx, opt_khz) in self.acoustics.spectrogram.ridge_khz.iter().enumerate() {
                if let Some(khz) = opt_khz {
                    if (*khz as f64) <= top_khz {
                        let fx = (f_idx as f32) / ((n_f - 1) as f32);
                        let fy = ((*khz as f64) / top_khz) as f32;
                        let p = Pos2::new(
                            inner.left() + fx * inner.width(),
                            inner.bottom() - fy * inner.height(),
                        );
                        painter.circle_filled(p, 1.8, Color32::from_rgb(56, 189, 248));
                    }
                }
            }
        }

        if let Some(hp) = resp.hover_pos() {
            if inner.contains(hp) {
                let nx = (hp.x - inner.left()) / inner.width();
                let ny = (hp.y - inner.top()) / inner.height();
                let (t_ms, f_khz, db) = self.acoustics.spectrogram.sample_dbfs_at(nx, ny, top_khz);
                let badge = format!(" {t_ms:.1} ms | {f_khz:.2} kHz | {db:.1} dBFS ");
                painter.text(
                    Pos2::new(inner.right() - 8.0, inner.top() + 10.0),
                    Align2::RIGHT_TOP,
                    badge,
                    FontId::monospace(10.5),
                    Color32::from_rgb(254, 240, 138),
                );
            }
        }
    }

    fn paint_batch_sweep_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Cut-In (m/s):");
            ui.add(egui::TextEdit::singleline(&mut self.batch_ui.cut_in_csv).desired_width(105.0));

            ui.label("Lights-Out %:");
            ui.add(
                egui::TextEdit::singleline(&mut self.batch_ui.lights_out_csv).desired_width(95.0),
            );

            ui.label("+Patch:");
            ui.add(egui::TextEdit::singleline(&mut self.batch_ui.milkweed_csv).desired_width(58.0));

            ui.label("dTemp:");
            ui.add(egui::TextEdit::singleline(&mut self.batch_ui.temp_csv).desired_width(58.0));

            ui.label("Smoke:");
            ui.add(egui::TextEdit::singleline(&mut self.batch_ui.smoke_csv).desired_width(68.0));

            let btn_lbl = if self.batch_ui.running_sweep {
                "Running Rayon Sweep..."
            } else {
                "Run Rayon v0.6 Sweep"
            };
            if ui
                .add_enabled(!self.batch_ui.running_sweep, egui::Button::new(btn_lbl))
                .clicked()
            {
                let cut_ins = parse_ui_list::<f64>(&self.batch_ui.cut_in_csv, &[3.5, 5.0, 6.0]);
                let lights =
                    parse_ui_list::<f64>(&self.batch_ui.lights_out_csv, &[0.0, 50.0, 85.0]);
                let mw_adds = parse_ui_list::<usize>(&self.batch_ui.milkweed_csv, &[0, 6]);
                let t_offs = parse_ui_list::<f64>(&self.batch_ui.temp_csv, &[0.0, 2.0]);
                let smks = parse_ui_list::<f64>(&self.batch_ui.smoke_csv, &[0.0, 180.0]);

                self.batch_ui.running_sweep = true;
                let region = self.params.region;
                let doy = self.engine.metrics.day_of_year;
                let ssp = self.params.cmip6_scenario;
                let hours = self.batch_ui.sweep_hours;
                let wind_mult = self.params.wind_850_multiplier;
                let precip = self.params.precip_front_mm_hr;
                let turbines = self.turbines.clone();
                let milkweed = self.milkweed.clone();
                let observations = self.observations.clone();
                let weather = self.engine.weather_nodes.clone();
                let tx = self.batch_ui.sweep_tx.clone();
                let egui_ctx = ctx.clone();

                std::thread::spawn(move || {
                    let res = execute_batch_sweep_grid(
                        region,
                        doy,
                        ssp,
                        hours,
                        &cut_ins,
                        &lights,
                        &mw_adds,
                        &t_offs,
                        &smks,
                        wind_mult,
                        precip,
                        &turbines,
                        &milkweed,
                        &observations,
                        &weather,
                    );
                    let _ = tx.send(res);
                    egui_ctx.request_repaint();
                });
            }
        });

        ui.add_space(4.0);
        if self.batch_ui.outcomes.is_empty() {
            ui.small(
                RichText::new(
                    "Click 'Run Rayon v0.6 Sweep' to jointly evaluate turbine bat curtailment, BirdCast Lights-Out building strike reduction, CMIP6 mismatch, and smoke effects.",
                )
                .color(Color32::from_rgb(148, 163, 184)),
            );
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("batch_sweep_results_grid")
                .num_columns(10)
                .striped(true)
                .spacing([12.0, 4.0])
                .show(ui, |ui| {
                    for hdr in [
                        "ID",
                        "Cut-In",
                        "LtsOut",
                        "+Patch",
                        "PM2.5",
                        "Avoid Bat%",
                        "Avoid Strike%",
                        "Pareto",
                        "Mismatch",
                        "North Front",
                    ] {
                        ui.label(RichText::new(hdr).strong().monospace());
                    }
                    ui.end_row();

                    for r in &self.batch_ui.outcomes {
                        ui.monospace(&r.scenario_id);
                        ui.monospace(format!("{:.1} m/s", r.turbine_cut_in_mps));
                        ui.monospace(format!("{:.0}%", r.lights_out_pct));
                        ui.monospace(format!("+{}", r.virtual_milkweed_added));
                        ui.monospace(format!("{:.0}", r.smoke_pm25_ugm3));
                        ui.colored_label(
                            Color32::from_rgb(52, 211, 153),
                            format!("{:.1}%", r.avoided_bat_pct),
                        );
                        ui.colored_label(
                            Color32::from_rgb(244, 114, 182),
                            format!("{:.1}%", r.avoided_urban_strikes_pct),
                        );
                        ui.colored_label(
                            Color32::from_rgb(56, 189, 248),
                            format!("{:.1}", r.pareto_protection_per_mwh_pct),
                        );
                        ui.colored_label(
                            Color32::from_rgb(251, 146, 60),
                            format!("{:+.1}d ({:.0}%)", r.trophic_mismatch_days, r.trophic_overlap_pct),
                        );
                        ui.colored_label(
                            Color32::from_rgb(251, 191, 36),
                            format!("{:.2} N", r.monarch_north_lat),
                        );
                        ui.end_row();
                    }
                });
        });
    }

    fn render_central_map(&mut self, ctx: &egui::Context) {
        let mut frame = egui::Frame::default();
        frame.fill = Color32::from_rgb(8, 13, 21);

        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            let canvas_rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(canvas_rect, Sense::click_and_drag());
            let painter = ui.painter_at(canvas_rect);

            if response.hovered() {
                let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_y.abs() > 0.1 {
                    if let Some(hover_pos) = response.hover_pos() {
                        self.map
                            .zoom_around((scroll_y as f64) * 0.0045, hover_pos, canvas_rect);
                    }
                }
            }

            match self.tool_mode {
                MapToolMode::PanAndInspect => {
                    if response.dragged_by(egui::PointerButton::Primary) {
                        self.map.pan_by_pixels(response.drag_delta());
                    }
                    if response.double_clicked() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lat, lon) = self.map.screen_to_lat_lon(pos, canvas_rect);
                            self.map.jump_to(lat, lon, (self.map.zoom + 1.0).min(14.5));
                        }
                    }
                }
                MapToolMode::DrawTransect => {
                    if response.drag_started_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let pt = self.map.screen_to_lat_lon(pos, canvas_rect);
                            self.transect_drag_anchor = Some(pt);
                            self.params.transect_start = pt;
                            self.params.transect_end = pt;
                            self.engine.cached_transect_dem = None;
                        }
                    } else if response.dragged_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            self.params.transect_end = self.map.screen_to_lat_lon(pos, canvas_rect);
                        }
                    } else if response.drag_stopped() {
                        self.transect_drag_anchor = None;
                    }
                }
                MapToolMode::PlantVirtualMilkweed => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let (lat, lon) = self.map.screen_to_lat_lon(pos, canvas_rect);
                            let label =
                                format!("Virtual Stepping-Stone #{}", self.virtual_patch_counter);
                            self.virtual_patch_counter += 1;
                            let _ = self
                                .store
                                .insert_milkweed_patch(&label, lat, lon, 25.0, true, 155.0);
                            self.milkweed = self.store.load_milkweed_patches();
                        }
                    }
                }
            }

            let tiles = self.map.visible_raster_tiles(ctx, canvas_rect);
            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            let tint = match self.map.provider {
                BasemapProvider::OpenStreetMap | BasemapProvider::UsgsNationalTopo => {
                    Color32::from_rgba_unmultiplied(185, 198, 215, 210)
                }
                _ => Color32::WHITE,
            };
            for (rect, tex) in tiles {
                painter.image(tex.id(), rect, uv, tint);
            }

            self.paint_vector_basemap(&painter, canvas_rect);

            if self.layers.hex_heatmap {
                self.paint_hex_choropleth(&painter, canvas_rect);
            }

            if self.layers.skyglow_domes {
                self.paint_skyglow_domes(&painter, canvas_rect);
            }

            if self.layers.radar_rings {
                self.paint_radar_stations(&painter, canvas_rect);
            }

            if self.layers.wind_barbs {
                self.paint_synoptic_wind_vectors(&painter, canvas_rect);
            }

            if self.layers.milkweed {
                for p in &self.milkweed {
                    let sp = self.map.lat_lon_to_screen(p.lat, p.lon, canvas_rect);
                    if !canvas_rect.expand(30.0).contains(sp) {
                        continue;
                    }
                    let rad = ((p.area_ha.sqrt() as f32) * 1.4).clamp(4.0, 12.0);
                    painter.circle_filled(
                        sp,
                        rad,
                        Color32::from_rgba_unmultiplied(16, 185, 129, 65),
                    );
                    painter.circle_stroke(sp, rad, Stroke::new(1.3, Color32::from_rgb(52, 211, 153)));
                }
            }

            if self.layers.turbines {
                for t in &self.turbines {
                    let sp = self.map.lat_lon_to_screen(t.lat, t.lon, canvas_rect);
                    if !canvas_rect.expand(24.0).contains(sp) {
                        continue;
                    }
                    painter.circle_stroke(
                        sp,
                        5.5,
                        Stroke::new(1.2, Color32::from_rgba_unmultiplied(244, 63, 94, 195)),
                    );
                    painter.circle_filled(sp, 2.0, Color32::from_rgb(251, 113, 133));
                }
            }

            if self.layers.observations {
                for obs in &self.observations {
                    let sp = self.map.lat_lon_to_screen(obs.lat, obs.lon, canvas_rect);
                    if !canvas_rect.expand(12.0).contains(sp) {
                        continue;
                    }
                    let is_acoustic = obs.source_api.starts_with("WAV-STFT")
                        || obs.source_api.starts_with("CLI-WAV");
                    let col = taxon_color(obs.taxon, 225);
                    if is_acoustic {
                        painter.circle_stroke(
                            sp,
                            7.5,
                            Stroke::new(1.6, Color32::from_rgb(254, 240, 138)),
                        );
                    }
                    let r_rect = Rect::from_center_size(sp, Vec2::splat(5.5));
                    painter.rect_stroke(
                        r_rect,
                        CornerRadius::same(1),
                        Stroke::new(1.2, col),
                        StrokeKind::Inside,
                    );
                }
            }

            if self.gpu_mode_active() {
                let aridity = self.engine.metrics.wetland_aridity_index;
                let (gpu_wx, gpu_turb, gpu_att) = pack_gpu_scene_buffers(
                    self.params.region,
                    aridity,
                    self.engine.metrics.hour_of_day,
                    self.engine.metrics.is_nocturnal,
                    self.params.lights_out_curfew_pct,
                    self.params.skyglow_intensity_mult,
                    &self.engine.weather_nodes,
                    &self.turbines,
                    &self.milkweed,
                );
                let step_active = self.last_frame_sim_dt_hr > 0.0;
                let uniforms = pack_gpu_uniforms(
                    self.map.center_lat,
                    self.map.center_lon,
                    self.map.zoom,
                    canvas_rect.width(),
                    canvas_rect.height(),
                    self.last_frame_sim_dt_hr,
                    self.engine.metrics.hour_of_day,
                    self.engine.metrics.day_of_year,
                    self.engine.metrics.mean_corridor_pm25_ugm3,
                    aridity,
                    self.engine.metrics.radar_peak_alt_m,
                    self.engine.metrics.is_nocturnal,
                    step_active,
                    self.layers.birds,
                    self.layers.bats,
                    self.layers.butterflies,
                    self.layers.wake_trails,
                    self.gpu_agent_count,
                    self.gpu_tick_seed,
                    self.engine.metrics.enkf_gain,
                    &self.params,
                    gpu_wx.len(),
                    gpu_turb.len(),
                    gpu_att.len(),
                );
                painter.add(egui_wgpu::Callback::new_paint_callback(
                    canvas_rect,
                    MapGpuSimAndPaintCallback {
                        uniforms,
                        weather_nodes: gpu_wx,
                        turbines: gpu_turb,
                        attractors: gpu_att,
                        region: self.params.region,
                        season: self.params.season,
                    },
                ));
            } else {
                for agent in &self.engine.agents {
                    let enabled = match agent.taxon {
                        Taxon::Bird => self.layers.birds,
                        Taxon::Bat => self.layers.bats,
                        Taxon::Butterfly => self.layers.butterflies,
                    };
                    if !enabled {
                        continue;
                    }
                    let sp = self.map.lat_lon_to_screen(agent.lat, agent.lon, canvas_rect);
                    if !canvas_rect.expand(20.0).contains(sp) {
                        continue;
                    }
                    let radius = if agent.in_fallout_stopover { 3.2 } else { 2.2 };
                    let fill_col = if agent.taxon == Taxon::Bird && agent.skyglow_entrained {
                        Color32::from_rgba_unmultiplied(244, 114, 182, 235)
                    } else if agent.smoke_disrupted && agent.taxon == Taxon::Bird {
                        Color32::from_rgba_unmultiplied(249, 115, 22, 235)
                    } else {
                        taxon_color(agent.taxon, 235)
                    };
                    painter.circle_filled(sp, radius, fill_col);
                }
            }

            if self.layers.landmarks {
                self.paint_landmarks(&painter, canvas_rect);
            }

            let pa = self.map.lat_lon_to_screen(
                self.params.transect_start.0,
                self.params.transect_start.1,
                canvas_rect,
            );
            let pb = self.map.lat_lon_to_screen(
                self.params.transect_end.0,
                self.params.transect_end.1,
                canvas_rect,
            );
            painter.line_segment([pa, pb], Stroke::new(2.0, Color32::from_rgb(52, 211, 153)));
            painter.circle_filled(pa, 5.0, Color32::from_rgb(16, 185, 129));
            painter.circle_filled(pb, 5.0, Color32::from_rgb(16, 185, 129));
            painter.text(
                pa + Vec2::new(7.0, -7.0),
                Align2::LEFT_BOTTOM,
                "A",
                FontId::monospace(12.0),
                Color32::from_rgb(110, 231, 183),
            );
            painter.text(
                pb + Vec2::new(7.0, -7.0),
                Align2::LEFT_BOTTOM,
                "B",
                FontId::monospace(12.0),
                Color32::from_rgb(110, 231, 183),
            );

            self.paint_map_hud(&painter, canvas_rect, response.hover_pos());
        });
    }

    fn paint_skyglow_domes(&self, painter: &egui::Painter, canvas_rect: Rect) {
        let mpp = self.map.meters_per_pixel().max(10.0);
        for dome in skyglow_domes_for_region(self.params.region) {
            let sp = self.map.lat_lon_to_screen(dome.lat, dome.lon, canvas_rect);
            let eff_nw = effective_skyglow_radiance_nw(
                dome.peak_radiance_nw_cm2_sr,
                self.engine.metrics.hour_of_day,
                self.engine.metrics.is_nocturnal,
                self.params.lights_out_curfew_pct,
                self.params.skyglow_intensity_mult,
            );
            let dim_scale = (eff_nw / dome.peak_radiance_nw_cm2_sr.max(1.0)).clamp(0.18, 1.4);
            let px_r = (((dome.radius_km * 1000.0 * dim_scale.sqrt()) / mpp) as f32).clamp(9.0, 95.0);
            if !canvas_rect.expand(px_r).contains(sp) {
                continue;
            }
            let alpha = if self.engine.metrics.is_nocturnal {
                (14.0 + (eff_nw * 0.42)).clamp(12.0, 78.0) as u8
            } else {
                10_u8
            };
            painter.circle_filled(
                sp,
                px_r,
                Color32::from_rgba_unmultiplied(244, 114, 182, alpha),
            );
            painter.circle_stroke(
                sp,
                px_r,
                Stroke::new(1.1, Color32::from_rgba_unmultiplied(251, 191, 36, 130)),
            );
            if self.map.zoom >= 5.2 {
                painter.text(
                    sp + Vec2::new(0.0, -px_r - 4.0),
                    Align2::CENTER_BOTTOM,
                    format!("[VIIRS {:.0} nW] {}", eff_nw, dome.city_name),
                    FontId::monospace(9.5),
                    Color32::from_rgb(251, 207, 232),
                );
            }
        }
    }

    fn paint_radar_stations(&self, painter: &egui::Painter, canvas_rect: Rect) {
        let mpp = self.map.meters_per_pixel().max(10.0);
        for prof in &self.engine.radar_profiles {
            let sp = self.map.lat_lon_to_screen(prof.lat, prof.lon, canvas_rect);
            let ring_px = (((95_000.0) / mpp) as f32).clamp(12.0, 88.0);
            if !canvas_rect.expand(ring_px).contains(sp) {
                continue;
            }
            painter.circle_stroke(
                sp,
                ring_px,
                Stroke::new(1.1, Color32::from_rgba_unmultiplied(34, 211, 238, 125)),
            );
            painter.circle_filled(sp, 3.8, Color32::from_rgb(34, 211, 238));
            painter.text(
                sp + Vec2::new(6.0, 0.0),
                Align2::LEFT_CENTER,
                format!("[RADAR {}] {:.0} MTR", prof.station_code, prof.mtr_birds_km_hr),
                FontId::monospace(10.0),
                Color32::from_rgb(103, 232, 249),
            );
        }
    }

    fn paint_vector_basemap(&self, painter: &egui::Painter, canvas_rect: Rect) {
        let graticule_stroke = Stroke::new(0.6, Color32::from_rgba_unmultiplied(51, 65, 85, 95));
        for lat_i in (-60..=72).step_by(4) {
            let p1 = self.map.lat_lon_to_screen(lat_i as f64, -175.0, canvas_rect);
            let p2 = self.map.lat_lon_to_screen(lat_i as f64, 175.0, canvas_rect);
            painter.line_segment([p1, p2], graticule_stroke);
        }
        for lon_i in (-176..=176).step_by(4) {
            let p1 = self.map.lat_lon_to_screen(-60.0, lon_i as f64, canvas_rect);
            let p2 = self.map.lat_lon_to_screen(72.0, lon_i as f64, canvas_rect);
            painter.line_segment([p1, p2], graticule_stroke);
        }

        if self.params.region == FlywayRegion::PacificAmericas {
            let draw_poly = |coords: &[(f64, f64)], stroke: Stroke| {
                let pts: Vec<Pos2> = coords
                    .iter()
                    .map(|(lt, ln)| self.map.lat_lon_to_screen(*lt, *ln, canvas_rect))
                    .collect();
                painter.add(egui::Shape::line(pts, stroke));
            };

            for border in STATE_BOUNDARIES {
                draw_poly(
                    border,
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(100, 116, 139, 135)),
                );
            }
            draw_poly(
                PACIFIC_COASTLINE,
                Stroke::new(2.0, Color32::from_rgba_unmultiplied(71, 135, 179, 220)),
            );
            draw_poly(
                PUGET_SOUND_RING,
                Stroke::new(1.4, Color32::from_rgba_unmultiplied(71, 135, 179, 185)),
            );
            draw_poly(
                COLUMBIA_RIVER,
                Stroke::new(1.8, Color32::from_rgba_unmultiplied(34, 211, 238, 165)),
            );
            draw_poly(
                WILLAMETTE_RIVER,
                Stroke::new(1.6, Color32::from_rgba_unmultiplied(56, 189, 248, 175)),
            );
            draw_poly(
                KLAMATH_SACRAMENTO_CORRIDOR,
                Stroke::new(1.2, Color32::from_rgba_unmultiplied(56, 189, 248, 120)),
            );
        }
    }

    fn paint_hex_choropleth(&self, painter: &egui::Painter, canvas_rect: Rect) {
        let counts = self.engine.hex_density_counts();
        let px_radius = ((24_000.0 / self.map.meters_per_pixel()) as f32 * 0.42).clamp(4.0, 38.0);

        for (hex_key, count) in counts {
            if count < 2 {
                continue;
            }
            let Some((lat, lon)) = hex_bin_center(&hex_key) else {
                continue;
            };
            let center = self.map.lat_lon_to_screen(lat, lon, canvas_rect);
            if !canvas_rect.expand(px_radius).contains(center) {
                continue;
            }

            let alpha = (18 + count * 4).min(95) as u8;
            let mut verts = Vec::with_capacity(6);
            for k in 0..6 {
                let ang = (k as f32) * std::f32::consts::TAU / 6.0 + std::f32::consts::FRAC_PI_6;
                verts.push(Pos2::new(
                    center.x + px_radius * ang.cos(),
                    center.y + px_radius * ang.sin(),
                ));
            }
            painter.add(egui::Shape::convex_polygon(
                verts,
                Color32::from_rgba_unmultiplied(56, 189, 248, alpha),
                Stroke::new(0.7, Color32::from_rgba_unmultiplied(56, 189, 248, 65)),
            ));
        }
    }

    fn paint_synoptic_wind_vectors(&self, painter: &egui::Painter, canvas_rect: Rect) {
        for n in &self.engine.weather_nodes {
            let origin = self.map.lat_lon_to_screen(n.lat, n.lon, canvas_rect);
            if !canvas_rect.contains(origin) {
                continue;
            }
            let rad = (n.wind_850_deg + 180.0).to_radians() as f32;
            let mag = (n.wind_850_mps * self.params.wind_850_multiplier) as f32;
            let vec_px = Vec2::new(rad.sin() * mag * 2.8, -rad.cos() * mag * 2.8);
            painter.arrow(
                origin,
                vec_px,
                Stroke::new(1.5, Color32::from_rgba_unmultiplied(148, 163, 184, 190)),
            );
        }
    }

    fn paint_landmarks(&self, painter: &egui::Painter, canvas_rect: Rect) {
        for lm in landmarks_for_region(self.params.region) {
            let sp = self.map.lat_lon_to_screen(lm.lat, lm.lon, canvas_rect);
            if !canvas_rect.contains(sp) {
                continue;
            }
            let (col, prefix) = match lm.kind {
                LandmarkKind::WildlifeRefuge => (Color32::from_rgb(74, 222, 128), "NWR"),
                LandmarkKind::MotusStation => (Color32::from_rgb(34, 211, 238), "MOTUS"),
                LandmarkKind::MonarchSanctuary => (Color32::from_rgb(251, 191, 36), "ROOST"),
                LandmarkKind::CascadePeak => (Color32::from_rgb(203, 213, 225), "PEAK"),
            };
            painter.circle_stroke(sp, 4.5, Stroke::new(1.4, col));
            if self.map.zoom >= 5.2 {
                painter.text(
                    sp + Vec2::new(7.0, 0.0),
                    Align2::LEFT_CENTER,
                    format!("[{prefix}] {}", lm.name),
                    FontId::proportional(11.0),
                    col,
                );
            }
        }
    }

    fn paint_map_hud(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        hover_pos: Option<Pos2>,
    ) {
        let mpp = self.map.meters_per_pixel();
        let target_km = if mpp * 110.0 > 100_000.0 {
            200.0
        } else if mpp * 110.0 > 40_000.0 {
            50.0
        } else if mpp * 110.0 > 10_000.0 {
            15.0
        } else {
            5.0
        };
        let bar_px = ((target_km * 1000.0) / mpp) as f32;

        let hud_rect = Rect::from_min_size(
            Pos2::new(canvas_rect.left() + 12.0, canvas_rect.bottom() - 54.0),
            Vec2::new(bar_px.max(230.0) + 24.0, 42.0),
        );
        painter.rect_filled(
            hud_rect,
            CornerRadius::same(6),
            Color32::from_rgba_unmultiplied(15, 23, 42, 215),
        );
        painter.rect_stroke(
            hud_rect,
            CornerRadius::same(6),
            Stroke::new(1.0, Color32::from_rgb(51, 65, 85)),
            StrokeKind::Inside,
        );

        let bar_y = hud_rect.top() + 14.0;
        let bar_x0 = hud_rect.left() + 12.0;
        painter.line_segment(
            [Pos2::new(bar_x0, bar_y), Pos2::new(bar_x0 + bar_px, bar_y)],
            Stroke::new(2.2, Color32::from_rgb(226, 232, 240)),
        );
        painter.text(
            Pos2::new(bar_x0 + bar_px + 8.0, bar_y),
            Align2::LEFT_CENTER,
            format!("{target_km:.0} km (z={:.1})", self.map.zoom),
            FontId::monospace(10.5),
            Color32::from_rgb(203, 213, 225),
        );

        if let Some(hp) = hover_pos {
            let (hlat, hlon) = self.map.screen_to_lat_lon(hp, canvas_rect);
            painter.text(
                Pos2::new(bar_x0, hud_rect.bottom() - 11.0),
                Align2::LEFT_CENTER,
                format!("Cursor: {hlat:.4} N, {hlon:.4} E"),
                FontId::monospace(10.5),
                Color32::from_rgb(148, 163, 184),
            );
        } else {
            painter.text(
                Pos2::new(bar_x0, hud_rect.bottom() - 11.0),
                Align2::LEFT_CENTER,
                format!(
                    "Center: {:.4} N, {:.4} E",
                    self.map.center_lat, self.map.center_lon
                ),
                FontId::monospace(10.5),
                Color32::from_rgb(148, 163, 184),
            );
        }
    }
}

impl eframe::App for AeroFlywayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_keyboard_shortcuts(ctx);
        self.handle_dropped_wav_files(ctx);
        self.poll_background_channels();
        self.last_frame_sim_dt_hr = 0.0;

        if self.running && self.stage == AppStage::Workbench {
            let real_dt = ctx.input(|i| i.stable_dt as f64).clamp(0.004, 0.1);
            let sim_dt_hr = real_dt * self.sim_speed_hr_per_sec;
            self.last_frame_sim_dt_hr = sim_dt_hr;
            self.gpu_tick_seed = self.gpu_tick_seed.wrapping_add(1);

            self.engine.step(
                sim_dt_hr,
                &self.params,
                &self.turbines,
                &self.milkweed,
                &self.observations,
            );
            self.snapshot_accum_hr += sim_dt_hr;
            if self.snapshot_accum_hr >= 2.0 {
                self.snapshot_accum_hr = 0.0;
                self.store.record_snapshot(&ModelSnapshot {
                    tick_hour: self.engine.metrics.sim_hours,
                    enkf_rmse: self.engine.metrics.enkf_rmse,
                    avian_flux_index: self.engine.metrics.avian_flux_agents_airborne as f64,
                    bat_rotor_contacts: self.engine.metrics.cumulative_bat_rotor_contacts,
                    monarch_front_lat: self.engine.metrics.monarch_northern_front_lat,
                });
            }
            ctx.request_repaint();
        }

        self.render_native_menu_bar(ctx);
        self.render_help_modal(ctx);

        match self.stage {
            AppStage::StartupHub => {
                self.render_startup_hub(ctx);
            }
            AppStage::Workbench => {
                self.render_top_bar(ctx);
                self.render_left_dock(ctx);
                self.render_bottom_dock(ctx);
                self.render_central_map(ctx);
            }
        }
    }
}

fn parse_ui_list<T: std::str::FromStr + Copy>(raw: &str, fallback: &[T]) -> Vec<T> {
    let parsed: Vec<T> = raw
        .split(',')
        .filter_map(|p| p.trim().parse::<T>().ok())
        .collect();
    if parsed.is_empty() {
        fallback.to_vec()
    } else {
        parsed
    }
}

fn taxon_color(taxon: Taxon, alpha: u8) -> Color32 {
    match taxon {
        Taxon::Bird => Color32::from_rgba_unmultiplied(56, 189, 248, alpha),
        Taxon::Bat => Color32::from_rgba_unmultiplied(192, 132, 252, alpha),
        Taxon::Butterfly => Color32::from_rgba_unmultiplied(251, 191, 36, alpha),
    }
}

fn configure_dark_telemetry_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(13, 19, 30);
    visuals.window_fill = Color32::from_rgb(15, 23, 42);
    visuals.extreme_bg_color = Color32::from_rgb(8, 12, 20);
    ctx.set_visuals(visuals);
}

fn main() -> eframe::Result<()> {
    let skip_hub = match try_dispatch_cli() {
        Ok(CliDispatchOutcome::HandledExit) => return Ok(()),
        Ok(CliDispatchOutcome::LaunchGui { skip_startup_hub }) => skip_startup_hub,
        Err(err) => {
            eprintln!("[AeroFlyway CLI Error] {err}");
            std::process::exit(2);
        }
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1580.0, 980.0])
            .with_min_inner_size([1040.0, 700.0])
            .with_drag_and_drop(true)
            .with_title(
                "AeroFlyway v0.6 — Startup Hub, VIIRS Lights-Out, NEXRAD/Aloft VPTS & WAV STFT",
            ),
        renderer: eframe::Renderer::Wgpu,
        multisampling: 0,
        depth_buffer: 0,
        stencil_buffer: 0,
        ..Default::default()
    };

    eframe::run_native(
        "AeroFlyway",
        native_options,
        Box::new(move |cc| Ok(Box::new(AeroFlywayApp::new(cc, skip_hub)))),
    )
}
