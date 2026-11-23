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

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! AeroFlyway v0.6.0 — Native Rust `wgpu` + WGSL 100k-Agent Multi-Flyway GIS Workstation:
//! - Startup Campaign Hub (`AppStage::StartupHub`) + Native Persistent Menu Bar (`egui::menu::bar`)
//! - VIIRS Nighttime Skyglow Urban Collision Modeling & BirdCast `Lights-Out` Curfew Sliders
//! - NEXRAD WSR-88D & ENRAM Aloft Vertical Radar Profile (`VPTS`) Ingestion & Altitude Explorer
//! - Continuous 365-Day DOY Phenology, Open-Meteo CMIP6 + CAMS PM2.5 Smoke Forcing & WAV STFT Lab.

use aeroflyway::acoustics::{
    compute_stft_spectrogram, load_wav_file, magma_lut, parse_wav_bytes,
    synthesize_preset_recording, write_time_expanded_wav_16bit, write_wav_16bit,
    SpectrogramResult, SynthPreset, WavRecording, SPECTROGRAM_DB_CEIL,
};
use aeroflyway::basemap::{
    landmarks_for_region, radar_stations_for_region, skyglow_domes_for_region,
    vector_basemap_for_region, FlywayRegion, LandmarkKind, MilkweedPatch, Taxon, WindTurbine,
};
use aeroflyway::cli::{
    execute_batch_sweep_grid, try_dispatch_cli, write_batch_sweep_reports, BatchScenarioOutcome,
    CliDispatchOutcome,
};
use aeroflyway::db::{
    hex_bin_center, spatial_hex_bin, CampaignSummary, FlywayStore, ModelSnapshot, ObservationRecord,
};
use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2,
};
use eframe::egui_wgpu;
use aeroflyway::export::{export_corridor_csv_bundle, export_corridor_geojson_to_path};
use aeroflyway::gpu_pipeline::{
    pack_gpu_scene_buffers, pack_gpu_uniforms, FlywayGpuResources, GpuReadbackSummary,
    MapGpuSimAndPaintCallback, SharedReadbackHandle, TransectGpuPaintCallback, DEFAULT_GPU_AGENTS,
    MAX_GPU_AGENTS,
};
use aeroflyway::ingest::{
    build_regional_radar_profiles, load_vpts_csv_file, spawn_live_ingest, Cmip6Scenario,
    IngestPayload,
};
use aeroflyway::map_engine::{BasemapProvider, MapLabelPlacer, MapViewport};
use aeroflyway::physics::{
    compute_gpu_scale_factor, compute_lights_out_collision_reduction_pct,
    default_synoptic_nodes_for_region, effective_skyglow_radiance_nw, AeroPhysicsEngine,
    SeasonPreset, SimulationParams,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScenarioPreset {
    AutumnLightsOutPeak,
    MegafireSmokeSsp585,
    SpringFalloutUnmitigated,
}

impl ScenarioPreset {
    fn label(self) -> &'static str {
        match self {
            Self::AutumnLightsOutPeak => "Autumn Lights-Out",
            Self::MegafireSmokeSsp585 => "Megafire SSP5-8.5",
            Self::SpringFalloutUnmitigated => "Spring Fallout",
        }
    }

    fn hover_description(self) -> &'static str {
        match self {
            Self::AutumnLightsOutPeak => {
                "DOY 272 peak autumn flight with 85% BirdCast Lights-Out curfew, 5.5 m/s smart bat curtailment, and 1.15x 850 hPa tailwind."
            }
            Self::MegafireSmokeSsp585 => {
                "DOY 258 CMIP6 SSP5-8.5 (+4.1 C) with 195 ug/m^3 wildfire smoke plume forcing >1,850 m AGL climbs and trophic mismatch."
            }
            Self::SpringFalloutUnmitigated => {
                "DOY 128 spring northbound push under SSP2-4.5 with 2.6 mm/hr frontal rain fallout, 0% Lights-Out, and 1.6x skyglow."
            }
        }
    }
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
            hex_heatmap: false,
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
    selected_pulse_idx: Option<usize>,
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
            selected_pulse_idx: None,
        }
    }

    fn load_new_recording(&mut self, rec: WavRecording) {
        let nyquist_khz = if rec.nyquist_khz().is_finite() {
            rec.nyquist_khz().max(0.5)
        } else {
            0.5
        };
        let min_khz = nyquist_khz.min(2.0);
        let default_top = if nyquist_khz > 40.0 {
            nyquist_khz.min(95.0)
        } else {
            nyquist_khz.min(14.0)
        };
        self.view_max_khz = default_top.clamp(min_khz, nyquist_khz);
        self.spectrogram = compute_stft_spectrogram(&rec, self.fft_size);
        self.recording = rec;
        self.selected_pulse_idx = None;
        self.texture_dirty = true;
    }

    fn recompute_stft(&mut self) {
        self.spectrogram = compute_stft_spectrogram(&self.recording, self.fft_size);
        self.selected_pulse_idx = None;
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
    show_about_modal: bool,
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
    bottom_dock_default_height: f32,
    last_transect_size_pts: (f32, f32),
    active_scenario_preset: Option<ScenarioPreset>,
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
        Self::new_with_wgpu(
            &cc.egui_ctx,
            cc.wgpu_render_state.as_ref(),
            PathBuf::from("aeroflyway.db"),
            skip_startup_hub,
        )
    }

    fn new_with_wgpu(
        ctx: &egui::Context,
        wgpu_render_state: Option<&egui_wgpu::RenderState>,
        db_path: PathBuf,
        skip_startup_hub: bool,
    ) -> Self {
        configure_dark_telemetry_visuals(ctx);

        let db_name = db_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("aeroflyway.db")
            .to_string();
        let store = FlywayStore::open(&db_path).expect("Failed to initialize aeroflyway.db");
        let active_campaign = store.load_campaign_summary(&db_name);
        let observations = store.load_observations();
        let turbines = store.load_turbines();
        let milkweed = store.load_milkweed_patches();

        let params = SimulationParams::default();
        let gpu_readback = SharedReadbackHandle::new();
        let mut gpu_available = false;

        if let Some(wgpu_render_state) = wgpu_render_state {
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
            campaign_db_path_input: db_name,
            active_campaign,
            show_help_modal: false,
            show_about_modal: false,
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
            bottom_dock_default_height: 255.0,
            last_transect_size_pts: (840.0, 180.0),
            active_scenario_preset: None,
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
        compute_gpu_scale_factor(
            self.gpu_mode_active(),
            self.gpu_agent_count,
            self.engine.agents.len(),
        )
    }

    fn open_or_switch_campaign_db(&mut self, db_file: &str, title: &str, region: FlywayRegion) {
        let trimmed = db_file.trim();
        if trimmed.is_empty() {
            self.status_line = "Campaign DB path cannot be empty.".to_string();
            return;
        }
        let p = PathBuf::from(trimmed);
        match FlywayStore::open(&p) {
            Ok(new_store) => {
                self.store = new_store;
                self.campaign_db_path_input = trimmed.to_string();
                self.campaign_name_input = title.to_string();
                self.apply_region_switch(region);
                if let Err(err) = self.store.save_campaign_metadata(
                    title,
                    self.params.region,
                    self.params.day_of_year,
                    self.params.cmip6_scenario.code(),
                    self.params.lights_out_curfew_pct,
                ) {
                    self.status_line = format!(
                        "Opened campaign '{}' ({}) with metadata save warning: {err}",
                        title, trimmed
                    );
                } else {
                    self.status_line = format!("Opened campaign '{}' ({})", title, trimmed);
                }
                self.active_campaign = self.store.load_campaign_summary(trimmed);
            }
            Err(err) => {
                self.status_line = format!("Failed opening campaign DB '{}': {err}", p.display());
            }
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

        let seed_err = self.store.switch_region_seed(new_region).err();
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
        self.status_line = if let Some(err) = seed_err {
            format!(
                "Switched flyway corridor -> {} (SQLite seed warning: {err})",
                new_region.label()
            )
        } else {
            format!(
                "Switched flyway corridor -> {} (DOY {:.0}, Lights-Out {:.0}%)",
                new_region.label(),
                self.params.day_of_year,
                self.params.lights_out_curfew_pct
            )
        };
    }

    fn save_active_campaign_metadata(&mut self) {
        match self.store.save_campaign_metadata(
            &self.campaign_name_input,
            self.params.region,
            self.params.day_of_year,
            self.params.cmip6_scenario.code(),
            self.params.lights_out_curfew_pct,
        ) {
            Ok(()) => {
                self.active_campaign = self
                    .store
                    .load_campaign_summary(&self.campaign_db_path_input);
                self.status_line = format!(
                    "Saved campaign '{}' into {}",
                    self.campaign_name_input, self.campaign_db_path_input
                );
            }
            Err(err) => {
                self.status_line = format!("Failed saving campaign metadata to SQLite: {err}");
            }
        }
    }

    fn reset_camera(&mut self) {
        let (clat, clon, zoom) = self.params.region.default_viewport();
        self.map.jump_to(clat, clon, zoom);
        self.map.pitch_deg = 42.0;
        self.map.bearing_deg = -18.0;
        self.map.alt_exaggeration = 1.0;
        self.status_line = format!(
            "Reset camera to {} default ({clat:.2}, {clon:.2}, z={zoom:.1}, 2.5D)",
            self.params.region.code()
        );
    }

    fn toggle_lights_out_curfew(&mut self) {
        let enable = self.params.lights_out_curfew_pct < 70.0;
        self.params.lights_out_curfew_pct = if enable { 85.0 } else { 0.0 };
        let est_reduction =
            compute_lights_out_collision_reduction_pct(self.params.lights_out_curfew_pct);
        self.status_line = if enable {
            format!(
                "Lights-Out curfew ON: 85% dimmed (23:00-06:00, ~{est_reduction:.0}% strike reduction)"
            )
        } else {
            "Lights-Out curfew OFF: 0% dimmed (unmitigated urban skyglow)".to_string()
        };
    }

    fn apply_scenario_preset(&mut self, preset: ScenarioPreset) {
        match preset {
            ScenarioPreset::AutumnLightsOutPeak => {
                self.params.season = SeasonPreset::AutumnFalloutAndBatPulse;
                self.params.day_of_year = 272.0;
                self.params.cmip6_scenario = Cmip6Scenario::BaselineHistorical;
                self.params.lights_out_curfew_pct = 85.0;
                self.params.skyglow_intensity_mult = 1.0;
                self.params.turbine_cut_in_mps = 5.5;
                self.params.smoke_pm25_offset_ugm3 = 0.0;
                self.params.precip_front_mm_hr = 0.4;
                self.params.wind_850_multiplier = 1.15;
                self.params.radar_nudge_weight = 0.65;
            }
            ScenarioPreset::MegafireSmokeSsp585 => {
                self.params.season = SeasonPreset::AutumnFalloutAndBatPulse;
                self.params.day_of_year = 258.0;
                self.params.cmip6_scenario = Cmip6Scenario::Ssp585High;
                self.params.lights_out_curfew_pct = 35.0;
                self.params.skyglow_intensity_mult = 1.0;
                self.params.turbine_cut_in_mps = 3.5;
                self.params.smoke_pm25_offset_ugm3 = 195.0;
                self.params.precip_front_mm_hr = 0.0;
                self.params.wind_850_multiplier = 1.0;
                self.params.radar_nudge_weight = 0.65;
            }
            ScenarioPreset::SpringFalloutUnmitigated => {
                self.params.season = SeasonPreset::SpringNorthbound;
                self.params.day_of_year = 128.0;
                self.params.cmip6_scenario = Cmip6Scenario::Ssp245Intermediate;
                self.params.lights_out_curfew_pct = 0.0;
                self.params.skyglow_intensity_mult = 1.6;
                self.params.turbine_cut_in_mps = 3.0;
                self.params.smoke_pm25_offset_ugm3 = 0.0;
                self.params.precip_front_mm_hr = 2.6;
                self.params.wind_850_multiplier = 1.45;
                self.params.radar_nudge_weight = 0.65;
            }
        }
        self.engine.weather_nodes =
            default_synoptic_nodes_for_region(self.params.region, self.params.cmip6_scenario);
        self.engine.radar_profiles = build_regional_radar_profiles(
            None,
            self.params.region,
            &self.engine.weather_nodes,
            self.params.smoke_pm25_offset_ugm3,
        );
        self.engine
            .reseed_agents(self.params.region, self.params.season);
        self.reset_all_counters();
        self.gpu_tick_seed = self.gpu_tick_seed.wrapping_add(1);
        self.active_scenario_preset = Some(preset);
        self.status_line = format!(
            "Applied scenario '{}': DOY {:.0}, {}, Lights-Out {:.0}%, Smoke +{:.0} ug/m3",
            preset.label(),
            self.params.day_of_year,
            self.params.cmip6_scenario.code().to_uppercase(),
            self.params.lights_out_curfew_pct,
            self.params.smoke_pm25_offset_ugm3
        );
    }

    fn handle_keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        let wants_kb = ctx.wants_keyboard_input();
        let (esc_pressed, save_pressed) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Escape),
                (i.modifiers.command || i.modifiers.ctrl) && i.key_pressed(egui::Key::S),
            )
        });
        if esc_pressed {
            if self.show_about_modal {
                self.show_about_modal = false;
            } else if self.show_help_modal {
                self.show_help_modal = false;
            } else if !wants_kb {
                self.tool_mode = MapToolMode::PanAndInspect;
                self.transect_drag_anchor = None;
            }
        }
        if save_pressed {
            self.save_active_campaign_metadata();
        }
        if wants_kb {
            return;
        }
        let (
            space_pressed,
            f1_pressed,
            h_pressed,
            cam_reset_pressed,
            tab_1,
            tab_2,
            tab_3,
            tab_4,
            t_pressed,
            m_pressed,
            l_pressed,
        ) = ctx.input(|i| {
            let no_mod = !i.modifiers.ctrl && !i.modifiers.command && !i.modifiers.alt;
            if !no_mod {
                return (
                    false, false, false, false, false, false, false, false, false, false, false,
                );
            }
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::F1),
                i.key_pressed(egui::Key::H),
                i.key_pressed(egui::Key::Num0) || i.key_pressed(egui::Key::R),
                i.key_pressed(egui::Key::Num1),
                i.key_pressed(egui::Key::Num2),
                i.key_pressed(egui::Key::Num3),
                i.key_pressed(egui::Key::Num4),
                i.key_pressed(egui::Key::T),
                i.key_pressed(egui::Key::M),
                i.key_pressed(egui::Key::L),
            )
        });

        if space_pressed {
            self.running = !self.running;
            self.status_line = if self.running {
                "Simulation resumed.".to_string()
            } else {
                "Simulation paused.".to_string()
            };
        }
        if f1_pressed {
            self.show_help_modal = !self.show_help_modal;
        }
        if h_pressed {
            self.stage = match self.stage {
                AppStage::StartupHub => AppStage::Workbench,
                AppStage::Workbench => AppStage::StartupHub,
            };
        }
        if cam_reset_pressed {
            self.reset_camera();
        }
        if tab_1 {
            self.stage = AppStage::Workbench;
            self.bottom_tab = BottomDockTab::AirspaceTransect;
            self.status_line = "Tab [1]: Airspace Cross-Section (0-4,200 m ASL)".to_string();
        } else if tab_2 {
            self.stage = AppStage::Workbench;
            self.bottom_tab = BottomDockTab::RadarVerticalProfile;
            self.status_line = "Tab [2]: NEXRAD / Aloft Vertical Profile (VPTS)".to_string();
        } else if tab_3 {
            self.stage = AppStage::Workbench;
            self.bottom_tab = BottomDockTab::AcousticSpectrogram;
            self.status_line = "Tab [3]: WAV STFT Bioacoustic Inspector".to_string();
        } else if tab_4 {
            self.stage = AppStage::Workbench;
            self.bottom_tab = BottomDockTab::BatchSweepLab;
            self.status_line = "Tab [4]: Multi-Scenario Batch Sweep Lab".to_string();
        }
        if t_pressed {
            self.tool_mode = if self.tool_mode == MapToolMode::DrawTransect {
                MapToolMode::PanAndInspect
            } else {
                MapToolMode::DrawTransect
            };
            self.transect_drag_anchor = None;
            self.status_line = match self.tool_mode {
                MapToolMode::DrawTransect => {
                    "Tool mode: Draw A-B Transect (primary-drag on map)".to_string()
                }
                _ => "Tool mode: Pan & Inspect".to_string(),
            };
        }
        if m_pressed {
            self.tool_mode = if self.tool_mode == MapToolMode::PlantVirtualMilkweed {
                MapToolMode::PanAndInspect
            } else {
                MapToolMode::PlantVirtualMilkweed
            };
            self.transect_drag_anchor = None;
            self.status_line = match self.tool_mode {
                MapToolMode::PlantVirtualMilkweed => {
                    "Tool mode: Plant Virtual Milkweed Host Patch (click map)".to_string()
                }
                _ => "Tool mode: Pan & Inspect".to_string(),
            };
        }
        if l_pressed {
            self.toggle_lights_out_curfew();
        }
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
                        match load_vpts_csv_file(path, st) {
                            Ok(prof) => {
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
                            Err(err) => {
                                self.status_line = format!("VPTS CSV load error: {err}");
                                return;
                            }
                        }
                    } else {
                        self.status_line =
                            "VPTS CSV load error: no radar station configured for region"
                                .to_string();
                        return;
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

        match self.store.upsert_observations(&[rec]) {
            Ok(_) => {
                self.observations = self.store.load_observations();
                self.map.jump_to(lat, lon, self.map.zoom.max(7.8));
                self.status_line = format!(
                    "Pinned '{}' at ({lat:.4}, {lon:.4}) into SQLite & GIS map",
                    spec.guild_label
                );
            }
            Err(err) => {
                self.status_line = format!("Failed pinning acoustic detection to SQLite: {err}");
            }
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
            let mut db_warn: Option<String> = None;
            if !payload.observations.is_empty() {
                if let Err(err) = self.store.upsert_observations(&payload.observations) {
                    db_warn = Some(format!("SQLite observation upsert failed: {err}"));
                } else {
                    self.observations = self.store.load_observations();
                }
            }
            if !payload.turbines.is_empty() {
                if let Err(err) = self.store.upsert_turbines(&payload.turbines) {
                    db_warn = Some(format!("SQLite turbine upsert failed: {err}"));
                } else {
                    self.turbines = self.store.load_turbines();
                }
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
            self.status_line = match db_warn {
                Some(warn) => format!("{} ({warn})", payload.status_summary),
                None => payload.status_summary,
            };
        }

        while let Ok(sweep_res) = self.batch_ui.sweep_rx.try_recv() {
            self.batch_ui.running_sweep = false;
            let out_dir = self.export_dir_input.trim().to_string();
            match write_batch_sweep_reports(&out_dir, &sweep_res) {
                Ok((csv_p, _)) => {
                    self.status_line = format!(
                        "Completed {} Rayon v0.6 sweep scenarios -> {}",
                        sweep_res.len(),
                        csv_p.display()
                    );
                }
                Err(err) => {
                    self.status_line = format!(
                        "Completed {} scenarios, but sweep export failed: {err}",
                        sweep_res.len()
                    );
                }
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
                    if ui.button("Save Campaign Metadata to SQLite (Ctrl+S)").clicked() {
                        self.save_active_campaign_metadata();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Export RFC 7946 GeoJSON...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Save RFC 7946 Corridor GeoJSON")
                            .add_filter("GeoJSON", &["geojson", "json"])
                            .set_file_name("aeroflyway_corridor.geojson")
                            .save_file()
                        {
                            match export_corridor_geojson_to_path(
                                &path,
                                &self.engine,
                                &self.params,
                                &self.observations,
                                &self.turbines,
                                &self.milkweed,
                                self.current_scale_factor(),
                            ) {
                                Ok((p, n)) => {
                                    self.status_line =
                                        format!("Exported {n} GeoJSON features -> {}", p.display());
                                }
                                Err(err) => {
                                    self.status_line = format!("GeoJSON export error: {err}");
                                }
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("Export Corridor CSV Bundle...").clicked() {
                        if let Some(dir) = rfd::FileDialog::new()
                            .set_title("Select Export Folder for Corridor CSV Bundle")
                            .pick_folder()
                        {
                            self.export_dir_input = dir.display().to_string();
                            match export_corridor_csv_bundle(
                                &dir,
                                &self.engine,
                                &self.params,
                                &self.observations,
                                self.current_scale_factor(),
                            ) {
                                Ok((hp, _, n)) => {
                                    self.status_line =
                                        format!("Exported {n} CSV rows -> {}", hp.display());
                                }
                                Err(err) => {
                                    self.status_line = format!("CSV bundle export error: {err}");
                                }
                            }
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
                    if ui.button("Toggle BirdCast 85% Lights-Out Curfew (L)").clicked() {
                        self.toggle_lights_out_curfew();
                        ui.close_menu();
                    }
                    if ui.button("Reset Camera to Corridor 2.5D Default (0 / R)").clicked() {
                        self.reset_camera();
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
                        self.status_line = "Applied +195 ug/m3 wildfire smoke plume offset.".to_string();
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
                        self.status_line = "Cleared wildfire smoke PM2.5 offset.".to_string();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Open Radar Vertical Profile Tab (2)").clicked() {
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
                    if ui.button("About AeroFlyway...").clicked() {
                        self.show_about_modal = true;
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
                    if ui.button("About").clicked() {
                        self.show_about_modal = true;
                    }
                    ui.small(
                        RichText::new("Copyright CRV Science Network")
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.separator();
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
                        ui.horizontal_wrapped(|ui| {
                            ui.label("SQLite `.db` File:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.campaign_db_path_input)
                                    .desired_width(175.0),
                            );
                            if ui.button("Browse .db...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Open SQLite Campaign Workspace")
                                    .add_filter("SQLite Campaign", &["db", "sqlite", "sqlite3"])
                                    .pick_file()
                                {
                                    let db_f = path.display().to_string();
                                    self.campaign_db_path_input = db_f.clone();
                                    let title = self.campaign_name_input.clone();
                                    let reg = self.params.region;
                                    self.open_or_switch_campaign_db(&db_f, &title, reg);
                                }
                            }
                            if ui.button("New .db...").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Create SQLite Campaign Workspace")
                                    .add_filter("SQLite Campaign", &["db", "sqlite"])
                                    .set_file_name("aeroflyway_campaign.db")
                                    .save_file()
                                {
                                    let db_f = path.display().to_string();
                                    self.campaign_db_path_input = db_f.clone();
                                    let title = self.campaign_name_input.clone();
                                    let reg = self.params.region;
                                    self.open_or_switch_campaign_db(&db_f, &title, reg);
                                }
                            }
                            if ui.button("Open / Create .db").clicked() {
                                let db_f = self.campaign_db_path_input.clone();
                                let title = self.campaign_name_input.clone();
                                let reg = self.params.region;
                                self.open_or_switch_campaign_db(&db_f, &title, reg);
                            }
                        });

                        ui.add_space(4.0);
                        ui.small(
                            RichText::new(format!(
                                "Active Campaign: '{}' ({}) | Region: {} | DOY {:.0} | {} | Lights-Out {:.0}% | {} obs | Updated: {}",
                                self.active_campaign.campaign_name,
                                self.active_campaign.db_path,
                                self.active_campaign.region_code,
                                self.active_campaign.doy,
                                self.active_campaign.cmip6_code,
                                self.active_campaign.lights_out_pct,
                                self.active_campaign.observation_count,
                                self.active_campaign.updated_at,
                            ))
                            .monospace()
                            .color(Color32::from_rgb(148, 163, 184)),
                        );

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
                        )
                        .on_hover_text("Continuous 365-day Day-of-Year controlling solar declination, photoperiod, and northbound/southbound corridor bearing.");
                        ui.add(
                            egui::Slider::new(&mut self.params.lights_out_curfew_pct, 0.0..=100.0)
                                .text("Lights-Out Curfew (23:00-06:00)")
                                .suffix("%"),
                        )
                        .on_hover_text("Municipal BirdCast Lights-Out dimming percentage during 23:00-06:00 (0.88 curfew factor; 0.32 early-dusk factor).");
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
                                .stroke(Stroke::new(1.2_f32, border_c))
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

                ui.add_space(10.0);
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.small(
                        RichText::new("Copyright CRV Science Network (GPL-3.0-or-later)")
                            .color(Color32::from_rgb(148, 163, 184)),
                    );
                    ui.separator();
                    if ui.small_button("About AeroFlyway").clicked() {
                        self.show_about_modal = true;
                    }
                    ui.hyperlink_to("https://crvscience.com", "https://crvscience.com");
                    ui.hyperlink_to("https://crvscience.com/apps", "https://crvscience.com/apps");
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
                ui.monospace("  0 / R : Reset camera to active corridor default 2.5D view");
                ui.monospace("  1 - 4 : Switch bottom dock tabs (1 Transect, 2 VPTS, 3 WAV, 4 Batch)");
                ui.monospace("  T / M : Toggle Draw A-B Transect (T) or Plant Host Patch (M) tool");
                ui.monospace("  L     : Toggle 85% BirdCast Lights-Out curfew vs 0% unmitigated");
                ui.monospace("  Ctrl+S: Save active campaign metadata to SQLite");
                ui.monospace("  Esc   : Dismiss About/Help modal or reset map tool to Pan & Inspect");
                ui.monospace("  F1    : Show / hide this reference window");
                ui.separator();
                ui.label(RichText::new("Empirical Data & Model Grounding").strong());
                ui.small("- VIIRS Nighttime Lights & Building Collisions: EOG VIIRS DNB radiance (nW/cm^2/sr) coupled to Van Doren et al. (2021, PNAS) McCormick Place window-lighting collision model & BirdCast/Audubon Lights-Out (23:00-06:00).");
                ui.small("- Weather Radar Vertical Profiles (VPTS): Unidata NEXRAD Level-II S3 (KRTX, KMAX, KUEX, KDDC, KEWX, KDIX) & ENRAM Aloft OPERA S3 (nlhrw, bejab, behel) vol2bird VPTS CSV format.");
                ui.small("- Climate & Megafire Smoke: Open-Meteo Forecast, CMIP6 Climate (/v1/climate), CAMS Air Quality (/v1/air-quality), Copernicus GLO-90 DEM (/v1/elevation), and Overton et al. (2022, Ecology) PM2.5 >= 160 ug/m^3 ceiling climbs.");
            });
        self.show_help_modal = open;
    }

    fn render_about_modal(&mut self, ctx: &egui::Context) {
        if !self.show_about_modal {
            return;
        }
        let mut open = self.show_about_modal;
        egui::Window::new("About AeroFlyway")
            .open(&mut open)
            .resizable(false)
            .default_width(500.0)
            .show(ctx, |ui| {
                ui.heading(
                    RichText::new(format!("AeroFlyway v{}", env!("CARGO_PKG_VERSION")))
                        .color(Color32::from_rgb(56, 189, 248)),
                );
                ui.label(
                    RichText::new("Global Aeroecology, VIIRS Lights-Out, NEXRAD/Aloft Radar & Bioacoustic Workstation")
                        .color(Color32::from_rgb(203, 213, 225)),
                );
                ui.add_space(6.0);
                ui.label(RichText::new("Copyright CRV Science Network").strong());
                ui.small(
                    "Licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). \
                     This program comes with ABSOLUTELY NO WARRANTY.",
                );
                ui.separator();
                ui.label(RichText::new("CRV Science Network Resources").strong());
                ui.hyperlink_to("CRV Science Network (https://crvscience.com)", "https://crvscience.com");
                ui.hyperlink_to("CRV Science Apps Hub (https://crvscience.com/apps)", "https://crvscience.com/apps");
                ui.hyperlink_to(
                    "Companion Preprint (APP-PREPRINT-13)",
                    "https://crvscience.com/apps/preprints/aeroflyway-avian-migration-radar",
                );
            });
        self.show_about_modal = open;
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
                        if self.gpu_agent_count >= 1000 {
                            format!("WGSL GPU ({:.1}k)", self.gpu_agent_count as f32 / 1000.0)
                        } else {
                            format!("WGSL GPU ({})", self.gpu_agent_count)
                        }
                    } else {
                        format!("CPU Rayon ({})", self.engine.agents.len())
                    };
                    if ui.selectable_label(self.use_gpu_compute, badge).clicked() {
                        self.use_gpu_compute = !self.use_gpu_compute;
                    }
                }

                ui.separator();
                let is_25d = self.map.is_25d();
                let cam_lbl = if is_25d {
                    format!("2D / 2.5D ({:.0}°)", self.map.pitch_deg)
                } else {
                    "2D / 2.5D (2D)".to_string()
                };
                if ui.selectable_label(is_25d, cam_lbl).clicked() {
                    self.map.toggle_25d();
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
                for (mode, txt, tip) in [
                    (
                        MapToolMode::PanAndInspect,
                        "Pan / Inspect",
                        "Primary-drag to pan, secondary-drag to orbit/tilt 2.5D camera, scroll to zoom (Esc).",
                    ),
                    (
                        MapToolMode::DrawTransect,
                        "Draw A-B Transect (T)",
                        "Primary-drag on map to reposition the A -> B vertical airspace cross-section (T).",
                    ),
                    (
                        MapToolMode::PlantVirtualMilkweed,
                        "Plant Host Patch (M)",
                        "Click on map to insert a virtual Asclepias stepping-stone persisted in SQLite (M).",
                    ),
                ] {
                    if ui
                        .selectable_label(self.tool_mode == mode, txt)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        self.tool_mode = mode;
                        self.transect_drag_anchor = None;
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
        let mut chosen_preset: Option<ScenarioPreset> = None;

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
                            egui::Slider::new(&mut self.gpu_agent_count, 512..=MAX_GPU_AGENTS)
                                .step_by(256.0)
                                .suffix(" agents"),
                        )
                        .on_hover_text("Number of VRAM-resident multi-taxa agents simulated per tick by compute.wgsl and drawn via zero-copy instancing.");
                    }

                    ui.add_space(5.0);
                    ui.separator();
                    ui.heading("2.5D Oblique Camera (Right-Drag Map)");
                    ui.add(
                        egui::Slider::new(&mut self.map.pitch_deg, 0.0..=62.0)
                            .prefix("Pitch ")
                            .suffix(" deg")
                            .fixed_decimals(0),
                    )
                    .on_hover_text("Camera tilt angle (0 deg = 2D Plan, 42-62 deg = 2.5D perspective oblique with ground shadows and drop-lines).");
                    ui.add(
                        egui::Slider::new(&mut self.map.bearing_deg, -180.0..=180.0)
                            .prefix("Bearing ")
                            .suffix(" deg")
                            .fixed_decimals(0),
                    )
                    .on_hover_text("Azimuthal map camera rotation around corridor center (degrees clockwise from North).");
                    ui.add(
                        egui::Slider::new(&mut self.map.alt_exaggeration, 0.0..=2.5)
                            .prefix("Alt Exag ")
                            .suffix("x")
                            .fixed_decimals(2),
                    )
                    .on_hover_text("Vertical exaggeration applied to airborne altitude (m AGL) in 2.5D oblique projection.");
                    ui.horizontal(|ui| {
                        if ui.button("2.5D Oblique").clicked() {
                            self.map.pitch_deg = 42.0;
                            self.map.bearing_deg = -18.0;
                        }
                        if ui.button("2D Plan").clicked() {
                            self.map.pitch_deg = 0.0;
                            self.map.bearing_deg = 0.0;
                        }
                        if ui
                            .button("Reset (0/R)")
                            .on_hover_text("Reset camera to active corridor default viewport & 2.5D angle (hotkey: 0 or R).")
                            .clicked()
                        {
                            self.reset_camera();
                        }
                    });

                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("Scenario Presets");
                    ui.horizontal_wrapped(|ui| {
                        for preset in [
                            ScenarioPreset::AutumnLightsOutPeak,
                            ScenarioPreset::MegafireSmokeSsp585,
                            ScenarioPreset::SpringFalloutUnmitigated,
                        ] {
                            let is_active = self.active_scenario_preset == Some(preset);
                            if ui
                                .selectable_label(is_active, preset.label())
                                .on_hover_text(preset.hover_description())
                                .clicked()
                            {
                                chosen_preset = Some(preset);
                            }
                        }
                    });

                    ui.add_space(6.0);
                    ui.separator();
                    ui.heading("VIIRS Skyglow, Lights-Out & Radar");
                    ui.label("Municipal Lights-Out Curfew (23:00-06:00)");
                    ui.add(
                        egui::Slider::new(&mut self.params.lights_out_curfew_pct, 0.0..=100.0)
                            .suffix("% dimmed"),
                    )
                    .on_hover_text("Fraction of urban building window lighting extinguished during nocturnal migration (23:00-06:00 curfew factor 0.88; early dusk shoulder 0.32).");
                    ui.horizontal(|ui| {
                        if ui.button("Unmitigated (0%)").clicked() {
                            self.params.lights_out_curfew_pct = 0.0;
                            self.status_line = "Lights-Out curfew OFF: 0% dimmed.".to_string();
                        }
                        if ui.button("50% Window Dim").clicked() {
                            self.params.lights_out_curfew_pct = 50.0;
                            self.status_line = "Lights-Out curfew: 50% window dimming applied.".to_string();
                        }
                        if ui.button("BirdCast Alert (85%)").clicked() {
                            self.params.lights_out_curfew_pct = 85.0;
                            self.status_line = "Lights-Out curfew ON: 85% BirdCast alert applied.".to_string();
                        }
                    });
                    ui.label("VIIRS Anthropogenic Skyglow Multiplier");
                    ui.add(
                        egui::Slider::new(&mut self.params.skyglow_intensity_mult, 0.0..=2.5)
                            .suffix("x"),
                    )
                    .on_hover_text("Scalar multiplier (0.0-2.5x) on EOG VIIRS DNB upward urban skyglow radiance (nW/cm^2/sr); amplifies nocturnal phototactic entrainment (<300 m AGL) and building-strike hazard (^1.68 exponent).");
                    ui.label("NEXRAD / Aloft VPTS Altitude Assimilation");
                    ui.add(
                        egui::Slider::new(&mut self.params.radar_nudge_weight, 0.0..=1.0)
                            .fixed_decimals(2),
                    )
                    .on_hover_text("Data assimilation weight (0.00-1.00) nudging avian cruise altitudes toward observed NEXRAD/Aloft VPTS peak layer height and constraining EnKF RMSE.");

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
                        .on_hover_text("Annual Day-of-Year (1-365) driving solar declination, day length, seasonal temperature modulation, and flyway bearing.")
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
                        .on_hover_text("Wildfire smoke PM2.5 concentration offset; triggers behavioral ceiling climb and avoidance above 160 ug/m^3")
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
                    )
                    .on_hover_text("Minimum hub-height wind speed before wind turbine blades begin spinning; higher cut-in speeds reduce nocturnal bat rotor mortality");

                    ui.label("850 hPa Low-Level Jet Multiplier");
                    ui.add(
                        egui::Slider::new(&mut self.params.wind_850_multiplier, 0.2..=2.5)
                            .suffix("x"),
                    )
                    .on_hover_text("Scaling factor (0.2-2.5x) on synoptic 850 hPa low-level jet wind vectors (u850, v850) governing avian groundspeed, drift, and wind-amplified urban strike hazard.");

                    ui.label("Precipitation Fallout Front (mm/hr)");
                    ui.add(
                        egui::Slider::new(&mut self.params.precip_front_mm_hr, 0.0..=5.0)
                            .suffix(" mm/h"),
                    )
                    .on_hover_text("Frontal rainfall intensity forcing nocturnal migration ceiling suppression and coastal/stopover fallout");

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

        if let Some(preset) = chosen_preset {
            self.apply_scenario_preset(preset);
        }
        if let Some(reg) = jump_region {
            if reg != self.params.region {
                self.apply_region_switch(reg);
            }
        }
    }

    fn render_bottom_dock(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("bottom_workbench_dock")
            .resizable(true)
            .default_height(self.bottom_dock_default_height)
            .min_height(185.0)
            .max_height(440.0)
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::AirspaceTransect,
                        "[1] Airspace Cross-Section (0-4,200 m ASL)",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::RadarVerticalProfile,
                        "[2] NEXRAD / Aloft Vertical Profile (VPTS)",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::AcousticSpectrogram,
                        "[3] WAV STFT Bioacoustic Inspector",
                    );
                    ui.selectable_value(
                        &mut self.bottom_tab,
                        BottomDockTab::BatchSweepLab,
                        "[4] Multi-Scenario Batch Sweep Lab",
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

    fn paint_transect_tab(&mut self, ui: &mut egui::Ui) {
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
            Stroke::new(1.0_f32, Color32::from_rgb(38, 52, 73)),
            StrokeKind::Inside,
        );

        let inner = Rect::from_min_max(
            Pos2::new(plot_rect.min.x + 48.0, plot_rect.min.y + 10.0),
            Pos2::new(plot_rect.max.x - 16.0, plot_rect.max.y - 22.0),
        );
        if inner.width() < 40.0 || inner.height() < 30.0 {
            return;
        }
        self.last_transect_size_pts = (inner.width().max(1.0), inner.height().max(1.0));

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
                Stroke::new(1.1_f32, Color32::from_rgba_unmultiplied(34, 211, 238, 150)),
            );
        }

        for elev in [1000.0, 2000.0, 3000.0, 4000.0] {
            let y = map_xz(0.0, elev).y;
            painter.line_segment(
                [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
                Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 90)),
            );
            painter.text(
                Pos2::new(inner.left() - 6.0, y),
                Align2::RIGHT_CENTER,
                format!("{elev:.0}m"),
                FontId::monospace(10.0),
                Color32::from_rgb(148, 163, 184),
            );
        }

        // X-axis distance ticks (0 .. total_km)
        for step in 0..=4 {
            let frac = step as f32 / 4.0;
            let x = inner.left() + frac * inner.width();
            let dist_km = (frac as f64) * total_km;
            if (1..=3).contains(&step) {
                painter.line_segment(
                    [Pos2::new(x, inner.top()), Pos2::new(x, inner.bottom())],
                    Stroke::new(0.7_f32, Color32::from_rgba_unmultiplied(71, 85, 105, 60)),
                );
            }
            painter.line_segment(
                [Pos2::new(x, inner.bottom()), Pos2::new(x, inner.bottom() + 4.0)],
                Stroke::new(1.0_f32, Color32::from_rgb(100, 116, 139)),
            );
            let (align, label) = match step {
                0 => (Align2::LEFT_TOP, "0 km (A)".to_string()),
                4 => (Align2::RIGHT_TOP, format!("{total_km:.0} km (B)")),
                _ => (Align2::CENTER_TOP, format!("{dist_km:.0} km")),
            };
            painter.text(
                Pos2::new(x, inner.bottom() + 5.0),
                align,
                label,
                FontId::monospace(9.5),
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
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 120)),
            ));
        }

        let mut terr_poly = Vec::with_capacity(terrain.len());
        for (d, elev) in &terrain {
            let top_p = map_xz(*d, *elev);
            let bot_p = Pos2::new(top_p.x, inner.bottom());
            painter.line_segment(
                [bot_p, top_p],
                Stroke::new(3.0_f32, Color32::from_rgba_unmultiplied(30, 58, 47, 155)),
            );
            terr_poly.push(top_p);
        }
        painter.add(egui::Shape::line(
            terr_poly,
            Stroke::new(1.8_f32, Color32::from_rgb(74, 222, 128)),
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
            if ui.button("Browse .csv...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open NEXRAD / Aloft VPTS CSV")
                    .add_filter("VPTS Radar CSV", &["csv", "txt"])
                    .pick_file()
                {
                    let p_str = path.display().to_string();
                    self.vpts_csv_path_input = p_str.clone();
                    import_path = Some(p_str);
                }
            }
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
        let (outer, resp) = ui.allocate_exact_size(avail, Sense::hover());
        let painter = ui.painter_at(outer);

        painter.rect_filled(outer, CornerRadius::same(4), Color32::from_rgb(8, 13, 22));
        painter.rect_stroke(
            outer,
            CornerRadius::same(4),
            Stroke::new(1.0_f32, Color32::from_rgb(38, 52, 73)),
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
                Stroke::new(0.7_f32, Color32::from_rgba_unmultiplied(51, 65, 85, 110)),
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
        let row_pick_half_h = ((inner.height() / (active_prof.bins.len().max(1) as f32)) * 0.55)
            .max(bar_h * 0.75)
            .max(6.0);
        let mut hovered_bin: Option<(usize, f32)> = None;
        if let Some(hp) = resp.hover_pos() {
            if hp.x >= inner.left() - 6.0
                && hp.x <= outer.right() - 4.0
                && hp.y >= inner.top() - 8.0
                && hp.y <= inner.bottom() + 8.0
            {
                let mut best_dy = f32::MAX;
                for (b_idx, b) in active_prof.bins.iter().enumerate() {
                    let fy = (b.height_m_asl / max_h).clamp(0.0, 1.0) as f32;
                    let cy = inner.bottom() - fy * inner.height();
                    let dy = (hp.y - cy).abs();
                    if dy <= row_pick_half_h && dy < best_dy {
                        best_dy = dy;
                        hovered_bin = Some((b_idx, cy));
                    }
                }
            }
        }

        for (b_idx, b) in active_prof.bins.iter().enumerate() {
            let fy = (b.height_m_asl / max_h).clamp(0.0, 1.0) as f32;
            let cy = inner.bottom() - fy * inner.height();
            let w_frac = (b.bird_density_per_km3 / max_dens).clamp(0.0, 1.0) as f32;
            let bar_rect = Rect::from_min_max(
                Pos2::new(inner.left(), cy - bar_h * 0.5),
                Pos2::new(inner.left() + w_frac * inner.width(), cy + bar_h * 0.5),
            );
            let is_hovered = hovered_bin.map(|(idx, _)| idx == b_idx).unwrap_or(false);
            if is_hovered {
                let row_rect = Rect::from_min_max(
                    Pos2::new(inner.left(), cy - bar_h * 0.5 - 2.0),
                    Pos2::new(outer.right() - 8.0, cy + bar_h * 0.5 + 2.0),
                );
                painter.rect_filled(
                    row_rect,
                    CornerRadius::same(2),
                    Color32::from_rgba_unmultiplied(34, 211, 238, 28),
                );
            }
            let is_peak = (b.height_m_asl - active_prof.peak_altitude_m_asl).abs() < 110.0;
            let fill = if is_hovered {
                Color32::from_rgb(103, 232, 249)
            } else if is_peak {
                Color32::from_rgb(34, 211, 238)
            } else {
                Color32::from_rgba_unmultiplied(56, 189, 248, 185)
            };
            painter.rect_filled(bar_rect, CornerRadius::same(2), fill);
            if is_hovered {
                painter.rect_stroke(
                    bar_rect.expand(1.0),
                    CornerRadius::same(2),
                    Stroke::new(1.2_f32, Color32::from_rgb(224, 242, 254)),
                    StrokeKind::Outside,
                );
            }

            // Wind vector barb on right column
            let barb_origin = Pos2::new(inner.right() + 48.0, cy);
            let vec_px = Vec2::new((b.u_mps as f32) * 2.2, -(b.v_mps as f32) * 2.2);
            painter.arrow(
                barb_origin,
                vec_px,
                Stroke::new(
                    if is_hovered { 1.8 } else { 1.2_f32 },
                    Color32::from_rgba_unmultiplied(167, 243, 208, if is_hovered { 255 } else { 195 }),
                ),
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

        if let Some((b_idx, _)) = hovered_bin {
            if let Some(b) = active_prof.bins.get(b_idx) {
                let speed_mps = b.u_mps.hypot(b.v_mps);
                let dir_deg = (b.u_mps.atan2(b.v_mps).to_degrees() + 360.0) % 360.0;
                resp.on_hover_text(format!(
                    "{} @ {:.0} m ASL\nDensity: {:.2} birds/km^3\nReflectivity eta: {:.1} cm^2/km^3\nGround speed: {:.1} m/s ({:.0} km/h, track {:.0}°)\nWind (u, v): ({:+.1}, {:+.1}) m/s\nsd_vvp: {:.2} m/s",
                    active_prof.station_code,
                    b.height_m_asl,
                    b.bird_density_per_km3,
                    b.reflectivity_eta,
                    speed_mps,
                    speed_mps * 3.6,
                    dir_deg,
                    b.u_mps,
                    b.v_mps,
                    b.sd_vvp_mps,
                ));
            }
        }
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
                egui::TextEdit::singleline(&mut self.acoustics.wav_path_input).desired_width(135.0),
            );
            if ui.button("Browse .wav...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open Bioacoustic WAV Recording")
                    .add_filter("WAV Audio", &["wav", "WAV"])
                    .pick_file()
                {
                    self.acoustics.wav_path_input = path.display().to_string();
                    match load_wav_file(&path) {
                        Ok(rec) => {
                            self.status_line = format!(
                                "Loaded WAV '{}' ({:.1} kHz, {:.0} ms)",
                                rec.source_label,
                                rec.sample_rate_hz as f64 / 1000.0,
                                rec.duration_ms()
                            );
                            self.acoustics.load_new_recording(rec);
                        }
                        Err(e) => self.status_line = e,
                    }
                }
            }
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

            let coords = self
                .acoustics
                .recording
                .guano_lat_lon
                .or(Some((self.map.center_lat, self.map.center_lon)));

            if ui.button("Save Sample .wav").clicked() {
                if let Some(wav_out) = rfd::FileDialog::new()
                    .set_title("Save 16-Bit GUANO WAV")
                    .add_filter("WAV Audio", &["wav"])
                    .set_file_name(&self.acoustics.recording.source_label)
                    .save_file()
                {
                    match write_wav_16bit(&wav_out, &self.acoustics.recording, coords) {
                        Ok(()) => {
                            self.status_line =
                                format!("Saved 16-bit GUANO WAV -> {}", wav_out.display());
                        }
                        Err(e) => self.status_line = e,
                    }
                }
            }

            if ui.button("Export 10x Audible .wav").clicked() {
                let stem = self
                    .acoustics
                    .recording
                    .source_label
                    .strip_suffix(".wav")
                    .or_else(|| self.acoustics.recording.source_label.strip_suffix(".WAV"))
                    .unwrap_or(&self.acoustics.recording.source_label);
                let default_name = format!("{stem}_10x_audible.wav");
                if let Some(wav_out) = rfd::FileDialog::new()
                    .set_title("Export 10x Time-Expanded Audible WAV")
                    .add_filter("WAV Audio", &["wav"])
                    .set_file_name(&default_name)
                    .save_file()
                {
                    let orig_khz = self.acoustics.recording.sample_rate_hz as f64 / 1000.0;
                    match write_time_expanded_wav_16bit(
                        &wav_out,
                        &self.acoustics.recording,
                        10,
                        coords,
                    ) {
                        Ok(expanded_sr) => {
                            let actual_factor =
                                self.acoustics.recording.sample_rate_hz as f64 / expanded_sr.max(1) as f64;
                            let exp_dur_ms = self.acoustics.recording.duration_ms() * actual_factor;
                            self.status_line = format!(
                                "Exported 10x audible WAV ({orig_khz:.1} -> {:.1} kHz, {exp_dur_ms:.0} ms) -> {}",
                                expanded_sr as f64 / 1000.0,
                                wav_out.display()
                            );
                        }
                        Err(e) => self.status_line = e,
                    }
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

            if ui
                .add(
                    egui::Slider::new(&mut self.acoustics.db_floor, -95.0..=-35.0)
                        .text("dB floor")
                        .fixed_decimals(0),
                )
                .on_hover_text("Minimum spectral power floor (dBFS) mapped to the bottom of the Magma colormap.")
                .changed()
            {
                self.acoustics.texture_dirty = true;
            }

            let nyquist_khz = if self.acoustics.spectrogram.max_freq_khz.is_finite() {
                self.acoustics.spectrogram.max_freq_khz.max(0.5)
            } else {
                0.5
            };
            let min_khz = nyquist_khz.min(2.0);
            self.acoustics.view_max_khz = if self.acoustics.view_max_khz.is_finite() {
                self.acoustics.view_max_khz
            } else {
                nyquist_khz
            }
            .clamp(min_khz, nyquist_khz);
            if ui
                .add(
                    egui::Slider::new(&mut self.acoustics.view_max_khz, min_khz..=nyquist_khz)
                        .text("Max kHz")
                        .fixed_decimals(1),
                )
                .on_hover_text("Upper displayed frequency bound (kHz, up to Nyquist sample_rate / 2).")
                .changed()
            {
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
        let right_w = (avail.x * 0.34).clamp(250.0, 440.0);
        let left_w = (avail.x - right_w - 8.0).max(260.0);
        let row_h = avail.y.max(100.0);

        ui.horizontal_top(|ui| {
            // Left: STFT Spectrogram + Y kHz ticks + X ms ticks + Magma dBFS colorbar + pulse highlight
            let (outer, resp) = ui.allocate_exact_size(Vec2::new(left_w, row_h), Sense::hover());
            let painter = ui.painter_at(outer);

            painter.rect_filled(outer, CornerRadius::same(4), Color32::from_rgb(8, 11, 24));
            painter.rect_stroke(
                outer,
                CornerRadius::same(4),
                Stroke::new(1.0_f32, Color32::from_rgb(45, 58, 82)),
                StrokeKind::Inside,
            );

            let inner = Rect::from_min_max(
                Pos2::new(outer.min.x + 52.0, outer.min.y + 8.0),
                Pos2::new(outer.max.x - 68.0, outer.max.y - 24.0),
            );
            if inner.width() >= 40.0 && inner.height() >= 24.0 {
                let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                painter.image(tex_id, inner, uv, Color32::WHITE);

                let nyquist_khz = if self.acoustics.spectrogram.max_freq_khz.is_finite() {
                    self.acoustics.spectrogram.max_freq_khz.max(0.5)
                } else {
                    0.5
                };
                let min_khz = nyquist_khz.min(2.0);
                let top_khz = if self.acoustics.view_max_khz.is_finite() {
                    self.acoustics.view_max_khz
                } else {
                    nyquist_khz
                }
                .clamp(min_khz, nyquist_khz);
                let dur_ms = self.acoustics.spectrogram.duration_ms.max(1.0);

                // Y-axis frequency ticks (kHz)
                for step in 0..=4 {
                    let frac = step as f32 / 4.0;
                    let y = inner.bottom() - frac * inner.height();
                    let khz = (frac as f64) * top_khz;
                    painter.line_segment(
                        [Pos2::new(inner.left() - 4.0, y), Pos2::new(inner.left(), y)],
                        Stroke::new(1.0_f32, Color32::from_rgb(100, 116, 139)),
                    );
                    painter.text(
                        Pos2::new(inner.left() - 6.0, y),
                        Align2::RIGHT_CENTER,
                        format!("{khz:.1}k"),
                        FontId::monospace(9.5),
                        Color32::from_rgb(203, 213, 225),
                    );
                }

                // X-axis time ticks (ms)
                for step in 0..=4 {
                    let frac = step as f32 / 4.0;
                    let x = inner.left() + frac * inner.width();
                    let t_ms = (frac as f64) * dur_ms;
                    painter.line_segment(
                        [Pos2::new(x, inner.bottom()), Pos2::new(x, inner.bottom() + 4.0)],
                        Stroke::new(1.0_f32, Color32::from_rgb(100, 116, 139)),
                    );
                    let align = match step {
                        0 => Align2::LEFT_TOP,
                        4 => Align2::RIGHT_TOP,
                        _ => Align2::CENTER_TOP,
                    };
                    painter.text(
                        Pos2::new(x, inner.bottom() + 5.0),
                        align,
                        format!("{t_ms:.0} ms"),
                        FontId::monospace(9.5),
                        Color32::from_rgb(203, 213, 225),
                    );
                }

                // Magma dBFS vertical colorbar
                let cbar_rect = Rect::from_min_max(
                    Pos2::new(inner.right() + 10.0, inner.top()),
                    Pos2::new(inner.right() + 24.0, inner.bottom()),
                );
                let steps = 32usize;
                let strip_h = cbar_rect.height() / (steps as f32);
                for s in 0..steps {
                    let t = 1.0 - (s as f32 + 0.5) / (steps as f32);
                    let (r, g, b) = magma_lut(t);
                    let y0 = cbar_rect.top() + (s as f32) * strip_h;
                    let y1 = (y0 + strip_h + 0.5).min(cbar_rect.bottom());
                    painter.rect_filled(
                        Rect::from_min_max(Pos2::new(cbar_rect.left(), y0), Pos2::new(cbar_rect.right(), y1)),
                        CornerRadius::same(0),
                        Color32::from_rgb(r, g, b),
                    );
                }
                painter.rect_stroke(
                    cbar_rect,
                    CornerRadius::same(1),
                    Stroke::new(1.0_f32, Color32::from_rgb(71, 85, 105)),
                    StrokeKind::Inside,
                );
                let db_mid = 0.5 * (SPECTROGRAM_DB_CEIL + self.acoustics.db_floor);
                for (y, lbl) in [
                    (cbar_rect.top(), format!("{:.0} dB", SPECTROGRAM_DB_CEIL)),
                    (cbar_rect.center().y, format!("{db_mid:.0}")),
                    (cbar_rect.bottom(), format!("{:.0} dB", self.acoustics.db_floor)),
                ] {
                    painter.text(
                        Pos2::new(cbar_rect.right() + 4.0, y),
                        Align2::LEFT_CENTER,
                        lbl,
                        FontId::monospace(9.0),
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

                // Pulse bounding boxes & selected pulse highlight
                for (p_idx, pulse) in self.acoustics.spectrogram.pulses.iter().enumerate() {
                    if pulse.f_min_khz > top_khz {
                        continue;
                    }
                    let x0 = inner.left() + ((pulse.start_ms / dur_ms).clamp(0.0, 1.0) as f32) * inner.width();
                    let x1 = inner.left()
                        + (((pulse.start_ms + pulse.duration_ms) / dur_ms).clamp(0.0, 1.0) as f32)
                            * inner.width();
                    let y_top = inner.bottom()
                        - ((pulse.f_max_khz.min(top_khz) / top_khz).clamp(0.0, 1.0) as f32)
                            * inner.height();
                    let y_bot = inner.bottom()
                        - ((pulse.f_min_khz.min(top_khz) / top_khz).clamp(0.0, 1.0) as f32)
                            * inner.height();
                    let p_rect = Rect::from_min_max(
                        Pos2::new(x0.min(x1 - 2.0), y_top.min(y_bot - 4.0)),
                        Pos2::new(x1.max(x0 + 3.0), y_bot.max(y_top + 4.0)),
                    );
                    let is_selected = self.acoustics.selected_pulse_idx == Some(p_idx);
                    if is_selected {
                        painter.rect_filled(
                            p_rect.expand(3.0),
                            CornerRadius::same(2),
                            Color32::from_rgba_unmultiplied(250, 204, 21, 38),
                        );
                        painter.rect_stroke(
                            p_rect.expand(3.0),
                            CornerRadius::same(2),
                            Stroke::new(1.8_f32, Color32::from_rgb(250, 204, 21)),
                            StrokeKind::Outside,
                        );
                        painter.text(
                            Pos2::new(p_rect.center().x, (p_rect.top() - 4.0).max(inner.top() + 10.0)),
                            Align2::CENTER_BOTTOM,
                            format!("P{} Fc={:.1}k", p_idx + 1, pulse.f_char_khz),
                            FontId::monospace(9.5),
                            Color32::from_rgb(254, 240, 138),
                        );
                    } else {
                        painter.rect_stroke(
                            p_rect,
                            CornerRadius::same(1),
                            Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(148, 163, 184, 75)),
                            StrokeKind::Outside,
                        );
                    }
                }

                if let Some(hp) = resp.hover_pos() {
                    if inner.contains(hp) {
                        let nx = (hp.x - inner.left()) / inner.width();
                        let ny = (hp.y - inner.top()) / inner.height();
                        let (t_ms, f_khz, db) =
                            self.acoustics.spectrogram.sample_dbfs_at(nx, ny, top_khz);
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

            // Right: Clickable Call Pulses Table (`spec.pulses`)
            ui.allocate_ui_with_layout(
                Vec2::new(right_w, row_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    let pulses = &self.acoustics.spectrogram.pulses;
                    ui.horizontal(|ui| {
                        let hdr = if pulses.is_empty() {
                            "No pulses above gate".to_string()
                        } else {
                            format!("Detected Call Pulses ({})", pulses.len())
                        };
                        ui.label(RichText::new(hdr).strong().color(Color32::from_rgb(226, 232, 240)));
                        if self.acoustics.selected_pulse_idx.is_some()
                            && ui.small_button("Clear").clicked()
                        {
                            self.acoustics.selected_pulse_idx = None;
                        }
                    });
                    egui::ScrollArea::vertical()
                        .id_salt("acoustic_pulses_scroll")
                        .max_height((row_h - 24.0).max(60.0))
                        .show(ui, |ui| {
                            egui::Grid::new("acoustic_pulses_grid")
                                .num_columns(6)
                                .striped(true)
                                .spacing([8.0, 3.0])
                                .show(ui, |ui| {
                                    for h in ["#", "Start", "Dur", "Fc", "Band", "Peak"] {
                                        ui.small(RichText::new(h).strong().monospace());
                                    }
                                    ui.end_row();
                                    for (idx, p) in pulses.iter().enumerate() {
                                        let sel = self.acoustics.selected_pulse_idx == Some(idx);
                                        if ui
                                            .selectable_label(sel, format!("P{}", idx + 1))
                                            .clicked()
                                        {
                                            self.acoustics.selected_pulse_idx =
                                                if sel { None } else { Some(idx) };
                                        }
                                        ui.small(
                                            RichText::new(format!("{:.1}ms", p.start_ms)).monospace(),
                                        );
                                        ui.small(
                                            RichText::new(format!("{:.1}ms", p.duration_ms))
                                                .monospace(),
                                        );
                                        ui.small(
                                            RichText::new(format!("{:.1}k", p.f_char_khz))
                                                .monospace()
                                                .color(Color32::from_rgb(56, 189, 248)),
                                        );
                                        ui.small(
                                            RichText::new(format!(
                                                "{:.1}-{:.1}",
                                                p.f_min_khz, p.f_max_khz
                                            ))
                                            .monospace(),
                                        );
                                        ui.small(
                                            RichText::new(format!("{:.0}dB", p.peak_dbfs))
                                                .monospace(),
                                        );
                                        ui.end_row();
                                    }
                                });
                        });
                },
            );
        });
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

            ui.label("Hours:");
            ui.add(
                egui::DragValue::new(&mut self.batch_ui.sweep_hours)
                    .range(6.0..=240.0)
                    .speed(1.0)
                    .suffix(" h"),
            )
            .on_hover_text("Simulation duration per sweep scenario (6.0–240.0 h)");

            if ui.button("Output Folder...").clicked() {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Select Batch Sweep Output Folder")
                    .pick_folder()
                {
                    self.export_dir_input = dir.display().to_string();
                    self.status_line = format!("Batch sweep export folder -> {}", self.export_dir_input);
                }
            }

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

            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                self.map.bearing_deg =
                    MapViewport::wrap_bearing(self.map.bearing_deg + delta.x as f64 * 0.35);
                self.map.pitch_deg =
                    (self.map.pitch_deg - delta.y as f64 * 0.28).clamp(0.0, 62.0);
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
                            match self
                                .store
                                .insert_milkweed_patch(&label, lat, lon, 25.0, true, 155.0)
                            {
                                Ok(_) => {
                                    self.milkweed = self.store.load_milkweed_patches();
                                    self.status_line =
                                        format!("Planted virtual stepping-stone '{label}' at ({lat:.4}, {lon:.4})");
                                }
                                Err(err) => {
                                    self.status_line =
                                        format!("Failed saving virtual stepping-stone to SQLite: {err}");
                                }
                            }
                        }
                    }
                }
            }

            let tiles = self.map.visible_world_tiles(ctx, canvas_rect);
            let base_rgba = match self.map.provider {
                BasemapProvider::OpenStreetMap | BasemapProvider::UsgsNationalTopo => {
                    [185_u8, 198, 215, 210]
                }
                _ => [255_u8, 255, 255, 255],
            };
            for tile in &tiles {
                let mesh = build_tilted_tile_mesh_4x4(&self.map, tile, base_rgba, canvas_rect);
                painter.add(egui::Shape::mesh(mesh));
            }

            self.paint_vector_basemap(&painter, canvas_rect);

            if self.layers.hex_heatmap {
                self.paint_hex_choropleth(&painter, canvas_rect);
            }

            let mut label_candidates: Vec<MapLabelCandidate> = Vec::new();

            if self.layers.skyglow_domes {
                self.paint_skyglow_domes(&painter, canvas_rect, &mut label_candidates);
            }

            if self.layers.radar_rings {
                self.paint_radar_stations(&painter, canvas_rect, &mut label_candidates);
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
                    painter.circle_stroke(sp, rad, Stroke::new(1.3_f32, Color32::from_rgb(52, 211, 153)));
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
                        Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(244, 63, 94, 195)),
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
                            Stroke::new(1.6_f32, Color32::from_rgb(254, 240, 138)),
                        );
                    }
                    let r_rect = Rect::from_center_size(sp, Vec2::splat(5.5));
                    painter.rect_stroke(
                        r_rect,
                        CornerRadius::same(1),
                        Stroke::new(1.2_f32, col),
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
                    self.map.pitch_deg,
                    self.map.bearing_deg,
                    self.map.alt_exaggeration,
                    canvas_rect.width(),
                    canvas_rect.height(),
                    self.last_transect_size_pts.0,
                    self.last_transect_size_pts.1,
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
                let heading_rad = aero_heading_rad(&self.params, self.engine.metrics.day_of_year);
                let cb = self.map.bearing_deg.to_radians() as f32;
                let sb = cb.sin();
                let cb = cb.cos();
                let cp = (self.map.pitch_deg.clamp(0.0, 62.0).to_radians() as f32).cos();
                for agent in &self.engine.agents {
                    let enabled = match agent.taxon {
                        Taxon::Bird => self.layers.birds,
                        Taxon::Bat => self.layers.bats,
                        Taxon::Butterfly => self.layers.butterflies,
                    };
                    if !enabled {
                        continue;
                    }
                    let (ground_sp, _) =
                        self.map
                            .project_ground_with_scale(agent.lat, agent.lon, 0.0, canvas_rect);
                    let (sp, persp) = self.map.project_ground_with_scale(
                        agent.lat,
                        agent.lon,
                        agent.alt_agl_m,
                        canvas_rect,
                    );
                    if !canvas_rect.expand(36.0).contains(sp) {
                        continue;
                    }
                    if self.map.pitch_deg > 2.0 && (sp.y - ground_sp.y).abs() > 3.0 {
                        let stem_alpha = if agent.is_roosting { 18 } else { 42 };
                        painter.line_segment(
                            [ground_sp, sp],
                            Stroke::new(0.8_f32, Color32::from_rgba_unmultiplied(148, 163, 184, stem_alpha)),
                        );
                        painter.circle_filled(
                            ground_sp,
                            1.4,
                            Color32::from_rgba_unmultiplied(15, 23, 42, 90),
                        );
                    }

                    let spd = agent.vel_u_mps.hypot(agent.vel_v_mps);
                    let (vu, vv) = if spd > 0.15 {
                        ((agent.vel_u_mps / spd) as f32, (agent.vel_v_mps / spd) as f32)
                    } else {
                        (heading_rad.sin() as f32, heading_rad.cos() as f32)
                    };
                    let rot_u = vu * cb - vv * sb;
                    let rot_v = (vu * sb + vv * cb) * cp;
                    let fwd = Vec2::new(rot_u, -rot_v).normalized();
                    let right = Vec2::new(-fwd.y, fwd.x);

                    let alpha = if agent.is_roosting { 56 } else { 209 }; // 0.22 vs 0.82
                    let fill_col = if agent.taxon == Taxon::Bird && agent.skyglow_entrained {
                        Color32::from_rgba_unmultiplied(244, 114, 182, alpha)
                    } else if agent.smoke_disrupted && agent.taxon == Taxon::Bird {
                        Color32::from_rgba_unmultiplied(249, 115, 22, alpha)
                    } else {
                        taxon_color(agent.taxon, alpha)
                    };
                    let size_pt = ((3.4 + (self.map.zoom as f32 - 3.5).max(0.0) * 0.36).clamp(3.3, 5.2))
                        * persp.clamp(0.78, 1.25)
                        * if agent.is_roosting { 0.8 } else { 1.0 };
                    paint_taxon_silhouette(&painter, agent.taxon, sp, fwd, right, size_pt, fill_col);
                }
            }

            if self.layers.landmarks {
                self.paint_landmarks(&painter, canvas_rect, &mut label_candidates);
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
            painter.line_segment([pa, pb], Stroke::new(2.0_f32, Color32::from_rgb(52, 211, 153)));
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

            let hud_rect = self.paint_map_hud(&painter, canvas_rect, response.hover_pos());
            let legend_rect = self.paint_taxon_legend_pill(&painter, canvas_rect);
            let transect_a_rect = Rect::from_min_max(pa + Vec2::new(4.0, -24.0), pa + Vec2::new(24.0, -4.0));
            let transect_b_rect = Rect::from_min_max(pb + Vec2::new(4.0, -24.0), pb + Vec2::new(24.0, -4.0));
            self.paint_decluttered_map_labels(
                &painter,
                canvas_rect,
                label_candidates,
                &[hud_rect, legend_rect, transect_a_rect, transect_b_rect],
                response.hover_pos(),
            );
        });
    }

    fn paint_skyglow_domes(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        label_candidates: &mut Vec<MapLabelCandidate>,
    ) {
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
                Stroke::new(1.1_f32, Color32::from_rgba_unmultiplied(251, 191, 36, 130)),
            );
            if self.map.zoom >= 5.2 && canvas_rect.contains(sp) {
                label_candidates.push(MapLabelCandidate {
                    anchor: sp,
                    marker_radius_px: (px_r * 0.35).clamp(6.0, 18.0),
                    priority: 1,
                    text: format!("[VIIRS {:.0} nW] {}", eff_nw, dome.city_name),
                    font: FontId::monospace(9.5),
                    color: Color32::from_rgb(251, 207, 232),
                });
            }
        }
    }

    fn paint_radar_stations(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        label_candidates: &mut Vec<MapLabelCandidate>,
    ) {
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
                Stroke::new(1.1_f32, Color32::from_rgba_unmultiplied(34, 211, 238, 125)),
            );
            painter.circle_filled(sp, 3.8, Color32::from_rgb(34, 211, 238));
            if canvas_rect.contains(sp) {
                label_candidates.push(MapLabelCandidate {
                    anchor: sp,
                    marker_radius_px: 5.0,
                    priority: 0,
                    text: format!("[RADAR {}] {:.0} MTR", prof.station_code, prof.mtr_birds_km_hr),
                    font: FontId::monospace(10.0),
                    color: Color32::from_rgb(103, 232, 249),
                });
            }
        }
    }

    fn paint_vector_basemap(&self, painter: &egui::Painter, canvas_rect: Rect) {
        let graticule_stroke = Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(51, 65, 85, 95));
        let (min_lat_b, max_lat_b, min_lon_b, max_lon_b) = self.params.region.corridor_bounds();
        let lat_lo = ((min_lat_b - 18.0).max(-64.0) / 4.0).floor() as i32 * 4;
        let lat_hi = ((max_lat_b + 18.0).min(76.0) / 4.0).ceil() as i32 * 4;
        let lon_lo = ((min_lon_b - 24.0).max(-176.0) / 4.0).floor() as i32 * 4;
        let lon_hi = ((max_lon_b + 24.0).min(176.0) / 4.0).ceil() as i32 * 4;

        for lat_i in (lat_lo..=lat_hi).step_by(4) {
            let pts: Vec<Pos2> = (lon_lo..=lon_hi)
                .step_by(2)
                .map(|ln| self.map.lat_lon_to_screen(lat_i as f64, ln as f64, canvas_rect))
                .collect();
            painter.add(egui::Shape::line(pts, graticule_stroke));
        }
        for lon_i in (lon_lo..=lon_hi).step_by(4) {
            let pts: Vec<Pos2> = (lat_lo..=lat_hi)
                .step_by(2)
                .map(|lt| self.map.lat_lon_to_screen(lt as f64, lon_i as f64, canvas_rect))
                .collect();
            painter.add(egui::Shape::line(pts, graticule_stroke));
        }

        let basemap = vector_basemap_for_region(self.params.region);
        let draw_poly = |coords: &[(f64, f64)], stroke: Stroke| {
            let pts: Vec<Pos2> = coords
                .iter()
                .map(|(lt, ln)| self.map.lat_lon_to_screen(*lt, *ln, canvas_rect))
                .collect();
            painter.add(egui::Shape::line(pts, stroke));
        };

        for border in basemap.borders {
            draw_poly(
                border,
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(100, 116, 139, 135)),
            );
        }
        for (idx, coastline) in basemap.coastlines.iter().enumerate() {
            let width = if idx == 0 { 2.0 } else { 1.5_f32 };
            let alpha = if idx == 0 { 220 } else { 185 };
            draw_poly(
                coastline,
                Stroke::new(width, Color32::from_rgba_unmultiplied(71, 135, 179, alpha)),
            );
        }
        for (idx, river) in basemap.rivers.iter().enumerate() {
            let (width, col) = if idx == 0 {
                (1.8, Color32::from_rgba_unmultiplied(34, 211, 238, 165))
            } else if idx == 1 {
                (1.6, Color32::from_rgba_unmultiplied(56, 189, 248, 175))
            } else {
                (1.2_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 125))
            };
            draw_poly(river, Stroke::new(width, col));
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
                Stroke::new(0.7_f32, Color32::from_rgba_unmultiplied(56, 189, 248, 65)),
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
                Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(148, 163, 184, 190)),
            );
        }
    }

    fn paint_landmarks(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        label_candidates: &mut Vec<MapLabelCandidate>,
    ) {
        for lm in landmarks_for_region(self.params.region) {
            let sp = self.map.lat_lon_to_screen(lm.lat, lm.lon, canvas_rect);
            if !canvas_rect.contains(sp) {
                continue;
            }
            let (col, prefix, prio) = match lm.kind {
                LandmarkKind::WildlifeRefuge => (Color32::from_rgb(74, 222, 128), "NWR", 2),
                LandmarkKind::MotusStation => (Color32::from_rgb(34, 211, 238), "MOTUS", 2),
                LandmarkKind::MonarchSanctuary => (Color32::from_rgb(251, 191, 36), "ROOST", 2),
                LandmarkKind::CascadePeak => (Color32::from_rgb(203, 213, 225), "PEAK", 3),
            };
            painter.circle_stroke(sp, 4.5, Stroke::new(1.4_f32, col));
            if self.map.zoom >= 5.2 {
                label_candidates.push(MapLabelCandidate {
                    anchor: sp,
                    marker_radius_px: 5.5,
                    priority: prio,
                    text: format!("[{prefix}] {}", lm.name),
                    font: FontId::proportional(11.0),
                    color: col,
                });
            }
        }
    }

    fn paint_decluttered_map_labels(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        candidates: Vec<MapLabelCandidate>,
        reserved_rects: &[Rect],
        hover_pos: Option<Pos2>,
    ) {
        let mut placer = MapLabelPlacer::new(canvas_rect);
        for &r in reserved_rects {
            placer.reserve_rect(r);
        }

        let mut indexed: Vec<(usize, MapLabelCandidate)> =
            candidates.into_iter().enumerate().collect();
        indexed.sort_by_key(|(idx, c)| (c.priority, *idx));

        let mut skipped_for_hover: Vec<(Pos2, std::sync::Arc<egui::Galley>, Color32, Vec2)> =
            Vec::new();

        for (_, cand) in indexed {
            let galley = painter.layout_no_wrap(cand.text, cand.font, cand.color);
            let box_size = galley.size() + Vec2::new(10.0, 4.0);
            if let Some(pill_rect) =
                placer.try_place(cand.anchor, cand.marker_radius_px, box_size)
            {
                let delta = pill_rect.center() - cand.anchor;
                if delta.length() > 12.0 {
                    let edge_pt = Pos2::new(
                        cand.anchor.x.clamp(pill_rect.left(), pill_rect.right()),
                        cand.anchor.y.clamp(pill_rect.top(), pill_rect.bottom()),
                    );
                    painter.line_segment(
                        [cand.anchor, edge_pt],
                        Stroke::new(
                            0.8_f32,
                            Color32::from_rgba_unmultiplied(
                                cand.color.r(),
                                cand.color.g(),
                                cand.color.b(),
                                135,
                            ),
                        ),
                    );
                }
                painter.rect_filled(
                    pill_rect,
                    CornerRadius::same(4),
                    Color32::from_rgba_unmultiplied(11, 17, 27, 205),
                );
                painter.rect_stroke(
                    pill_rect,
                    CornerRadius::same(4),
                    Stroke::new(
                        0.8_f32,
                        Color32::from_rgba_unmultiplied(
                            cand.color.r(),
                            cand.color.g(),
                            cand.color.b(),
                            110,
                        ),
                    ),
                    StrokeKind::Inside,
                );
                painter.galley(pill_rect.min + Vec2::new(5.0, 2.0), galley, cand.color);
            } else {
                skipped_for_hover.push((cand.anchor, galley, cand.color, box_size));
            }
        }

        if let Some(hp) = hover_pos {
            for (anchor, galley, color, box_size) in skipped_for_hover {
                if anchor.distance(hp) <= 12.0 {
                    let pill_pos = Pos2::new(
                        (anchor.x + 10.0).min(canvas_rect.right() - box_size.x - 6.0),
                        (anchor.y - box_size.y - 6.0).max(canvas_rect.top() + 6.0),
                    );
                    let pill_rect = Rect::from_min_size(pill_pos, box_size);
                    painter.rect_filled(
                        pill_rect,
                        CornerRadius::same(4),
                        Color32::from_rgba_unmultiplied(15, 23, 42, 240),
                    );
                    painter.rect_stroke(
                        pill_rect,
                        CornerRadius::same(4),
                        Stroke::new(1.0_f32, color),
                        StrokeKind::Inside,
                    );
                    painter.galley(pill_rect.min + Vec2::new(5.0, 2.0), galley, color);
                    break;
                }
            }
        }
    }

    fn paint_map_hud(
        &self,
        painter: &egui::Painter,
        canvas_rect: Rect,
        hover_pos: Option<Pos2>,
    ) -> Rect {
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
            Stroke::new(1.0_f32, Color32::from_rgb(51, 65, 85)),
            StrokeKind::Inside,
        );

        let bar_y = hud_rect.top() + 14.0;
        let bar_x0 = hud_rect.left() + 12.0;
        painter.line_segment(
            [Pos2::new(bar_x0, bar_y), Pos2::new(bar_x0 + bar_px, bar_y)],
            Stroke::new(2.2_f32, Color32::from_rgb(226, 232, 240)),
        );
        painter.text(
            Pos2::new(bar_x0 + bar_px + 8.0, bar_y),
            Align2::LEFT_CENTER,
            format!(
                "{target_km:.0} km (z={:.1} | P={:.0}° B={:.0}°)",
                self.map.zoom, self.map.pitch_deg, self.map.bearing_deg
            ),
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
        hud_rect
    }

    fn paint_taxon_legend_pill(&self, painter: &egui::Painter, canvas_rect: Rect) -> Rect {
        let pill_w = 248.0_f32;
        let pill_h = 74.0_f32;
        let pill_rect = Rect::from_min_size(
            Pos2::new(
                canvas_rect.right() - pill_w - 12.0,
                canvas_rect.bottom() - pill_h - 12.0,
            ),
            Vec2::new(pill_w, pill_h),
        );
        painter.rect_filled(
            pill_rect,
            CornerRadius::same(7),
            Color32::from_rgba_unmultiplied(15, 23, 42, 220),
        );
        painter.rect_stroke(
            pill_rect,
            CornerRadius::same(7),
            Stroke::new(1.0_f32, Color32::from_rgb(51, 65, 85)),
            StrokeKind::Inside,
        );

        let rows = [
            (
                Taxon::Bird,
                Color32::from_rgb(56, 189, 248),
                "Avian Streams (800-3500 m AGL)",
            ),
            (
                Taxon::Bat,
                Color32::from_rgb(192, 132, 252),
                "Chiroptera Rotor Zone (40-300 m)",
            ),
            (
                Taxon::Butterfly,
                Color32::from_rgb(251, 146, 60),
                "Lepidoptera Thermal (5-120 m)",
            ),
        ];
        for (idx, (taxon, col, label)) in rows.iter().enumerate() {
            let cy = pill_rect.top() + 15.0 + (idx as f32) * 21.0;
            let icon_center = Pos2::new(pill_rect.left() + 18.0, cy);
            paint_taxon_silhouette(
                painter,
                *taxon,
                icon_center,
                Vec2::new(0.0, -1.0),
                Vec2::new(1.0, 0.0),
                6.2,
                *col,
            );
            painter.text(
                Pos2::new(pill_rect.left() + 31.0, cy),
                Align2::LEFT_CENTER,
                *label,
                FontId::proportional(11.0),
                Color32::from_rgb(226, 232, 240),
            );
        }
        pill_rect
    }
}

struct MapLabelCandidate {
    anchor: Pos2,
    marker_radius_px: f32,
    priority: u8,
    text: String,
    font: FontId,
    color: Color32,
}

impl AeroFlywayApp {
    fn render_ui(&mut self, ctx: &egui::Context) {
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
        self.render_about_modal(ctx);

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

impl eframe::App for AeroFlywayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_ui(ctx);
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
        Taxon::Butterfly => Color32::from_rgba_unmultiplied(251, 146, 60, alpha),
    }
}

fn aero_heading_rad(params: &SimulationParams, active_doy: f64) -> f64 {
    aeroflyway::physics::annual_flyway_heading_deg(params.region, active_doy).to_radians()
}

/// Computes atmospheric horizon fade factor (`0.14..=1.0`) for 2.5D tilted basemap vertices.
/// Returns `1.0` in 2D plan view (`pitch_deg <= 1.0`) and gently attenuates far-horizon tile
/// vertices (`persp < 0.78` or near top viewport edge) into the dark background sky.
fn compute_tile_horizon_fade(
    pitch_deg: f64,
    persp: f32,
    screen_y: f32,
    canvas_rect: Rect,
) -> f32 {
    if pitch_deg <= 1.0 {
        return 1.0;
    }
    let pitch_t = ((pitch_deg as f32 - 1.0) / 61.0).clamp(0.0, 1.0);
    let persp_fade = ((persp - 0.34) / 0.44).clamp(0.0, 1.0);
    let top_norm = ((screen_y - canvas_rect.top()) / (canvas_rect.height().max(1.0) * 0.28))
        .clamp(0.0, 1.0);
    let horizon_atten = persp_fade.min(0.30 + 0.70 * top_norm);
    (1.0 - pitch_t * (1.0 - horizon_atten)).clamp(0.14, 1.0)
}

/// Subdivides a projected Web Mercator raster basemap tile into a `4x4` grid (`32` triangles)
/// with per-vertex atmospheric horizon fade so oblique 2.5D perspective distortion stays smooth.
fn build_tilted_tile_mesh_4x4(
    map: &MapViewport,
    tile: &aeroflyway::map_engine::ProjectedTile,
    base_rgba: [u8; 4],
    canvas_rect: Rect,
) -> egui::epaint::Mesh {
    const SUBDIV: u32 = 4;
    let stride = SUBDIV + 1;
    let step = 1.0_f32 / (SUBDIV as f32);
    let mut mesh = egui::epaint::Mesh::with_texture(tile.texture.id());
    mesh.vertices.reserve((stride * stride) as usize);

    for iy in 0..=SUBDIV {
        let v = (iy as f32) * step;
        let wy = tile.min_wy + (tile.max_wy - tile.min_wy) * (v as f64);
        for ix in 0..=SUBDIV {
            let u = (ix as f32) * step;
            let wx = tile.min_wx + (tile.max_wx - tile.min_wx) * (u as f64);
            let (sp, persp) = map.project_world_px(wx, wy, 0.0, canvas_rect);
            let fade = compute_tile_horizon_fade(map.pitch_deg, persp, sp.y, canvas_rect);
            let alpha = ((base_rgba[3] as f32) * fade).round().clamp(0.0, 255.0) as u8;
            let color =
                Color32::from_rgba_unmultiplied(base_rgba[0], base_rgba[1], base_rgba[2], alpha);
            mesh.vertices.push(egui::epaint::Vertex {
                pos: sp,
                uv: Pos2::new(u, v),
                color,
            });
        }
    }
    for iy in 0..SUBDIV {
        for ix in 0..SUBDIV {
            let i0 = iy * stride + ix;
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;
            mesh.add_triangle(i0, i1, i3);
            mesh.add_triangle(i0, i3, i2);
        }
    }
    mesh
}

fn paint_taxon_silhouette(
    painter: &egui::Painter,
    taxon: Taxon,
    center: Pos2,
    fwd: Vec2,
    right: Vec2,
    half_px: f32,
    color: Color32,
) {
    let pt = |u: f32, v: f32| -> Pos2 { center + right * (u * half_px) + fwd * (v * half_px) };
    match taxon {
        Taxon::Bird => {
            // Swept chevron wing + forked swallow tail
            let pts = vec![
                pt(0.0, 0.92),
                pt(0.20, 0.25),
                pt(0.95, -0.35),
                pt(0.55, -0.45),
                pt(0.14, -0.15),
                pt(0.28, -0.88),
                pt(0.0, -0.52),
                pt(-0.28, -0.88),
                pt(-0.14, -0.15),
                pt(-0.55, -0.45),
                pt(-0.95, -0.35),
                pt(-0.20, 0.25),
            ];
            painter.add(egui::Shape::convex_polygon(
                vec![pt(0.0, 0.92), pt(0.95, -0.35), pt(0.0, -0.18), pt(-0.95, -0.35)],
                color,
                Stroke::NONE,
            ));
            painter.add(egui::Shape::line(
                pts,
                Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), color.a())),
            ));
        }
        Taxon::Bat => {
            // Twin ears + 3 scalloped trailing cutouts
            let wing_pts = vec![
                pt(0.0, 0.42),
                pt(0.16, 0.78),
                pt(0.26, 0.34),
                pt(0.94, 0.14),
                pt(0.72, -0.30),
                pt(0.52, -0.12),
                pt(0.32, -0.34),
                pt(0.14, -0.14),
                pt(0.0, -0.48),
                pt(-0.14, -0.14),
                pt(-0.32, -0.34),
                pt(-0.52, -0.12),
                pt(-0.72, -0.30),
                pt(-0.94, 0.14),
                pt(-0.26, 0.34),
                pt(-0.16, 0.78),
                pt(0.0, 0.42),
            ];
            painter.add(egui::Shape::convex_polygon(
                vec![pt(0.0, 0.50), pt(0.92, 0.12), pt(0.0, -0.42), pt(-0.92, 0.12)],
                color,
                Stroke::NONE,
            ));
            painter.add(egui::Shape::line(wing_pts, Stroke::new(1.1_f32, color)));
        }
        Taxon::Butterfly => {
            // 4-lobed forewing/hindwing butterfly + thorax
            let fore_l = vec![
                pt(0.06, 0.18),
                pt(0.45, 0.72),
                pt(0.92, 0.46),
                pt(0.84, 0.06),
                pt(0.12, -0.02),
            ];
            let fore_r = vec![
                pt(-0.06, 0.18),
                pt(-0.12, -0.02),
                pt(-0.84, 0.06),
                pt(-0.92, 0.46),
                pt(-0.45, 0.72),
            ];
            let hind_l = vec![
                pt(0.08, -0.04),
                pt(0.70, -0.14),
                pt(0.56, -0.68),
                pt(0.14, -0.58),
            ];
            let hind_r = vec![
                pt(-0.08, -0.04),
                pt(-0.14, -0.58),
                pt(-0.56, -0.68),
                pt(-0.70, -0.14),
            ];
            for lobe in [fore_l, fore_r, hind_l, hind_r] {
                painter.add(egui::Shape::convex_polygon(lobe, color, Stroke::NONE));
            }
            painter.line_segment([pt(0.0, -0.64), pt(0.0, 0.62)], Stroke::new(1.4_f32, color));
        }
    }
}

fn configure_dark_telemetry_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(13, 19, 30);
    visuals.window_fill = Color32::from_rgb(15, 23, 42);
    visuals.extreme_bg_color = Color32::from_rgb(8, 12, 20);
    ctx.set_visuals(visuals);
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    struct ThreadUnparker(std::thread::Thread);
    impl std::task::Wake for ThreadUnparker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(ThreadUnparker(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

fn run_headless_screenshot(out_path: &std::path::Path, view: u8) -> Result<(), String> {
    use eframe::egui_wgpu::wgpu;

    const WIDTH: u32 = 1600;
    const HEIGHT: u32 = 1000;
    const BYTES_PER_ROW: u32 = WIDTH * 4;

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .or_else(|| {
        block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
    })
    .ok_or_else(|| "No wgpu adapter found for offscreen screenshot.".to_string())?;

    let (device, queue) =
        block_on_future(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .map_err(|e| format!("Failed to request wgpu device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);
    let render_state = egui_wgpu::RenderState {
        adapter,
        available_adapters: Vec::new(),
        device: device.clone(),
        queue: queue.clone(),
        target_format: format,
        renderer: Arc::new(egui::mutex::RwLock::new(renderer)),
    };

    let temp_db =
        std::env::temp_dir().join(format!("aeroflyway_headless_{}.db", std::process::id()));
    let _ = std::fs::remove_file(&temp_db);

    let ctx = egui::Context::default();
    let mut app = AeroFlywayApp::new_with_wgpu(&ctx, Some(&render_state), temp_db.clone(), true);
    app.campaign_db_path_input = "aeroflyway.db".to_string();
    app.gpu_agent_count = DEFAULT_GPU_AGENTS;
    app.sim_speed_hr_per_sec = 6.0;
    app.map.zoom = 5.95;
    app.engine.step(
        3.0,
        &app.params,
        &app.turbines,
        &app.milkweed,
        &app.observations,
    );
    match view {
        2 => {
            app.bottom_tab = BottomDockTab::RadarVerticalProfile;
            app.bottom_dock_default_height = 375.0;
            app.selected_radar_idx = 0;
            app.layers.wind_barbs = true;
            app.layers.radar_rings = true;
            if let Some(prof) = app.engine.radar_profiles.first() {
                app.map.jump_to(prof.lat, prof.lon, 6.75);
            }
        }
        3 => {
            app.bottom_tab = BottomDockTab::AcousticSpectrogram;
            app.bottom_dock_default_height = 395.0;
            let rec = synthesize_preset_recording(SynthPreset::MyotisBuzzUltrasonic250k);
            app.acoustics.fft_size = 512;
            app.acoustics.load_new_recording(rec);
            app.acoustics.show_ridge_overlay = true;
            app.pin_current_acoustic_detection();
            app.map.zoom = 6.15;
        }
        _ => {}
    }

    let extent = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let render_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("aeroflyway_headless_target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let render_view = render_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aeroflyway_headless_readback"),
        size: (BYTES_PER_ROW as u64) * (HEIGHT as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    for frame_idx in 0..4_u32 {
        let raw_input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new(WIDTH as f32, HEIGHT as f32),
            )),
            predicted_dt: 0.08,
            time: Some((frame_idx as f64) * 0.08),
            ..Default::default()
        };

        let full_output = ctx.run(raw_input, |ctx| {
            app.render_ui(ctx);
        });
        let paint_jobs = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [WIDTH, HEIGHT],
            pixels_per_point: full_output.pixels_per_point,
        };

        let mut renderer_guard = render_state.renderer.write();
        for (id, delta) in &full_output.textures_delta.set {
            renderer_guard.update_texture(&device, &queue, *id, delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aeroflyway_headless_encoder"),
        });
        let user_cmds = renderer_guard.update_buffers(
            &device,
            &queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("aeroflyway_headless_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 8.0 / 255.0,
                            g: 13.0 / 255.0,
                            b: 21.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut rpass = rpass.forget_lifetime();
            renderer_guard.render(&mut rpass, &paint_jobs, &screen_descriptor);
        }

        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &render_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(BYTES_PER_ROW),
                        rows_per_image: Some(HEIGHT),
                    },
                },
                extent,
            );
        }

        queue.submit(user_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);

        for id in &full_output.textures_delta.free {
            renderer_guard.free_texture(id);
        }
    }

    let slice = staging_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("Readback channel failed: {e}"))?
        .map_err(|e| format!("Buffer map failed: {e}"))?;

    let rgba = slice.get_mapped_range().to_vec();
    let _ = slice;
    staging_buf.unmap();

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {e}"))?;
        }
    }
    {
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        use image::ImageEncoder;
        let file = std::fs::File::create(out_path)
            .map_err(|e| format!("Failed to create PNG screenshot: {e}"))?;
        let encoder = PngEncoder::new_with_quality(
            std::io::BufWriter::new(file),
            CompressionType::Best,
            FilterType::Adaptive,
        );
        encoder
            .write_image(&rgba, WIDTH, HEIGHT, image::ExtendedColorType::Rgba8)
            .map_err(|e| format!("Failed to write PNG screenshot: {e}"))?;
    }

    let _ = std::fs::remove_file(&temp_db);
    let _ = std::fs::remove_file(format!("{}-wal", temp_db.display()));
    let _ = std::fs::remove_file(format!("{}-shm", temp_db.display()));

    Ok(())
}

#[cfg(windows)]
fn attach_parent_console_if_cli() {
    if std::env::args_os().len() > 1 {
        // SAFETY: Win32 AttachConsole(ATTACH_PARENT_PROCESS) binds stdout/stderr to an
        // existing parent terminal when CLI arguments are present and is a no-op otherwise.
        unsafe {
            let _ = windows_sys::Win32::System::Console::AttachConsole(
                windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS,
            );
        }
    }
}

#[cfg(not(windows))]
fn attach_parent_console_if_cli() {}

fn load_embedded_app_icon() -> Option<Arc<egui::IconData>> {
    const ICON_PNG: &[u8] = include_bytes!("../assets/aeroflyway_icon_64.png");
    let img = image::load_from_memory(ICON_PNG).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(Arc::new(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }))
}

fn main() -> eframe::Result<()> {
    attach_parent_console_if_cli();

    let skip_hub = match try_dispatch_cli() {
        Ok(CliDispatchOutcome::HandledExit) => return Ok(()),
        Ok(CliDispatchOutcome::Screenshot { out_path, view }) => {
            if let Err(err) = run_headless_screenshot(&out_path, view) {
                eprintln!("[AeroFlyway Screenshot Error] {err}");
                std::process::exit(1);
            }
            println!(
                "Saved {} (1600x1000 RGBA offscreen screenshot, view {view})",
                out_path.display()
            );
            return Ok(());
        }
        Ok(CliDispatchOutcome::LaunchGui { skip_startup_hub }) => skip_startup_hub,
        Err(err) => {
            eprintln!("[AeroFlyway CLI Error] {err}");
            std::process::exit(2);
        }
    };

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1580.0, 980.0])
        .with_min_inner_size([1040.0, 700.0])
        .with_drag_and_drop(true)
        .with_title(
            "AeroFlyway v0.6 — Startup Hub, VIIRS Lights-Out, NEXRAD/Aloft VPTS & WAV STFT",
        );
    if let Some(icon) = load_embedded_app_icon() {
        viewport = viewport.with_icon(icon);
    }

    let native_options = eframe::NativeOptions {
        viewport,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("aeroflyway_gui_test_{tag}_{}.db", std::process::id()))
    }

    fn cleanup_db(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    fn step_app_with_input(
        app: &mut AeroFlywayApp,
        ctx: &egui::Context,
        modifiers: egui::Modifiers,
        events: Vec<egui::Event>,
        dropped_files: Vec<egui::DroppedFile>,
    ) {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                Vec2::new(1280.0, 800.0),
            )),
            modifiers,
            events,
            dropped_files,
            ..Default::default()
        };
        let _ = ctx.run(raw_input, |c| {
            app.render_ui(c);
        });
    }

    #[test]
    fn dropped_invalid_csv_reports_vpts_error_without_wav_fallthrough() {
        let db_path = temp_db("csv_drop");
        cleanup_db(&db_path);
        let bad_csv = std::env::temp_dir().join(format!(
            "aeroflyway_bad_vpts_{}.csv",
            std::process::id()
        ));
        std::fs::write(&bad_csv, "not,a,valid,vpts,header\n1,2,3,4,5\n").expect("write temp csv");

        let ctx = egui::Context::default();
        let mut app = AeroFlywayApp::new_with_wgpu(&ctx, None, db_path.clone(), true);
        app.running = false;

        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            Vec::new(),
            vec![egui::DroppedFile {
                path: Some(bad_csv.clone()),
                ..Default::default()
            }],
        );

        assert!(
            app.status_line.starts_with("VPTS CSV load error:"),
            "expected VPTS CSV error, got: {}",
            app.status_line
        );
        assert!(!app.status_line.contains("WAV"));

        let _ = std::fs::remove_file(&bad_csv);
        drop(app);
        cleanup_db(&db_path);
    }

    #[test]
    fn headless_escape_layering_and_ctrl_s_save_campaign() {
        let db_path = temp_db("shortcuts");
        cleanup_db(&db_path);

        let ctx = egui::Context::default();
        let mut app = AeroFlywayApp::new_with_wgpu(&ctx, None, db_path.clone(), true);
        app.running = false;
        app.show_about_modal = true;
        app.show_help_modal = true;
        app.tool_mode = MapToolMode::DrawTransect;
        app.campaign_name_input = "Shortcut Persisted Campaign".to_string();

        let esc = || egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };

        // 1st Escape dismisses About modal first
        step_app_with_input(&mut app, &ctx, egui::Modifiers::NONE, vec![esc()], Vec::new());
        assert!(!app.show_about_modal);
        assert!(app.show_help_modal);
        assert_eq!(app.tool_mode, MapToolMode::DrawTransect);

        // 2nd Escape dismisses Help modal next
        step_app_with_input(&mut app, &ctx, egui::Modifiers::NONE, vec![esc()], Vec::new());
        assert!(!app.show_help_modal);
        assert_eq!(app.tool_mode, MapToolMode::DrawTransect);

        // 3rd Escape resets map tool to PanAndInspect
        step_app_with_input(&mut app, &ctx, egui::Modifiers::NONE, vec![esc()], Vec::new());
        assert_eq!(app.tool_mode, MapToolMode::PanAndInspect);

        // Ctrl+S saves campaign metadata and refreshes active_campaign + status_line
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::CTRL,
            vec![egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::CTRL,
            }],
            Vec::new(),
        );
        assert_eq!(app.active_campaign.campaign_name, "Shortcut Persisted Campaign");
        assert!(app.status_line.contains("Saved campaign 'Shortcut Persisted Campaign'"));
        assert_ne!(app.active_campaign.updated_at, "2026-11-14T04:10Z");

        drop(app);
        cleanup_db(&db_path);
    }

    #[test]
    fn headless_new_shortcuts_text_guard_scenarios_and_4x4_tile_mesh() {
        let db_path = temp_db("new_ux");
        cleanup_db(&db_path);

        let ctx = egui::Context::default();
        let mut app = AeroFlywayApp::new_with_wgpu(&ctx, None, db_path.clone(), true);
        app.running = false;

        let key_evt = |key: egui::Key| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };

        // Tabs 1-4
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::Num2)],
            Vec::new(),
        );
        assert_eq!(app.bottom_tab, BottomDockTab::RadarVerticalProfile);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::Num3)],
            Vec::new(),
        );
        assert_eq!(app.bottom_tab, BottomDockTab::AcousticSpectrogram);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::Num4)],
            Vec::new(),
        );
        assert_eq!(app.bottom_tab, BottomDockTab::BatchSweepLab);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::Num1)],
            Vec::new(),
        );
        assert_eq!(app.bottom_tab, BottomDockTab::AirspaceTransect);

        // T / M tool mode toggles
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::T)],
            Vec::new(),
        );
        assert_eq!(app.tool_mode, MapToolMode::DrawTransect);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::T)],
            Vec::new(),
        );
        assert_eq!(app.tool_mode, MapToolMode::PanAndInspect);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::M)],
            Vec::new(),
        );
        assert_eq!(app.tool_mode, MapToolMode::PlantVirtualMilkweed);
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::M)],
            Vec::new(),
        );
        assert_eq!(app.tool_mode, MapToolMode::PanAndInspect);

        // L Lights-Out toggle updates status_line
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::L)],
            Vec::new(),
        );
        assert!((app.params.lights_out_curfew_pct - 85.0).abs() < 1e-6);
        assert!(app.status_line.contains("Lights-Out curfew ON"));
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::L)],
            Vec::new(),
        );
        assert!((app.params.lights_out_curfew_pct - 0.0).abs() < 1e-6);
        assert!(app.status_line.contains("Lights-Out curfew OFF"));

        // 0 and R camera reset
        app.map.jump_to(40.0, -120.0, 9.5);
        app.map.pitch_deg = 10.0;
        app.map.bearing_deg = 55.0;
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::Num0)],
            Vec::new(),
        );
        let (def_lat, def_lon, def_zoom) = app.params.region.default_viewport();
        assert!((app.map.center_lat - def_lat).abs() < 1e-6);
        assert!((app.map.center_lon - def_lon).abs() < 1e-6);
        assert!((app.map.zoom - def_zoom).abs() < 1e-6);
        assert!((app.map.pitch_deg - 42.0).abs() < 1e-6);
        assert!((app.map.bearing_deg - (-18.0)).abs() < 1e-6);

        app.map.jump_to(39.0, -119.0, 8.5);
        app.map.pitch_deg = 0.0;
        step_app_with_input(
            &mut app,
            &ctx,
            egui::Modifiers::NONE,
            vec![key_evt(egui::Key::R)],
            Vec::new(),
        );
        assert!((app.map.center_lat - def_lat).abs() < 1e-6);
        assert!((app.map.pitch_deg - 42.0).abs() < 1e-6);

        // TextEdit focus guard prevents single-key shortcuts from firing
        let text_id = egui::Id::new("focused_text_guard_test");
        let mut buf = String::new();
        let _ = ctx.run(egui::RawInput::default(), |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let resp = ui.add(egui::TextEdit::singleline(&mut buf).id(text_id));
                resp.request_focus();
            });
        });
        assert!(ctx.wants_keyboard_input());
        app.map.pitch_deg = 12.0;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                events: vec![
                    key_evt(egui::Key::Num2),
                    key_evt(egui::Key::T),
                    key_evt(egui::Key::L),
                    key_evt(egui::Key::R),
                ],
                ..Default::default()
            },
            |c| {
                egui::CentralPanel::default().show(c, |ui| {
                    ui.add(egui::TextEdit::singleline(&mut buf).id(text_id));
                });
                app.handle_keyboard_shortcuts(c);
            },
        );
        assert_eq!(app.bottom_tab, BottomDockTab::AirspaceTransect);
        assert_eq!(app.tool_mode, MapToolMode::PanAndInspect);
        assert!((app.params.lights_out_curfew_pct - 0.0).abs() < 1e-6);
        assert!((app.map.pitch_deg - 12.0).abs() < 1e-6);

        // 3 Scenario presets
        app.apply_scenario_preset(ScenarioPreset::MegafireSmokeSsp585);
        assert_eq!(app.params.cmip6_scenario, Cmip6Scenario::Ssp585High);
        assert!((app.params.smoke_pm25_offset_ugm3 - 195.0).abs() < 1e-6);
        app.apply_scenario_preset(ScenarioPreset::SpringFalloutUnmitigated);
        assert_eq!(app.params.season, SeasonPreset::SpringNorthbound);
        assert!((app.params.lights_out_curfew_pct - 0.0).abs() < 1e-6);
        app.apply_scenario_preset(ScenarioPreset::AutumnLightsOutPeak);
        assert!((app.params.lights_out_curfew_pct - 85.0).abs() < 1e-6);

        // 4x4 tilted tile mesh & horizon fade checks
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 700.0));
        let fade_2d = compute_tile_horizon_fade(0.0, 0.35, 10.0, canvas);
        assert!((fade_2d - 1.0).abs() < 1e-6);
        let fade_center = compute_tile_horizon_fade(48.0, 1.0, 350.0, canvas);
        let fade_horizon = compute_tile_horizon_fade(48.0, 0.36, 12.0, canvas);
        assert!(fade_horizon < fade_center && fade_horizon >= 0.14);

        let tex = ctx.load_texture(
            "test_tile_4x4",
            egui::ColorImage::example(),
            TextureOptions::LINEAR,
        );
        let tile = aeroflyway::map_engine::ProjectedTile {
            min_wx: 0.0,
            min_wy: 0.0,
            max_wx: 256.0,
            max_wy: 256.0,
            texture: tex,
        };
        let mesh = build_tilted_tile_mesh_4x4(&app.map, &tile, [255, 255, 255, 255], canvas);
        assert_eq!(mesh.vertices.len(), 25, "4x4 tile grid requires 5x5=25 vertices");
        assert_eq!(mesh.indices.len(), 96, "4x4 tile grid requires 32 triangles (96 indices)");

        drop(app);
        cleanup_db(&db_path);
    }
}
