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

//! Rayon-parallel mechanistic aeroecology, climate, VIIRS skyglow & radar solver (`src/physics.rs` — v0.6.0):
//! - Global multi-flyway region support (`FlywayRegion`)
//! - Continuous 365-day Day-of-Year (`day_of_year`) astronomical solar declination & photoperiod
//! - CMIP6 downscaled warming (`Cmip6Scenario`), photoperiod-vs-GDD tri-trophic mismatch, and wetland aridity
//! - Megafire smoke plume disruption (`PM2.5 >= 160 ug/m^3` staging halts & 4,000 m over-plume climbs)
//! - VIIRS nighttime skyglow (`UrbanSkyglowDome`) phototactic entrapment, building collisions, and
//!   BirdCast / Audubon `Lights-Out` curfew (`23:00-06:00`) mortality reduction (Van Doren et al. 2021, *PNAS*)
//! - NEXRAD WSR-88D / Aloft OPERA vertical radar profile (`VPTS`) altitude & velocity assimilation.

use crate::basemap::{
    landmarks_for_region, sample_terrain_elevation_m, skyglow_domes_for_region, FlywayRegion,
    LandmarkKind, MilkweedPatch, Taxon, UrbanSkyglowDome, WindTurbine,
};
use crate::db::{spatial_hex_bin, ObservationRecord};
use crate::ingest::{
    build_regional_radar_profiles, regional_synoptic_waypoints, Cmip6Scenario, SynopticNodeWeather,
    VerticalRadarProfile,
};
use crate::map_engine::haversine_km;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeasonPreset {
    SpringNorthbound,
    SummerBreeding,
    AutumnFalloutAndBatPulse,
    OverwinterRoost,
}

impl SeasonPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::SpringNorthbound => "Spring Northbound (DOY 118 / Late Apr)",
            Self::SummerBreeding => "Summer Breeding & Gen 2-3 (DOY 192 / Mid Jul)",
            Self::AutumnFalloutAndBatPulse => "Autumn Fallout & Bat Pulse (DOY 272 / Late Sep)",
            Self::OverwinterRoost => "Overwinter Sanctuary Roost (DOY 345 / Mid Dec)",
        }
    }

    pub fn canonical_doy(self) -> f64 {
        match self {
            Self::SpringNorthbound => 118.0,
            Self::SummerBreeding => 192.0,
            Self::AutumnFalloutAndBatPulse => 272.0,
            Self::OverwinterRoost => 345.0,
        }
    }

    pub fn from_doy(doy: f64) -> Self {
        let d = doy.rem_euclid(365.0);
        if (65.0..162.0).contains(&d) {
            Self::SpringNorthbound
        } else if (162.0..225.0).contains(&d) {
            Self::SummerBreeding
        } else if (225.0..320.0).contains(&d) {
            Self::AutumnFalloutAndBatPulse
        } else {
            Self::OverwinterRoost
        }
    }

    pub fn base_temp_c(self) -> f64 {
        match self {
            Self::SpringNorthbound => 15.5,
            Self::SummerBreeding => 23.5,
            Self::AutumnFalloutAndBatPulse => 14.0,
            Self::OverwinterRoost => 9.5,
        }
    }
}

/// Astronomical solar declination (radians) for continuous Day-of-Year `doy` (`1..=365`).
pub fn solar_declination_rad(doy: f64) -> f64 {
    let axial_tilt_rad = 23.44_f64.to_radians();
    axial_tilt_rad * ((2.0 * PI / 365.0) * (doy - 81.0)).sin()
}

/// Day length (hours of daylight between sunrise and sunset) at `lat_deg` on `doy`.
pub fn photoperiod_hours(lat_deg: f64, doy: f64) -> f64 {
    let phi = lat_deg.clamp(-66.0, 66.0).to_radians();
    let decl = solar_declination_rad(doy);
    let cos_ha = (-phi.tan() * decl.tan()).clamp(-1.0, 1.0);
    (24.0 / PI) * cos_ha.acos()
}

/// Continuous annual migratory target bearing (degrees clockwise from North) driven by DOY and region axis.
pub fn annual_flyway_heading_deg(region: FlywayRegion, doy: f64) -> f64 {
    let d = doy.rem_euclid(365.0);
    let spring_axis = region.spring_axis_bearing_deg();
    let autumn_axis = (spring_axis + 180.0).rem_euclid(360.0);

    if (60.0..165.0).contains(&d) {
        spring_axis
    } else if (165.0..220.0).contains(&d) {
        (spring_axis + 35.0).rem_euclid(360.0)
    } else if (220.0..325.0).contains(&d) {
        autumn_axis
    } else {
        (autumn_axis + 12.0).rem_euclid(360.0)
    }
}

/// Computes tri-trophic phenological mismatch (days) between temperature-cued insect prey (`-3.0 d/C`)
/// and photoperiod-anchored long-distance avian arrival (`-0.4 d/C` + megafire smoke staging lag).
pub fn compute_trophic_mismatch_days(total_delta_t_c: f64, effective_pm25_ugm3: f64) -> f64 {
    let prey_advance_days = -3.0 * total_delta_t_c;
    let smoke_lag_days = if effective_pm25_ugm3 >= 160.0 {
        2.2 + ((effective_pm25_ugm3 - 160.0) / 140.0).clamp(0.0, 1.5) * 2.4
    } else {
        (effective_pm25_ugm3 / 160.0).clamp(0.0, 1.0) * 0.6
    };
    let bird_shift_days = -0.4 * total_delta_t_c + smoke_lag_days;
    bird_shift_days - prey_advance_days
}

/// Returns whether the municipal BirdCast / Audubon `Lights-Out` curfew window (`23:00-06:00`) is active.
pub fn is_lights_out_curfew_active(hour_of_day: f64) -> bool {
    !(6.0..23.0).contains(&hour_of_day)
}

/// Computes the BirdCast / Audubon `Lights-Out` skyglow dimming multiplier (`0.12..=1.0`):
/// - During active curfew (`23:00-06:00`): `1.0 - 0.88 * (lights_out_curfew_pct / 100.0)`
/// - During early dusk/dawn nocturnal shoulder (`06:00-23:00`): `1.0 - 0.32 * (lights_out_curfew_pct / 100.0)`
/// - During daylight (`!is_nocturnal`): `0.0`
pub fn compute_skyglow_curfew_dim(
    hour_of_day: f64,
    is_nocturnal: bool,
    lights_out_curfew_pct: f64,
) -> f64 {
    if !is_nocturnal {
        return 0.0;
    }
    let frac = lights_out_curfew_pct.clamp(0.0, 100.0) / 100.0;
    if is_lights_out_curfew_active(hour_of_day) {
        1.0 - 0.88 * frac
    } else {
        // Early dusk shoulder dimming (35% of peak curfew compliance)
        1.0 - 0.32 * frac
    }
}

/// Computes effective VIIRS urban skyglow radiance (`nW/cm^2/sr`) accounting for nocturnal phase,
/// anthropogenic skyglow multiplier, and `Lights-Out` curfew window dimming (`0..=100%`).
pub fn effective_skyglow_radiance_nw(
    peak_radiance_nw: f64,
    hour_of_day: f64,
    is_nocturnal: bool,
    lights_out_curfew_pct: f64,
    skyglow_mult: f64,
) -> f64 {
    let curfew_dim = compute_skyglow_curfew_dim(hour_of_day, is_nocturnal, lights_out_curfew_pct);
    (peak_radiance_nw * skyglow_mult.clamp(0.0, 3.0) * curfew_dim).max(0.0)
}

/// Continuous-time hourly bat rotor-zone collision hazard rate (`hr^-1`) used by `compute.wgsl`.
pub const BAT_TURBINE_COLLISION_RATE_PER_HR: f64 = 0.16;

/// Continuous-time Poisson step probability `1 - exp(-rate * dt)` invariant under sub-stepping.
pub fn turbine_collision_step_prob(rate_per_hr: f64, dt_hours: f64) -> f64 {
    1.0 - (-rate_per_hr.max(0.0) * dt_hours.max(0.0)).exp()
}

/// Metric `cos(lat)`-scaled longitude half-width (deg) matching a `half_lat_deg` latitude box.
pub fn turbine_box_half_lon_deg(half_lat_deg: f64, lat_deg: f64) -> f64 {
    let cos_lat = lat_deg.to_radians().cos().abs().max(0.2);
    half_lat_deg / cos_lat
}

/// Analytical expectation of nocturnal building-collision mortality avoided (%) for a given
/// `Lights-Out` curtailment percentage (`Van Doren et al. 2021, PNAS`: 50% window darkening yields
/// ~59-64% overall mortality reduction via non-linear phototactic exponent `1.68`).
pub fn compute_lights_out_collision_reduction_pct(lights_out_pct: f64) -> f64 {
    let f_lit = (1.0 - 0.88 * (lights_out_pct.clamp(0.0, 100.0) / 100.0)).clamp(0.08, 1.0);
    (100.0 * (1.0 - f_lit.powf(1.68))).clamp(0.0, 95.0)
}

#[derive(Clone, Debug)]
pub struct SimulationParams {
    pub region: FlywayRegion,
    pub season: SeasonPreset,
    pub day_of_year: f64,
    pub advance_annual_calendar: bool,
    pub cmip6_scenario: Cmip6Scenario,
    pub wind_850_multiplier: f64,
    pub precip_front_mm_hr: f64,
    pub turbine_cut_in_mps: f64,
    pub temp_offset_c: f64,
    pub smoke_pm25_offset_ugm3: f64,
    pub lights_out_curfew_pct: f64,
    pub skyglow_intensity_mult: f64,
    pub radar_nudge_weight: f64,
    pub transect_start: (f64, f64),
    pub transect_end: (f64, f64),
    pub transect_half_width_km: f64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        let region = FlywayRegion::PacificAmericas;
        let season = SeasonPreset::AutumnFalloutAndBatPulse;
        let (ta, tb) = region.default_transect();
        Self {
            region,
            season,
            day_of_year: season.canonical_doy(),
            advance_annual_calendar: true,
            cmip6_scenario: Cmip6Scenario::BaselineHistorical,
            wind_850_multiplier: 1.0,
            precip_front_mm_hr: 0.4,
            turbine_cut_in_mps: 3.5,
            temp_offset_c: 0.0,
            smoke_pm25_offset_ugm3: 0.0,
            lights_out_curfew_pct: 35.0,
            skyglow_intensity_mult: 1.0,
            radar_nudge_weight: 0.65,
            transect_start: ta,
            transect_end: tb,
            transect_half_width_km: 38.0,
        }
    }
}

impl SimulationParams {
    pub fn total_warming_c(&self) -> f64 {
        self.cmip6_scenario.canonical_delta_t_c() + self.temp_offset_c
    }

    pub fn effective_aridity_index(&self, weather_nodes: &[SynopticNodeWeather]) -> f64 {
        let base_aridity = self.cmip6_scenario.canonical_aridity_index();
        if weather_nodes.is_empty() {
            return (base_aridity + (self.temp_offset_c * 0.04)).clamp(0.0, 0.95);
        }
        let mean_sm = weather_nodes
            .iter()
            .map(|n| n.soil_moisture_m3m3)
            .sum::<f64>()
            / (weather_nodes.len() as f64);
        let sm_deficit = ((0.30 - mean_sm) / 0.26).clamp(0.0, 0.85);
        (0.55 * base_aridity + 0.45 * sm_deficit + (self.temp_offset_c * 0.03)).clamp(0.0, 0.95)
    }
}

#[derive(Clone, Debug)]
pub struct FlywayAgent {
    pub id: u32,
    pub taxon: Taxon,
    pub lat: f64,
    pub lon: f64,
    pub alt_agl_m: f64,
    pub vel_u_mps: f64,
    pub vel_v_mps: f64,
    pub energy_reserve: f64,
    pub generation: u8,
    pub gdd_accum: f64,
    pub in_fallout_stopover: bool,
    pub smoke_disrupted: bool,
    pub skyglow_entrained: bool,
    pub is_roosting: bool,
    pub trail: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct TransectPoint {
    pub dist_km: f64,
    pub terrain_m: f64,
    pub alt_agl_m: f64,
    pub taxon: Taxon,
    pub smoke_disrupted: bool,
    pub skyglow_entrained: bool,
}

#[derive(Clone, Debug, Default)]
pub struct TelemetryMetrics {
    pub sim_hours: f64,
    pub hour_of_day: f64,
    pub day_of_year: f64,
    pub photoperiod_hr: f64,
    pub is_nocturnal: bool,
    pub avian_flux_agents_airborne: usize,
    pub fallout_stopover_birds: usize,
    pub smoke_disrupted_birds: usize,
    pub skyglow_entrained_birds: usize,
    pub cumulative_urban_strikes: u32,
    pub baseline_unmitigated_strikes: u32,
    pub avoided_urban_strikes_pct: f64,
    pub active_migrating_bats: usize,
    pub cumulative_bat_rotor_contacts: u32,
    pub baseline_uncurtailed_contacts: u32,
    pub avoided_bat_contacts_pct: f64,
    pub curtailed_energy_pct: f64,
    pub monarch_mean_gdd: f64,
    pub monarch_peak_generation: u8,
    pub monarch_northern_front_lat: f64,
    pub mean_corridor_pm25_ugm3: f64,
    pub trophic_mismatch_days: f64,
    pub trophic_overlap_pct: f64,
    pub wetland_aridity_index: f64,
    pub radar_mtr_birds_km_hr: f64,
    pub radar_peak_alt_m: f64,
    pub enkf_gain: f64,
    pub enkf_rmse: f64,
    pub rmse_history: Vec<f64>,
}

pub struct AeroPhysicsEngine {
    pub agents: Vec<FlywayAgent>,
    pub weather_nodes: Vec<SynopticNodeWeather>,
    pub radar_profiles: Vec<VerticalRadarProfile>,
    pub cached_transect_dem: Option<Vec<(f64, f64)>>,
    pub metrics: TelemetryMetrics,
    enkf_ensemble_gains: Vec<f64>,
    rng_tick: u64,
}

pub const CPU_SWARM_AGENT_COUNT: usize = 480;

/// Computes the multiplier scaling CPU swarm counts (`cpu_agent_count`, normally `480`)
/// or 480-agent hex aggregates to the active GPU swarm size (`gpu_agent_count`).
pub fn compute_gpu_scale_factor(
    gpu_mode_active: bool,
    gpu_agent_count: u32,
    cpu_agent_count: usize,
) -> f64 {
    if gpu_mode_active {
        (gpu_agent_count as f64) / (cpu_agent_count.max(1) as f64)
    } else {
        1.0
    }
}

impl AeroPhysicsEngine {
    pub fn new() -> Self {
        let region = FlywayRegion::PacificAmericas;
        let season = SeasonPreset::AutumnFalloutAndBatPulse;
        let doy = season.canonical_doy();
        let (clat, _, _) = region.default_viewport();
        let weather_nodes = default_synoptic_nodes_for_region(region, Cmip6Scenario::BaselineHistorical);
        let radar_profiles = build_regional_radar_profiles(None, region, &weather_nodes, 0.0);
        let (mean_mtr, peak_alt) = summarize_radar_corridor(&radar_profiles);

        let mut engine = Self {
            agents: Vec::with_capacity(CPU_SWARM_AGENT_COUNT),
            weather_nodes,
            radar_profiles,
            cached_transect_dem: None,
            metrics: TelemetryMetrics {
                sim_hours: 23.25,
                hour_of_day: 23.25,
                day_of_year: doy,
                photoperiod_hr: photoperiod_hours(clat, doy),
                is_nocturnal: true,
                avoided_urban_strikes_pct: compute_lights_out_collision_reduction_pct(35.0),
                trophic_overlap_pct: 100.0,
                radar_mtr_birds_km_hr: mean_mtr,
                radar_peak_alt_m: peak_alt,
                enkf_gain: 1.0,
                enkf_rmse: 0.128,
                ..Default::default()
            },
            enkf_ensemble_gains: (0..48)
                .map(|i| 0.82 + (i as f64 / 47.0) * 0.36)
                .collect(),
            rng_tick: 1,
        };
        engine.reseed_agents(region, season);
        engine
    }

    pub fn reseed_agents(&mut self, region: FlywayRegion, season: SeasonPreset) {
        self.agents.clear();
        let (min_lat, max_lat, min_lon, max_lon) = region.corridor_bounds();
        let lat_span = (max_lat - min_lat).max(2.0);
        let lon_span = (max_lon - min_lon).max(2.0);
        let mid_lon = 0.5 * (min_lon + max_lon);

        let mut id = 0_u32;

        for i in 0..220 {
            let h1 = deterministic_unit(i as u64 * 31 + 7);
            let h2 = deterministic_unit(i as u64 * 67 + 13);
            let h3 = deterministic_unit(i as u64 * 101 + 29);

            let lat_frac = match season {
                SeasonPreset::SpringNorthbound => 0.12 + h1 * 0.62,
                SeasonPreset::OverwinterRoost => 0.08 + h1 * 0.45,
                _ => 0.24 + h1 * 0.68,
            };
            let lat = min_lat + lat_frac * lat_span;
            let lane_offset = if i % 3 != 0 { -0.16 } else { 0.18 };
            let lon = (mid_lon + lane_offset * lon_span + (h2 - 0.5) * 0.36 * lon_span)
                .clamp(min_lon + 0.15, max_lon - 0.15);
            let alt = 480.0 + h3 * 1250.0;

            self.agents.push(FlywayAgent {
                id,
                taxon: Taxon::Bird,
                lat,
                lon,
                alt_agl_m: alt,
                vel_u_mps: 0.0,
                vel_v_mps: 0.0,
                energy_reserve: 0.48 + 0.48 * h2,
                generation: 1,
                gdd_accum: 0.0,
                in_fallout_stopover: false,
                smoke_disrupted: false,
                skyglow_entrained: false,
                is_roosting: false,
                trail: vec![(lat, lon)],
            });
            id += 1;
        }

        for i in 0..140 {
            let h1 = deterministic_unit(i as u64 * 43 + 1009);
            let h2 = deterministic_unit(i as u64 * 89 + 2017);
            let h3 = deterministic_unit(i as u64 * 127 + 3001);

            let lat = min_lat + (0.18 + h1 * 0.72) * lat_span;
            let lon =
                (mid_lon + (h2 - 0.5) * 0.55 * lon_span).clamp(min_lon + 0.15, max_lon - 0.15);

            self.agents.push(FlywayAgent {
                id,
                taxon: Taxon::Bat,
                lat,
                lon,
                alt_agl_m: 55.0 + h3 * 240.0,
                vel_u_mps: 0.0,
                vel_v_mps: 0.0,
                energy_reserve: 0.6 + 0.35 * h1,
                generation: 1,
                gdd_accum: 0.0,
                in_fallout_stopover: false,
                smoke_disrupted: false,
                skyglow_entrained: false,
                is_roosting: false,
                trail: vec![(lat, lon)],
            });
            id += 1;
        }

        for i in 0..120 {
            let h1 = deterministic_unit(i as u64 * 53 + 5003);
            let h2 = deterministic_unit(i as u64 * 97 + 6011);
            let h3 = deterministic_unit(i as u64 * 139 + 7027);

            let (lat_frac, gen, gdd) = match season {
                SeasonPreset::OverwinterRoost => (0.06 + h1 * 0.28, 4, 310.0 + h3 * 40.0),
                SeasonPreset::SpringNorthbound => (0.14 + h1 * 0.45, 1, 140.0 + h3 * 190.0),
                _ => (0.28 + h1 * 0.52, 3, 210.0 + h3 * 135.0),
            };
            let lat = min_lat + lat_frac * lat_span;
            let lon =
                (mid_lon + (h2 - 0.5) * 0.48 * lon_span).clamp(min_lon + 0.15, max_lon - 0.15);

            self.agents.push(FlywayAgent {
                id,
                taxon: Taxon::Butterfly,
                lat,
                lon,
                alt_agl_m: 12.0 + h3 * 95.0,
                vel_u_mps: 0.0,
                vel_v_mps: 0.0,
                energy_reserve: 0.7,
                generation: gen,
                gdd_accum: gdd,
                in_fallout_stopover: false,
                smoke_disrupted: false,
                skyglow_entrained: false,
                is_roosting: false,
                trail: vec![(lat, lon)],
            });
            id += 1;
        }
        debug_assert_eq!(self.agents.len(), CPU_SWARM_AGENT_COUNT);
    }

    /// Advances the multi-taxa IBM forward by `dt_hours` in parallel via Rayon and updates metrics.
    pub fn step(
        &mut self,
        dt_hours: f64,
        params: &SimulationParams,
        turbines: &[WindTurbine],
        milkweed: &[MilkweedPatch],
        observations: &[ObservationRecord],
    ) {
        self.rng_tick = self.rng_tick.wrapping_add(1);
        self.metrics.sim_hours += dt_hours;
        self.metrics.hour_of_day = self.metrics.sim_hours.rem_euclid(24.0);

        let active_doy = if params.advance_annual_calendar {
            (params.day_of_year + self.metrics.sim_hours / 24.0 - 1.0).rem_euclid(365.0) + 1.0
        } else {
            params.day_of_year.clamp(1.0, 365.0)
        };
        self.metrics.day_of_year = active_doy;

        let (center_lat, _, _) = params.region.default_viewport();
        let day_len_hr = photoperiod_hours(center_lat, active_doy);
        self.metrics.photoperiod_hr = day_len_hr;

        let sunrise_hr = (12.0 - 0.5 * day_len_hr).clamp(3.5, 9.5);
        let sunset_hr = (12.0 + 0.5 * day_len_hr).clamp(15.0, 21.5);
        let hr = self.metrics.hour_of_day;
        let is_nocturnal = hr < sunrise_hr || hr > sunset_hr;
        let is_dusk_window = hr >= (sunset_hr - 1.2) && hr <= (sunset_hr + 4.2).min(23.9);
        self.metrics.is_nocturnal = is_nocturnal;

        let mean_node_pm25 = if self.weather_nodes.is_empty() {
            10.0
        } else {
            self.weather_nodes.iter().map(|n| n.pm25_ugm3).sum::<f64>()
                / (self.weather_nodes.len() as f64)
        };
        let eff_corridor_pm25 = (mean_node_pm25 + params.smoke_pm25_offset_ugm3).max(0.0);
        self.metrics.mean_corridor_pm25_ugm3 = eff_corridor_pm25;

        let total_dt = params.total_warming_c();
        let mismatch_days = compute_trophic_mismatch_days(total_dt, eff_corridor_pm25);
        let trophic_overlap = (-0.5 * (mismatch_days / 11.5).powi(2)).exp().clamp(0.08, 1.0);
        let aridity = params.effective_aridity_index(&self.weather_nodes);

        self.metrics.trophic_mismatch_days = mismatch_days;
        self.metrics.trophic_overlap_pct = trophic_overlap * 100.0;
        self.metrics.wetland_aridity_index = aridity;

        let (mean_mtr, radar_peak_alt_asl) = summarize_radar_corridor(&self.radar_profiles);
        self.metrics.radar_mtr_birds_km_hr = mean_mtr;
        self.metrics.radar_peak_alt_m = radar_peak_alt_asl;

        let refueling_mult = trophic_overlap * (1.0 - 0.55 * aridity).clamp(0.25, 1.0);
        let enkf_gain = self.metrics.enkf_gain;
        let weather_snapshot = self.weather_nodes.clone();
        let skyglow_domes = skyglow_domes_for_region(params.region);
        let tick_seed = self.rng_tick;
        let heading_rad = annual_flyway_heading_deg(params.region, active_doy).to_radians();
        let target_u = heading_rad.sin();
        let target_v = heading_rad.cos();

        let refuges: Vec<(f64, f64)> = landmarks_for_region(params.region)
            .iter()
            .filter(|l| {
                l.kind == LandmarkKind::WildlifeRefuge || l.kind == LandmarkKind::MonarchSanctuary
            })
            .map(|l| (l.lat, l.lon))
            .collect();

        let (min_lat, max_lat, min_lon, max_lon) = params.region.corridor_bounds();
        let mid_lon = 0.5 * (min_lon + max_lon);
        let radar_nudge = params.radar_nudge_weight.clamp(0.0, 1.0);
        let _ = is_dusk_window;

        let contact_flags: Vec<(u32, u32, u32, u32)> = self
            .agents
            .par_iter_mut()
            .map(|agent| {
                let jitter_a = deterministic_unit(
                    (agent.id as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(tick_seed),
                ) - 0.5;
                let jitter_b = deterministic_unit(
                    (agent.id as u64)
                        .wrapping_mul(1442695040888963407)
                        .wrapping_add(tick_seed * 7),
                ) - 0.5;

                let phase_hr = (deterministic_unit(
                    (agent.id as u64)
                        .wrapping_mul(747796405)
                        .wrapping_add(2891336453),
                ) - 0.5)
                    * 2.2;
                let halo_deg = 0.18
                    + 0.27
                        * deterministic_unit(
                            (agent.id as u64)
                                .wrapping_mul(277803737)
                                .wrapping_add(1442695041),
                        );
                let halo_spin = if agent.id % 2 == 0 { 1.0 } else { -1.0 };
                let local_hr = (hr + (agent.lon - mid_lon) / 15.0 + phase_hr).rem_euclid(24.0);
                let nocturnal_drive = if local_hr > 12.0 {
                    smoothstep_f64(sunset_hr - 1.25, sunset_hr + 0.95, local_hr)
                } else {
                    1.0 - smoothstep_f64(sunrise_hr - 0.95, sunrise_hr + 1.25, local_hr)
                };
                let dusk_pulse = smoothstep_f64(sunset_hr - 1.5, sunset_hr, local_hr)
                    * (1.0 - smoothstep_f64(sunset_hr + 0.5, sunset_hr + 3.0, local_hr));
                let diurnal_drive = 1.0 - nocturnal_drive;
                let bat_drive = (nocturnal_drive + 0.55 * dusk_pulse).clamp(0.0, 1.0);

                let (temp_2m, precip_local, pm25_local, u80, v80, ws80, u850, v850) =
                    interpolate_weather(&weather_snapshot, agent.lat, agent.lon, params, active_doy);

                let mut actual_bat_contact = 0_u32;
                let mut baseline_bat_contact = 0_u32;
                let mut actual_urban_strike = 0_u32;
                let mut baseline_urban_strike = 0_u32;

                match agent.taxon {
                    Taxon::Bird => {
                        let total_precip = precip_local + params.precip_front_mm_hr;
                        let severe_smoke = pm25_local >= 160.0;
                        agent.smoke_disrupted = severe_smoke;
                        agent.skyglow_entrained = false;

                        let smoke_halt = severe_smoke && (agent.id % 5 < 2);
                        let is_diurnal_bird = (agent.id % 20) >= 13;
                        let circadian_drive = if is_diurnal_bird {
                            diurnal_drive
                        } else {
                            nocturnal_drive
                        };

                        if circadian_drive > 0.25 && total_precip < 2.2 && !smoke_halt && agent.energy_reserve > 0.12 {
                            agent.in_fallout_stopover = false;
                            agent.is_roosting = false;
                            let smoke_drag = if severe_smoke { 0.78 } else { 1.0 };
                            let speed_ramp = 0.48 + 0.52 * circadian_drive;
                            let airspeed = 11.5 * enkf_gain * smoke_drag * speed_ramp;
                            let lateral_disp = if severe_smoke { 4.8 } else { 2.2 };

                            // Check VIIRS urban skyglow phototactic pull & building collision risk
                            let (glow_u, glow_v, eff_rad_nw, base_rad_nw, hazard_idx) = if !is_diurnal_bird {
                                sample_skyglow_pull(
                                    agent.lat,
                                    agent.lon,
                                    &skyglow_domes,
                                    local_hr,
                                    nocturnal_drive > 0.35,
                                    params.lights_out_curfew_pct,
                                    params.skyglow_intensity_mult,
                                )
                            } else {
                                (0.0, 0.0, 0.0, 0.0, 0.0)
                            };

                            if eff_rad_nw > 14.0 && !severe_smoke {
                                agent.skyglow_entrained = true;
                                let pull_strength = (eff_rad_nw / 110.0).clamp(0.12, 0.88);
                                let orbit_u = -glow_v * halo_spin;
                                let orbit_v = glow_u * halo_spin;
                                agent.vel_u_mps = airspeed
                                    * ((1.0 - 0.55 * pull_strength) * target_u
                                        + 0.45 * pull_strength * glow_u
                                        + 0.42 * pull_strength * orbit_u)
                                    + 0.7 * u850
                                    + jitter_a * 4.2;
                                agent.vel_v_mps = airspeed
                                    * ((1.0 - 0.55 * pull_strength) * target_v
                                        + 0.45 * pull_strength * glow_v
                                        + 0.42 * pull_strength * orbit_v)
                                    + 0.7 * v850
                                    + jitter_b * 4.2;
                                // Skyglow disorientation draws nocturnal songbirds down into <260 m AGL facade zone
                                let target_canopy_agl = 85.0 + 115.0 * (1.0 - pull_strength);
                                agent.alt_agl_m = (0.68 * agent.alt_agl_m + 0.32 * target_canopy_agl + jitter_a * 24.0)
                                    .clamp(28.0, 320.0);
                                agent.energy_reserve = (agent.energy_reserve - 0.025 * dt_hours).max(0.04);

                                // Van Doren et al. (2021, PNAS): windy nocturnal migration + lit windows -> glass collisions
                                let wind_amp = (1.0 + 0.035 * u850.hypot(v850)).clamp(1.0, 1.6);
                                let base_prob = 0.095 * hazard_idx * (base_rad_nw / 100.0).powf(1.68) * wind_amp * dt_hours.min(1.5);
                                let eff_prob = 0.095 * hazard_idx * (eff_rad_nw / 100.0).powf(1.68) * wind_amp * dt_hours.min(1.5);
                                let roll = deterministic_unit((agent.id as u64).wrapping_mul(2654435761) ^ tick_seed);
                                if roll < base_prob {
                                    baseline_urban_strike = 1;
                                }
                                if roll < eff_prob {
                                    actual_urban_strike = 1;
                                }
                            } else {
                                agent.vel_u_mps = airspeed * target_u + u850 + jitter_a * lateral_disp;
                                agent.vel_v_mps = airspeed * target_v + v850 + jitter_b * lateral_disp;
                                if severe_smoke {
                                    agent.alt_agl_m = (agent.alt_agl_m + 240.0 * dt_hours.max(0.2) + jitter_a * 65.0)
                                        .clamp(1850.0, 3850.0);
                                } else {
                                    let ground_m = sample_terrain_elevation_m(agent.lat, agent.lon);
                                    let radar_target_agl = (radar_peak_alt_asl - ground_m).clamp(480.0, 3200.0);
                                    let free_agl = (agent.alt_agl_m + jitter_a * 65.0).clamp(520.0, 3400.0);
                                    agent.alt_agl_m = (1.0 - 0.22 * radar_nudge) * free_agl
                                        + (0.22 * radar_nudge) * radar_target_agl;
                                }
                                let burn_rate = if severe_smoke { 0.026 } else { 0.018 };
                                agent.energy_reserve = (agent.energy_reserve - burn_rate * dt_hours).max(0.04);
                            }
                        } else {
                            agent.in_fallout_stopover = total_precip >= 2.2 || smoke_halt;
                            agent.is_roosting = true;
                            let (pull_u, pull_v) =
                                halo_attractor_vec(agent.lat, agent.lon, &refuges, halo_deg, halo_spin);
                            agent.vel_u_mps = pull_u * 2.2 + 0.35 * target_u + jitter_a * 0.85;
                            agent.vel_v_mps = pull_v * 2.2 + 0.35 * target_v + jitter_b * 0.85;
                            agent.alt_agl_m = (agent.alt_agl_m * 0.72).clamp(15.0, 140.0);
                            agent.energy_reserve =
                                (agent.energy_reserve + 0.048 * refueling_mult * dt_hours).min(1.0);
                        }
                    }
                    Taxon::Bat => {
                        agent.smoke_disrupted = pm25_local >= 160.0;
                        agent.skyglow_entrained = false;
                        let warm_enough = temp_2m > 7.5;
                        let low_wind_pulse = ws80 < 6.8;
                        let smoke_suppression = if pm25_local >= 160.0 { 0.62 } else { 1.0 };
                        if bat_drive > 0.22 && warm_enough {
                            agent.is_roosting = false;
                            let pulse_boost = (if low_wind_pulse { 1.35 } else { 0.55 }) * (0.45 + 0.55 * bat_drive);
                            let airspeed = 6.8 * pulse_boost * smoke_suppression * enkf_gain;
                            agent.vel_u_mps = airspeed * target_u + 0.45 * u80 + jitter_a * 1.8;
                            agent.vel_v_mps = airspeed * target_v + 0.45 * v80 + jitter_b * 1.8;
                            agent.alt_agl_m = (agent.alt_agl_m + jitter_b * 28.0).clamp(40.0, 300.0);

                            for t in turbines {
                                let cos_t_lat = t.lat.to_radians().cos().abs().max(0.2);
                                if (agent.lat - t.lat).abs() < 0.045
                                    && (agent.lon - t.lon).abs() * cos_t_lat < 0.045
                                {
                                    let r_rad = t.rotor_diameter_m * 0.5;
                                    let z_min = (t.hub_height_m - r_rad).max(15.0);
                                    let z_max = t.hub_height_m + r_rad;
                                    if agent.alt_agl_m >= z_min && agent.alt_agl_m <= z_max {
                                        if ws80 >= 3.0 && ws80 <= 25.0 {
                                            baseline_bat_contact = 1;
                                        }
                                        if ws80 >= params.turbine_cut_in_mps && ws80 <= 25.0 {
                                            actual_bat_contact = 1;
                                        }
                                        break;
                                    }
                                }
                            }
                        } else {
                            agent.is_roosting = true;
                            agent.vel_u_mps = 0.45 * target_u + jitter_a * 0.65;
                            agent.vel_v_mps = 0.45 * target_v + jitter_b * 0.65;
                            agent.alt_agl_m = (agent.alt_agl_m * 0.65).clamp(12.0, 42.0);
                        }
                    }
                    Taxon::Butterfly => {
                        agent.smoke_disrupted = pm25_local >= 160.0;
                        agent.skyglow_entrained = false;
                        let gdd_rate_per_hr = (temp_2m - 11.5).max(0.0) / 24.0;
                        agent.gdd_accum += gdd_rate_per_hr * dt_hours;
                        if agent.gdd_accum >= 352.0 {
                            agent.gdd_accum -= 352.0;
                            agent.generation = (agent.generation % 4) + 1;
                        }

                        let (mw_u, mw_v) = nearest_milkweed_halo_vec(
                            agent.lat,
                            agent.lon,
                            milkweed,
                            aridity,
                            halo_deg,
                            halo_spin,
                        );
                        if diurnal_drive > 0.22 && (12.5..=35.5).contains(&temp_2m) && pm25_local < 220.0 {
                            agent.is_roosting = false;
                            let airspeed = 4.2 * enkf_gain * (0.42 + 0.58 * diurnal_drive);
                            agent.vel_u_mps = airspeed * (0.68 * target_u + 0.32 * mw_u)
                                + 0.30 * u80
                                + jitter_a * 1.4;
                            agent.vel_v_mps = airspeed * (0.68 * target_v + 0.32 * mw_v)
                                + 0.30 * v80
                                + jitter_b * 1.4;
                            agent.alt_agl_m =
                                (14.0 + (temp_2m - 12.0) * 4.2 + jitter_a * 16.0).clamp(5.0, 120.0);
                        } else {
                            agent.is_roosting = true;
                            agent.vel_u_mps = 0.85 * mw_u + 0.45 * target_u + jitter_a * 0.45;
                            agent.vel_v_mps = 0.85 * mw_v + 0.45 * target_v + jitter_b * 0.45;
                            agent.alt_agl_m = 5.0;
                        }
                    }
                }

                let dt_sec = dt_hours * 3600.0;
                let dlat = (agent.vel_v_mps * dt_sec) / 111_320.0;
                let cos_lat = agent.lat.to_radians().cos().abs().max(0.2);
                let dlon = (agent.vel_u_mps * dt_sec) / (111_320.0 * cos_lat);

                let mut next_lat = agent.lat + dlat;
                if next_lat < min_lat {
                    next_lat = max_lat - 0.6;
                    agent.trail.clear();
                } else if next_lat > max_lat {
                    next_lat = min_lat + 0.6;
                    agent.trail.clear();
                }
                agent.lat = next_lat;
                agent.lon = (agent.lon + dlon).clamp(min_lon, max_lon);

                if tick_seed % 2 == 0 {
                    agent.trail.push((agent.lat, agent.lon));
                    if agent.trail.len() > 14 {
                        agent.trail.remove(0);
                    }
                }

                (
                    actual_bat_contact,
                    baseline_bat_contact,
                    actual_urban_strike,
                    baseline_urban_strike,
                )
            })
            .collect();

        let step_bat_actual: u32 = contact_flags.iter().map(|c| c.0).sum();
        let step_bat_base: u32 = contact_flags.iter().map(|c| c.1).sum();
        let step_urb_actual: u32 = contact_flags.iter().map(|c| c.2).sum();
        let step_urb_base: u32 = contact_flags.iter().map(|c| c.3).sum();

        self.metrics.cumulative_bat_rotor_contacts = self
            .metrics
            .cumulative_bat_rotor_contacts
            .saturating_add(step_bat_actual);
        self.metrics.baseline_uncurtailed_contacts = self
            .metrics
            .baseline_uncurtailed_contacts
            .saturating_add(step_bat_base);

        self.metrics.cumulative_urban_strikes = self
            .metrics
            .cumulative_urban_strikes
            .saturating_add(step_urb_actual);
        self.metrics.baseline_unmitigated_strikes = self
            .metrics
            .baseline_unmitigated_strikes
            .saturating_add(step_urb_base);

        if self.metrics.baseline_uncurtailed_contacts > 0 {
            let saved = self
                .metrics
                .baseline_uncurtailed_contacts
                .saturating_sub(self.metrics.cumulative_bat_rotor_contacts) as f64;
            self.metrics.avoided_bat_contacts_pct =
                (100.0 * saved / self.metrics.baseline_uncurtailed_contacts as f64).clamp(0.0, 100.0);
        } else {
            let cut_delta = (params.turbine_cut_in_mps - 3.0).max(0.0);
            self.metrics.avoided_bat_contacts_pct = (1.0 - (-0.52 * cut_delta).exp()) * 88.0;
        }

        if self.metrics.baseline_unmitigated_strikes > 8 {
            let saved_urb = self
                .metrics
                .baseline_unmitigated_strikes
                .saturating_sub(self.metrics.cumulative_urban_strikes) as f64;
            self.metrics.avoided_urban_strikes_pct =
                (100.0 * saved_urb / self.metrics.baseline_unmitigated_strikes as f64)
                    .clamp(0.0, 100.0);
        } else {
            self.metrics.avoided_urban_strikes_pct =
                compute_lights_out_collision_reduction_pct(params.lights_out_curfew_pct);
        }

        let cut_excess = (params.turbine_cut_in_mps - 3.0).max(0.0);
        self.metrics.curtailed_energy_pct = (0.42 * cut_excess.powf(2.15)).clamp(0.0, 18.5);

        let mut birds_airborne = 0usize;
        let mut birds_fallout = 0usize;
        let mut birds_smoke = 0usize;
        let mut birds_skyglow = 0usize;
        let mut bats_active = 0usize;
        let mut gdd_sum = 0.0_f64;
        let mut bfly_count = 0usize;
        let mut max_gen = 1_u8;
        let mut north_lat = min_lat;

        for a in &self.agents {
            match a.taxon {
                Taxon::Bird => {
                    if a.alt_agl_m > 160.0 || a.skyglow_entrained {
                        birds_airborne += 1;
                    }
                    if a.in_fallout_stopover {
                        birds_fallout += 1;
                    }
                    if a.smoke_disrupted {
                        birds_smoke += 1;
                    }
                    if a.skyglow_entrained {
                        birds_skyglow += 1;
                    }
                }
                Taxon::Bat => {
                    if a.alt_agl_m > 36.0 {
                        bats_active += 1;
                    }
                }
                Taxon::Butterfly => {
                    gdd_sum += a.gdd_accum;
                    bfly_count += 1;
                    max_gen = max_gen.max(a.generation);
                    north_lat = north_lat.max(a.lat);
                }
            }
        }

        self.metrics.avian_flux_agents_airborne = birds_airborne;
        self.metrics.fallout_stopover_birds = birds_fallout;
        self.metrics.smoke_disrupted_birds = birds_smoke;
        self.metrics.skyglow_entrained_birds = birds_skyglow;
        self.metrics.active_migrating_bats = bats_active;
        self.metrics.monarch_mean_gdd = if bfly_count > 0 {
            gdd_sum / bfly_count as f64
        } else {
            0.0
        };
        self.metrics.monarch_peak_generation = max_gen;
        self.metrics.monarch_northern_front_lat = north_lat;

        if self.rng_tick % 4 == 0 {
            self.assimilate_enkf(observations, radar_nudge);
        }
    }

    fn assimilate_enkf(&mut self, observations: &[ObservationRecord], radar_nudge: f64) {
        if observations.is_empty() {
            return;
        }
        let mut obs_bins: HashMap<String, f64> = HashMap::new();
        for o in observations {
            *obs_bins.entry(o.hex_bin.clone()).or_insert(0.0) += (o.count as f64).ln_1p() + 1.0;
        }
        let obs_norm: f64 = obs_bins.values().sum::<f64>().max(1.0);

        let mut sim_bins: HashMap<String, f64> = HashMap::new();
        for a in &self.agents {
            let b = spatial_hex_bin(a.lat, a.lon);
            *sim_bins.entry(b).or_insert(0.0) += 1.0;
        }
        let sim_norm: f64 = sim_bins.values().sum::<f64>().max(1.0);

        let mut sq_err = 0.0_f64;
        let mut n = 0usize;
        for (bin, obs_val) in &obs_bins {
            let y_obs = obs_val / obs_norm;
            let y_sim = sim_bins.get(bin).copied().unwrap_or(0.0) / sim_norm;
            let diff = y_obs - y_sim;
            sq_err += diff * diff;
            n += 1;
        }
        let raw_rmse = if n > 0 { (sq_err / n as f64).sqrt() } else { 0.1 };
        // Vertical radar profile (`VPTS`) constraint reduces altitude-velocity state uncertainty
        let rmse = (raw_rmse * (1.0 - 0.18 * radar_nudge)).max(0.025);

        let mean_g =
            self.enkf_ensemble_gains.iter().sum::<f64>() / (self.enkf_ensemble_gains.len() as f64);
        let var_g = self
            .enkf_ensemble_gains
            .iter()
            .map(|g| (g - mean_g).powi(2))
            .sum::<f64>()
            / (self.enkf_ensemble_gains.len() as f64);
        let obs_var = 0.025_f64 * (1.0 - 0.25 * radar_nudge).max(0.5);
        let kalman_k = var_g / (var_g + obs_var);
        let innovation = (0.12 - rmse).clamp(-0.08, 0.08);

        for (idx, g) in self.enkf_ensemble_gains.iter_mut().enumerate() {
            let pert = (deterministic_unit(self.rng_tick * 97 + idx as u64) - 0.5) * 0.012;
            *g = (*g + kalman_k * innovation + pert).clamp(0.65, 1.45);
        }

        self.metrics.enkf_gain =
            self.enkf_ensemble_gains.iter().sum::<f64>() / (self.enkf_ensemble_gains.len() as f64);
        self.metrics.enkf_rmse = rmse;
        self.metrics.rmse_history.push(rmse);
        if self.metrics.rmse_history.len() > 120 {
            self.metrics.rmse_history.remove(0);
        }
    }

    /// Samples terrain profile (`(dist_km, terrain_m)`) using cached Copernicus GLO-90 `/v1/elevation`
    /// points when available or multi-continent offline DEM fallback otherwise.
    pub fn compute_transect_slice(
        &self,
        params: &SimulationParams,
    ) -> (f64, Vec<(f64, f64)>, Vec<TransectPoint>) {
        let (lat_a, lon_a) = params.transect_start;
        let (lat_b, lon_b) = params.transect_end;
        let total_km = haversine_km(lat_a, lon_a, lat_b, lon_b).max(1.0);

        let elev_at_frac = |t: f64| -> f64 {
            if let Some(dem) = &self.cached_transect_dem {
                if dem.len() >= 2 {
                    let pos = (t.clamp(0.0, 1.0)) * ((dem.len() - 1) as f64);
                    let idx0 = (pos.floor() as usize).min(dem.len() - 1);
                    let idx1 = (idx0 + 1).min(dem.len() - 1);
                    let frac = pos - (idx0 as f64);
                    return dem[idx0].1 * (1.0 - frac) + dem[idx1].1 * frac;
                }
            }
            let lat = lat_a + (lat_b - lat_a) * t;
            let lon = lon_a + (lon_b - lon_a) * t;
            sample_terrain_elevation_m(lat, lon)
        };

        let steps = 72usize;
        let mut terrain_profile = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            terrain_profile.push((t * total_km, elev_at_frac(t)));
        }

        let dlat = lat_b - lat_a;
        let dlon = lon_b - lon_a;
        let seg_len_sq = (dlat * dlat + dlon * dlon).max(1e-9);

        let mut projected = Vec::new();
        for a in &self.agents {
            let t = (((a.lat - lat_a) * dlat + (a.lon - lon_a) * dlon) / seg_len_sq).clamp(0.0, 1.0);
            let proj_lat = lat_a + t * dlat;
            let proj_lon = lon_a + t * dlon;
            let cross_km = haversine_km(a.lat, a.lon, proj_lat, proj_lon);
            if cross_km <= params.transect_half_width_km {
                projected.push(TransectPoint {
                    dist_km: t * total_km,
                    terrain_m: elev_at_frac(t),
                    alt_agl_m: a.alt_agl_m,
                    taxon: a.taxon,
                    smoke_disrupted: a.smoke_disrupted,
                    skyglow_entrained: a.skyglow_entrained,
                });
            }
        }

        (total_km, terrain_profile, projected)
    }

    pub fn hex_density_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for a in &self.agents {
            *counts.entry(spatial_hex_bin(a.lat, a.lon)).or_insert(0) += 1;
        }
        counts
    }
}

pub fn summarize_radar_corridor(profiles: &[VerticalRadarProfile]) -> (f64, f64) {
    if profiles.is_empty() {
        return (1850.0, 980.0);
    }
    let n = profiles.len() as f64;
    let mtr = profiles.iter().map(|p| p.mtr_birds_km_hr).sum::<f64>() / n;
    let peak = profiles.iter().map(|p| p.peak_altitude_m_asl).sum::<f64>() / n;
    (mtr, peak)
}

fn sample_skyglow_pull(
    lat: f64,
    lon: f64,
    domes: &[UrbanSkyglowDome],
    hour_of_day: f64,
    is_nocturnal: bool,
    lights_out_pct: f64,
    skyglow_mult: f64,
) -> (f64, f64, f64, f64, f64) {
    if !is_nocturnal || domes.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut best_eff = 0.0_f64;
    let mut best_base = 0.0_f64;
    let mut best_u = 0.0_f64;
    let mut best_v = 0.0_f64;
    let mut best_haz = 0.0_f64;

    for dome in domes {
        let dist_km = haversine_km(lat, lon, dome.lat, dome.lon);
        let reach_km = dome.radius_km * 1.85;
        if dist_km <= reach_km {
            let decay = (-0.5 * (dist_km / dome.radius_km.max(8.0)).powi(2)).exp();
            let base_rad = dome.peak_radiance_nw_cm2_sr * skyglow_mult.clamp(0.0, 3.0) * decay;
            let eff_rad = effective_skyglow_radiance_nw(
                dome.peak_radiance_nw_cm2_sr,
                hour_of_day,
                is_nocturnal,
                lights_out_pct,
                skyglow_mult,
            ) * decay;
            if eff_rad > best_eff {
                best_eff = eff_rad;
                best_base = base_rad;
                best_haz = dome.glass_hazard_index;
                let dlat = dome.lat - lat;
                let dlon = dome.lon - lon;
                let len = (dlat * dlat + dlon * dlon).sqrt().max(1e-5);
                best_u = dlon / len;
                best_v = dlat / len;
            }
        }
    }
    (best_u, best_v, best_eff, best_base, best_haz)
}

fn interpolate_weather(
    nodes: &[SynopticNodeWeather],
    lat: f64,
    lon: f64,
    params: &SimulationParams,
    doy: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let seasonal_mod_c = 6.5 * ((2.0 * PI / 365.0) * (doy - 110.0)).sin();
    let total_warming = params.total_warming_c();

    if nodes.is_empty() {
        let base_t = 15.0 + seasonal_mod_c + total_warming;
        return (
            base_t,
            0.0,
            (10.0 + params.smoke_pm25_offset_ugm3).max(0.0),
            1.2,
            -3.5,
            3.7,
            2.5,
            -7.5,
        );
    }

    let mut w_sum = 0.0;
    let mut temp = 0.0;
    let mut precip = 0.0;
    let mut pm25 = 0.0;
    let mut u80 = 0.0;
    let mut v80 = 0.0;
    let mut u850 = 0.0;
    let mut v850 = 0.0;

    for n in nodes {
        let d2 = ((lat - n.lat).powi(2) + (lon - n.lon).powi(2)).max(0.02);
        let w = 1.0 / d2;
        w_sum += w;
        temp += w * n.temp_2m_c;
        precip += w * n.precip_mm_hr;
        pm25 += w * n.pm25_ugm3;

        let r80 = (n.wind_80m_deg + 180.0).to_radians();
        u80 += w * n.wind_80m_mps * r80.sin();
        v80 += w * n.wind_80m_mps * r80.cos();

        let r850 = (n.wind_850_deg + 180.0).to_radians();
        u850 += w * n.wind_850_mps * r850.sin();
        v850 += w * n.wind_850_mps * r850.cos();
    }

    let inv = 1.0 / w_sum.max(1e-9);
    let t_final = temp * inv + 0.35 * seasonal_mod_c + total_warming;
    let p_final = precip * inv;
    let pm_final = (pm25 * inv + params.smoke_pm25_offset_ugm3).max(0.0);
    let u80_f = u80 * inv;
    let v80_f = v80 * inv;
    let ws80_f = (u80_f * u80_f + v80_f * v80_f).sqrt();
    let u850_f = u850 * inv * params.wind_850_multiplier;
    let v850_f = v850 * inv * params.wind_850_multiplier;

    (t_final, p_final, pm_final, u80_f, v80_f, ws80_f, u850_f, v850_f)
}

#[inline]
fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn halo_steer(dlat: f64, dlon: f64, halo_deg: f64, spin: f64) -> (f64, f64) {
    let dist = dlon.hypot(dlat).max(1e-5);
    let ru = dlon / dist;
    let rv = dlat / dist;
    let tu = -rv * spin;
    let tv = ru * spin;
    let radial_push = if dist > halo_deg {
        ((dist - halo_deg) / halo_deg).clamp(0.0, 1.0)
    } else {
        -((halo_deg - dist) / halo_deg).clamp(0.0, 0.72)
    };
    let tan_w = 0.85 * (1.0 - (dist / (halo_deg * 2.4)).clamp(0.0, 1.0)) + 0.15;
    let vx = radial_push * ru + tan_w * tu;
    let vy = radial_push * rv + tan_w * tv;
    let norm = vx.hypot(vy).max(1e-5);
    (vx / norm, vy / norm)
}

fn halo_attractor_vec(
    lat: f64,
    lon: f64,
    targets: &[(f64, f64)],
    halo_deg: f64,
    spin: f64,
) -> (f64, f64) {
    let mut best_d2 = f64::MAX;
    let mut best = (0.0, 0.0);
    for &(tlat, tlon) in targets {
        let dlat = tlat - lat;
        let dlon = tlon - lon;
        let d2 = dlat * dlat + dlon * dlon;
        if d2 < best_d2 {
            best_d2 = d2;
            best = halo_steer(dlat, dlon, halo_deg, spin);
        }
    }
    best
}

fn nearest_milkweed_halo_vec(
    lat: f64,
    lon: f64,
    patches: &[MilkweedPatch],
    aridity_index: f64,
    halo_deg: f64,
    spin: f64,
) -> (f64, f64) {
    let mut best_score = f64::MAX;
    let mut best = (0.0, 0.0);
    let drought_scale = (1.0 - 0.55 * aridity_index).clamp(0.25, 1.0);
    for p in patches {
        let dlat = p.lat - lat;
        let dlon = p.lon - lon;
        let d2 = (dlat * dlat + dlon * dlon).max(1e-6);
        let eff_area = (p.area_ha * drought_scale).max(1.0);
        let weighted = d2 / eff_area.sqrt();
        if weighted < best_score {
            best_score = weighted;
            best = halo_steer(dlat, dlon, halo_deg, spin);
        }
    }
    best
}

fn deterministic_unit(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1_u64 << 53) as f64)
}

pub fn default_synoptic_nodes() -> Vec<SynopticNodeWeather> {
    default_synoptic_nodes_for_region(FlywayRegion::PacificAmericas, Cmip6Scenario::BaselineHistorical)
}

pub fn default_synoptic_nodes_for_region(
    region: FlywayRegion,
    ssp: Cmip6Scenario,
) -> Vec<SynopticNodeWeather> {
    let waypoints = regional_synoptic_waypoints(region);
    let sm = (0.28 * (1.0 - 0.55 * ssp.canonical_aridity_index())).clamp(0.04, 0.45);
    let base_pm = match ssp {
        Cmip6Scenario::Ssp585High => 42.0,
        Cmip6Scenario::Ssp370Regional => 26.0,
        _ => 11.5,
    };

    waypoints
        .into_iter()
        .enumerate()
        .map(|(idx, (name, lat, lon))| SynopticNodeWeather {
            name: name.to_string(),
            lat,
            lon,
            temp_2m_c: 15.8 - (lat.abs() - 38.0) * 0.35,
            precip_mm_hr: if idx == 0 { 0.2 } else { 0.0 },
            wind_80m_mps: 4.2 + (idx as f64) * 0.35,
            wind_80m_deg: 335.0,
            wind_850_mps: 8.4 + (idx as f64) * 0.45,
            wind_850_deg: 345.0,
            pm25_ugm3: base_pm,
            aod_550nm: 0.14,
            soil_moisture_m3m3: sm,
            cmip6_delta_t_c: ssp.canonical_delta_t_c(),
        })
        .collect()
}
