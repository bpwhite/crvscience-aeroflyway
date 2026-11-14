// AeroFlyway v0.6.0 — WGSL Compute Shader (`@workgroup_size(256)`)
// Steps up to 100,000 VRAM-resident multi-taxa agents across any global flyway region:
// - Inverse-distance Open-Meteo + CMIP6 + CAMS PM2.5 interpolation
// - Continuous 365-day DOY photoperiod-vs-GDD trophic mismatch & wetland drought refueling penalty
// - VIIRS nighttime urban skyglow phototaxis (`kind == 2u`), low-canopy disorientation (`flags | 8u`),
//   and BirdCast / Audubon Lights-Out building-collision mitigation (Van Doren et al. 2021, PNAS)
// - NEXRAD WSR-88D & Aloft OPERA vertical radar (`VPTS`) cruise-altitude nudging
// - Megafire smoke plume disruption (`PM2.5 >= 160 ug/m^3`: staging halts & 4,000 m over-plume climbs)
// - Chiroptera dusk/nocturnal pulse & 3D wind-turbine rotor collision / smart curtailment checks

struct SimUniforms {
    center_lat: f32,
    center_lon: f32,
    center_merc_n: f32,
    zoom_scale_px: f32,

    canvas_w_pts: f32,
    canvas_h_pts: f32,
    transect_lat_a: f32,
    transect_lon_a: f32,

    transect_lat_b: f32,
    transect_lon_b: f32,
    transect_total_km: f32,
    transect_half_width_km: f32,

    dt_hours: f32,
    hour_of_day: f32,
    heading_rad: f32,
    wind_850_mult: f32,

    precip_front_mm: f32,
    turbine_cut_in_mps: f32,
    temp_offset_c: f32,
    enkf_gain: f32,

    agent_count: u32,
    layer_bits: u32,
    tick_seed: u32,
    reseed_season_code: u32,

    num_weather_nodes: u32,
    num_turbines: u32,
    num_attractors: u32,
    region_code: u32,

    day_of_year: f32,
    smoke_pm25_ugm3: f32,
    corridor_min_lat: f32,
    corridor_max_lat: f32,

    corridor_min_lon: f32,
    corridor_max_lon: f32,
    drought_aridity_index: f32,
    trophic_mismatch_days: f32,

    lights_out_pct: f32,
    radar_peak_alt_m: f32,
    radar_nudge_weight: f32,
    skyglow_intensity_mult: f32,
};

struct GpuAgent {
    lat: f32,
    lon: f32,
    alt_agl_m: f32,
    vel_u_mps: f32,

    vel_v_mps: f32,
    prev_lat: f32,
    prev_lon: f32,
    energy: f32,

    gdd_accum: f32,
    taxon: u32,
    generation: u32,
    flags: u32,
};

struct GpuWeatherNode {
    lat: f32,
    lon: f32,
    temp_2m_c: f32,
    precip_mm_hr: f32,

    u80_mps: f32,
    v80_mps: f32,
    u850_mps: f32,
    v850_mps: f32,

    pm25_ugm3: f32,
    aod_550nm: f32,
    soil_moisture_m3m3: f32,
    cmip6_delta_t_c: f32,
};

struct GpuTurbine {
    lat: f32,
    lon: f32,
    hub_height_m: f32,
    rotor_radius_m: f32,
};

struct GpuAttractor {
    lat: f32,
    lon: f32,
    weight: f32,
    kind: u32,
};

struct GpuTelemetryCounters {
    birds_airborne: atomic<u32>,
    birds_fallout: atomic<u32>,
    bats_flying: atomic<u32>,
    bat_rotor_hits: atomic<u32>,
    bat_baseline_hits: atomic<u32>,
    bfly_airborne: atomic<u32>,
    gdd_sum_tenths: atomic<u32>,
    max_north_lat_milli: atomic<u32>,
    birds_skyglow: atomic<u32>,
    urban_strikes: atomic<u32>,
    urban_baseline_strikes: atomic<u32>,
    pad_word: atomic<u32>,
};

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read_write> agents: array<GpuAgent>;
@group(0) @binding(2) var<storage, read> weather_nodes: array<GpuWeatherNode>;
@group(0) @binding(3) var<storage, read> turbines: array<GpuTurbine>;
@group(0) @binding(4) var<storage, read> attractors: array<GpuAttractor>;
@group(0) @binding(5) var<storage, read_write> counters: GpuTelemetryCounters;

fn pcg_hash(input_val: u32) -> u32 {
    let state = input_val * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_unit(seed_val: u32) -> f32 {
    return f32(pcg_hash(seed_val)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= sim.agent_count) {
        return;
    }

    var ag = agents[idx];

    let seed_a = idx * 1664525u + sim.tick_seed * 1013904223u + 17u;
    let seed_b = idx * 22695477u + sim.tick_seed * 69069u + 53u;
    let jitter_a = rand_unit(seed_a) - 0.5;
    let jitter_b = rand_unit(seed_b) - 0.5;

    let is_nocturnal = (sim.layer_bits & 8u) != 0u;
    let is_running = (sim.layer_bits & 16u) != 0u;
    let is_dusk = (sim.hour_of_day >= 17.5) && (sim.hour_of_day <= 23.8);

    let seasonal_mod_c = 6.5 * sin(0.017214206 * (sim.day_of_year - 110.0));

    var w_sum = 0.0;
    var temp_acc = 0.0;
    var precip_acc = 0.0;
    var pm25_acc = 0.0;
    var u80_acc = 0.0;
    var v80_acc = 0.0;
    var u850_acc = 0.0;
    var v850_acc = 0.0;

    let n_wx = sim.num_weather_nodes;
    for (var w_i: u32 = 0u; w_i < n_wx; w_i = w_i + 1u) {
        let wn = weather_nodes[w_i];
        let dlat = ag.lat - wn.lat;
        let dlon = ag.lon - wn.lon;
        let d2 = max(dlat * dlat + dlon * dlon, 0.02);
        let inv_d2 = 1.0 / d2;
        w_sum = w_sum + inv_d2;
        temp_acc = temp_acc + inv_d2 * wn.temp_2m_c;
        precip_acc = precip_acc + inv_d2 * wn.precip_mm_hr;
        pm25_acc = pm25_acc + inv_d2 * wn.pm25_ugm3;
        u80_acc = u80_acc + inv_d2 * wn.u80_mps;
        v80_acc = v80_acc + inv_d2 * wn.v80_mps;
        u850_acc = u850_acc + inv_d2 * wn.u850_mps;
        v850_acc = v850_acc + inv_d2 * wn.v850_mps;
    }

    var temp_2m = 15.0 + 0.35 * seasonal_mod_c + sim.temp_offset_c;
    var precip_local = 0.0;
    var pm25_local = max(10.0 + sim.smoke_pm25_ugm3, 0.0);
    var u80 = 1.0;
    var v80 = -3.2;
    var u850 = 2.2 * sim.wind_850_mult;
    var v850 = -7.0 * sim.wind_850_mult;

    if (w_sum > 1e-6) {
        let inv_w = 1.0 / w_sum;
        temp_2m = temp_acc * inv_w + 0.35 * seasonal_mod_c + sim.temp_offset_c;
        precip_local = precip_acc * inv_w;
        pm25_local = max(pm25_acc * inv_w + sim.smoke_pm25_ugm3, 0.0);
        u80 = u80_acc * inv_w;
        v80 = v80_acc * inv_w;
        u850 = u850_acc * inv_w * sim.wind_850_mult;
        v850 = v850_acc * inv_w * sim.wind_850_mult;
    }
    let ws80 = sqrt(max(u80 * u80 + v80 * v80, 0.0));
    let ws850 = sqrt(max(u850 * u850 + v850 * v850, 0.0));

    let dir_u = sin(sim.heading_rad);
    let dir_v = cos(sim.heading_rad);

    let mm_norm = sim.trophic_mismatch_days / 11.5;
    let trophic_overlap = clamp(exp(-0.5 * mm_norm * mm_norm), 0.08, 1.0);
    let refueling_mult = trophic_overlap * clamp(1.0 - 0.55 * sim.drought_aridity_index, 0.25, 1.0);

    if (is_running && sim.dt_hours > 0.0) {
        ag.prev_lat = ag.lat;
        ag.prev_lon = ag.lon;

        if (ag.taxon == 0u) {
            let total_precip = precip_local + sim.precip_front_mm;
            let severe_smoke = pm25_local >= 160.0;
            ag.flags = ag.flags & (~12u);
            if (severe_smoke) {
                ag.flags = ag.flags | 4u;
            }
            let smoke_halt = severe_smoke && ((idx % 5u) < 2u);

            if (is_nocturnal && (total_precip < 2.2) && (!smoke_halt) && (ag.energy > 0.12)) {
                ag.flags = ag.flags & (~1u);
                var smoke_drag = 1.0;
                var lat_disp = 2.4;
                if (severe_smoke) {
                    smoke_drag = 0.78;
                    lat_disp = 4.8;
                }
                let airspeed = 11.5 * sim.enkf_gain * smoke_drag;

                // Evaluate VIIRS urban skyglow dome phototaxis (`kind == 2u`, `weight == eff_radiance_nw`)
                var best_glow = 0.0;
                var glow_u = 0.0;
                var glow_v = 0.0;
                if (!severe_smoke) {
                    for (var g_i: u32 = 0u; g_i < sim.num_attractors; g_i = g_i + 1u) {
                        let dome = attractors[g_i];
                        if (dome.kind == 2u && dome.weight > 4.0) {
                            let dly = dome.lat - ag.lat;
                            let dlx = dome.lon - ag.lon;
                            let cos_c = max(abs(cos(ag.lat * 0.01745329252)), 0.2);
                            let dist_km = 111.32 * sqrt(dly * dly + (dlx * cos_c) * (dlx * cos_c));
                            if (dist_km < 92.0) {
                                let decay = exp(-0.5 * (dist_km / 46.0) * (dist_km / 46.0));
                                let eff_rad = dome.weight * decay;
                                if (eff_rad > best_glow) {
                                    best_glow = eff_rad;
                                    let inv_d = 1.0 / max(sqrt(dlx * dlx + dly * dly), 1e-5);
                                    glow_u = dlx * inv_d;
                                    glow_v = dly * inv_d;
                                }
                            }
                        }
                    }
                }

                if (best_glow > 14.0) {
                    ag.flags = ag.flags | 8u;
                    let pull_s = clamp(best_glow / 110.0, 0.12, 0.88);
                    ag.vel_u_mps = airspeed * ((1.0 - 0.55 * pull_s) * dir_u + 0.75 * pull_s * glow_u) + 0.7 * u850 + jitter_a * 4.2;
                    ag.vel_v_mps = airspeed * ((1.0 - 0.55 * pull_s) * dir_v + 0.75 * pull_s * glow_v) + 0.7 * v850 + jitter_b * 4.2;
                    let target_canopy = 85.0 + 115.0 * (1.0 - pull_s);
                    ag.alt_agl_m = clamp(0.68 * ag.alt_agl_m + 0.32 * target_canopy + jitter_a * 24.0, 28.0, 320.0);
                    ag.energy = max(ag.energy - 0.025 * sim.dt_hours, 0.04);

                    // Non-linear Lights-Out vs unmitigated collision counter (Van Doren et al. 2021)
                    let curfew_dim = max(1.0 - 0.88 * clamp(sim.lights_out_pct / 100.0, 0.0, 1.0), 0.10);
                    let unmit_glow = best_glow / curfew_dim;
                    let wind_amp = clamp(1.0 + 0.035 * ws850, 1.0, 1.6);
                    let p_base = 0.0032 * pow(max(unmit_glow / 100.0, 0.05), 1.68) * wind_amp * min(sim.dt_hours, 1.2);
                    let p_eff = 0.0032 * pow(max(best_glow / 100.0, 0.05), 1.68) * wind_amp * min(sim.dt_hours, 1.2);
                    let u_roll = rand_unit(seed_b ^ 0x5A5AA5A5u);
                    if (u_roll < p_base) {
                        atomicAdd(&counters.urban_baseline_strikes, 1u);
                    }
                    if (u_roll < p_eff) {
                        atomicAdd(&counters.urban_strikes, 1u);
                    }
                } else {
                    ag.vel_u_mps = airspeed * dir_u + u850 + jitter_a * lat_disp;
                    ag.vel_v_mps = airspeed * dir_v + v850 + jitter_b * lat_disp;
                    if (severe_smoke) {
                        ag.alt_agl_m = clamp(ag.alt_agl_m + 240.0 * max(sim.dt_hours, 0.2) + jitter_a * 65.0, 1850.0, 3850.0);
                        ag.energy = max(ag.energy - 0.026 * sim.dt_hours, 0.04);
                    } else {
                        let radar_agl = clamp(sim.radar_peak_alt_m - 140.0, 380.0, 1950.0);
                        let free_agl = clamp(ag.alt_agl_m + jitter_a * 60.0, 380.0, 1950.0);
                        let nw = clamp(sim.radar_nudge_weight, 0.0, 1.0) * 0.22;
                        ag.alt_agl_m = (1.0 - nw) * free_agl + nw * radar_agl;
                        ag.energy = max(ag.energy - 0.018 * sim.dt_hours, 0.05);
                    }
                }
            } else {
                if ((total_precip >= 2.2) || smoke_halt) {
                    ag.flags = ag.flags | 1u;
                } else {
                    ag.flags = ag.flags & (~1u);
                }
                var best_d2 = 1e9;
                var pull_u = 0.0;
                var pull_v = 0.0;
                for (var a_i: u32 = 0u; a_i < sim.num_attractors; a_i = a_i + 1u) {
                    let att = attractors[a_i];
                    if (att.kind == 0u) {
                        let dl_y = att.lat - ag.lat;
                        let dl_x = att.lon - ag.lon;
                        let dist2 = dl_x * dl_x + dl_y * dl_y;
                        if (dist2 < best_d2 && dist2 > 1e-7) {
                            best_d2 = dist2;
                            let inv_l = 1.0 / sqrt(dist2);
                            pull_u = dl_x * inv_l;
                            pull_v = dl_y * inv_l;
                        }
                    }
                }
                ag.vel_u_mps = pull_u * 2.6 + jitter_a * 0.85;
                ag.vel_v_mps = pull_v * 2.6 + jitter_b * 0.85;
                ag.alt_agl_m = clamp(ag.alt_agl_m * 0.74, 15.0, 140.0);
                ag.energy = min(ag.energy + 0.048 * refueling_mult * sim.dt_hours, 1.0);
            }
        } else if (ag.taxon == 1u) {
            ag.flags = ag.flags & (~2u);
            if ((is_nocturnal || is_dusk) && (temp_2m > 7.5)) {
                var pulse_boost = 0.55;
                if (ws80 < 6.8) {
                    pulse_boost = 1.35;
                }
                var smoke_supp = 1.0;
                if (pm25_local >= 160.0) {
                    smoke_supp = 0.62;
                }
                let airspeed = 6.8 * pulse_boost * smoke_supp * sim.enkf_gain;
                ag.vel_u_mps = airspeed * dir_u + 0.45 * u80 + jitter_a * 1.9;
                ag.vel_v_mps = airspeed * dir_v + 0.45 * v80 + jitter_b * 1.9;
                ag.alt_agl_m = clamp(ag.alt_agl_m + jitter_b * 30.0, 38.0, 340.0);

                for (var t_i: u32 = 0u; t_i < sim.num_turbines; t_i = t_i + 1u) {
                    let tb = turbines[t_i];
                    if ((abs(ag.lat - tb.lat) < 0.032) && (abs(ag.lon - tb.lon) < 0.042)) {
                        let z_lo = max(tb.hub_height_m - tb.rotor_radius_m, 15.0);
                        let z_hi = tb.hub_height_m + tb.rotor_radius_m;
                        if ((ag.alt_agl_m >= z_lo) && (ag.alt_agl_m <= z_hi)) {
                            ag.flags = ag.flags | 2u;
                            if (rand_unit(seed_a ^ 0xA5A55A5Au) < 0.08) {
                                if ((ws80 >= 3.0) && (ws80 <= 25.0)) {
                                    atomicAdd(&counters.bat_baseline_hits, 1u);
                                }
                                if ((ws80 >= sim.turbine_cut_in_mps) && (ws80 <= 25.0)) {
                                    atomicAdd(&counters.bat_rotor_hits, 1u);
                                }
                            }
                            break;
                        }
                    }
                }
            } else {
                ag.vel_u_mps = jitter_a * 0.25;
                ag.vel_v_mps = jitter_b * 0.25;
                ag.alt_agl_m = clamp(ag.alt_agl_m * 0.65, 8.0, 35.0);
            }
        } else {
            let gdd_hr = max(temp_2m - 11.5, 0.0) / 24.0;
            ag.gdd_accum = ag.gdd_accum + gdd_hr * sim.dt_hours;
            if (ag.gdd_accum >= 352.0) {
                ag.gdd_accum = ag.gdd_accum - 352.0;
                ag.generation = (ag.generation % 4u) + 1u;
            }

            if ((!is_nocturnal) && (temp_2m >= 12.5) && (temp_2m <= 35.5) && (pm25_local < 220.0)) {
                var best_score = 1e9;
                var mw_u = 0.0;
                var mw_v = 0.0;
                for (var m_i: u32 = 0u; m_i < sim.num_attractors; m_i = m_i + 1u) {
                    let patch_item = attractors[m_i];
                    if (patch_item.kind == 1u) {
                        let dl_y = patch_item.lat - ag.lat;
                        let dl_x = patch_item.lon - ag.lon;
                        let d2 = max(dl_x * dl_x + dl_y * dl_y, 1e-6);
                        let score = d2 / max(patch_item.weight, 0.5);
                        if (score < best_score) {
                            best_score = score;
                            let inv_l = 1.0 / sqrt(d2);
                            mw_u = dl_x * inv_l;
                            mw_v = dl_y * inv_l;
                        }
                    }
                }
                let airspeed = 4.2 * sim.enkf_gain;
                ag.vel_u_mps = airspeed * (0.65 * dir_u + 0.35 * mw_u) + 0.30 * u80 + jitter_a * 1.4;
                ag.vel_v_mps = airspeed * (0.65 * dir_v + 0.35 * mw_v) + 0.30 * v80 + jitter_b * 1.4;
                ag.alt_agl_m = clamp(18.0 + (temp_2m - 12.0) * 4.5 + jitter_a * 18.0, 5.0, 145.0);
            } else {
                ag.vel_u_mps = 0.0;
                ag.vel_v_mps = 0.0;
                ag.alt_agl_m = 4.0;
            }
        }

        let dt_sec = sim.dt_hours * 3600.0;
        let dlat_deg = (ag.vel_v_mps * dt_sec) / 111320.0;
        let cos_lat = max(abs(cos(ag.lat * 0.01745329252)), 0.2);
        let dlon_deg = (ag.vel_u_mps * dt_sec) / (111320.0 * cos_lat);

        var next_lat = ag.lat + dlat_deg;
        var next_lon = ag.lon + dlon_deg;

        if (next_lat < sim.corridor_min_lat) {
            next_lat = sim.corridor_max_lat - rand_unit(seed_b) * 1.2;
            ag.prev_lat = next_lat;
        } else if (next_lat > sim.corridor_max_lat) {
            next_lat = sim.corridor_min_lat + rand_unit(seed_b) * 1.2;
            ag.prev_lat = next_lat;
        }
        next_lon = clamp(next_lon, sim.corridor_min_lon, sim.corridor_max_lon);

        ag.lat = next_lat;
        ag.lon = next_lon;
        agents[idx] = ag;
    }

    if (ag.taxon == 0u) {
        if ((ag.alt_agl_m > 160.0) || ((ag.flags & 8u) != 0u)) {
            atomicAdd(&counters.birds_airborne, 1u);
        }
        if ((ag.flags & 1u) != 0u) {
            atomicAdd(&counters.birds_fallout, 1u);
        }
        if ((ag.flags & 8u) != 0u) {
            atomicAdd(&counters.birds_skyglow, 1u);
        }
    } else if (ag.taxon == 1u) {
        if (ag.alt_agl_m > 36.0) {
            atomicAdd(&counters.bats_flying, 1u);
        }
    } else {
        if (ag.alt_agl_m > 6.0) {
            atomicAdd(&counters.bfly_airborne, 1u);
        }
        let gdd_t = u32(clamp(ag.gdd_accum, 0.0, 1000.0));
        atomicAdd(&counters.gdd_sum_tenths, gdd_t);
        let lat_milli = u32(clamp(ag.lat * 1000.0, 0.0, 90000.0));
        atomicMax(&counters.max_north_lat_milli, lat_milli);
    }
}
