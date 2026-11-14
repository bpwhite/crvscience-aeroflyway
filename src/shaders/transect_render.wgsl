// AeroFlyway v0.6.0 — Zero-Copy Instanced Vertical Airspace Transect Shader (`transect_render.wgsl`)
// Projects up to 100,000 GPU agents onto the A -> B cross-section (`0-4,200 m ASL`) across all
// 5 global flyway regions, highlighting 4,000 m smoke climbs (coral) and <300 m skyglow entrapment (magenta).

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

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read> agents: array<GpuAgent>;

struct TransectOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn quad_uv(v_idx: u32) -> vec2<f32> {
    switch v_idx {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0,  1.0); }
    }
}

fn terrain_elev_m(lat: f32, lon: f32) -> f32 {
    if ((lon >= -126.0) && (lon <= -115.0) && (lat >= 32.0) && (lat <= 51.0)) {
        if (lon < -124.18) { return 0.0; }
        let cr_x = (lon + 123.62) / 0.24;
        let coast_range = 620.0 * exp(-(cr_x * cr_x)) * (0.65 + 0.35 * abs(sin((lat - 42.0) * 2.1)));
        var valley_floor = 180.0;
        if ((lon > -123.45) && (lon < -122.55) && (lat > 43.8) && (lat < 47.5)) {
            valley_floor = 68.0;
        }
        let cas_x = (lon + 121.85) / 0.42;
        let cascade_ridge = 1850.0 * exp(-(cas_x * cas_x));
        let hd_y = (lat - 45.374) / 0.12;
        let hd_x = (lon + 121.696) / 0.14;
        let hood = 1550.0 * exp(-(hd_y * hd_y + hd_x * hd_x));
        let gc_y = (lat - 45.65) / 0.11;
        let gorge_cut = 1.0 - 0.82 * exp(-(gc_y * gc_y));
        var east_plateau = 0.0;
        if (lon > -121.3) {
            let ep_x = (lon + 121.3) / 0.6;
            east_plateau = 980.0 * (1.0 - exp(-(ep_x * ep_x)));
        }
        return clamp(valley_floor + coast_range + (cascade_ridge + hood) * gorge_cut + east_plateau, 0.0, 4350.0);
    }
    if ((lon >= -105.0) && (lon < -90.0) && (lat >= 25.0) && (lat <= 48.0)) {
        return clamp(max(-lon - 94.0, 0.0) * 105.0, 15.0, 1650.0);
    }
    if ((lon >= -83.0) && (lon <= -68.0) && (lat >= 24.0) && (lat <= 46.0)) {
        let app_x = (lon + 78.6 + (lat - 38.0) * 0.45) / 0.95;
        return clamp(45.0 + 920.0 * exp(-(app_x * app_x)), 0.0, 2100.0);
    }
    if ((lon >= -12.0) && (lon <= 18.0) && (lat >= 33.0) && (lat <= 58.0)) {
        let pyr_y = (lat - 42.8) / 0.55;
        let pyr_x = (lon + 0.5) / 2.2;
        let bet_y = (lat - 37.05) / 0.45;
        let bet_x = (lon + 3.4) / 1.6;
        return clamp(75.0 + 1950.0 * exp(-(pyr_y * pyr_y + pyr_x * pyr_x)) + 1280.0 * exp(-(bet_y * bet_y + bet_x * bet_x)), 0.0, 3400.0);
    }
    if ((lon >= 98.0) && (lon <= 127.0) && (lat >= 20.0) && (lat <= 50.0)) {
        var step_m = 38.0;
        if ((lat > 41.5) || (lon < 112.0)) {
            let mg = max(lat - 39.5, 0.0) / 3.2;
            step_m = 1180.0 * (1.0 - exp(-(mg * mg)));
        }
        return clamp(step_m, 0.0, 3200.0);
    }
    return 120.0;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vert_idx: u32,
    @builtin(instance_index) inst_idx: u32,
) -> TransectOut {
    var out: TransectOut;
    if (inst_idx >= sim.agent_count) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let ag = agents[inst_idx];
    let taxon_bit = 1u << min(ag.taxon, 2u);
    if ((sim.layer_bits & taxon_bit) == 0u) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let dlat = sim.transect_lat_b - sim.transect_lat_a;
    let dlon = sim.transect_lon_b - sim.transect_lon_a;
    let seg_sq = max(dlat * dlat + dlon * dlon, 1e-8);
    let t_proj = ((ag.lat - sim.transect_lat_a) * dlat + (ag.lon - sim.transect_lon_a) * dlon) / seg_sq;

    if ((t_proj < 0.0) || (t_proj > 1.0)) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let proj_lat = sim.transect_lat_a + t_proj * dlat;
    let proj_lon = sim.transect_lon_a + t_proj * dlon;

    let ky = (ag.lat - proj_lat) * 111.32;
    let kx = (ag.lon - proj_lon) * 111.32 * max(cos(proj_lat * 0.01745329252), 0.2);
    let cross_km = sqrt(max(kx * kx + ky * ky, 0.0));
    if (cross_km > sim.transect_half_width_km) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    let alt_asl = terrain_elev_m(proj_lat, proj_lon) + ag.alt_agl_m;
    let fy = clamp(alt_asl / 4200.0, 0.0, 1.0);

    let uv = quad_uv(vert_idx);
    let r_ndc_x = 2.4 / 420.0;
    let r_ndc_y = 2.4 / 90.0;
    let ndc_x = (t_proj * 2.0 - 1.0) + uv.x * r_ndc_x;
    let ndc_y = (fy * 2.0 - 1.0) + uv.y * r_ndc_y;

    var col = vec4<f32>(0.22, 0.74, 0.97, 0.70);
    if (ag.taxon == 0u && (ag.flags & 8u) != 0u) {
        col = vec4<f32>(0.99, 0.42, 0.78, 0.92);
    } else if (ag.taxon == 0u && (ag.flags & 4u) != 0u) {
        col = vec4<f32>(0.99, 0.52, 0.24, 0.88);
    } else if (ag.taxon == 1u) {
        col = vec4<f32>(0.75, 0.52, 0.99, 0.78);
    } else if (ag.taxon == 2u) {
        col = vec4<f32>(0.99, 0.75, 0.15, 0.82);
    }

    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local_uv = uv;
    out.color = col;
    return out;
}

@fragment
fn fs_main(in_frag: TransectOut) -> @location(0) vec4<f32> {
    let r2 = dot(in_frag.local_uv, in_frag.local_uv);
    if (r2 > 1.0) {
        discard;
    }
    let soft = clamp((1.0 - r2) * 2.4, 0.0, 1.0);
    let a = in_frag.color.a * soft;
    return vec4<f32>(in_frag.color.rgb * a, a);
}
