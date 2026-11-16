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

// AeroFlyway v0.6.0 — Zero-Copy Instanced Map Render Shader (`map_render.wgsl`)
// Renders up to 100,000 VRAM-resident agents across any global flyway using relative-to-center
// spherical Web Mercator projection, highlighting rain fallout (emerald), smoke climb (coral),
// and VIIRS nighttime skyglow disorientation (magenta-rose `flags & 8u`).

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

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) taxon: u32,
};

fn quad_corner(v_idx: u32) -> vec2<f32> {
    switch v_idx {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>( 1.0,  1.0); }
        case 3u: { return vec2<f32>(-1.0, -1.0); }
        case 4u: { return vec2<f32>( 1.0,  1.0); }
        default: { return vec2<f32>(-1.0,  1.0); }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vert_idx: u32,
    @builtin(instance_index) inst_idx: u32,
) -> VsOut {
    var out: VsOut;
    if (inst_idx >= sim.agent_count) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.taxon = 0u;
        return out;
    }

    let ag = agents[inst_idx];
    let taxon_bit = 1u << min(ag.taxon, 2u);
    if ((sim.layer_bits & taxon_bit) == 0u) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.taxon = ag.taxon;
        return out;
    }

    var dlon = ag.lon - sim.center_lon;
    if (dlon > 180.0) { dlon = dlon - 360.0; }
    if (dlon < -180.0) { dlon = dlon + 360.0; }

    let dx_pts = (dlon / 360.0) * sim.zoom_scale_px;
    let lat_rad = clamp(ag.lat, -85.0511, 85.0511) * 0.01745329252;
    let merc_n = log(tan(0.7853981634 + 0.5 * lat_rad));
    let dy_pts = -((merc_n - sim.center_merc_n) / 6.2831853072) * sim.zoom_scale_px;

    let half_w = max(sim.canvas_w_pts * 0.5, 1.0);
    let half_h = max(sim.canvas_h_pts * 0.5, 1.0);

    if ((abs(dx_pts) > half_w + 24.0) || (abs(dy_pts) > half_h + 24.0)) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.taxon = ag.taxon;
        return out;
    }

    let speed = sqrt(max(ag.vel_u_mps * ag.vel_u_mps + ag.vel_v_mps * ag.vel_v_mps, 0.0));
    var fwd_screen = vec2<f32>(0.0, -1.0);
    if (speed > 0.15) {
        fwd_screen = vec2<f32>(ag.vel_u_mps / speed, -ag.vel_v_mps / speed);
    }
    let right_screen = vec2<f32>(-fwd_screen.y, fwd_screen.x);

    let show_trails = (sim.layer_bits & 32u) != 0u;
    var tail_stretch = 1.0;
    if (show_trails) {
        tail_stretch = clamp(1.0 + speed * 0.11, 1.0, 2.4);
    }

    let base_px = clamp(1.15 + log2(max(sim.zoom_scale_px / 256.0, 1.0)) * 0.16, 1.2, 2.8);
    let uv = quad_corner(vert_idx);

    var along = uv.y * base_px;
    if (uv.y < 0.0) {
        along = uv.y * base_px * tail_stretch;
    }
    let across = uv.x * base_px;

    let pt_x = dx_pts + right_screen.x * across + fwd_screen.x * along;
    let pt_y = dy_pts + right_screen.y * across + fwd_screen.y * along;

    let ndc_x = pt_x / half_w;
    let ndc_y = -pt_y / half_h;

    var col = vec4<f32>(0.22, 0.74, 0.97, 0.78);
    if (ag.taxon == 0u) {
        if ((ag.flags & 8u) != 0u) {
            col = vec4<f32>(0.99, 0.42, 0.78, 0.92);
        } else if ((ag.flags & 4u) != 0u) {
            col = vec4<f32>(0.99, 0.52, 0.24, 0.90);
        } else if ((ag.flags & 1u) != 0u) {
            col = vec4<f32>(0.30, 0.95, 0.75, 0.90);
        } else {
            col = vec4<f32>(0.22, 0.74, 0.97, 0.76);
        }
    } else if (ag.taxon == 1u) {
        if ((ag.flags & 2u) != 0u) {
            col = vec4<f32>(0.98, 0.28, 0.42, 0.95);
        } else {
            col = vec4<f32>(0.75, 0.52, 0.99, 0.80);
        }
    } else {
        let gen_f = clamp(f32(ag.generation) * 0.25, 0.25, 1.0);
        col = vec4<f32>(0.99, 0.82 - 0.26 * gen_f, 0.16, 0.84);
    }

    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local_uv = uv;
    out.color = col;
    out.taxon = ag.taxon;
    return out;
}

@fragment
fn fs_main(in_frag: VsOut) -> @location(0) vec4<f32> {
    let u = in_frag.local_uv.x;
    let v = in_frag.local_uv.y;

    var shape_d = 0.0;
    if (in_frag.taxon == 0u) {
        let nose = abs(u) * 0.85 + max(v, -0.35 * abs(u));
        shape_d = max(abs(u), nose);
    } else if (in_frag.taxon == 1u) {
        let wing = u * u * 0.75 + (v - 0.15 * (1.0 - u * u)) * (v - 0.15 * (1.0 - u * u));
        shape_d = sqrt(max(wing, 0.0));
    } else {
        shape_d = abs(u) * 0.92 + abs(v) * 0.92;
    }

    if (shape_d > 1.0) {
        discard;
    }

    let edge_fade = clamp((1.0 - shape_d) * 3.2, 0.0, 1.0);
    var tail_fade = 1.0;
    if (v < 0.0) {
        tail_fade = clamp(1.0 + v * 0.78, 0.14, 1.0);
    }

    let alpha = in_frag.color.a * edge_fade * tail_fade;
    return vec4<f32>(in_frag.color.rgb * alpha, alpha);
}
