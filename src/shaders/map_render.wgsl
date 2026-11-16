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

    cam_pitch_rad: f32,
    cam_bearing_rad: f32,
    cam_alt_exag: f32,
    _cam_pad: f32,
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
    @location(3) wing_phase: f32,
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

    let ge = (dlon / 360.0) * sim.zoom_scale_px;
    let lat_rad = clamp(ag.lat, -85.0511, 85.0511) * 0.01745329252;
    let merc_n = log(tan(0.7853981634 + 0.5 * lat_rad));
    let gn = ((merc_n - sim.center_merc_n) / 6.2831853072) * sim.zoom_scale_px;

    let cb = cos(sim.cam_bearing_rad);
    let sb = sin(sim.cam_bearing_rad);
    let cp = cos(sim.cam_pitch_rad);
    let sp = sin(sim.cam_pitch_rad);

    let rx = ge * cb - gn * sb;
    let fy = ge * sb + gn * cb;
    let elev_z = (ag.alt_agl_m / 36.0) * max(sim.cam_alt_exag, 0.0);
    let cam_d = max(sim.canvas_h_pts * 1.15, 680.0);
    let depth = fy * sp - elev_z * cp;
    let persp = cam_d / max(cam_d + depth, cam_d * 0.22);

    let sx_pts = rx * persp;
    let sy_pts = -((fy * cp + elev_z * sp) * persp);

    let half_w = max(sim.canvas_w_pts * 0.5, 1.0);
    let half_h = max(sim.canvas_h_pts * 0.5, 1.0);

    if ((abs(sx_pts) > half_w + 48.0) || (abs(sy_pts) > half_h + 48.0)) {
        out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.local_uv = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.taxon = ag.taxon;
        out.wing_phase = 0.0;
        return out;
    }

    let speed = sqrt(max(ag.vel_u_mps * ag.vel_u_mps + ag.vel_v_mps * ag.vel_v_mps, 0.0));
    var vu = sin(sim.heading_rad);
    var vv = cos(sim.heading_rad);
    if (speed > 0.15) {
        vu = ag.vel_u_mps / speed;
        vv = ag.vel_v_mps / speed;
    }
    let rot_u = vu * cb - vv * sb;
    let rot_v = (vu * sb + vv * cb) * cp;
    let fwd_raw = vec2<f32>(rot_u, -rot_v);
    let fwd_len = max(length(fwd_raw), 1e-4);
    let fwd_screen = fwd_raw / fwd_len;
    let right_screen = vec2<f32>(-fwd_screen.y, fwd_screen.x);

    let is_roost = (ag.flags & 16u) != 0u;
    let zoom_lvl = log2(max(sim.zoom_scale_px / 256.0, 1.0));
    let quad_diam_pt = clamp(6.8 + max(zoom_lvl - 3.5, 0.0) * 0.72, 6.5, 10.5) * select(1.0, 0.80, is_roost);
    let base_px = quad_diam_pt * 0.56 * clamp(persp, 0.78, 1.25);

    let uv = quad_corner(vert_idx);
    let along = uv.y * base_px;
    let across = uv.x * base_px;

    let pt_x = sx_pts + right_screen.x * across + fwd_screen.x * along;
    let pt_y = sy_pts + right_screen.y * across + fwd_screen.y * along;

    let ndc_x = pt_x / half_w;
    let ndc_y = -pt_y / half_h;

    var col = vec4<f32>(0.22, 0.74, 0.97, 0.82);
    if (ag.taxon == 0u) {
        if ((ag.flags & 8u) != 0u) {
            col = vec4<f32>(0.98, 0.44, 0.76, 0.82);
        } else if ((ag.flags & 4u) != 0u) {
            col = vec4<f32>(0.98, 0.54, 0.24, 0.82);
        } else if ((ag.flags & 1u) != 0u) {
            col = vec4<f32>(0.28, 0.90, 0.72, 0.82);
        } else {
            // Cyan #38BDF8
            col = vec4<f32>(0.22, 0.74, 0.97, 0.82);
        }
    } else if (ag.taxon == 1u) {
        if ((ag.flags & 2u) != 0u) {
            col = vec4<f32>(0.98, 0.30, 0.44, 0.82);
        } else {
            // Violet #C084FC
            col = vec4<f32>(0.75, 0.52, 0.99, 0.82);
        }
    } else {
        // Monarch Gold/Orange #FB923C
        let gen_f = clamp(f32(ag.generation) * 0.25, 0.25, 1.0);
        col = vec4<f32>(0.98, 0.62 - 0.10 * gen_f, 0.22, 0.82);
    }
    if (is_roost) {
        col.a = 0.22;
    }

    out.clip_pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.local_uv = uv;
    out.color = col;
    out.taxon = ag.taxon;
    out.wing_phase = sin(sim.hour_of_day * 16.0 + f32(inst_idx) * 1.91);
    return out;
}

fn sd_capsule_pt(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

@fragment
fn fs_main(in_frag: VsOut) -> @location(0) vec4<f32> {
    let ax = abs(in_frag.local_uv.x);
    let y = in_frag.local_uv.y;
    let p = vec2<f32>(ax, y);

    var sdf = 1.0;
    if (in_frag.taxon == 0u) {
        // BIRD: Swept chevron wings + wingbeat flex + fuselage + forked swallow tail
        let flex = in_frag.wing_phase * 0.14;
        let wing_d = sd_capsule_pt(p, vec2<f32>(0.04, 0.24), vec2<f32>(0.88, -0.30 + flex)) - (0.14 - 0.085 * ax);
        let body_d = sd_capsule_pt(p, vec2<f32>(0.0, -0.36), vec2<f32>(0.0, 0.64)) - 0.105;
        let tail_d = sd_capsule_pt(p, vec2<f32>(0.02, -0.32), vec2<f32>(0.24, -0.78)) - 0.045;
        sdf = min(wing_d, min(body_d, tail_d));
    } else if (in_frag.taxon == 1u) {
        // BAT: Twin pointed ears + 3 scalloped trailing wing cutouts
        let body_d = length(vec2<f32>(ax / 0.13, (y - 0.06) / 0.36)) - 1.0;
        let ear_d = sd_capsule_pt(p, vec2<f32>(0.07, 0.34), vec2<f32>(0.15, 0.68)) - 0.045;
        var wing_d = 1.0;
        if (ax <= 0.90) {
            let y_lead = 0.34 - 0.24 * pow(ax, 1.35);
            let scallop = 0.13 * abs(sin(ax * 10.4));
            let y_trail = -0.34 + 0.14 * ax + scallop;
            let mid_y = 0.5 * (y_lead + y_trail);
            let half_span = max(0.5 * (y_lead - y_trail), 0.02);
            let dy = abs(y - mid_y) - half_span;
            let dx = ax - 0.88;
            wing_d = max(dy, dx * 0.7);
        }
        sdf = min(wing_d, min(body_d * 0.14, ear_d));
    } else {
        // LEPIDOPTERA: 4-lobed forewing/hindwing butterfly + slender thorax
        let body_d = sd_capsule_pt(p, vec2<f32>(0.0, -0.52), vec2<f32>(0.0, 0.48)) - 0.058;
        let ant_d = sd_capsule_pt(p, vec2<f32>(0.02, 0.46), vec2<f32>(0.16, 0.74)) - 0.024;
        let fore_p = vec2<f32>((ax - 0.44) / 0.40, (y - 0.24 - 0.10 * ax) / 0.34);
        let fore_d = (length(fore_p) - 1.0) * 0.34;
        let hind_p = vec2<f32>((ax - 0.33) / 0.30, (y + 0.30) / 0.28);
        let hind_d = (length(hind_p) - 1.0) * 0.28;
        sdf = min(min(fore_d, hind_d), min(body_d, ant_d));
    }

    if (sdf > 0.06) {
        discard;
    }
    let aa = 1.0 - smoothstep(-0.04, 0.06, sdf);
    let alpha = in_frag.color.a * aa;
    return vec4<f32>(in_frag.color.rgb * alpha, alpha);
}
