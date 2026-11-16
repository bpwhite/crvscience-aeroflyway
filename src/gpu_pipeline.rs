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

//! `wgpu` + WGSL Compute & Zero-Copy Instanced Rendering Pipeline (`gpu_pipeline.rs` — v0.6.0).
//! Manages up to `100,000` GPU-resident multi-taxa agents in VRAM (`4.8 MB`), executing:
//! 1. `@compute @workgroup_size(256)` global-region DOY phenology, CMIP6 trophic mismatch,
//!    VIIRS nighttime skyglow phototaxis & Lights-Out building-collision mitigation (`kind == 2u`),
//!    NEXRAD/Aloft `VPTS` radar altitude nudging, `PM2.5 >= 160 ug/m^3` smoke ceiling & rotor solver
//! 2. Zero-copy instanced map chevron + wake renderer (`draw(0..6, 0..N)`)
//! 3. Zero-copy instanced vertical airspace transect renderer (`draw(0..6, 0..N)`, `0-4,200 m ASL`)
//! 4. Asynchronous non-blocking 48-byte (`12 x u32`) atomic telemetry counter readback.

use crate::basemap::{
    landmarks_for_region, skyglow_domes_for_region, FlywayRegion, LandmarkKind, MilkweedPatch,
    WindTurbine,
};
use crate::ingest::SynopticNodeWeather;
use crate::map_engine::haversine_km;
use crate::physics::{
    annual_flyway_heading_deg, compute_trophic_mismatch_days, effective_skyglow_radiance_nw,
    SeasonPreset, SimulationParams,
};
use bytemuck::{Pod, Zeroable};
use eframe::wgpu::util::DeviceExt;
use eframe::{egui, egui_wgpu, wgpu};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub const MAX_GPU_AGENTS: u32 = 100_000;
pub const DEFAULT_GPU_AGENTS: u32 = 2_560;
const MAX_WEATHER_NODES: usize = 16;
const MAX_TURBINES: usize = 512;
const MAX_ATTRACTORS: usize = 256;

/// 44-scalar (`176-byte`, 11 x `vec4`) uniform block shared identically across `compute.wgsl`,
/// `map_render.wgsl`, and `transect_render.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSimUniforms {
    pub center_lat: f32,
    pub center_lon: f32,
    pub center_merc_n: f32,
    pub zoom_scale_px: f32,

    pub canvas_w_pts: f32,
    pub canvas_h_pts: f32,
    pub transect_lat_a: f32,
    pub transect_lon_a: f32,

    pub transect_lat_b: f32,
    pub transect_lon_b: f32,
    pub transect_total_km: f32,
    pub transect_half_width_km: f32,

    pub dt_hours: f32,
    pub hour_of_day: f32,
    pub heading_rad: f32,
    pub wind_850_mult: f32,

    pub precip_front_mm: f32,
    pub turbine_cut_in_mps: f32,
    pub temp_offset_c: f32,
    pub enkf_gain: f32,

    pub agent_count: u32,
    pub layer_bits: u32,
    pub tick_seed: u32,
    pub reseed_season_code: u32,

    pub num_weather_nodes: u32,
    pub num_turbines: u32,
    pub num_attractors: u32,
    pub region_code: u32,

    pub day_of_year: f32,
    pub smoke_pm25_ugm3: f32,
    pub corridor_min_lat: f32,
    pub corridor_max_lat: f32,

    pub corridor_min_lon: f32,
    pub corridor_max_lon: f32,
    pub drought_aridity_index: f32,
    pub trophic_mismatch_days: f32,

    pub lights_out_pct: f32,
    pub radar_peak_alt_m: f32,
    pub radar_nudge_weight: f32,
    pub skyglow_intensity_mult: f32,

    pub cam_pitch_rad: f32,
    pub cam_bearing_rad: f32,
    pub cam_alt_exag: f32,
    pub _cam_pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuAgent {
    pub lat: f32,
    pub lon: f32,
    pub alt_agl_m: f32,
    pub vel_u_mps: f32,

    pub vel_v_mps: f32,
    pub prev_lat: f32,
    pub prev_lon: f32,
    pub energy: f32,

    pub gdd_accum: f32,
    pub taxon: u32,
    pub generation: u32,
    pub flags: u32,
}

/// 12-scalar (`48-byte`, 3 x `vec4`) synoptic + CMIP6 + CAMS smoke node.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuWeatherNode {
    pub lat: f32,
    pub lon: f32,
    pub temp_2m_c: f32,
    pub precip_mm_hr: f32,

    pub u80_mps: f32,
    pub v80_mps: f32,
    pub u850_mps: f32,
    pub v850_mps: f32,

    pub pm25_ugm3: f32,
    pub aod_550nm: f32,
    pub soil_moisture_m3m3: f32,
    pub cmip6_delta_t_c: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuTurbine {
    pub lat: f32,
    pub lon: f32,
    pub hub_height_m: f32,
    pub rotor_radius_m: f32,
}

/// `kind`: `0 = refuge/roost`, `1 = host-plant patch`, `2 = VIIRS urban skyglow dome` (`weight = eff_radiance_nw`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuAttractor {
    pub lat: f32,
    pub lon: f32,
    pub weight: f32,
    pub kind: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuReadbackSummary {
    pub birds_airborne: u32,
    pub birds_fallout: u32,
    pub bats_flying: u32,
    pub bat_rotor_hits_step: u32,
    pub bat_baseline_hits_step: u32,
    pub bfly_airborne: u32,
    pub mean_gdd: f64,
    pub max_north_lat: f64,
    pub birds_skyglow_entrained: u32,
    pub urban_strikes_step: u32,
    pub urban_baseline_strikes_step: u32,
}

pub struct SharedReadbackHandle {
    pub latest: Mutex<Option<GpuReadbackSummary>>,
    pub map_in_flight: AtomicBool,
    pub map_ready: AtomicBool,
}

impl SharedReadbackHandle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            latest: Mutex::new(None),
            map_in_flight: AtomicBool::new(false),
            map_ready: AtomicBool::new(false),
        })
    }
}

pub struct FlywayGpuResources {
    pub uniform_buffer: wgpu::Buffer,
    pub agent_buffer: wgpu::Buffer,
    pub weather_buffer: wgpu::Buffer,
    pub turbine_buffer: wgpu::Buffer,
    pub attractor_buffer: wgpu::Buffer,
    pub counter_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,

    pub compute_pipeline: wgpu::ComputePipeline,
    pub compute_bind_group: wgpu::BindGroup,

    pub map_render_pipeline: wgpu::RenderPipeline,
    pub transect_render_pipeline: wgpu::RenderPipeline,
    pub render_bind_group: wgpu::BindGroup,

    pub readback: Arc<SharedReadbackHandle>,
    pub frame_counter: u64,
    pub current_reseed_key: u32,
}

impl FlywayGpuResources {
    pub fn initialize(
        render_state: &egui_wgpu::RenderState,
        region: FlywayRegion,
        season: SeasonPreset,
        readback: Arc<SharedReadbackHandle>,
    ) -> Self {
        let device = &render_state.device;
        let target_format = render_state.target_format;

        let initial_agents = build_initial_gpu_agents(region, season);
        let initial_uniforms = GpuSimUniforms::zeroed();
        let zero_weather = vec![GpuWeatherNode::zeroed(); MAX_WEATHER_NODES];
        let zero_turbines = vec![GpuTurbine::zeroed(); MAX_TURBINES];
        let zero_attractors = vec![GpuAttractor::zeroed(); MAX_ATTRACTORS];
        let zero_counters = [0_u32; 12];

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_sim_uniforms_v06"),
            contents: bytemuck::bytes_of(&initial_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let agent_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_gpu_agents_100k"),
            contents: bytemuck::cast_slice(&initial_agents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let weather_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_weather_nodes"),
            contents: bytemuck::cast_slice(&zero_weather),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let turbine_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_turbines"),
            contents: bytemuck::cast_slice(&zero_turbines),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let attractor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_attractors_and_skyglow"),
            contents: bytemuck::cast_slice(&zero_attractors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let counter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("aeroflyway_atomic_counters_48b"),
            contents: bytemuck::cast_slice(&zero_counters),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aeroflyway_counter_staging_48b"),
            size: 48,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aeroflyway_compute_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aeroflyway_compute_bgl"),
            entries: &[
                bgl_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Uniform),
                bgl_entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                bgl_entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                bgl_entry(
                    3,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                bgl_entry(
                    4,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                bgl_entry(
                    5,
                    wgpu::ShaderStages::COMPUTE,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aeroflyway_compute_layout"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aeroflyway_compute_pipeline"),
            layout: Some(&compute_layout),
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aeroflyway_compute_bg"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weather_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: turbine_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: attractor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counter_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aeroflyway_render_bgl"),
            entries: &[
                bgl_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                ),
                bgl_entry(
                    1,
                    wgpu::ShaderStages::VERTEX,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
            ],
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aeroflyway_render_layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        let map_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aeroflyway_map_render_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/map_render.wgsl").into()),
        });

        let transect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("aeroflyway_transect_render_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/transect_render.wgsl").into()),
        });

        let map_render_pipeline = create_instanced_pipeline(
            device,
            &render_layout,
            &map_shader,
            target_format,
            "map_instanced",
        );
        let transect_render_pipeline = create_instanced_pipeline(
            device,
            &render_layout,
            &transect_shader,
            target_format,
            "transect_instanced",
        );

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aeroflyway_render_bg"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            uniform_buffer,
            agent_buffer,
            weather_buffer,
            turbine_buffer,
            attractor_buffer,
            counter_buffer,
            staging_buffer,
            compute_pipeline,
            compute_bind_group,
            map_render_pipeline,
            transect_render_pipeline,
            render_bind_group,
            readback,
            frame_counter: 0,
            current_reseed_key: make_reseed_key(region, season),
        }
    }
}

fn bgl_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_instanced_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

pub struct MapGpuSimAndPaintCallback {
    pub uniforms: GpuSimUniforms,
    pub weather_nodes: Vec<GpuWeatherNode>,
    pub turbines: Vec<GpuTurbine>,
    pub attractors: Vec<GpuAttractor>,
    pub region: FlywayRegion,
    pub season: SeasonPreset,
}

impl egui_wgpu::CallbackTrait for MapGpuSimAndPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<FlywayGpuResources>() else {
            return Vec::new();
        };

        let desired_key = make_reseed_key(self.region, self.season);
        if res.current_reseed_key != desired_key {
            let reseeded = build_initial_gpu_agents(self.region, self.season);
            queue.write_buffer(&res.agent_buffer, 0, bytemuck::cast_slice(&reseeded));
            res.current_reseed_key = desired_key;
        }

        if res.readback.map_ready.swap(false, Ordering::AcqRel) {
            let slice = res.staging_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let raw_words: &[u32] = bytemuck::cast_slice(&mapped);
            if raw_words.len() >= 11 {
                let bfly_total = (self.uniforms.agent_count as f64 * 0.28).max(1.0);
                let summary = GpuReadbackSummary {
                    birds_airborne: raw_words[0],
                    birds_fallout: raw_words[1],
                    bats_flying: raw_words[2],
                    bat_rotor_hits_step: raw_words[3],
                    bat_baseline_hits_step: raw_words[4],
                    bfly_airborne: raw_words[5],
                    mean_gdd: (raw_words[6] as f64) / bfly_total,
                    max_north_lat: (raw_words[7] as f64) / 1000.0,
                    birds_skyglow_entrained: raw_words[8],
                    urban_strikes_step: raw_words[9],
                    urban_baseline_strikes_step: raw_words[10],
                };
                if let Ok(mut guard) = res.readback.latest.lock() {
                    *guard = Some(summary);
                }
            }
            drop(mapped);
            res.staging_buffer.unmap();
            res.readback.map_in_flight.store(false, Ordering::Release);
        }

        queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        if !self.weather_nodes.is_empty() {
            queue.write_buffer(
                &res.weather_buffer,
                0,
                bytemuck::cast_slice(&self.weather_nodes),
            );
        }
        if !self.turbines.is_empty() {
            queue.write_buffer(&res.turbine_buffer, 0, bytemuck::cast_slice(&self.turbines));
        }
        if !self.attractors.is_empty() {
            queue.write_buffer(
                &res.attractor_buffer,
                0,
                bytemuck::cast_slice(&self.attractors),
            );
        }
        queue.write_buffer(&res.counter_buffer, 0, bytemuck::cast_slice(&[0_u32; 12]));

        res.frame_counter = res.frame_counter.wrapping_add(1);
        let should_stage =
            res.frame_counter % 12 == 0 && !res.readback.map_in_flight.load(Ordering::Acquire);

        let mut compute_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("aeroflyway_compute_encoder"),
        });

        let count = self.uniforms.agent_count.min(MAX_GPU_AGENTS);
        if count > 0 {
            let workgroups = (count + 255) / 256;
            let mut cpass = compute_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aeroflyway_wgsl_compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&res.compute_pipeline);
            cpass.set_bind_group(0, &res.compute_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if should_stage {
            compute_enc.copy_buffer_to_buffer(&res.counter_buffer, 0, &res.staging_buffer, 0, 48);
        }

        queue.submit(std::iter::once(compute_enc.finish()));

        if should_stage && !res.readback.map_in_flight.swap(true, Ordering::AcqRel) {
            let handle = Arc::clone(&res.readback);
            res.staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        handle.map_ready.store(true, Ordering::Release);
                    } else {
                        handle.map_in_flight.store(false, Ordering::Release);
                    }
                });
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<FlywayGpuResources>() else {
            return;
        };
        let count = self.uniforms.agent_count.min(MAX_GPU_AGENTS);
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&res.map_render_pipeline);
        render_pass.set_bind_group(0, &res.render_bind_group, &[]);
        render_pass.draw(0..6, 0..count);
    }
}

pub struct TransectGpuPaintCallback {
    pub agent_count: u32,
}

impl egui_wgpu::CallbackTrait for TransectGpuPaintCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<FlywayGpuResources>() else {
            return;
        };
        let count = self.agent_count.min(MAX_GPU_AGENTS);
        if count == 0 {
            return;
        }
        render_pass.set_pipeline(&res.transect_render_pipeline);
        render_pass.set_bind_group(0, &res.render_bind_group, &[]);
        render_pass.draw(0..6, 0..count);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn pack_gpu_uniforms(
    center_lat: f64,
    center_lon: f64,
    zoom: f64,
    pitch_deg: f64,
    bearing_deg: f64,
    alt_exaggeration: f64,
    canvas_w_pts: f32,
    canvas_h_pts: f32,
    dt_hours: f64,
    hour_of_day: f64,
    active_doy: f64,
    mean_corridor_pm25: f64,
    aridity_index: f64,
    radar_peak_alt_m: f64,
    is_nocturnal: bool,
    running: bool,
    show_birds: bool,
    show_bats: bool,
    show_butterflies: bool,
    show_trails: bool,
    agent_count: u32,
    tick_seed: u32,
    enkf_gain: f64,
    params: &SimulationParams,
    num_weather_nodes: usize,
    num_turbines: usize,
    num_attractors: usize,
) -> GpuSimUniforms {
    let clamped_lat = center_lat.clamp(-85.0511, 85.0511);
    let center_merc_n = ((PI / 4.0) + (clamped_lat.to_radians() / 2.0)).tan().ln() as f32;
    let zoom_scale_px = (256.0 * 2.0_f64.powf(zoom)) as f32;
    let total_km = haversine_km(
        params.transect_start.0,
        params.transect_start.1,
        params.transect_end.0,
        params.transect_end.1,
    )
    .max(1.0) as f32;

    let mut layer_bits = 0_u32;
    if show_birds {
        layer_bits |= 1;
    }
    if show_bats {
        layer_bits |= 2;
    }
    if show_butterflies {
        layer_bits |= 4;
    }
    if is_nocturnal {
        layer_bits |= 8;
    }
    if running {
        layer_bits |= 16;
    }
    if show_trails {
        layer_bits |= 32;
    }

    let (min_lat, max_lat, min_lon, max_lon) = params.region.corridor_bounds();
    let heading_rad = annual_flyway_heading_deg(params.region, active_doy).to_radians() as f32;
    let total_warming = params.total_warming_c();
    let mismatch_days = compute_trophic_mismatch_days(total_warming, mean_corridor_pm25);

    GpuSimUniforms {
        center_lat: center_lat as f32,
        center_lon: center_lon as f32,
        center_merc_n,
        zoom_scale_px,
        canvas_w_pts,
        canvas_h_pts,
        transect_lat_a: params.transect_start.0 as f32,
        transect_lon_a: params.transect_start.1 as f32,
        transect_lat_b: params.transect_end.0 as f32,
        transect_lon_b: params.transect_end.1 as f32,
        transect_total_km: total_km,
        transect_half_width_km: params.transect_half_width_km as f32,
        dt_hours: dt_hours as f32,
        hour_of_day: hour_of_day as f32,
        heading_rad,
        wind_850_mult: params.wind_850_multiplier as f32,
        precip_front_mm: params.precip_front_mm_hr as f32,
        turbine_cut_in_mps: params.turbine_cut_in_mps as f32,
        temp_offset_c: total_warming as f32,
        enkf_gain: enkf_gain as f32,
        agent_count: agent_count.min(MAX_GPU_AGENTS),
        layer_bits,
        tick_seed,
        reseed_season_code: make_reseed_key(params.region, params.season),
        num_weather_nodes: (num_weather_nodes.min(MAX_WEATHER_NODES)) as u32,
        num_turbines: (num_turbines.min(MAX_TURBINES)) as u32,
        num_attractors: (num_attractors.min(MAX_ATTRACTORS)) as u32,
        region_code: params.region.gpu_code(),
        day_of_year: active_doy as f32,
        smoke_pm25_ugm3: params.smoke_pm25_offset_ugm3 as f32,
        corridor_min_lat: min_lat as f32,
        corridor_max_lat: max_lat as f32,
        corridor_min_lon: min_lon as f32,
        corridor_max_lon: max_lon as f32,
        drought_aridity_index: aridity_index as f32,
        trophic_mismatch_days: mismatch_days as f32,
        lights_out_pct: params.lights_out_curfew_pct as f32,
        radar_peak_alt_m: radar_peak_alt_m as f32,
        radar_nudge_weight: params.radar_nudge_weight as f32,
        skyglow_intensity_mult: params.skyglow_intensity_mult as f32,
        cam_pitch_rad: pitch_deg.clamp(0.0, 62.0).to_radians() as f32,
        cam_bearing_rad: bearing_deg.to_radians() as f32,
        cam_alt_exag: alt_exaggeration.max(0.0) as f32,
        _cam_pad: 0.0,
    }
}

pub fn pack_gpu_scene_buffers(
    region: FlywayRegion,
    aridity_index: f64,
    hour_of_day: f64,
    is_nocturnal: bool,
    lights_out_pct: f64,
    skyglow_mult: f64,
    weather: &[SynopticNodeWeather],
    turbines: &[WindTurbine],
    milkweed: &[MilkweedPatch],
) -> (Vec<GpuWeatherNode>, Vec<GpuTurbine>, Vec<GpuAttractor>) {
    let mut gpu_wx = Vec::with_capacity(weather.len().min(MAX_WEATHER_NODES));
    for n in weather.iter().take(MAX_WEATHER_NODES) {
        let r80 = (n.wind_80m_deg + 180.0).to_radians();
        let r850 = (n.wind_850_deg + 180.0).to_radians();
        gpu_wx.push(GpuWeatherNode {
            lat: n.lat as f32,
            lon: n.lon as f32,
            temp_2m_c: n.temp_2m_c as f32,
            precip_mm_hr: n.precip_mm_hr as f32,
            u80_mps: (n.wind_80m_mps * r80.sin()) as f32,
            v80_mps: (n.wind_80m_mps * r80.cos()) as f32,
            u850_mps: (n.wind_850_mps * r850.sin()) as f32,
            v850_mps: (n.wind_850_mps * r850.cos()) as f32,
            pm25_ugm3: n.pm25_ugm3 as f32,
            aod_550nm: n.aod_550nm as f32,
            soil_moisture_m3m3: n.soil_moisture_m3m3 as f32,
            cmip6_delta_t_c: n.cmip6_delta_t_c as f32,
        });
    }

    let mut gpu_turb = Vec::with_capacity(turbines.len().min(MAX_TURBINES));
    for t in turbines.iter().take(MAX_TURBINES) {
        gpu_turb.push(GpuTurbine {
            lat: t.lat as f32,
            lon: t.lon as f32,
            hub_height_m: t.hub_height_m as f32,
            rotor_radius_m: (t.rotor_diameter_m * 0.5) as f32,
        });
    }

    let drought_scale = (1.0 - 0.55 * aridity_index).clamp(0.25, 1.0) as f32;
    let mut gpu_att = Vec::with_capacity(MAX_ATTRACTORS);
    for lm in landmarks_for_region(region) {
        if lm.kind == LandmarkKind::WildlifeRefuge || lm.kind == LandmarkKind::MonarchSanctuary {
            gpu_att.push(GpuAttractor {
                lat: lm.lat as f32,
                lon: lm.lon as f32,
                weight: 10.0 * drought_scale,
                kind: 0,
            });
        }
    }
    for p in milkweed {
        if gpu_att.len() >= MAX_ATTRACTORS {
            break;
        }
        gpu_att.push(GpuAttractor {
            lat: p.lat as f32,
            lon: p.lon as f32,
            weight: (p.area_ha.sqrt().max(1.0) as f32) * drought_scale,
            kind: 1,
        });
    }
    for dome in skyglow_domes_for_region(region) {
        if gpu_att.len() >= MAX_ATTRACTORS {
            break;
        }
        let eff_nw = effective_skyglow_radiance_nw(
            dome.peak_radiance_nw_cm2_sr,
            hour_of_day,
            is_nocturnal,
            lights_out_pct,
            skyglow_mult,
        );
        gpu_att.push(GpuAttractor {
            lat: dome.lat as f32,
            lon: dome.lon as f32,
            weight: eff_nw as f32,
            kind: 2,
        });
    }

    (gpu_wx, gpu_turb, gpu_att)
}

fn make_reseed_key(region: FlywayRegion, season: SeasonPreset) -> u32 {
    let s_code = match season {
        SeasonPreset::SpringNorthbound => 0,
        SeasonPreset::SummerBreeding => 1,
        SeasonPreset::AutumnFalloutAndBatPulse => 2,
        SeasonPreset::OverwinterRoost => 3,
    };
    (region.gpu_code() << 8) | s_code
}

pub fn build_initial_gpu_agents(region: FlywayRegion, season: SeasonPreset) -> Vec<GpuAgent> {
    let (min_lat, max_lat, min_lon, max_lon) = region.corridor_bounds();
    let lat_span = (max_lat - min_lat).max(2.0) as f32;
    let lon_span = (max_lon - min_lon).max(2.0) as f32;
    let min_lat_f = min_lat as f32;
    let min_lon_f = min_lon as f32;
    let max_lon_f = max_lon as f32;
    let mid_lon_f = 0.5 * (min_lon_f + max_lon_f);

    let mut out = Vec::with_capacity(MAX_GPU_AGENTS as usize);
    for i in 0..MAX_GPU_AGENTS {
        let slot = i % 25;
        let taxon = if slot < 11 {
            0_u32
        } else if slot < 18 {
            1_u32
        } else {
            2_u32
        };

        let h1 = splitmix_f32(i as u64 * 31 + 17 + (region.gpu_code() as u64) * 997);
        let h2 = splitmix_f32(i as u64 * 67 + 29 + (region.gpu_code() as u64) * 1999);
        let h3 = splitmix_f32(i as u64 * 103 + 43);

        let (lat, lon, alt, gen, gdd) = if taxon == 0 {
            let frac = match season {
                SeasonPreset::SpringNorthbound => 0.10 + h1 * 0.65,
                SeasonPreset::OverwinterRoost => 0.08 + h1 * 0.48,
                _ => 0.22 + h1 * 0.68,
            };
            let lat = min_lat_f + frac * lat_span;
            let lane = if i % 3 != 0 { -0.15 } else { 0.17 };
            let lon = (mid_lon_f + lane * lon_span + (h2 - 0.5) * 0.38 * lon_span)
                .clamp(min_lon_f + 0.1, max_lon_f - 0.1);
            (lat, lon, 450.0 + h3 * 1350.0, 1_u32, 0.0)
        } else if taxon == 1 {
            let lat = min_lat_f + (0.16 + h1 * 0.74) * lat_span;
            let lon = (mid_lon_f + (h2 - 0.5) * 0.56 * lon_span)
                .clamp(min_lon_f + 0.1, max_lon_f - 0.1);
            (lat, lon, 52.0 + h3 * 250.0, 1_u32, 0.0)
        } else {
            let (frac, gen, gdd) = match season {
                SeasonPreset::OverwinterRoost => (0.06 + h1 * 0.28, 4_u32, 305.0 + h3 * 42.0),
                SeasonPreset::SpringNorthbound => (0.12 + h1 * 0.48, 1_u32, 135.0 + h3 * 195.0),
                _ => (0.26 + h1 * 0.54, 3_u32, 205.0 + h3 * 140.0),
            };
            let lat = min_lat_f + frac * lat_span;
            let lon = (mid_lon_f + (h2 - 0.5) * 0.50 * lon_span)
                .clamp(min_lon_f + 0.1, max_lon_f - 0.1);
            (lat, lon, 14.0 + h3 * 95.0, gen, gdd)
        };

        out.push(GpuAgent {
            lat,
            lon,
            alt_agl_m: alt,
            vel_u_mps: 0.0,
            vel_v_mps: 0.0,
            prev_lat: lat,
            prev_lon: lon,
            energy: 0.5 + 0.45 * h2,
            gdd_accum: gdd,
            taxon,
            generation: gen,
            flags: 0,
        });
    }
    out
}

fn splitmix_f32(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / ((1_u64 << 53) as f64)) as f32
}
