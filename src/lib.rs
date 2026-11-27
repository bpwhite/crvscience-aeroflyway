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

#![forbid(unsafe_code)]

pub mod acoustics;
pub mod basemap;
pub mod cli;
pub mod db;
pub mod export;
pub mod gpu_pipeline;
pub mod ingest;
pub mod map_engine;
pub mod physics;

pub use export::{
    evaluate_multi_corridor_lights_out_summary, export_multi_corridor_lights_out_summary,
    format_multi_corridor_lights_out_csv, CorridorLightsOutInterventionRow,
    MultiCorridorLightsOutSummary,
};
