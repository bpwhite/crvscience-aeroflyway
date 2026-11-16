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

//! Global multi-flyway GIS region presets (`FlywayRegion`), offline coastlines/rivers,
//! bottleneck refuges, Motus stations, VIIRS DNB urban skyglow domes (`UrbanSkyglowDome`),
//! NEXRAD WSR-88D & Aloft OPERA radar stations (`RadarStationMeta`), wind-turbine corridors,
//! host-plant patches, and multi-continent DEM profiles (`src/basemap.rs` — AeroFlyway v0.6.0).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Taxon {
    Bird,
    Bat,
    Butterfly,
}

impl Taxon {
    pub fn label(self) -> &'static str {
        match self {
            Taxon::Bird => "Birds (Aves)",
            Taxon::Bat => "Bats (Chiroptera)",
            Taxon::Butterfly => "Butterflies (Lepidoptera)",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Taxon::Bird => "BIRD",
            Taxon::Bat => "BAT",
            Taxon::Butterfly => "BUTTERFLY",
        }
    }

    pub fn from_code(code: &str) -> Self {
        match code {
            "BAT" => Taxon::Bat,
            "BUTTERFLY" => Taxon::Butterfly,
            _ => Taxon::Bird,
        }
    }
}

/// Five major global flyway corridors supported natively in AeroFlyway v0.6.0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlywayRegion {
    PacificAmericas,
    CentralAmericasPlatte,
    AtlanticAmericas,
    EastAtlanticPalearctic,
    EastAsianAustralasian,
}

impl FlywayRegion {
    pub const ALL: [FlywayRegion; 5] = [
        FlywayRegion::PacificAmericas,
        FlywayRegion::CentralAmericasPlatte,
        FlywayRegion::AtlanticAmericas,
        FlywayRegion::EastAtlanticPalearctic,
        FlywayRegion::EastAsianAustralasian,
    ];

    pub fn code(self) -> &'static str {
        match self {
            Self::PacificAmericas => "pacific",
            Self::CentralAmericasPlatte => "central",
            Self::AtlanticAmericas => "atlantic",
            Self::EastAtlanticPalearctic => "palearctic",
            Self::EastAsianAustralasian => "east_asia",
        }
    }

    pub fn gpu_code(self) -> u32 {
        match self {
            Self::PacificAmericas => 0,
            Self::CentralAmericasPlatte => 1,
            Self::AtlanticAmericas => 2,
            Self::EastAtlanticPalearctic => 3,
            Self::EastAsianAustralasian => 4,
        }
    }

    pub fn from_str_code(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "pacific" | "pacific_americas" | "pnw" => Some(Self::PacificAmericas),
            "central" | "central_americas" | "platte" | "mississippi" => {
                Some(Self::CentralAmericasPlatte)
            }
            "atlantic" | "atlantic_americas" | "capemay" => Some(Self::AtlanticAmericas),
            "palearctic" | "east_atlantic" | "europe" | "gibraltar" => {
                Some(Self::EastAtlanticPalearctic)
            }
            "east_asia" | "eaa" | "australasian" | "poyang" => Some(Self::EastAsianAustralasian),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PacificAmericas => "Pacific Americas (Willamette / Gorge / Klamath)",
            Self::CentralAmericasPlatte => "Central & Mississippi (Platte / Bracken / Gulf)",
            Self::AtlanticAmericas => "Atlantic Americas (Cape May / Delaware / Keys)",
            Self::EastAtlanticPalearctic => "East Atlantic Palearctic (Falsterbo / Donana / Gibraltar)",
            Self::EastAsianAustralasian => "East Asian-Australasian (Bohai / Yellow Sea / Poyang)",
        }
    }

    pub fn short_card_summary(self) -> &'static str {
        match self {
            Self::PacificAmericas => "NEXRAD KRTX/KMAX | Finley NWR & Columbia Gorge | Portland/Seattle VIIRS",
            Self::CentralAmericasPlatte => "NEXRAD KUEX/KDDC/KEWX | Platte Cranes & Bracken Bats | DFW/Austin VIIRS",
            Self::AtlanticAmericas => "NEXRAD KDIX/KDOX | Cape May & Delaware Bay | NYC/Philadelphia VIIRS",
            Self::EastAtlanticPalearctic => "Aloft nlhrw/bejab/behel | Falsterbo to Gibraltar | Randstad/Seville VIIRS",
            Self::EastAsianAustralasian => "C-Band Bohai/Poyang | Ugii Lake to Yangtze | Shanghai/Tianjin VIIRS",
        }
    }

    /// Default `(center_lat, center_lon, zoom)` when switching to this flyway region.
    pub fn default_viewport(self) -> (f64, f64, f64) {
        match self {
            Self::PacificAmericas => (44.48, -123.28, 6.8),
            Self::CentralAmericasPlatte => (36.80, -98.45, 5.6),
            Self::AtlanticAmericas => (37.60, -75.80, 5.8),
            Self::EastAtlanticPalearctic => (44.20, 0.80, 5.0),
            Self::EastAsianAustralasian => (34.20, 117.50, 5.2),
        }
    }

    /// Bounding `(min_lat, max_lat, min_lon, max_lon)` for spherical corridor agent wrapping.
    pub fn corridor_bounds(self) -> (f64, f64, f64, f64) {
        match self {
            Self::PacificAmericas => (34.3, 49.0, -124.4, -116.8),
            Self::CentralAmericasPlatte => (26.0, 46.5, -103.5, -94.0),
            Self::AtlanticAmericas => (25.0, 45.2, -81.8, -70.0),
            Self::EastAtlanticPalearctic => (34.5, 56.5, -10.0, 14.5),
            Self::EastAsianAustralasian => (22.2, 48.5, 102.2, 124.8),
        }
    }

    /// Default A -> B 3D airspace transect endpoints `((lat_a, lon_a), (lat_b, lon_b))`.
    pub fn default_transect(self) -> ((f64, f64), (f64, f64)) {
        match self {
            Self::PacificAmericas => ((44.622, -124.045), (45.638, -120.580)),
            Self::CentralAmericasPlatte => ((40.668, -98.892), (29.688, -98.352)),
            Self::AtlanticAmericas => ((39.086, -74.723), (37.820, -76.080)),
            Self::EastAtlanticPalearctic => ((55.383, 12.817), (35.900, -5.300)),
            Self::EastAsianAustralasian => ((47.755, 102.652), (29.150, 116.217)),
        }
    }

    /// Dominant corridor orientation offset (degrees clockwise from North during northbound spring).
    pub fn spring_axis_bearing_deg(self) -> f64 {
        match self {
            Self::PacificAmericas => 352.0,
            Self::CentralAmericasPlatte => 2.0,
            Self::AtlanticAmericas => 24.0,
            Self::EastAtlanticPalearctic => 21.0,
            Self::EastAsianAustralasian => 14.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkKind {
    WildlifeRefuge,
    MotusStation,
    MonarchSanctuary,
    CascadePeak,
}

#[derive(Clone, Debug)]
pub struct FlywayLandmark {
    pub name: &'static str,
    pub lat: f64,
    pub lon: f64,
    pub elevation_m: f64,
    pub kind: LandmarkKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindTurbine {
    pub case_id: i64,
    pub project_name: String,
    pub lat: f64,
    pub lon: f64,
    pub hub_height_m: f64,
    pub rotor_diameter_m: f64,
    pub capacity_kw: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MilkweedPatch {
    pub id: i64,
    pub label: String,
    pub lat: f64,
    pub lon: f64,
    pub area_ha: f64,
    pub is_virtual: bool,
    pub gdd_threshold: f64,
}

/// VIIRS Day/Night Band (DNB) urban artificial-light-at-night (ALAN) skyglow dome (`nW/cm^2/sr`).
/// Drives nocturnal avian phototactic attraction, low-canopy disorientation, and glass-facade strikes
/// (Van Doren et al. 2021, *PNAS*), mitigated by Lights-Out curfew dimming (`23:00-06:00`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UrbanSkyglowDome {
    pub city_name: &'static str,
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f64,
    pub peak_radiance_nw_cm2_sr: f64,
    pub glass_hazard_index: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadarNetwork {
    NexradWsr88d,
    AloftOpera,
    EastAsianDoppler,
}

impl RadarNetwork {
    pub fn label(self) -> &'static str {
        match self {
            Self::NexradWsr88d => "NOAA/Unidata NEXRAD WSR-88D (S-Band)",
            Self::AloftOpera => "ENRAM Aloft / OPERA (C-Band VPTS)",
            Self::EastAsianDoppler => "EAAF S/C-Band Aeroecology Profiler",
        }
    }
}

/// Weather surveillance radar station metadata for vertical biological profile (`VPTS`) ingestion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RadarStationMeta {
    pub code: &'static str,
    pub name: &'static str,
    pub network: RadarNetwork,
    pub lat: f64,
    pub lon: f64,
    pub antenna_elev_m: f64,
    pub wavelength_cm: f64,
    pub range_km: f64,
}

pub const PACIFIC_COASTLINE: &[(f64, f64)] = &[
    (48.386, -124.725), (47.908, -124.638), (47.275, -124.218), (46.904, -124.120),
    (46.258, -124.053), (45.892, -123.961), (45.468, -123.972), (44.962, -124.018),
    (44.636, -124.062), (44.016, -124.136), (43.345, -124.331), (42.837, -124.564),
    (42.052, -124.279), (41.755, -124.203), (41.056, -124.148), (40.439, -124.409),
    (39.442, -123.805), (38.995, -123.702), (38.305, -123.056), (37.996, -123.023),
    (37.775, -122.512), (37.152, -122.361), (36.951, -122.027), (36.602, -121.895),
    (36.270, -121.808), (35.658, -121.215), (35.126, -120.632), (34.448, -120.471),
];

pub const PUGET_SOUND_RING: &[(f64, f64)] = &[
    (48.386, -124.725), (48.150, -123.430), (48.120, -122.760), (47.606, -122.332),
    (47.045, -122.902), (47.330, -123.120), (47.900, -122.650), (48.750, -122.500),
];

pub const COLUMBIA_RIVER: &[(f64, f64)] = &[
    (46.246, -124.052), (46.188, -123.831), (46.142, -123.210), (45.862, -122.785),
    (45.618, -122.671), (45.568, -122.402), (45.642, -121.936), (45.709, -121.521),
    (45.602, -121.183), (45.662, -120.825), (45.735, -120.205), (45.918, -119.328),
    (46.212, -119.122),
];

pub const WILLAMETTE_RIVER: &[(f64, f64)] = &[
    (43.985, -123.012), (44.272, -123.172), (44.418, -123.256), (44.564, -123.260),
    (44.639, -123.105), (44.775, -123.148), (44.943, -123.042), (45.162, -123.008),
    (45.289, -122.768), (45.356, -122.608), (45.515, -122.675), (45.649, -122.764),
];

pub const KLAMATH_SACRAMENTO_CORRIDOR: &[(f64, f64)] = &[
    (42.225, -121.782), (41.968, -121.564), (41.522, -122.120), (40.586, -122.392),
    (39.728, -121.838), (39.140, -121.618), (38.582, -121.494), (38.045, -121.895),
];

pub const STATE_BOUNDARIES: &[&[(f64, f64)]] = &[
    &[(42.000, -124.212), (42.000, -120.001), (42.000, -117.026)],
    &[
        (42.000, -117.026), (43.800, -117.026), (44.300, -117.220),
        (45.600, -116.463), (46.000, -116.916),
    ],
    &[(45.918, -119.328), (46.000, -118.980), (46.000, -116.916)],
];

pub const FLYWAY_LANDMARKS: &[FlywayLandmark] = &[
    FlywayLandmark { name: "William L. Finley NWR", lat: 44.4128, lon: -123.3285, elevation_m: 86.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Baskett Slough NWR", lat: 44.9658, lon: -123.2575, elevation_m: 65.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Ankeny NWR", lat: 44.7876, lon: -123.0768, elevation_m: 62.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Ridgefield NWR (Motus)", lat: 45.8080, lon: -122.7475, elevation_m: 12.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Hatfield Marine Motus", lat: 44.6225, lon: -124.0445, elevation_m: 6.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Malheur NWR", lat: 43.2660, lon: -118.8442, elevation_m: 1255.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Lower Klamath NWR", lat: 41.9675, lon: -121.5640, elevation_m: 1245.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Siskiyou Flyway Saddle", lat: 42.0550, lon: -122.6050, elevation_m: 1314.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Marys Peak (Coast Range)", lat: 44.5043, lon: -123.5512, elevation_m: 1249.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Mt. Hood", lat: 45.3736, lon: -121.6960, elevation_m: 3429.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "South Sister", lat: 44.1034, lon: -121.7693, elevation_m: 3159.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Mt. Shasta", lat: 41.4092, lon: -122.1949, elevation_m: 4322.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Santa Cruz Monarch Grove", lat: 36.9515, lon: -122.0268, elevation_m: 18.0, kind: LandmarkKind::MonarchSanctuary },
    FlywayLandmark { name: "Pismo Monarch Grove", lat: 35.1258, lon: -120.6319, elevation_m: 14.0, kind: LandmarkKind::MonarchSanctuary },
];

pub const CENTRAL_LANDMARKS: &[FlywayLandmark] = &[
    FlywayLandmark { name: "Rowe Sanctuary (Platte River)", lat: 40.6680, lon: -98.8920, elevation_m: 635.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Quivira / Cheyenne Bottoms", lat: 38.1850, lon: -98.4950, elevation_m: 545.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Bracken Cave Bat Preserve", lat: 29.6877, lon: -98.3522, elevation_m: 280.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Aransas NWR (Whooping Crane)", lat: 28.2300, lon: -96.8050, elevation_m: 5.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Wichita Mountains Corridor", lat: 34.7450, lon: -98.7120, elevation_m: 755.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Texas Hill Country Monarch Funnel", lat: 30.2800, lon: -99.1200, elevation_m: 590.0, kind: LandmarkKind::MonarchSanctuary },
];

pub const ATLANTIC_LANDMARKS: &[FlywayLandmark] = &[
    FlywayLandmark { name: "Cape May Point NWR", lat: 38.9593, lon: -74.9274, elevation_m: 4.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Avalon / Delaware Bay Motus", lat: 39.0863, lon: -74.7229, elevation_m: 3.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Blackwater / Chesapeake Staging", lat: 38.4450, lon: -76.1200, elevation_m: 2.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Hawk Mountain Sanctuary", lat: 40.6350, lon: -75.9880, elevation_m: 463.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Outer Banks Flyway Node", lat: 35.7680, lon: -75.5280, elevation_m: 4.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Everglades Coastal Roost", lat: 25.3200, lon: -80.9300, elevation_m: 2.0, kind: LandmarkKind::MonarchSanctuary },
];

pub const PALEARCTIC_LANDMARKS: &[FlywayLandmark] = &[
    FlywayLandmark { name: "Falsterbo Bird Observatory", lat: 55.3830, lon: 12.8170, elevation_m: 5.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Wadden Sea Tidal Staging", lat: 53.4200, lon: 6.1500, elevation_m: 2.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Donana National Park Marshes", lat: 36.9800, lon: -6.3700, elevation_m: 8.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Strait of Gibraltar Bottleneck", lat: 35.9000, lon: -5.3000, elevation_m: 42.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Western Pyrenees Pass", lat: 43.0300, lon: -1.3200, elevation_m: 1420.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Andalusian Painted Lady Meadow", lat: 36.6800, lon: -5.9200, elevation_m: 120.0, kind: LandmarkKind::MonarchSanctuary },
];

pub const EAST_ASIA_LANDMARKS: &[FlywayLandmark] = &[
    FlywayLandmark { name: "Poyang Lake Waterfowl Basin", lat: 29.1500, lon: 116.2170, elevation_m: 18.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Ugii Lake Crane Staging", lat: 47.7552, lon: 102.6520, elevation_m: 1332.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Yellow Sea / Yancheng Mudflats", lat: 33.3800, lon: 120.5200, elevation_m: 3.0, kind: LandmarkKind::WildlifeRefuge },
    FlywayLandmark { name: "Bohai Bay Shorebird Funnel", lat: 39.1500, lon: 118.2500, elevation_m: 4.0, kind: LandmarkKind::MotusStation },
    FlywayLandmark { name: "Lushan Flyway Ridge", lat: 29.5600, lon: 115.9800, elevation_m: 1165.0, kind: LandmarkKind::CascadePeak },
    FlywayLandmark { name: "Yangtze Estuary Nectar Marsh", lat: 31.5200, lon: 121.9200, elevation_m: 5.0, kind: LandmarkKind::MonarchSanctuary },
];

pub fn landmarks_for_region(region: FlywayRegion) -> &'static [FlywayLandmark] {
    match region {
        FlywayRegion::PacificAmericas => FLYWAY_LANDMARKS,
        FlywayRegion::CentralAmericasPlatte => CENTRAL_LANDMARKS,
        FlywayRegion::AtlanticAmericas => ATLANTIC_LANDMARKS,
        FlywayRegion::EastAtlanticPalearctic => PALEARCTIC_LANDMARKS,
        FlywayRegion::EastAsianAustralasian => EAST_ASIA_LANDMARKS,
    }
}

/// Returns VIIRS Day/Night Band urban skyglow domes (`nW/cm^2/sr`) intersecting each flyway corridor.
pub fn skyglow_domes_for_region(region: FlywayRegion) -> Vec<UrbanSkyglowDome> {
    match region {
        FlywayRegion::PacificAmericas => vec![
            UrbanSkyglowDome { city_name: "Portland-Vancouver Metro", lat: 45.5152, lon: -122.6784, radius_km: 42.0, peak_radiance_nw_cm2_sr: 78.5, glass_hazard_index: 0.74 },
            UrbanSkyglowDome { city_name: "Seattle-Tacoma Puget Core", lat: 47.6062, lon: -122.3321, radius_km: 52.0, peak_radiance_nw_cm2_sr: 94.0, glass_hazard_index: 0.84 },
            UrbanSkyglowDome { city_name: "Eugene-Springfield Basin", lat: 44.0521, lon: -123.0868, radius_km: 24.0, peak_radiance_nw_cm2_sr: 38.0, glass_hazard_index: 0.42 },
            UrbanSkyglowDome { city_name: "Sacramento Metro Core", lat: 38.5816, lon: -121.4944, radius_km: 44.0, peak_radiance_nw_cm2_sr: 74.0, glass_hazard_index: 0.68 },
        ],
        FlywayRegion::CentralAmericasPlatte => vec![
            UrbanSkyglowDome { city_name: "Dallas-Fort Worth Metroplex", lat: 32.7767, lon: -96.7970, radius_km: 64.0, peak_radiance_nw_cm2_sr: 116.0, glass_hazard_index: 0.91 },
            UrbanSkyglowDome { city_name: "San Antonio-Austin Corridor", lat: 29.5800, lon: -98.2900, radius_km: 48.0, peak_radiance_nw_cm2_sr: 88.0, glass_hazard_index: 0.79 },
            UrbanSkyglowDome { city_name: "Oklahoma City Metro", lat: 35.4676, lon: -97.5164, radius_km: 38.0, peak_radiance_nw_cm2_sr: 68.0, glass_hazard_index: 0.63 },
            UrbanSkyglowDome { city_name: "Omaha-Lincoln Platte East", lat: 41.1800, lon: -96.1500, radius_km: 34.0, peak_radiance_nw_cm2_sr: 58.0, glass_hazard_index: 0.55 },
        ],
        FlywayRegion::AtlanticAmericas => vec![
            UrbanSkyglowDome { city_name: "New York-Newark Harbor", lat: 40.7128, lon: -74.0060, radius_km: 68.0, peak_radiance_nw_cm2_sr: 145.0, glass_hazard_index: 0.97 },
            UrbanSkyglowDome { city_name: "Philadelphia-Delaware Valley", lat: 39.9526, lon: -75.1652, radius_km: 52.0, peak_radiance_nw_cm2_sr: 108.0, glass_hazard_index: 0.88 },
            UrbanSkyglowDome { city_name: "Baltimore-Washington Core", lat: 39.0458, lon: -76.6413, radius_km: 56.0, peak_radiance_nw_cm2_sr: 98.0, glass_hazard_index: 0.85 },
            UrbanSkyglowDome { city_name: "Hampton Roads / Norfolk", lat: 36.8529, lon: -76.2859, radius_km: 36.0, peak_radiance_nw_cm2_sr: 64.0, glass_hazard_index: 0.60 },
        ],
        FlywayRegion::EastAtlanticPalearctic => vec![
            UrbanSkyglowDome { city_name: "Randstad (Amsterdam-Rotterdam)", lat: 52.1500, lon: 4.5800, radius_km: 58.0, peak_radiance_nw_cm2_sr: 122.0, glass_hazard_index: 0.86 },
            UrbanSkyglowDome { city_name: "Copenhagen-Malmo Oresund", lat: 55.6400, lon: 12.6800, radius_km: 40.0, peak_radiance_nw_cm2_sr: 76.0, glass_hazard_index: 0.69 },
            UrbanSkyglowDome { city_name: "Bordeaux Gironde Hub", lat: 44.8378, lon: -0.5792, radius_km: 34.0, peak_radiance_nw_cm2_sr: 62.0, glass_hazard_index: 0.57 },
            UrbanSkyglowDome { city_name: "Seville-Cadiz Guadalquivir", lat: 37.3891, lon: -5.9845, radius_km: 38.0, peak_radiance_nw_cm2_sr: 72.0, glass_hazard_index: 0.64 },
        ],
        FlywayRegion::EastAsianAustralasian => vec![
            UrbanSkyglowDome { city_name: "Shanghai-Yangtze Megacity", lat: 31.2304, lon: 121.4737, radius_km: 72.0, peak_radiance_nw_cm2_sr: 138.0, glass_hazard_index: 0.94 },
            UrbanSkyglowDome { city_name: "Tianjin-Bohai Industrial Rim", lat: 39.1200, lon: 117.3600, radius_km: 62.0, peak_radiance_nw_cm2_sr: 124.0, glass_hazard_index: 0.89 },
            UrbanSkyglowDome { city_name: "Nanchang-Poyang South", lat: 28.6829, lon: 115.8579, radius_km: 38.0, peak_radiance_nw_cm2_sr: 69.0, glass_hazard_index: 0.62 },
            UrbanSkyglowDome { city_name: "Qingdao Jiaozhou Coast", lat: 36.0671, lon: 120.3826, radius_km: 42.0, peak_radiance_nw_cm2_sr: 82.0, glass_hazard_index: 0.72 },
        ],
    }
}

/// Returns verified NEXRAD WSR-88D and Aloft OPERA radar profiler sites per flyway corridor.
pub fn radar_stations_for_region(region: FlywayRegion) -> Vec<RadarStationMeta> {
    match region {
        FlywayRegion::PacificAmericas => vec![
            RadarStationMeta { code: "KRTX", name: "Portland / Scappoose OR", network: RadarNetwork::NexradWsr88d, lat: 45.7150, lon: -122.9650, antenna_elev_m: 527.0, wavelength_cm: 10.6, range_km: 120.0 },
            RadarStationMeta { code: "KMAX", name: "Medford / Mt. Ashland OR", network: RadarNetwork::NexradWsr88d, lat: 42.0811, lon: -122.7172, antenna_elev_m: 2290.0, wavelength_cm: 10.6, range_km: 120.0 },
        ],
        FlywayRegion::CentralAmericasPlatte => vec![
            RadarStationMeta { code: "KUEX", name: "Hastings / Platte River NE", network: RadarNetwork::NexradWsr88d, lat: 40.3208, lon: -98.4419, antenna_elev_m: 627.0, wavelength_cm: 10.6, range_km: 130.0 },
            RadarStationMeta { code: "KDDC", name: "Dodge City / Quivira KS", network: RadarNetwork::NexradWsr88d, lat: 37.7608, lon: -99.9689, antenna_elev_m: 790.0, wavelength_cm: 10.6, range_km: 130.0 },
            RadarStationMeta { code: "KEWX", name: "Austin-San Antonio / Bracken TX", network: RadarNetwork::NexradWsr88d, lat: 29.7041, lon: -98.0286, antenna_elev_m: 233.7, wavelength_cm: 10.6, range_km: 130.0 },
        ],
        FlywayRegion::AtlanticAmericas => vec![
            RadarStationMeta { code: "KDIX", name: "Philadelphia / Fort Dix NJ", network: RadarNetwork::NexradWsr88d, lat: 39.9471, lon: -74.4107, antenna_elev_m: 70.1, wavelength_cm: 10.6, range_km: 125.0 },
            RadarStationMeta { code: "KAKQ", name: "Wakefield / Chesapeake VA", network: RadarNetwork::NexradWsr88d, lat: 36.9840, lon: -77.0074, antenna_elev_m: 42.0, wavelength_cm: 10.6, range_km: 125.0 },
        ],
        FlywayRegion::EastAtlanticPalearctic => vec![
            RadarStationMeta { code: "nlhrw", name: "Herwijnen NL (OPERA/Aloft)", network: RadarNetwork::AloftOpera, lat: 51.8371, lon: 5.1380, antenna_elev_m: 25.0, wavelength_cm: 5.3, range_km: 110.0 },
            RadarStationMeta { code: "bejab", name: "Jabbeke BE (OPERA/Aloft)", network: RadarNetwork::AloftOpera, lat: 51.1919, lon: 3.0641, antenna_elev_m: 50.0, wavelength_cm: 5.3, range_km: 110.0 },
            RadarStationMeta { code: "behel", name: "Helchteren BE (OPERA/Aloft)", network: RadarNetwork::AloftOpera, lat: 51.0702, lon: 5.4054, antenna_elev_m: 144.0, wavelength_cm: 5.3, range_km: 110.0 },
        ],
        FlywayRegion::EastAsianAustralasian => vec![
            RadarStationMeta { code: "Z9791", name: "Nanchang / Poyang Doppler", network: RadarNetwork::EastAsianDoppler, lat: 28.6950, lon: 115.9100, antenna_elev_m: 68.0, wavelength_cm: 10.0, range_km: 125.0 },
            RadarStationMeta { code: "Z9220", name: "Tianjin / Bohai Bay Profiler", network: RadarNetwork::EastAsianDoppler, lat: 39.0440, lon: 117.7170, antenna_elev_m: 32.0, wavelength_cm: 10.0, range_km: 125.0 },
        ],
    }
}

pub fn seed_wind_turbines() -> Vec<WindTurbine> {
    seed_wind_turbines_for_region(FlywayRegion::PacificAmericas)
}

pub fn seed_wind_turbines_for_region(region: FlywayRegion) -> Vec<WindTurbine> {
    let clusters: &[(&str, f64, f64, f64, f64, f64, usize)] = match region {
        FlywayRegion::PacificAmericas => &[
            ("Biglow Canyon Wind Farm", 45.638, -120.605, 80.0, 93.0, 2300.0, 8),
            ("Klondike Wind Power", 45.582, -120.512, 80.0, 82.5, 1500.0, 6),
            ("Shepherds Flat", 45.702, -120.068, 84.0, 100.0, 2500.0, 9),
            ("Windy Point / Goldendale", 45.764, -120.735, 82.0, 96.0, 2300.0, 7),
            ("Stateline / Vansycle Ridge", 46.028, -118.812, 78.0, 90.0, 1800.0, 6),
            ("Montague Wind Facility", 45.495, -120.185, 89.0, 116.0, 2800.0, 6),
            ("Wheatridge Corridor", 45.420, -119.560, 90.0, 127.0, 3000.0, 6),
        ],
        FlywayRegion::CentralAmericasPlatte => &[
            ("Smoky Hills Kansas Wind", 38.920, -98.350, 85.0, 108.0, 2500.0, 8),
            ("Cimarron Bend Corridor", 37.620, -99.850, 88.0, 116.0, 2800.0, 8),
            ("Sweetwater Texas Ridge", 32.320, -100.410, 84.0, 105.0, 2400.0, 8),
            ("Anacacho / South Texas Wind", 29.180, -99.650, 86.0, 112.0, 2600.0, 6),
        ],
        FlywayRegion::AtlanticAmericas => &[
            ("Atlantic City / Jersey Coastal", 39.375, -74.452, 82.0, 98.0, 2200.0, 6),
            ("Allegheny Front Ridge", 39.210, -79.310, 85.0, 108.0, 2500.0, 8),
            ("Pocono / Kittatinny Ridge", 40.980, -75.720, 84.0, 100.0, 2300.0, 6),
        ],
        FlywayRegion::EastAtlanticPalearctic => &[
            ("Oresund / Falsterbo Array", 55.520, 12.760, 90.0, 120.0, 3600.0, 7),
            ("German Bight / Frisian Corridor", 53.780, 7.220, 95.0, 128.0, 4000.0, 8),
            ("Tarifa / Cadiz Strait Ridge", 36.065, -5.660, 78.0, 92.0, 2000.0, 8),
        ],
        FlywayRegion::EastAsianAustralasian => &[
            ("Yancheng / Rudong Intertidal Wind", 33.580, 120.820, 95.0, 132.0, 4200.0, 8),
            ("Bohai Tangshan Coastal Array", 39.040, 118.520, 90.0, 122.0, 3400.0, 7),
            ("Poyang Hukou Channel Wind", 29.680, 116.240, 85.0, 110.0, 2600.0, 6),
        ],
    };

    let mut out = Vec::new();
    let mut id_counter = 900_001_i64 + (region.gpu_code() as i64) * 10_000;
    for &(name, base_lat, base_lon, hh, rd, cap, count) in clusters {
        for i in 0..count {
            let angle = (i as f64) * 0.85;
            let rad = 0.022 + (i as f64) * 0.009;
            out.push(WindTurbine {
                case_id: id_counter,
                project_name: name.to_string(),
                lat: base_lat + rad * angle.sin() * 0.65,
                lon: base_lon + rad * angle.cos(),
                hub_height_m: hh,
                rotor_diameter_m: rd,
                capacity_kw: cap,
            });
            id_counter += 1;
        }
    }
    out
}

pub fn seed_milkweed_patches() -> Vec<MilkweedPatch> {
    seed_milkweed_patches_for_region(FlywayRegion::PacificAmericas)
}

pub fn seed_milkweed_patches_for_region(region: FlywayRegion) -> Vec<MilkweedPatch> {
    let raw: &[(&str, f64, f64, f64, f64)] = match region {
        FlywayRegion::PacificAmericas => &[
            ("Finley Wet Prairie Asclepias", 44.415, -123.318, 18.5, 175.0),
            ("Baskett Slough Uplands", 44.969, -123.251, 14.0, 185.0),
            ("Fern Ridge / Eugene Corridor", 44.112, -123.295, 22.0, 170.0),
            ("Rogue Valley Nectar Stepping-Stone", 42.392, -122.875, 31.0, 145.0),
            ("Klamath Basin Showy Milkweed", 42.025, -121.612, 28.0, 195.0),
            ("Umpqua Oak Savanna Patch", 43.248, -123.355, 16.0, 160.0),
            ("Columbia Confluence Meadow", 45.695, -121.885, 12.5, 190.0),
            ("Sacramento Valley Spring Breeder", 39.512, -121.855, 44.0, 110.0),
        ],
        FlywayRegion::CentralAmericasPlatte => &[
            ("Texas Hill Country Antelope-Horns", 30.420, -98.620, 38.0, 115.0),
            ("Red River Tallgrass Prairie", 33.850, -97.120, 29.0, 135.0),
            ("Flint Hills Common Milkweed", 38.420, -96.550, 42.0, 165.0),
            ("Platte Wet Meadow Asclepias", 40.685, -98.840, 26.0, 180.0),
        ],
        FlywayRegion::AtlanticAmericas => &[
            ("Cape May Dune Goldenrod & Milkweed", 38.965, -74.915, 19.0, 160.0),
            ("Delmarva Coastal Nectar Strip", 37.880, -75.540, 24.0, 145.0),
            ("Shenandoah Piedmont Meadow", 38.520, -78.350, 21.0, 170.0),
            ("Outer Banks Barrier Nectar", 35.620, -75.480, 17.5, 130.0),
        ],
        FlywayRegion::EastAtlanticPalearctic => &[
            ("Guadalquivir / Donana Thistle Step", 37.040, -6.280, 34.0, 120.0),
            ("Aquitaine / Gironde Meadow", 45.120, -0.780, 27.0, 155.0),
            ("Scheldt / Flanders Nectar Corridor", 51.310, 4.260, 22.0, 175.0),
            ("Skane / Falsterbo Coastal Heath", 55.410, 12.880, 18.0, 190.0),
        ],
        FlywayRegion::EastAsianAustralasian => &[
            ("Poyang Floodplain Nectar Meadow", 29.210, 116.310, 32.0, 125.0),
            ("Yangtze Chongming Wetland", 31.560, 121.840, 28.0, 145.0),
            ("Shandong Peninsula Stepping-Stone", 37.420, 121.450, 24.0, 170.0),
            ("Liaodong / Bohai Meadow", 40.620, 122.150, 21.0, 185.0),
        ],
    };

    raw.iter()
        .enumerate()
        .map(|(idx, &(label, lat, lon, area_ha, gdd_threshold))| MilkweedPatch {
            id: (idx + 1) as i64,
            label: label.to_string(),
            lat,
            lon,
            area_ha,
            is_virtual: false,
            gdd_threshold,
        })
        .collect()
}

/// Multi-continent offline digital elevation fallback (meters ASL) used when Copernicus GLO-90
/// (`/v1/elevation`) has not yet been fetched for the active transect.
pub fn sample_terrain_elevation_m(lat: f64, lon: f64) -> f64 {
    if (-126.0..=-115.0).contains(&lon) && (32.0..=51.0).contains(&lat) {
        if lon < -124.18 {
            return 0.0;
        }
        let coast_range = 620.0 * (-((lon + 123.62) / 0.24).powi(2)).exp()
            * (0.65 + 0.35 * ((lat - 42.0) * 2.1).sin().abs());
        let valley_floor = if lon > -123.45 && lon < -122.55 && lat > 43.8 && lat < 47.5 {
            68.0
        } else {
            180.0
        };
        let cascade_ridge = 1850.0 * (-((lon + 121.85) / 0.42).powi(2)).exp();
        let hood = 1550.0
            * (-(((lat - 45.374) / 0.12).powi(2) + ((lon + 121.696) / 0.14).powi(2))).exp();
        let sisters = 1320.0
            * (-(((lat - 44.103) / 0.15).powi(2) + ((lon + 121.769) / 0.15).powi(2))).exp();
        let shasta = 2350.0
            * (-(((lat - 41.409) / 0.16).powi(2) + ((lon + 122.195) / 0.16).powi(2))).exp();
        let east_plateau = if lon > -121.3 {
            980.0 * (1.0 - (-((lon + 121.3) / 0.6).powi(2)).exp())
        } else {
            0.0
        };
        let gorge_cut = 1.0 - 0.82 * (-((lat - 45.65) / 0.11).powi(2)).exp();
        let raw = valley_floor
            + coast_range
            + (cascade_ridge + hood + sisters + shasta) * gorge_cut
            + east_plateau;
        return raw.clamp(0.0, 4350.0);
    }

    if (-105.0..-90.0).contains(&lon) && (25.0..=48.0).contains(&lat) {
        let plains_slope = ((-lon - 94.0).max(0.0) * 105.0).clamp(15.0, 1450.0);
        let hill_country =
            320.0 * (-(((lat - 30.2) / 1.4).powi(2) + ((lon + 99.0) / 1.6).powi(2))).exp();
        return (plains_slope + hill_country).clamp(0.0, 2800.0);
    }

    if (-83.0..=-68.0).contains(&lon) && (24.0..=46.0).contains(&lat) {
        let appalachian =
            920.0 * (-((lon + 78.6 + (lat - 38.0) * 0.45) / 0.95).powi(2)).exp();
        let coastal = if lon > -75.4 { 12.0 } else { 85.0 };
        return (coastal + appalachian).clamp(0.0, 2100.0);
    }

    if (-12.0..=18.0).contains(&lon) && (33.0..=58.0).contains(&lat) {
        let pyrenees =
            1950.0 * (-((lat - 42.8) / 0.55).powi(2)).exp() * (-((lon + 0.5) / 2.2).powi(2)).exp();
        let betics =
            1280.0 * (-((lat - 37.05) / 0.45).powi(2)).exp() * (-((lon + 3.4) / 1.6).powi(2)).exp();
        let base = 55.0 + 90.0 * ((lat * 0.4).sin().abs());
        return (base + pyrenees + betics).clamp(0.0, 3400.0);
    }

    if (98.0..=127.0).contains(&lon) && (20.0..=50.0).contains(&lat) {
        let mongol_step = if lat > 41.5 || lon < 112.0 {
            1180.0 * (1.0 - (-((lat - 39.5).max(0.0) / 3.2).powi(2)).exp())
        } else {
            38.0
        };
        let lushan =
            980.0 * (-(((lat - 29.55) / 0.25).powi(2) + ((lon - 115.98) / 0.25).powi(2))).exp();
        return (mongol_step + lushan).clamp(0.0, 3200.0);
    }

    120.0
}
