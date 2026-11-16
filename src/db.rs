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

//! Bundled SQLite WAL spatiotemporal datastore (`aeroflyway.db` — AeroFlyway v0.6.0).
//! Stores multi-flyway public-data sightings, regional wind turbines, host-plant patches,
//! Startup Campaign Hub metadata (`workspace_campaigns`), persisted NEXRAD/Aloft vertical
//! radar profile bins (`radar_vpts_bins`), EnKF snapshots, and cached raster map tiles.

use crate::basemap::{
    seed_milkweed_patches_for_region, seed_wind_turbines_for_region, FlywayRegion, MilkweedPatch,
    Taxon, WindTurbine,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationRecord {
    pub obs_uid: String,
    pub taxon: Taxon,
    pub common_name: String,
    pub scientific_name: String,
    pub lat: f64,
    pub lon: f64,
    pub hex_bin: String,
    pub altitude_m: f64,
    pub count: u32,
    pub source_api: String,
    pub observed_at: String,
}

#[derive(Clone, Debug)]
pub struct ModelSnapshot {
    pub tick_hour: f64,
    pub enkf_rmse: f64,
    pub avian_flux_index: f64,
    pub bat_rotor_contacts: u32,
    pub monarch_front_lat: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CampaignSummary {
    pub campaign_name: String,
    pub db_path: String,
    pub region_code: String,
    pub doy: f64,
    pub cmip6_code: String,
    pub lights_out_pct: f64,
    pub observation_count: usize,
    pub updated_at: String,
}

/// Deterministic hexagonal grid bin key (~25 km cells) for zero-dependency H3-style aggregation.
pub fn spatial_hex_bin(lat: f64, lon: f64) -> String {
    let cell_deg_lat = 0.225_f64;
    let cos_lat = (lat.to_radians().cos()).abs().max(0.25);
    let cell_deg_lon = cell_deg_lat / cos_lat;

    let row = (lat / (cell_deg_lat * 0.866_025_4)).floor() as i32;
    let lon_offset = if row.rem_euclid(2) == 1 {
        0.5 * cell_deg_lon
    } else {
        0.0
    };
    let col = ((lon + lon_offset) / cell_deg_lon).floor() as i32;
    format!("H25:{row}:{col}")
}

/// Returns approximate center `(lat, lon)` of a deterministic `H25:row:col` hexagonal bin.
pub fn hex_bin_center(hex_key: &str) -> Option<(f64, f64)> {
    let mut parts = hex_key.split(':');
    let prefix = parts.next()?;
    if prefix != "H25" {
        return None;
    }
    let row: i32 = parts.next()?.parse().ok()?;
    let col: i32 = parts.next()?.parse().ok()?;

    let cell_deg_lat = 0.225_f64;
    let lat = ((row as f64) + 0.5) * (cell_deg_lat * 0.866_025_4);
    let cos_lat = (lat.to_radians().cos()).abs().max(0.25);
    let cell_deg_lon = cell_deg_lat / cos_lat;
    let lon_offset = if row.rem_euclid(2) == 1 {
        0.5 * cell_deg_lon
    } else {
        0.0
    };
    let lon = ((col as f64) + 0.5) * cell_deg_lon - lon_offset;
    Some((lat, lon))
}

pub struct FlywayStore {
    conn: Connection,
}

impl FlywayStore {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = db_path.as_ref();
        let conn = Connection::open(path_ref).map_err(|e| format!("SQLite open error: {e}"))?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS observations (
                obs_uid TEXT PRIMARY KEY,
                taxon TEXT NOT NULL,
                common_name TEXT NOT NULL,
                scientific_name TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                hex_bin TEXT NOT NULL,
                altitude_m REAL NOT NULL,
                count INTEGER NOT NULL,
                source_api TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_obs_taxon ON observations(taxon);
            CREATE INDEX IF NOT EXISTS idx_obs_hex ON observations(hex_bin);

            CREATE TABLE IF NOT EXISTS wind_turbines (
                case_id INTEGER PRIMARY KEY,
                project_name TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                hub_height_m REAL NOT NULL,
                rotor_diameter_m REAL NOT NULL,
                capacity_kw REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS milkweed_patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                area_ha REAL NOT NULL,
                is_virtual INTEGER NOT NULL DEFAULT 0,
                gdd_threshold REAL NOT NULL DEFAULT 165.0
            );

            CREATE TABLE IF NOT EXISTS model_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tick_hour REAL NOT NULL,
                enkf_rmse REAL NOT NULL,
                avian_flux_index REAL NOT NULL,
                bat_rotor_contacts INTEGER NOT NULL,
                monarch_front_lat REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workspace_campaigns (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                campaign_name TEXT NOT NULL,
                region_code TEXT NOT NULL,
                doy REAL NOT NULL,
                cmip6_code TEXT NOT NULL,
                lights_out_pct REAL NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS radar_vpts_bins (
                station_code TEXT NOT NULL,
                height_m_asl REAL NOT NULL,
                bird_density_km3 REAL NOT NULL,
                u_mps REAL NOT NULL,
                v_mps REAL NOT NULL,
                sd_vvp_mps REAL NOT NULL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (station_code, height_m_asl)
            );

            CREATE TABLE IF NOT EXISTS tile_cache (
                provider TEXT NOT NULL,
                z INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                png_blob BLOB NOT NULL,
                PRIMARY KEY (provider, z, x, y)
            );
            ",
        )
        .map_err(|e| format!("SQLite schema migration error: {e}"))?;

        let store = Self { conn };
        store.ensure_baseline_seed(FlywayRegion::PacificAmericas)?;
        Ok(store)
    }

    fn ensure_baseline_seed(&self, region: FlywayRegion) -> Result<(), String> {
        let obs_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
            .unwrap_or(0);

        if obs_count == 0 {
            let baseline = baseline_observations_for_region(region);
            self.upsert_observations(&baseline)?;
        }

        let turb_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM wind_turbines", [], |r| r.get(0))
            .unwrap_or(0);
        if turb_count == 0 {
            self.upsert_turbines(&seed_wind_turbines_for_region(region))?;
        }

        let patch_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM milkweed_patches", [], |r| r.get(0))
            .unwrap_or(0);
        if patch_count == 0 {
            for p in seed_milkweed_patches_for_region(region) {
                self.insert_milkweed_patch(
                    &p.label,
                    p.lat,
                    p.lon,
                    p.area_ha,
                    p.is_virtual,
                    p.gdd_threshold,
                )?;
            }
        }
        Ok(())
    }

    pub fn save_campaign_metadata(
        &self,
        campaign_name: &str,
        region: FlywayRegion,
        doy: f64,
        cmip6_code: &str,
        lights_out_pct: f64,
    ) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT INTO workspace_campaigns (id, campaign_name, region_code, doy, cmip6_code, lights_out_pct, updated_at)
                 VALUES (1, ?1, ?2, ?3, ?4, ?5, '2026-11-14T04:10Z')
                 ON CONFLICT(id) DO UPDATE SET
                    campaign_name = excluded.campaign_name,
                    region_code = excluded.region_code,
                    doy = excluded.doy,
                    cmip6_code = excluded.cmip6_code,
                    lights_out_pct = excluded.lights_out_pct,
                    updated_at = excluded.updated_at",
                params![campaign_name, region.code(), doy, cmip6_code, lights_out_pct],
            )
            .map_err(|e| format!("Save campaign metadata error: {e}"))?;
        Ok(())
    }

    pub fn load_campaign_summary(&self, db_path_label: &str) -> CampaignSummary {
        let obs_count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
            .unwrap_or(0);

        let row: Option<(String, String, f64, String, f64, String)> = self
            .conn
            .query_row(
                "SELECT campaign_name, region_code, doy, cmip6_code, lights_out_pct, updated_at
                 FROM workspace_campaigns WHERE id = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
            )
            .optional()
            .unwrap_or(None);

        if let Some((name, reg, doy, ssp, lo, updated)) = row {
            CampaignSummary {
                campaign_name: name,
                db_path: db_path_label.to_string(),
                region_code: reg,
                doy,
                cmip6_code: ssp,
                lights_out_pct: lo,
                observation_count: obs_count.max(0) as usize,
                updated_at: updated,
            }
        } else {
            CampaignSummary {
                campaign_name: "Default Pacific Flyway Campaign".to_string(),
                db_path: db_path_label.to_string(),
                region_code: "pacific".to_string(),
                doy: 272.0,
                cmip6_code: "baseline".to_string(),
                lights_out_pct: 35.0,
                observation_count: obs_count.max(0) as usize,
                updated_at: "2026-11-14T04:00Z".to_string(),
            }
        }
    }

    pub fn upsert_radar_bins(
        &self,
        station_code: &str,
        observed_at: &str,
        bins: &[(f64, f64, f64, f64, f64)],
    ) -> Result<usize, String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("SQLite radar tx error: {e}"))?;
        let mut count = 0usize;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO radar_vpts_bins (
                        station_code, height_m_asl, bird_density_km3, u_mps, v_mps, sd_vvp_mps, observed_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    ON CONFLICT(station_code, height_m_asl) DO UPDATE SET
                        bird_density_km3 = excluded.bird_density_km3,
                        u_mps = excluded.u_mps,
                        v_mps = excluded.v_mps,
                        sd_vvp_mps = excluded.sd_vvp_mps,
                        observed_at = excluded.observed_at",
                )
                .map_err(|e| format!("Prepare radar_vpts_bins error: {e}"))?;
            for &(h, dens, u, v, sd) in bins {
                stmt.execute(params![station_code, h, dens, u, v, sd, observed_at])
                    .map_err(|e| format!("Upsert radar bin error: {e}"))?;
                count += 1;
            }
        }
        tx.commit().map_err(|e| format!("Commit radar tx error: {e}"))?;
        Ok(count)
    }

    /// Resets regional turbines, baseline patches, and seed observations when switching global flyway presets.
    pub fn switch_region_seed(&self, region: FlywayRegion) -> Result<(), String> {
        let _ = self.conn.execute("DELETE FROM wind_turbines", []);
        let _ = self.conn.execute("DELETE FROM milkweed_patches", []);
        self.upsert_turbines(&seed_wind_turbines_for_region(region))?;
        for p in seed_milkweed_patches_for_region(region) {
            self.insert_milkweed_patch(
                &p.label,
                p.lat,
                p.lon,
                p.area_ha,
                p.is_virtual,
                p.gdd_threshold,
            )?;
        }
        let seed_obs = baseline_observations_for_region(region);
        self.upsert_observations(&seed_obs)?;
        Ok(())
    }

    pub fn upsert_observations(&self, records: &[ObservationRecord]) -> Result<usize, String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("SQLite tx error: {e}"))?;
        let mut count = 0usize;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO observations (
                        obs_uid, taxon, common_name, scientific_name, lat, lon,
                        hex_bin, altitude_m, count, source_api, observed_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                    ON CONFLICT(obs_uid) DO UPDATE SET
                        count = excluded.count,
                        observed_at = excluded.observed_at",
                )
                .map_err(|e| format!("Prepare obs error: {e}"))?;

            for rec in records {
                stmt.execute(params![
                    rec.obs_uid,
                    rec.taxon.code(),
                    rec.common_name,
                    rec.scientific_name,
                    rec.lat,
                    rec.lon,
                    rec.hex_bin,
                    rec.altitude_m,
                    rec.count,
                    rec.source_api,
                    rec.observed_at,
                ])
                .map_err(|e| format!("Insert obs error: {e}"))?;
                count += 1;
            }
        }
        tx.commit().map_err(|e| format!("Commit error: {e}"))?;
        Ok(count)
    }

    pub fn load_observations(&self) -> Vec<ObservationRecord> {
        let mut stmt = match self.conn.prepare(
            "SELECT obs_uid, taxon, common_name, scientific_name, lat, lon,
                    hex_bin, altitude_m, count, source_api, observed_at
             FROM observations
             ORDER BY observed_at DESC
             LIMIT 2500",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = stmt.query_map([], |row| {
            let t_str: String = row.get(1)?;
            Ok(ObservationRecord {
                obs_uid: row.get(0)?,
                taxon: Taxon::from_code(&t_str),
                common_name: row.get(2)?,
                scientific_name: row.get(3)?,
                lat: row.get(4)?,
                lon: row.get(5)?,
                hex_bin: row.get(6)?,
                altitude_m: row.get(7)?,
                count: row.get(8)?,
                source_api: row.get(9)?,
                observed_at: row.get(10)?,
            })
        });

        match rows {
            Ok(iter) => iter.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn upsert_turbines(&self, list: &[WindTurbine]) -> Result<usize, String> {
        let tx = self
            .conn
            .unchecked_transaction()
            .map_err(|e| format!("SQLite turbine tx error: {e}"))?;
        let mut count = 0usize;
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO wind_turbines (
                        case_id, project_name, lat, lon, hub_height_m, rotor_diameter_m, capacity_kw
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    ON CONFLICT(case_id) DO UPDATE SET
                        hub_height_m = excluded.hub_height_m,
                        rotor_diameter_m = excluded.rotor_diameter_m",
                )
                .map_err(|e| format!("Prepare turbine error: {e}"))?;

            for t in list {
                stmt.execute(params![
                    t.case_id,
                    t.project_name,
                    t.lat,
                    t.lon,
                    t.hub_height_m,
                    t.rotor_diameter_m,
                    t.capacity_kw,
                ])
                .map_err(|e| format!("Insert turbine error: {e}"))?;
                count += 1;
            }
        }
        tx.commit().map_err(|e| format!("Commit turbine error: {e}"))?;
        Ok(count)
    }

    pub fn load_turbines(&self) -> Vec<WindTurbine> {
        let mut stmt = match self.conn.prepare(
            "SELECT case_id, project_name, lat, lon, hub_height_m, rotor_diameter_m, capacity_kw
             FROM wind_turbines",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let res = match stmt.query_map([], |r| {
            Ok(WindTurbine {
                case_id: r.get(0)?,
                project_name: r.get(1)?,
                lat: r.get(2)?,
                lon: r.get(3)?,
                hub_height_m: r.get(4)?,
                rotor_diameter_m: r.get(5)?,
                capacity_kw: r.get(6)?,
            })
        }) {
            Ok(it) => it.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        };
        res
    }

    pub fn insert_milkweed_patch(
        &self,
        label: &str,
        lat: f64,
        lon: f64,
        area_ha: f64,
        is_virtual: bool,
        gdd_threshold: f64,
    ) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT INTO milkweed_patches (label, lat, lon, area_ha, is_virtual, gdd_threshold)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![label, lat, lon, area_ha, i32::from(is_virtual), gdd_threshold],
            )
            .map_err(|e| format!("Insert milkweed error: {e}"))?;
        Ok(())
    }

    pub fn clear_virtual_milkweed_patches(&self) -> Result<(), String> {
        self.conn
            .execute("DELETE FROM milkweed_patches WHERE is_virtual = 1", [])
            .map_err(|e| format!("Clear virtual milkweed error: {e}"))?;
        Ok(())
    }

    pub fn load_milkweed_patches(&self) -> Vec<MilkweedPatch> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, label, lat, lon, area_ha, is_virtual, gdd_threshold FROM milkweed_patches",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let res = match stmt.query_map([], |r| {
            let v_flag: i32 = r.get(5)?;
            Ok(MilkweedPatch {
                id: r.get(0)?,
                label: r.get(1)?,
                lat: r.get(2)?,
                lon: r.get(3)?,
                area_ha: r.get(4)?,
                is_virtual: v_flag != 0,
                gdd_threshold: r.get(6)?,
            })
        }) {
            Ok(it) => it.filter_map(Result::ok).collect(),
            Err(_) => Vec::new(),
        };
        res
    }

    pub fn record_snapshot(&self, snap: &ModelSnapshot) {
        let _ = self.conn.execute(
            "INSERT INTO model_snapshots (tick_hour, enkf_rmse, avian_flux_index, bat_rotor_contacts, monarch_front_lat)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                snap.tick_hour,
                snap.enkf_rmse,
                snap.avian_flux_index,
                snap.bat_rotor_contacts,
                snap.monarch_front_lat
            ],
        );
    }

    pub fn get_cached_tile(&self, provider: &str, z: u8, x: u32, y: u32) -> Option<Vec<u8>> {
        self.conn
            .query_row(
                "SELECT png_blob FROM tile_cache WHERE provider = ?1 AND z = ?2 AND x = ?3 AND y = ?4",
                params![provider, z, x, y],
                |r| r.get(0),
            )
            .optional()
            .ok()
            .flatten()
    }

    pub fn save_cached_tile(&self, provider: &str, z: u8, x: u32, y: u32, png_blob: &[u8]) {
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO tile_cache (provider, z, x, y, png_blob)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![provider, z, x, y, png_blob],
        );
    }
}

pub fn baseline_observations_for_region(region: FlywayRegion) -> Vec<ObservationRecord> {
    let raw: &[(&str, Taxon, &str, &str, f64, f64, f64, u32, &str, &str)] = match region {
        FlywayRegion::PacificAmericas => &[
            ("seed-pac-1", Taxon::Bird, "Cackling Goose", "Branta hutchinsii", 44.4128, -123.3285, 420.0, 640, "USFWS-Finley-Seed", "2026-10-24"),
            ("seed-pac-2", Taxon::Bird, "Swainson's Thrush", "Catharus ustulatus", 44.7876, -123.0768, 860.0, 48, "Motus-Willamette", "2026-09-29"),
            ("seed-pac-3", Taxon::Bird, "Sandhill Crane", "Antigone canadensis", 45.8080, -122.7475, 740.0, 310, "USFWS-Ridgefield", "2026-10-18"),
            ("seed-pac-4", Taxon::Bat, "Hoary Bat", "Lasiurus cinereus", 45.6380, -120.6050, 95.0, 14, "NABat-ColumbiaGorge", "2026-09-22"),
            ("seed-pac-5", Taxon::Bat, "Silver-haired Bat", "Lasionycteris noctivagans", 44.5043, -123.5512, 110.0, 19, "NABat-CoastRange", "2026-09-25"),
            ("seed-pac-6", Taxon::Butterfly, "Western Monarch", "Danaus plexippus", 42.3920, -122.8750, 28.0, 36, "Xerces-WesternMonarch", "2026-10-05"),
            ("seed-pac-7", Taxon::Butterfly, "Western Monarch", "Danaus plexippus", 36.9515, -122.0268, 15.0, 1420, "Xerces-SantaCruzRoost", "2026-11-08"),
        ],
        FlywayRegion::CentralAmericasPlatte => &[
            ("seed-cen-1", Taxon::Bird, "Sandhill Crane", "Antigone canadensis", 40.6680, -98.8920, 610.0, 18500, "Rowe-Platte-Survey", "2026-03-26"),
            ("seed-cen-2", Taxon::Bird, "Whooping Crane", "Grus americana", 38.1850, -98.4950, 780.0, 42, "USFWS-Quivira", "2026-04-08"),
            ("seed-cen-3", Taxon::Bat, "Mexican Free-tailed Bat", "Tadarida brasiliensis", 29.6877, -98.3522, 240.0, 45000, "BCI-BrackenEmergence", "2026-08-19"),
            ("seed-cen-4", Taxon::Butterfly, "Eastern Monarch", "Danaus plexippus", 34.7450, -98.7120, 45.0, 520, "JourneyNorth-Central", "2026-10-11"),
        ],
        FlywayRegion::AtlanticAmericas => &[
            ("seed-atl-1", Taxon::Bird, "Red Knot", "Calidris canutus", 39.0863, -74.7229, 320.0, 2100, "DelawareBay-Shorebird", "2026-05-21"),
            ("seed-atl-2", Taxon::Bird, "Broad-winged Hawk", "Buteo platypterus", 40.6350, -75.9880, 920.0, 840, "HawkMountain-Count", "2026-09-19"),
            ("seed-atl-3", Taxon::Bat, "Eastern Red Bat", "Lasiurus borealis", 38.9593, -74.9274, 88.0, 26, "NABat-CapeMay", "2026-09-28"),
            ("seed-atl-4", Taxon::Butterfly, "Atlantic Monarch", "Danaus plexippus", 38.9593, -74.9274, 18.0, 690, "CapeMay-MonarchProject", "2026-10-09"),
        ],
        FlywayRegion::EastAtlanticPalearctic => &[
            ("seed-pal-1", Taxon::Bird, "European Honey Buzzard", "Pernis apivorus", 55.3830, 12.8170, 890.0, 3400, "Falsterbo-Observatory", "2026-09-04"),
            ("seed-pal-2", Taxon::Bird, "Black Stork", "Ciconia nigra", 35.9000, -5.3000, 1120.0, 410, "MIGRES-Gibraltar", "2026-09-22"),
            ("seed-pal-3", Taxon::Bat, "Nathusius' Pipistrelle", "Pipistrellus nathusii", 53.4200, 6.1500, 74.0, 38, "EUROBATS-Wadden", "2026-09-14"),
            ("seed-pal-4", Taxon::Butterfly, "Painted Lady", "Vanessa cardui", 36.9800, -6.3700, 65.0, 950, "EBMS-Donana", "2026-04-12"),
        ],
        FlywayRegion::EastAsianAustralasian => &[
            ("seed-eaa-1", Taxon::Bird, "Siberian Crane", "Leucogeranus leucogeranus", 29.1500, 116.2170, 540.0, 1280, "EAAFP-Poyang", "2026-11-06"),
            ("seed-eaa-2", Taxon::Bird, "Bar-tailed Godwit", "Limosa lapponica", 39.1500, 118.2500, 1150.0, 4200, "EAAFP-BohaiBay", "2026-05-03"),
            ("seed-eaa-3", Taxon::Bat, "Asian Parti-colored Bat", "Vespertilio sinensis", 33.3800, 120.5200, 96.0, 22, "Acoustic-YellowSea", "2026-09-18"),
            ("seed-eaa-4", Taxon::Butterfly, "Chestnut Tiger", "Parantica sita", 29.5600, 115.9800, 140.0, 185, "Lushan-MigrationTag", "2026-10-16"),
        ],
    };

    raw.iter()
        .map(
            |&(uid, taxon, common, sci, lat, lon, alt, count, src, date)| ObservationRecord {
                obs_uid: uid.to_string(),
                taxon,
                common_name: common.to_string(),
                scientific_name: sci.to_string(),
                lat,
                lon,
                hex_bin: spatial_hex_bin(lat, lon),
                altitude_m: alt,
                count,
                source_api: src.to_string(),
                observed_at: date.to_string(),
            },
        )
        .collect()
}
