# AeroFlyway v0.6.0

Pure-Rust desktop aeroecology GIS, `wgpu`/WGSL 100,000-agent multi-flyway simulator, **Startup Campaign Hub**, **VIIRS nighttime skyglow `Lights-Out` urban collision model**, **NEXRAD WSR-88D / ENRAM Aloft vertical radar profile (`VPTS`) analyzer**, continuous 365-day phenology engine, and WAV/GUANO bioacoustic STFT workbench.

## What's New in v0.6.0
- **Startup Campaign Hub & Native Menu Bar (`src/main.rs`, `src/db.rs`)**:
  - Launches into an interactive **Startup Campaign Hub** (`AppStage::StartupHub`, bypassable via `aeroflyway --skip-hub`) with SQLite WAL `.db` campaign switching, 5-region global flyway cards, pre-flight climate/curfew sliders, and direct mode launchers (`3D GIS Map`, `NEXRAD/Aloft Radar Lab`, `WAV STFT Lab`, `Batch Sweep Lab`).
  - Persistent native top menu bar (`File / Corridor / Layers / Tools / Help`) plus keyboard shortcuts (`Space`, `H`, `R`, `L`, `F1`).
- **VIIRS Nighttime Skyglow & BirdCast `Lights-Out` Collision Mitigation (`src/basemap.rs`, `src/physics.rs`, `compute.wgsl`)**:
  - Regional VIIRS Day/Night Band (`nW/cm^2/sr`) urban skyglow domes across all 5 global flyways (Portland, Seattle, Sacramento; DFW, San Antonio-Austin; NYC-Newark, Philadelphia; Randstad, Copenhagen-Malmo, Seville; Shanghai, Tianjin-Bohai).
  - Models nocturnal phototactic attraction, `<300 m AGL` urban canopy disorientation (`flags & 8u`), and non-linear window-lit collision reduction (`Van Doren et al. 2021, PNAS`) controlled by the `23:00-06:00` **Lights-Out Curfew %** slider.
- **NEXRAD WSR-88D & ENRAM Aloft Vertical Radar Profiles (`VPTS` CSV + S3 Probes, `src/ingest.rs`)**:
  - Pure-Rust `VPTS` CSV parser (`radar,datetime,height,u,v,...,dens,sd_vvp`), public S3 endpoint probes (`unidata-nexrad-level2-chunks` for `KRTX`, `KMAX`, `KUEX`, `KDDC`, `KEWX`, `KDIX`; `aloftdata.s3-eu-west-1.amazonaws.com` for `nlhrw`, `bejab`, `behel`), Migration Traffic Rate (`MTR`, `birds/km/hr`) integration, and a dedicated **NEXRAD / Aloft Vertical Profile (VPTS)** dock tab.

## Quick Start & Headless CLI
```bash
# Launch desktop Startup Campaign Hub + 100k-agent WGSL GIS workstation
cargo run --release

# Launch directly into the 3D GIS map (skip Startup Hub)
cargo run --release -- --skip-hub

# Inspect regional NEXRAD / Aloft radar vertical profiles or parse a VPTS CSV
cargo run --release -- radar --region palearctic
cargo run --release -- radar --region pacific --vpts-csv sample_vpts.csv

# Run Rayon multi-scenario sweeps jointly varying turbine cut-in & Lights-Out curfew %
cargo run --release -- batch --region atlantic --ssp ssp245 --cut-in 3.5,5.0,6.0 --lights-out 0,50,85 --export-best-geojson
```

## Citation & CRV Science Apps Hub

- **CRV Science Apps Hub**: <https://crvscience.com/apps>
- **Companion Preprint (`APP-PREPRINT-01`)**: [AeroFlyway: GPU-Accelerated Continental Avian Migration Simulation, NEXRAD Radar Ingestion, and Bioacoustic Flight-Call Classification](https://crvscience.com/apps/preprints/aeroflyway-avian-migration-radar)
- **Zenodo Concept DOI**: [10.5281/zenodo.24222366](https://doi.org/10.5281/zenodo.24222366)
- **Validation Suite & Telemetry**: [`validation/README.md`](validation/README.md) (`cargo test`)
- **Machine-Readable Metadata**: [`CITATION.cff`](CITATION.cff)

```bibtex
@article{White2026AeroFlyway,
  author  = {White, Bryan P.},
  title   = {AeroFlyway: GPU-Accelerated Continental Avian Migration Simulation, NEXRAD Radar Ingestion, and Bioacoustic Flight-Call Classification},
  journal = {CRV Science Apps Preprints},
  year    = {2026},
  doi     = {10.5281/zenodo.24222366},
  url     = {https://crvscience.com/apps/preprints/aeroflyway-avian-migration-radar},
  note    = {Software v0.6.0; Preprint ID: APP-PREPRINT-01}
}
```
