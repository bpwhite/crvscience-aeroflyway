# AeroFlyway v0.6.0 User Guide

AeroFlyway is a desktop aeroecology workstation combining a `wgpu`/WGSL GPU agent simulation (up to `100,000` agents) with a Rayon CPU fallback (`480` agents), live weather/radar/biodiversity ingestion, a vertical airspace cross-section profiler, a NEXRAD/Aloft `VPTS` radar inspector, and a WAV/GUANO STFT bioacoustics lab.

---

## 1. Startup Campaign Hub

Launching `aeroflyway` (without `--skip-hub`) opens the **Startup Campaign Hub** (`AppStage::StartupHub`). Press `H` at any time to switch between the Startup Hub and the 3D GIS Workbench.

### SQLite Campaign Workspaces (Left Column)
- **Campaign Title & `.db` File**: Enter a campaign name and path (default `aeroflyway.db`), or use **Browse .db...**, **New .db...**, or **Open / Create .db** to switch SQLite WAL workspaces.
- **Quick Campaign Templates**: Load preconfigured regional campaigns with one click:
  - `aeroflyway.db` — *Pacific Willamette & Gorge Baseline* (`pacific`)
  - `platte_nexrad_spring.db` — *Platte Crane & Bracken NEXRAD* (`central`)
  - `atlantic_lights_out.db` — *NYC-Philly-Cape May Lights-Out* (`atlantic`)
  - `palearctic_aloft.db` — *Falsterbo-Gibraltar Aloft VPTS* (`palearctic`)
- **Pre-Flight Climate, Skyglow & Compute**: Adjust starting **Day of Year (DOY 1–365)**, **Lights-Out Curfew (0–100%)**, **CMIP6 SSP scenario**, optional **eBird 2.0 API key** (or `EBIRD_API_KEY` env var), and toggle `wgpu` WGSL GPU acceleration.

### Global Flyway Presets & Mode Launchers (Right Column)
Select any of the 5 global corridors, then jump straight into a workbench mode:
1. **Launch 3D Flyway GIS Workbench** (opens Tab `[1]` Airspace Cross-Section)
2. **Open NEXRAD / Aloft Radar Lab** (opens Tab `[2]` VPTS Profiler)
3. **Open WAV Bioacoustic STFT Lab** (opens Tab `[3]` WAV STFT Inspector)
4. **Open Multi-Scenario Batch Sweep** (opens Tab `[4]` Batch Sweep Lab)

---

## 2. 2.5D Pitched Map vs. 2D Plan Mode

AeroFlyway renders an interactive Web Mercator (`EPSG:3857`) map supporting both **2.5D oblique perspective** and **2D top-down plan** views:

- **Camera Controls**:
  - **Primary-drag**: Pan across the flyway corridor (when tool mode is `Pan / Inspect`).
  - **Secondary-drag**: Orbit camera **Bearing** (`-180°..+180°`) and tilt **Pitch** (`0°..62°`).
  - **Scroll wheel**: Cursor-anchored continuous zoom (`3.5..14.5`).
  - **Left Dock Sliders / Buttons**: Fine-tune `Pitch`, `Bearing`, and `Alt Exag` (`0.00x..2.50x`), switch quickly between **2.5D Oblique** (`42°` pitch, `-18°` bearing) and **2D Plan** (`0°` pitch, `0°` bearing), or click **Reset (0/R)** (or press `0` / `R`) to return to the active corridor's default 2.5D viewport.
- **Visual Depth & Basemap Rendering**:
  - In 2.5D pitched mode (`pitch > 2°`), airborne agents render **ground shadow ellipses** and **vertical altitude drop-line stems** anchoring their elevated positions to the ground plane, alongside normalized clip-space perspective depth (`clip_pos.z`).
  - Online raster basemaps (**Carto Dark Matter**, **USGS National Topo**, **OpenStreetMap**) are subdivided into **`4x4` triangle grids** (`32` triangles per tile) with **atmospheric horizon fade** when tilted so distant tiles blend cleanly into the background sky without seams. **Vector Dark (Offline)** renders offline coastlines, rivers, borders, and graticules without requiring network access.

---

## 3. Multi-Taxa Simulation, Scenario Presets & Environmental Controls

### Simulation Transport & Taxa Layers
- **Play / Pause (`Space`)**, **Step +1h**, and **Speed (`1x`, `4x`, `16x`)** live in the top command bar.
- Toggle **WGSL GPU** vs **CPU Rayon** compute in the top bar (and adjust VRAM swarm size `512..100,000` in the left dock).
- Use the **Layers** menu to show/hide:
  - **Birds** (cyan; turns **magenta** when skyglow-entrained `<300 m AGL`, **orange** when smoke-disrupted `>= 160 ug/m^3`, **emerald** in rain fallout stopover)
  - **Bats** (violet; turns **crimson** when entering turbine rotor collision boxes `|Δlat| < 0.032°`, `|Δlon| · cos(lat) < 0.032°`)
  - **Butterflies** (amber/gold; Monarchs accumulating GDD across generations `1..4` and seeking milkweed patches)
  - **VIIRS Nighttime Skyglow Domes**, **NEXRAD / Aloft Radar Rings**, **Wind Turbines & Rotors**, **Host-Plant Stepping-Stones**, **Spatial H25 Hex Choropleth**, **Synoptic 850 hPa Wind Vectors**, and **Refuges, Motus & Summits**.

### Three One-Click Scenario Presets
In the left dock under **Scenario Presets**, click any preset (hover for details):
1. **Autumn Lights-Out**: DOY `272`, Historical Baseline, `85%` BirdCast Lights-Out curfew, `5.5 m/s` smart bat turbine curtailment, `1.15x` 850 hPa tailwind.
2. **Megafire SSP5-8.5**: DOY `258`, CMIP6 `SSP5-8.5` (`+4.1 °C`), `+195 ug/m^3` wildfire smoke plume forcing `>1,850 m AGL` climbs and trophic mismatch.
3. **Spring Fallout**: DOY `128`, CMIP6 `SSP2-4.5` (`+2.4 °C`), `2.6 mm/hr` frontal precipitation forcing coastal/stopover fallout, `0%` Lights-Out curfew, `1.6x` VIIRS skyglow intensity.

All environmental sliders—including `wind_850_multiplier`, `radar_nudge_weight`, `skyglow_intensity_mult`, `lights_out_curfew_pct`, `smoke_pm25_offset_ugm3`, `turbine_cut_in_mps`, `precip_front_mm_hr`, and `day_of_year`—provide hover tooltips summarizing their ecological units and physical behavior.

---

## 4. Drawing Airspace Transects (`T`)

1. Press `T` (or click **Draw A-B Transect (T)** in the top bar) to enter transect drawing mode; press `T` again or `Esc` to return to `Pan / Inspect`.
2. **Primary-drag** across the map from origin **A** to endpoint **B** to reposition the cross-section swath (`±38 km` half-width).
3. Switch to bottom tab **`[1] Airspace Cross-Section (0-4,200 m ASL)`** (press `1`):
   - Displays terrain elevation (Copernicus **GLO-90** when synced online, or the built-in **Regional Analytical DEM** offline), turbine rotor envelope ceiling (`+135 m`), radar VPTS peak layer reference line, severe smoke layer (`>= 160 ug/m^3`), **X-axis distance ticks (`0 km (A)` to `total_km (B)`)**, and viewport aspect-corrected circular GPU point sprites.

---

## 5. Planting Virtual Milkweed Host Patches (`M`)

1. Press `M` (or click **Plant Host Patch (M)** in the top bar) to enter habitat planting mode; press `M` again or `Esc` to exit.
2. **Primary-click** anywhere on the map to plant a `25 ha` virtual *Asclepias* stepping-stone (`is_virtual = true`), immediately persisted into the active SQLite `.db` (`milkweed_patches` table) and uploaded to GPU/CPU attractor buffers so daytime Monarch butterflies steer toward corridor gaps.

---

## 6. NEXRAD / Aloft Radar Vertical Profiles (`[2]`)

Press `2` (or **Tools -> Open Radar Vertical Profile Tab (2)**) to inspect vertical migration profiles (`VPTS`):
- **Station Selector**: Click any regional NEXRAD (`KRTX`, `KMAX`, `KUEX`, `KDDC`, `KEWX`, `KDIX`, `KAKQ`), Aloft OPERA (`nlhrw`, `bejab`, `behel`), or EAAF station button to view its vertical profile and center the map on that radar.
- **VPTS CSV Import**: Click **Browse .csv...** / **Load VPTS**, or drag-and-drop any Aloft/bioRad `VPTS` CSV file (`radar,datetime,height,u,v,...,dens,sd_vvp`) directly onto the window. Clutter gates below the wavelength threshold (`sd_vvp < 2.0 m/s` S-band, `< 1.0 m/s` C-band) are filtered out automatically, and MTR is integrated using each gate's actual vertical thickness `Δh`.
- **Interactive Altitude Bin Hover Tooltips**: Hover over any horizontal density bar or `(u, v)` wind barb to inspect the highlighted gate's exact **altitude (`m ASL`)**, **bird density (`birds/km^3`)**, **reflectivity (`eta`, `cm^2/km^3`)**, **ground speed (`m/s` and `km/h`) & track heading**, **wind `(u, v)` components**, and **Doppler spectral width (`sd_vvp`)**.

---

## 7. WAV STFT Bioacoustics Workbench (`[3]`)

Press `3` to open **`[3] WAV STFT Bioacoustic Inspector`**:
- **Presets & Audio Loading**: Click **Hoary Bat (250 kHz)**, **Little Brown Myotis (250 kHz FM + Buzz)**, or **Thrush & Warbler NFC (48 kHz)**, use **Browse .wav...** / **Load .wav**, or drag-and-drop any 8/16/24/32-bit PCM or 32-bit IEEE float `.wav` file onto the window. Embedded `guan` (GUANO) `Loc Position:` GPS coordinates are extracted automatically.
- **Controls**: Select FFT size (`256`, `512`, `1024`, `2048`), adjust **dB floor** (`-95..-35 dBFS`), zoom **Max kHz** up to Nyquist, and toggle the **Ridge** frequency trace.
- **Metrics & Pulse Table**: Displays **Guild classification**, **Characteristic frequency (`Fc`)**, and **Pieretti power-based ACI**. Click any row (`P1`, `P2`, ...) in the **Detected Call Pulses** table to highlight its bounding box on the Magma spectrogram, or hover the spectrogram for live `(ms, kHz, dBFS)` readouts.
- **Export & Map Pinning**:
  - **Save Sample .wav**: Writes 16-bit mono PCM `.wav` with GUANO coordinates.
  - **Export 10x Audible .wav**: Writes a 10x time-expanded `.wav` (`sample_rate / 10`, floor `1,000 Hz`) so ultrasonic bat passes (`20–80 kHz`) become audible (`2–8 kHz`).
  - **Pin Call to Map**: Upserts the classified detection into SQLite (`observations`) and centers the GIS map on its GUANO coordinates.

---

## 8. Lights-Out Curfews & Batch Sweep Lab (`[4]`)

- **Lights-Out Controls**: Use `L` (prints live `status_line` confirmation with predicted strike reduction), left-dock preset buttons (**Unmitigated (0%)**, **50% Window Dim**, **BirdCast Alert (85%)**), or the **Lights-Out Curfew** slider (`0–100%`).
- **Batch Sweep Lab (`4`)**: Enter comma-separated parameter lists (`Cut-In m/s`, `Lights-Out %`, `+Patch`, `dTemp`, `Smoke`), set **Hours** (`6–240 h`), pick an output folder, and click **Run Rayon v0.6 Sweep** to compute Pareto-optimal risk reduction across every scenario combination (`batch_parameter_sweep.csv` and `.json`).
- **GIS Exports**: From **File**, export **RFC 7946 GeoJSON** (`aeroflyway_corridor.geojson`) or **CSV Bundle** (`corridor_hex_summary.csv`, `corridor_observations.csv`).

---

## 9. Complete Keyboard Shortcuts & CLI Reference

### Keyboard Shortcuts

| Key | Scope | Function |
|---|---|---|
| `Space` | Global (non-text) | Play / Pause simulation |
| `H` | Global (non-text) | Toggle Startup Campaign Hub <-> 3D GIS Workbench |
| `0` / `R` | Global (non-text) | Reset camera to active corridor default 2.5D view |
| `1` | Global (non-text) | Switch to `[1] Airspace Cross-Section` tab |
| `2` | Global (non-text) | Switch to `[2] NEXRAD / Aloft Vertical Profile (VPTS)` tab |
| `3` | Global (non-text) | Switch to `[3] WAV STFT Bioacoustic Inspector` tab |
| `4` | Global (non-text) | Switch to `[4] Multi-Scenario Batch Sweep Lab` tab |
| `T` | Global (non-text) | Toggle `Draw A-B Transect` tool mode |
| `M` | Global (non-text) | Toggle `Plant Virtual Milkweed Host Patch` tool mode |
| `L` | Global (non-text) | Toggle 85% BirdCast Lights-Out curfew vs 0% unmitigated |
| `Ctrl+K` / `Cmd+K` | Global (non-text) | Open fuzzy command palette (`Up`/`Down`/`Enter`/`Esc`) for views, corridors, simulation, and exports |
| `Ctrl+S` / `Cmd+S` | Global | Save campaign metadata into active SQLite `.db` |
| `Esc` | Global | Close palette or About/Help modal, or reset tool mode to `Pan / Inspect` |
| `F1` | Global (non-text) | Toggle Shortcuts & Scientific Grounding help modal |

### CLI Subcommands & Flags

- `aeroflyway [gui] [--skip-hub]`: Launch desktop GUI (optionally skipping Startup Hub).
- `aeroflyway --screenshot <out.png> [--screenshot-view <1|2|3>]`: Render 1600x1000 offscreen RGBA PNG via `wgpu`.
- `aeroflyway regions`: Print all 5 global flyway regions, bounds, refuges, radar stations, and VIIRS skyglow domes.
- `aeroflyway radar [--region <code>] [--vpts-csv <path>] [--smoke-pm25 <f64>] [--out-dir <dir>]`: Export `radar_vpts_profiles.csv`.
- `aeroflyway batch [--region <code>] [--doy <f64>] [--season <spring|summer|autumn|overwinter>] [--ssp <baseline|ssp126|ssp245|ssp370|ssp585>] [--lights-out <csv>] [--smoke-pm25 <csv>] [--hours <f64>] [--cut-in <csv>] [--milkweed-add <csv>] [--temp-offset <csv>] [--wind-mult <f64>] [--precip <f64>] [--db <path>] [--out-dir <dir>] [--export-best-geojson]`: Run parallel parameter sweep and write CSV/JSON/GeoJSON reports.
- `aeroflyway inspect-wav <path> [--fft <usize>] [--pin-db] [--export-pngs] [--default-lat <f64>] [--default-lon <f64>] [--db <path>] [--out-dir <dir>]`: Batch STFT analysis across `.wav` files/folders.
- `aeroflyway sync [--region <code>] [--ssp <code>] [--lat <f64>] [--lon <f64>] [--ebird-key <token>] [--warmup-hours <f64>] [--db <path>] [--export-dir <dir>]`: Headless live API + Radar S3 + GLO-90 sync and optional snapshot export.
- `aeroflyway compare-corridors [--doy <f64>] [--hours <f64>] [--lights-out <f64>] [--out-dir <dir>]`: Compare `0%` unmitigated vs. Lights-Out curfew urban strike & bat rotor mortality across all 5 global flyways (`aeroflyway_multi_corridor_lights_out.json` & `.csv`).
