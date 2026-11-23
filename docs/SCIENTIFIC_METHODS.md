# AeroFlyway v0.6.0 Scientific Methods & Numerical Formulation

This document specifies the physical, biological, signal-processing, and 3D/2D rendering formulations implemented across `src/physics.rs`, `src/gpu_pipeline.rs`, `src/shaders/compute.wgsl`, `src/shaders/map_render.wgsl`, `src/shaders/transect_render.wgsl`, `src/acoustics.rs`, `src/ingest.rs`, `src/map_engine.rs`, and `src/main.rs`.

---

## 1. Curfew-Adjusted VIIRS Skyglow & Urban Building Collision Model

### 1.1 Effective VIIRS Skyglow Radiance (`CPU` & `GPU` Parity)
Urban skyglow domes are parameterized by peak VIIRS Day/Night Band upward radiance $R_{\text{peak}}$ ($\text{nW}\,\text{cm}^{-2}\,\text{sr}^{-1}$), radius $R_d$ ($\text{km}$), and facade glass-hazard index $H_g$ (`src/basemap.rs`).

Given local simulation hour $h \in [0, 24)$, nocturnal flag $N \in \{\text{true}, \text{false}\}$, municipal BirdCast / Audubon `Lights-Out` curtailment percentage $p_{\text{LO}} \in [0, 100]$, and skyglow multiplier $m_{\text{glow}} \in [0, 3]$, the curfew dimming multiplier (`compute_skyglow_curfew_dim` in `src/physics.rs`) is:

$$
D_{\text{curfew}}(h, N, p_{\text{LO}}) =
\begin{cases}
0, & \neg N \\
1 - 0.88 \cdot \dfrac{p_{\text{LO}}}{100}, & N \land (h \ge 23 \lor h < 6) \quad \text{(active curfew window)} \\
1 - 0.32 \cdot \dfrac{p_{\text{LO}}}{100}, & N \land (6 \le h < 23) \quad \text{(early dusk / pre-dawn shoulder)}
\end{cases}
$$

The effective dome radiance (`effective_skyglow_radiance_nw`) is:

$$
R_{\text{eff}} = \max\!\big(0,\; R_{\text{peak}} \cdot \text{clamp}(m_{\text{glow}}, 0, 3) \cdot D_{\text{curfew}}\big)
$$

At geographic distance $d$ ($\text{km}$), Gaussian atmospheric glow decay yields $R_{\text{local}} = R_{\text{eff}} \exp\!\left(-\frac{1}{2}(d / R_{\sigma})^2\right)$.

### 1.2 Shared `SimUniforms` GPU Coupling
In `pack_gpu_scene_buffers` (`src/gpu_pipeline.rs`), each `kind == 2` skyglow attractor stores `weight` $= R_{\text{eff}}$. To eliminate dusk (`0.32`) vs. curfew (`0.88`) discrepancy when reconstructing unmitigated baseline risk inside `src/shaders/compute.wgsl`, `pack_gpu_uniforms` passes `curfew_dim` $= D_{\text{curfew}}$ directly inside the 192-byte (`48`-scalar, `12 x vec4`, `#[repr(C)] Pod + Zeroable`) `GpuSimUniforms` buffer:

$$
R_{\text{unmitigated}} = \frac{R_{\text{local}}}{\max(\text{sim.curfew\_dim}, 0.08)}
$$

Because $D_{\text{curfew}} \in [0.12, 1.0]$ whenever nocturnal attraction is active, `R_local / curfew_dim` recovers the exact unmitigated radiance at both `21:00` dusk ($D = 1 - 0.32 f$) and `01:00` curfew ($D = 1 - 0.88 f$).

### 1.3 Phototactic Entrainment & Non-Linear Window-Strike Probability
Nocturnal passerines (`taxon == 0u`, `best_glow > 14.0 nW/cm^2/sr`, non-smoke conditions) experience canopy descent toward $28\text{--}320\text{ m AGL}$ (`flags |= 8u`) and spiral attraction via `halo_steer`. Following Van Doren et al. (2021, *PNAS*), step collision probability scales with the `1.68` phototactic power-law exponent and 850 hPa wind amplification $A_w = \text{clamp}(1 + 0.035\,|\mathbf{v}_{850}|, 1.0, 1.6)$:

$$
p_{\text{strike}} = c \cdot \max\!\left(\frac{R}{100},\, 0.05\right)^{1.68} \cdot A_w \cdot \min(\Delta t,\, \Delta t_{\max})
$$

Analytical reduction across curtailment $p_{\text{LO}}$ (`compute_lights_out_collision_reduction_pct`) is:

$$
\text{Avoided}(\%) = \text{clamp}\!\left(100\left[1 - \left(1 - 0.88\frac{p_{\text{LO}}}{100}\right)^{1.68}\right],\, 0,\, 95\right)
$$

---

## 2. Bat Turbine Collision Geometry (`cos(lat)` Scaling) & Poisson Hazard Rate

### 2.1 Latitude-Aware Metric Encounter Box
Because longitudinal arc length shrinks with latitude as $\Delta x_{\text{km}} = 111.32\,|\Delta \lambda|\,\cos(\phi)$, a fixed degree threshold distorts the horizontal turbine encounter box across latitudes (e.g., `29°N` Texas vs. `55.5°N` Falsterbo). Both `src/shaders/compute.wgsl` and `src/physics.rs` scale longitudinal separation by $\cos(\phi_t)$ (`max(|cos(lat_rad)|, 0.2)`):

$$
|\phi_{\text{bat}} - \phi_{\text{turbine}}| < \Delta_{\phi}, \qquad
|\lambda_{\text{bat}} - \lambda_{\text{turbine}}| \cdot \max(|\cos(\phi_{\text{turbine}})|,\, 0.2) < \Delta_{\phi}
$$

with $\Delta_\phi = 0.032^\circ$ ($\approx 3.56\text{ km}$ half-width on GPU) and $\Delta_\phi = 0.045^\circ$ on CPU, paired with vertical rotor-swept bounds:

$$
z_{\text{lo}} = \max(H_{\text{hub}} - R_{\text{rotor}},\, 15.0\text{ m}), \qquad z_{\text{hi}} = H_{\text{hub}} + R_{\text{rotor}}
$$

### 2.2 Continuous-Time Poisson Collision Probability (`1 - exp(-rate * dt)`)
Rather than applying a frame-rate-dependent fixed probability per frame, `compute.wgsl` integrates the continuous-time hourly rotor collision hazard rate $\lambda_{\text{bat}} = 0.16\text{ hr}^{-1}$ (`BAT_TURBINE_COLLISION_RATE_PER_HR`) over `sim.dt_hours`:

$$
p_{\text{hit}}(\Delta t) = 1 - \exp\!\left(-\lambda_{\text{bat}} \cdot \max(\Delta t, 0)\right)
$$

This guarantees exact sub-stepping invariance: $1 - \big(1 - p_{\text{hit}}(\Delta t / N)\big)^N = p_{\text{hit}}(\Delta t)$. Hits are accumulated separately against the baseline cut-in ($3.0 \le |\mathbf{v}_{80}| \le 25.0\text{ m/s}$) and the active curtailment cut-in ($v_{\text{cut-in}} \le |\mathbf{v}_{80}| \le 25.0\text{ m/s}$).

---

## 3. Pieretti Power-Based Acoustic Complexity Index (ACI) & Ultrasonic Bat Pulse Retention

### 3.1 Hann-Windowed STFT & Power-Based Pieretti ACI
In `src/acoustics.rs`, signals are windowed by a length-$N_{\text{FFT}}$ Hann window $w[k]$ normalized by coherent gain $G_w = \frac{1}{2}\sum_k w[k]$, producing complex radix-2 DFT bins $X_t[b]$ with magnitude $M_{t,b} = |X_t[b]| / G_w$ and **spectral power / intensity**:

$$
I_{t,b} = M_{t,b}^2, \qquad \text{dBFS}_{t,b} = \text{clamp}\!\left(10 \log_{10}(I_{t,b} + 10^{-18}),\, -105,\, 0\right)
$$

Across all $T$ uniform-hop analysis frames ($t = 0, \dots, T-1$) and frequency bins $b \in [2, N_{\text{FFT}}/2)$, the Pieretti et al. (2011, *Ecological Indicators*) Acoustic Complexity Index is computed using spectral power $I_{t,b}$ and **including the final frame $T-1$ in the denominator**:

$$
\text{ACI} = \sum_{b=2}^{N_{\text{FFT}}/2 - 1} \frac{\sum_{t=0}^{T-2} \left|I_{t,b} - I_{t+1,b}\right|}{\sum_{t=0}^{T-1} I_{t,b}}
\quad \text{(for } \sum_{t=0}^{T-1} I_{t,b} > 10^{-12}\text{)}
$$

Including the final frame $t = T - 1$ guarantees the mathematical Pieretti upper bound $0 \le \text{ACI}_b \le 2$ per bin even when vocalizations occur at the trailing boundary.

### 3.2 Blind-Gap-Free Overlapping Analysis & Max-Hold Spectrogram Aggregation
Widening the STFT hop on multi-second `250–384 kHz` ultrasonic recordings to cap display width at 900 columns previously introduced inter-frame gaps wider than short (`2.0–4.8 ms`) *Myotis* echolocation pulses. `compute_stft_spectrogram` eliminates blind gaps by decoupling analysis resolution from UI column resolution:
1. **Overlapping Analysis Frames**: Uses `base_hop = max(N_FFT / 4, 64)` (`75%` overlap), widening at most to `N_FFT / 2` (`50%` overlap) only when `raw_frames > 24,000`. Every sample lies inside overlapping Hann windows, so short `~2 ms` bat pulses are always captured.
2. **Pulse Extraction & ACI on Fine Frames**: Spectral ridge detection (`gate_db = max(global_peak_db - 24, -58)`) and pulse bounds (`start_ms`, `duration_ms`, `Fc`, `Fmin`, `Fmax`, `peak_dbfs`) operate directly on the fine-resolution frames in O(`num_bins`) streaming memory.
3. **Max-Hold UI Spectrogram (`<= 900` Columns)**: When `analysis_frames > 900`, each UI column `c` (`0..900`) aggregates its spanned fine frames via per-bin **max-hold** (`max dBFS`), preserving visual visibility of millisecond bat pulses on multi-second displays.

---

## 4. VPTS Migration Traffic Rate (`MTR`) Over Variable Bin Heights `Δh`

Weather radar `VPTS` biological profiles (`src/ingest.rs`) filter non-avian insect/ground clutter by Doppler radial velocity standard deviation (`sd_vvp >= 2.0 m/s` for S-band `< 7.5 cm` NEXRAD/EAAF; `>= 1.0 m/s` for C-band Aloft OPERA). `parse_vpts_csv_text` preserves exact integer-meter gate heights (`height_m.round()`) rather than snapping to `200 m`.

In `summarize_profile_mtr_and_peak`, bins are sorted ascending by altitude $h_0 \le h_1 \le \dots \le h_{n-1}$, and each gate's vertical layer thickness $\Delta h_i$ (m) is computed from adjacent gate centers:

$$
\Delta h_i =
\begin{cases}
200\text{ m}, & n = 1 \\
h_1 - h_0, & i = 0 \\
\frac{1}{2}(h_{i+1} - h_{i-1}), & 0 < i < n - 1 \\
h_{n-1} - h_{n-2}, & i = n - 1
\end{cases}
$$

Migration Traffic Rate ($\text{birds}\,\text{km}^{-1}\,\text{hr}^{-1}$) integrates across all gates using exact layer thickness $\Delta h_i / 1000$ ($\text{km}$):

$$
\text{MTR} = \sum_{i=0}^{n-1} \rho_i \cdot \left(3.6\,\sqrt{u_i^2 + v_i^2}\right) \cdot \frac{\Delta h_i}{1000}
$$

This gives exact `5.0x` scaling between `100 m` and `500 m` gates of equal density/velocity, exact equality with `0.2 km` on uniform `200 m` NEXRAD gates, and seamless handling of irregular or unsorted height profiles.

---

## 5. 3D Perspective Projection, Cross-Section Sprites & Basemap Tiling

### 5.1 2.5D Oblique Perspective Depth & Ground Cues (`map_render.wgsl`)
Relative-to-center Web Mercator offsets $(g_e, g_n)$ in pixels are rotated by camera bearing $\beta$ and tilted by pitch $\theta \in [0, 62^\circ]$:

$$
r_x = g_e \cos\beta - g_n \sin\beta, \qquad
f_y = g_e \sin\beta + g_n \cos\beta, \qquad
z_{\text{elev}} = \frac{\text{alt\_agl\_m}}{36.0} \cdot \text{cam\_alt\_exag}
$$

With camera depth $d_c = \max(1.15\,H_{\text{canvas}}, 680)$, near plane $z_{\text{near}} = 0.22\,d_c$, and far plane $z_{\text{far}} = 8.0\,d_c$:

$$
v_z = \max\!\left(d_c + f_y \sin\theta - z_{\text{elev}} \cos\theta,\; z_{\text{near}}\right), \qquad
\text{persp} = \frac{d_c}{v_z}, \qquad
z_{\text{ndc}} = \text{clamp}\!\left(\frac{z_{\text{far}}(v_z - z_{\text{near}})}{v_z (z_{\text{far}} - z_{\text{near}})},\; 0.001,\; 0.999\right)
$$

In `gpu_pipeline.rs`, `map_render_pipeline` draws 3 instanced quads per agent (`0..18`, 6 vertices per part):
- `part == 0`: **Ground shadow ellipse** at ground footprint $(g_x, g_y, z_{\text{ndc},0})$, foreshortened vertically by $\cos\theta$.
- `part == 1`: **Vertical altitude drop-line stem** spanning ground footprint to airborne position with linearly interpolated depth $z_{\text{stem}}$.
- `part == 2`: **Taxon SDF silhouette** (animated bird chevron, scalloped bat, or 4-lobed butterfly) attenuated smoothly near the far horizon.

### 5.2 Aspect-Corrected Circular Transect Sprites (`transect_render.wgsl`)
`AeroFlywayApp` tracks the inner plot dimensions $(W_{\text{tr}}, H_{\text{tr}})$ of `[1] Airspace Cross-Section` and packs `transect_w_pts`, `transect_h_pts`, and `transect_aspect` $= W_{\text{tr}} / H_{\text{tr}}$ into `GpuSimUniforms`. `transect_render.wgsl` sizes each circular point sprite (`r_pts = 2.6`) in NDC as:

$$
r_{\text{ndc}, y} = \frac{2\,r_{\text{pts}}}{H_{\text{tr}}}, \qquad
r_{\text{ndc}, x} = \frac{r_{\text{ndc}, y}}{\text{transect\_aspect}}
$$

ensuring $1:1$ circular sprites across arbitrary dock heights and window aspect ratios.

### 5.3 `4x4` Tilted Basemap Subdivision & Horizon Fade (`src/main.rs`)
Raster map tiles are subdivided into `4x4` quad grids (`25` vertices, `32` triangles per tile) via `build_tilted_tile_mesh_4x4`. In 2D plan view (`pitch <= 1.0°`), vertex alpha stays unattenuated (`1.0`). When tilted (`pitch > 1.0°`), `compute_tile_horizon_fade` smoothly reduces alpha toward the upper viewport horizon (`>= 0.14`) as `persp` drops below `0.78`, preventing abrupt distant tile seams.
