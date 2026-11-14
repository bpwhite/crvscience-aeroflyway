# AeroFlyway Headless Validation Suite

This directory holds empirical validation artifacts generated deterministically by running the Rust integration suite in `tests/headless_validation.rs` against the compiled `aeroflyway` library (`src/lib.rs`).

## Running the Validation Suite

```bash
cargo test
# or target only the headless validation integration suite:
cargo test --test headless_validation
```

## Test Coverage & Generated Artifacts

1. `t1_lights_out_collision_mortality_curve`: Runs 36-hour nocturnal Atlantic Americas flyway physics simulations (`AeroPhysicsEngine`) across VIIRS skyglow Lights-Out curfews (`0%`, `20%`, `40%`, `60%`, `80%`, `100%`), tracking urban collision strikes, avoided mortality, peak skyglow entrainment, latitude bands, and altitude stratification -> writes [`lights_out_mortality_curve.csv`](lights_out_mortality_curve.csv).
2. `t2_nexrad_vpts_altitude_profile_integration`: Parses WSR-88D `KDOX` VPTS vertical reflectivity CSV data with S-band clutter filtering (`sd_vvp < 2.0 m/s`), integrates Migration Traffic Rate (`MTR`, birds/km/hr) across 200 m height gates alongside regional stations (`KDIX`, `KAKQ`) -> writes [`vpts_altitude_profile.csv`](vpts_altitude_profile.csv).
3. `t3_bioacoustic_nfc_stft_classification_and_sqlite_roundtrip`: Synthesizes a 48 kHz dual-guild nocturnal flight-call WAV (`PasserineNfcDuet48k`), verifies 16-bit PCM + GUANO coordinate roundtrip, validates Thrush (~3.2–3.5 kHz) vs. Warbler (~7.3 kHz) STFT spectrogram classification & band dominance, and audits `FlywayStore` SQLite persistence -> merges all empirical metrics into [`aeroflyway_validation_summary.json`](aeroflyway_validation_summary.json).
