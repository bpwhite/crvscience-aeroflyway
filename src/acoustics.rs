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

//! Pure-Rust RIFF/WAVE Audio Decoder/Synthesizer, Radix-2 Cooley-Tukey FFT, Hann-Windowed
//! STFT Spectrogram Generator, GUANO Metadata Extractor, and Bioacoustic Call Classifier.
//! Supports 8/16/24/32-bit PCM and 32-bit IEEE float `.wav` recordings from 16 kHz avian
//! recorders up to 384 kHz full-spectrum ultrasonic bat detectors (AudioMoth / Pettersson).

use crate::basemap::Taxon;
use eframe::egui::ColorImage;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthPreset {
    HoaryBatUltrasonic250k,
    MyotisBuzzUltrasonic250k,
    PasserineNfcDuet48k,
}

impl SynthPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::HoaryBatUltrasonic250k => "Hoary Bat (250 kHz QCF 27->19 kHz)",
            Self::MyotisBuzzUltrasonic250k => "Little Brown Myotis (250 kHz FM + Buzz)",
            Self::PasserineNfcDuet48k => "Thrush & Warbler NFC (48 kHz 2.9 & 6.8 kHz)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct WavRecording {
    pub source_label: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub samples_mono: Vec<f32>,
    pub guano_lat_lon: Option<(f64, f64)>,
}

impl WavRecording {
    pub fn duration_ms(&self) -> f64 {
        if self.sample_rate_hz == 0 {
            0.0
        } else {
            (self.samples_mono.len() as f64 * 1000.0) / (self.sample_rate_hz as f64)
        }
    }

    pub fn nyquist_khz(&self) -> f64 {
        (self.sample_rate_hz as f64) * 0.0005
    }
}

#[derive(Clone, Debug)]
pub struct CallPulse {
    pub start_ms: f64,
    pub duration_ms: f64,
    pub f_max_khz: f64,
    pub f_min_khz: f64,
    pub f_char_khz: f64,
    pub peak_dbfs: f64,
}

#[derive(Clone, Debug)]
pub struct SpectrogramResult {
    pub num_frames: usize,
    pub num_bins: usize,
    pub fft_size: usize,
    pub hop_size: usize,
    pub sample_rate_hz: u32,
    pub duration_ms: f64,
    pub max_freq_khz: f64,
    /// Row-major `[frame_idx * num_bins + bin_idx]` power in dBFS (`<= 0.0`).
    pub power_dbfs: Vec<f32>,
    /// Peak frequency (kHz) per frame where signal exceeds noise floor + 10 dB.
    pub ridge_khz: Vec<Option<f32>>,
    pub pulses: Vec<CallPulse>,
    pub aci_index: f64,
    pub characteristic_freq_khz: f64,
    pub f_max_khz: f64,
    pub f_min_khz: f64,
    pub mean_duration_ms: f64,
    pub classified_taxon: Taxon,
    pub guild_label: String,
    pub scientific_hint: String,
}

pub const SPECTROGRAM_DB_CEIL: f32 = -2.0;

impl SpectrogramResult {
    /// Converts the dBFS spectrogram matrix up to `view_max_khz` into an RGBA `ColorImage`
    /// (width = `num_frames`, height = displayed frequency bins, row 0 = highest frequency).
    pub fn to_color_image(&self, db_floor: f32, view_max_khz: f64) -> ColorImage {
        if self.num_frames == 0 || self.num_bins == 0 {
            return ColorImage::from_rgba_unmultiplied([1, 1], &[12, 18, 28, 255]);
        }

        let nyquist = if self.max_freq_khz.is_finite() {
            self.max_freq_khz.max(0.5)
        } else {
            0.5
        };
        let min_khz = nyquist.min(2.0);
        let sanitized_top = if view_max_khz.is_finite() {
            view_max_khz
        } else {
            nyquist
        };
        let clamped_top_khz = sanitized_top.clamp(min_khz, nyquist);
        let max_bin = (((clamped_top_khz / nyquist) * (self.num_bins as f64)).ceil() as usize)
            .clamp(8.min(self.num_bins), self.num_bins);

        let width = self.num_frames;
        let height = max_bin;
        let mut rgba = vec![0_u8; width * height * 4];
        let db_ceil = SPECTROGRAM_DB_CEIL;
        let span = (db_ceil - db_floor).max(6.0);

        for row in 0..height {
            // Top row corresponds to highest displayed frequency bin
            let bin = max_bin - 1 - row;
            for col in 0..width {
                let db = self.power_dbfs[col * self.num_bins + bin];
                let norm = ((db - db_floor) / span).clamp(0.0, 1.0);
                let (r, g, b) = magma_lut(norm);
                let off = (row * width + col) * 4;
                rgba[off] = r;
                rgba[off + 1] = g;
                rgba[off + 2] = b;
                rgba[off + 3] = 255;
            }
        }

        ColorImage::from_rgba_unmultiplied([width, height], &rgba)
    }

    pub fn sample_dbfs_at(&self, norm_x: f32, norm_y_from_top: f32, view_max_khz: f64) -> (f64, f64, f32) {
        if self.num_frames == 0 || self.num_bins == 0 {
            return (0.0, 0.0, -100.0);
        }
        let nyquist = if self.max_freq_khz.is_finite() {
            self.max_freq_khz.max(0.5)
        } else {
            0.5
        };
        let min_khz = nyquist.min(2.0);
        let sanitized_top = if view_max_khz.is_finite() {
            view_max_khz
        } else {
            nyquist
        };
        let top_khz = sanitized_top.clamp(min_khz, nyquist);
        let time_ms = (norm_x.clamp(0.0, 1.0) as f64) * self.duration_ms;
        let freq_khz = (1.0 - norm_y_from_top.clamp(0.0, 1.0) as f64) * top_khz;

        let col = ((norm_x.clamp(0.0, 0.9999) * self.num_frames as f32) as usize)
            .min(self.num_frames - 1);
        let bin = (((freq_khz / nyquist) * self.num_bins as f64) as usize).min(self.num_bins - 1);
        let db = self.power_dbfs[col * self.num_bins + bin];
        (time_ms, freq_khz, db)
    }
}

/// Reads and parses a `.wav` file from disk.
pub fn load_wav_file(path: impl AsRef<Path>) -> Result<WavRecording, String> {
    let p = path.as_ref();
    let bytes = fs::read(p).map_err(|e| format!("Failed reading {}: {e}", p.display()))?;
    let label = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("recording.wav")
        .to_string();
    parse_wav_bytes(&bytes, label)
}

/// Pure-Rust RIFF/WAVE decoder supporting PCM (8/16/24/32-bit), IEEE float (32-bit),
/// `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`), and embedded `guan` (GUANO) GPS metadata chunks.
pub fn parse_wav_bytes(bytes: &[u8], source_label: String) -> Result<WavRecording, String> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid RIFF/WAVE header".to_string());
    }

    let mut pos = 12usize;
    let mut format_tag = 0_u16;
    let mut channels = 1_u16;
    let mut sample_rate_hz = 48_000_u32;
    let mut bits_per_sample = 16_u16;
    let mut data_slice: Option<&[u8]> = None;
    let mut guano_lat_lon: Option<(f64, f64)> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_len = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(chunk_len).min(bytes.len());
        let body = &bytes[body_start..body_end];

        if chunk_id == b"fmt " && body.len() >= 16 {
            format_tag = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]).max(1);
            sample_rate_hz = u32::from_le_bytes([body[4], body[5], body[6], body[7]]).max(1000);
            bits_per_sample = u16::from_le_bytes([body[14], body[15]]);
            // WAVE_FORMAT_EXTENSIBLE stores subformat tag in bytes 24..26
            if format_tag == 0xFFFE && body.len() >= 26 {
                format_tag = u16::from_le_bytes([body[24], body[25]]);
            }
        } else if chunk_id == b"data" {
            data_slice = Some(body);
        } else if chunk_id == b"guan" {
            guano_lat_lon = parse_guano_coordinates(body);
        }

        // RIFF chunks are word-aligned (padded to even byte boundary)
        let padded = chunk_len + (chunk_len % 2);
        pos = body_start.saturating_add(padded);
    }

    let raw_pcm = data_slice.ok_or_else(|| "WAV file missing 'data' chunk".to_string())?;
    let ch = channels as usize;
    let bytes_per_sample = ((bits_per_sample as usize) + 7) / 8;
    let frame_stride = bytes_per_sample.saturating_mul(ch).max(1);

    // Cap decoded length to 12 seconds max so high-rate 384 kHz files stay snappy in RAM
    let max_frames = (sample_rate_hz as usize).saturating_mul(12);
    let total_frames = (raw_pcm.len() / frame_stride).min(max_frames);
    let mut samples_mono = Vec::with_capacity(total_frames);

    for f_idx in 0..total_frames {
        let base = f_idx * frame_stride;
        let mut acc = 0.0_f32;
        for c in 0..ch {
            let s_off = base + c * bytes_per_sample;
            let sample_f = match (format_tag, bits_per_sample) {
                (1, 8) => ((raw_pcm[s_off] as f32) - 128.0) / 128.0,
                (1, 16) if s_off + 2 <= raw_pcm.len() => {
                    let v = i16::from_le_bytes([raw_pcm[s_off], raw_pcm[s_off + 1]]);
                    (v as f32) / 32768.0
                }
                (1, 24) if s_off + 3 <= raw_pcm.len() => {
                    let b0 = raw_pcm[s_off] as i32;
                    let b1 = (raw_pcm[s_off + 1] as i32) << 8;
                    let b2 = ((raw_pcm[s_off + 2] as i8) as i32) << 16;
                    ((b0 | b1 | b2) as f32) / 8_388_608.0
                }
                (1, 32) if s_off + 4 <= raw_pcm.len() => {
                    let v = i32::from_le_bytes([
                        raw_pcm[s_off],
                        raw_pcm[s_off + 1],
                        raw_pcm[s_off + 2],
                        raw_pcm[s_off + 3],
                    ]);
                    (v as f32) / 2_147_483_648.0
                }
                (3, 32) if s_off + 4 <= raw_pcm.len() => f32::from_le_bytes([
                    raw_pcm[s_off],
                    raw_pcm[s_off + 1],
                    raw_pcm[s_off + 2],
                    raw_pcm[s_off + 3],
                ])
                .clamp(-1.0, 1.0),
                _ => 0.0,
            };
            acc += sample_f;
        }
        samples_mono.push(acc / (ch as f32));
    }

    if samples_mono.len() < 256 {
        return Err("WAV recording too short (< 256 samples)".to_string());
    }

    Ok(WavRecording {
        source_label,
        sample_rate_hz,
        channels,
        bits_per_sample,
        samples_mono,
        guano_lat_lon,
    })
}

/// Writes a mono 16-bit PCM `.wav` file with optional GUANO GPS coordinates (`Loc Position`).
pub fn write_wav_16bit(
    path: impl AsRef<Path>,
    recording: &WavRecording,
    coords: Option<(f64, f64)>,
) -> Result<(), String> {
    let pcm_bytes_len = (recording.samples_mono.len() * 2) as u32;
    let guano_str = coords
        .map(|(lat, lon)| {
            format!(
                "GUANO|Version: 1.0\nLoc Position: {lat:.5} {lon:.5}\nSamplerate: {}\n",
                recording.sample_rate_hz
            )
        })
        .unwrap_or_default();
    let guano_bytes = guano_str.as_bytes();
    let guano_padded_len = guano_bytes.len() + (guano_bytes.len() % 2);
    let guano_chunk_total = if guano_bytes.is_empty() {
        0_u32
    } else {
        (8 + guano_padded_len) as u32
    };

    let riff_size = 36_u32
        .saturating_add(pcm_bytes_len)
        .saturating_add(guano_chunk_total);

    let mut buf = Vec::with_capacity((riff_size + 8) as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk (16 bytes PCM)
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16_u32.to_le_bytes());
    buf.extend_from_slice(&1_u16.to_le_bytes()); // PCM = 1
    buf.extend_from_slice(&1_u16.to_le_bytes()); // Mono = 1
    buf.extend_from_slice(&recording.sample_rate_hz.to_le_bytes());
    let byte_rate = recording.sample_rate_hz * 2;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2_u16.to_le_bytes()); // block_align
    buf.extend_from_slice(&16_u16.to_le_bytes()); // bits_per_sample

    if !guano_bytes.is_empty() {
        buf.extend_from_slice(b"guan");
        buf.extend_from_slice(&(guano_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(guano_bytes);
        if guano_bytes.len() % 2 == 1 {
            buf.push(0);
        }
    }

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&pcm_bytes_len.to_le_bytes());
    for &s in &recording.samples_mono {
        let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&q.to_le_bytes());
    }

    fs::write(path.as_ref(), buf)
        .map_err(|e| format!("Failed writing {}: {e}", path.as_ref().display()))
}

/// Builds a time-expanded copy of `recording` with header sample rate divided by `expansion_factor`
/// (typically `10` for ultrasonic bat passes, shifting e.g. 250 kHz -> 25 kHz and 40 kHz -> 4 kHz).
pub fn make_time_expanded_recording(
    recording: &WavRecording,
    expansion_factor: u32,
) -> WavRecording {
    let factor = expansion_factor.max(1);
    let expanded_sr = (recording.sample_rate_hz / factor).max(1000);
    let stem = recording
        .source_label
        .strip_suffix(".wav")
        .or_else(|| recording.source_label.strip_suffix(".WAV"))
        .unwrap_or(&recording.source_label);
    WavRecording {
        source_label: format!("{stem}_{factor}x_audible.wav"),
        sample_rate_hz: expanded_sr,
        channels: 1,
        bits_per_sample: 16,
        samples_mono: recording.samples_mono.clone(),
        guano_lat_lon: recording.guano_lat_lon,
    }
}

/// Writes a time-expanded 16-bit mono PCM `.wav` file (`sample_rate_hz / expansion_factor`),
/// returning the expanded output sample rate in Hz.
pub fn write_time_expanded_wav_16bit(
    path: impl AsRef<Path>,
    recording: &WavRecording,
    expansion_factor: u32,
    coords: Option<(f64, f64)>,
) -> Result<u32, String> {
    let expanded = make_time_expanded_recording(recording, expansion_factor);
    write_wav_16bit(path, &expanded, coords)?;
    Ok(expanded.sample_rate_hz)
}

fn parse_guano_coordinates(body: &[u8]) -> Option<(f64, f64)> {
    let txt = std::str::from_utf8(body).ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("Loc Position:") {
            let mut parts = rest.split_whitespace();
            let lat: f64 = parts.next()?.parse().ok()?;
            let lon: f64 = parts.next()?.parse().ok()?;
            if (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon) {
                return Some((lat, lon));
            }
        }
    }
    None
}

/// Synthesizes realistic reference ultrasonic bat passes and nocturnal avian flight calls.
pub fn synthesize_preset_recording(preset: SynthPreset) -> WavRecording {
    match preset {
        SynthPreset::HoaryBatUltrasonic250k => {
            // 250 kHz full-spectrum ultrasonic recording (360 ms) containing 3 QCF/FM search pulses
            // sweeping from ~27.5 kHz down to 19.2 kHz Fc (diagnostic of Lasiurus cinereus < 23 kHz)
            let sr = 250_000_u32;
            let n = (sr as usize * 360) / 1000;
            let mut samples = vec![0.0_f32; n];
            let pulse_starts_ms = [42.0, 158.0, 272.0];

            for (i, s) in samples.iter_mut().enumerate() {
                let t_sec = i as f64 / sr as f64;
                let t_ms = t_sec * 1000.0;
                // Subtle ultrasonic ambient noise floor
                let mut val = (pseudo_noise(i as u64 + 19) - 0.5) * 0.012;

                for (p_idx, &p0) in pulse_starts_ms.iter().enumerate() {
                    let dur_ms = 14.5 + (p_idx as f64) * 1.2;
                    if t_ms >= p0 && t_ms <= p0 + dur_ms {
                        let u = (t_ms - p0) / dur_ms; // 0..1
                        // Exponential sweep flattening into QCF tail near 19.4 kHz
                        let f_start = 27_800.0 - (p_idx as f64) * 450.0;
                        let f_end = 19_300.0;
                        let tau = 0.28 * (dur_ms / 1000.0);
                        let dt = (t_ms - p0) / 1000.0;
                        let phase = 2.0
                            * PI
                            * (f_end * dt + (f_start - f_end) * tau * (1.0 - (-dt / tau).exp()));
                        let env = (PI * u).sin().powf(0.55);
                        val += 0.78 * env * phase.sin() + 0.14 * env * (2.0 * phase).sin();
                    }
                }
                *s = (val as f32).clamp(-1.0, 1.0);
            }

            WavRecording {
                source_label: "synth_hoary_bat_250kHz.wav".to_string(),
                sample_rate_hz: sr,
                channels: 1,
                bits_per_sample: 16,
                samples_mono: samples,
                guano_lat_lon: Some((45.6380, -120.6050)), // Columbia Gorge wind corridor
            }
        }
        SynthPreset::MyotisBuzzUltrasonic250k => {
            // 250 kHz recording (280 ms): steep Myotis lucifugus FM pulses (82 -> 42 kHz, 4.2 ms)
            // transitioning into a rapid foraging approach train
            let sr = 250_000_u32;
            let n = (sr as usize * 280) / 1000;
            let mut samples = vec![0.0_f32; n];
            let pulses: &[(f64, f64, f64, f64)] = &[
                (28.0, 4.8, 84_000.0, 42_500.0),
                (78.0, 4.5, 82_000.0, 43_000.0),
                (124.0, 4.1, 80_000.0, 43_500.0),
                (162.0, 3.6, 76_000.0, 44_000.0),
                (192.0, 3.0, 72_000.0, 44_500.0),
                (215.0, 2.5, 68_000.0, 43_000.0),
                (233.0, 2.2, 65_000.0, 41_500.0),
                (248.0, 2.0, 62_000.0, 40_500.0),
            ];

            for (i, s) in samples.iter_mut().enumerate() {
                let t_ms = (i as f64 * 1000.0) / sr as f64;
                let mut val = (pseudo_noise(i as u64 + 701) - 0.5) * 0.012;
                for &(p0, dur_ms, f0, f1) in pulses {
                    if t_ms >= p0 && t_ms <= p0 + dur_ms {
                        let u = (t_ms - p0) / dur_ms;
                        let dt = (t_ms - p0) / 1000.0;
                        let dur_s = dur_ms / 1000.0;
                        let k = (f1 - f0) / dur_s;
                        let phase = 2.0 * PI * (f0 * dt + 0.5 * k * dt * dt);
                        let env = (PI * u).sin().powf(0.65);
                        val += 0.76 * env * phase.sin();
                    }
                }
                *s = (val as f32).clamp(-1.0, 1.0);
            }

            WavRecording {
                source_label: "synth_myotis_lucifugus_250kHz.wav".to_string(),
                sample_rate_hz: sr,
                channels: 1,
                bits_per_sample: 16,
                samples_mono: samples,
                guano_lat_lon: Some((44.4128, -123.3285)), // William L. Finley NWR
            }
        }
        SynthPreset::PasserineNfcDuet48k => {
            // 48 kHz nocturnal flight call recording (650 ms):
            // Pulse 1: Swainson's Thrush upward-inflected liquid 'whit' (2.45 -> 3.65 kHz, 85 ms)
            // Pulse 2: Yellow-rumped Warbler modulated high 'zeep' (6.4 -> 7.6 kHz, 52 ms)
            let sr = 48_000_u32;
            let n = (sr as usize * 650) / 1000;
            let mut samples = vec![0.0_f32; n];

            for (i, s) in samples.iter_mut().enumerate() {
                let t_ms = (i as f64 * 1000.0) / sr as f64;
                let mut val = (pseudo_noise(i as u64 + 4099) - 0.5) * 0.010;

                // Swainson's Thrush NFC at 110..198 ms
                if (110.0..=198.0).contains(&t_ms) {
                    let u = (t_ms - 110.0) / 88.0;
                    let dt = (t_ms - 110.0) / 1000.0;
                    let phase = 2.0 * PI * (2480.0 * dt + 6800.0 * dt * dt);
                    let env = (PI * u).sin().powf(0.6);
                    val += 0.72 * env * phase.sin() + 0.16 * env * (2.0 * phase).sin();
                }

                // Yellow-rumped Warbler NFC at 390..446 ms with 115 Hz rapid FM vibrato
                if (390.0..=446.0).contains(&t_ms) {
                    let u = (t_ms - 390.0) / 56.0;
                    let dt = (t_ms - 390.0) / 1000.0;
                    let mod_term = 3.2 * (2.0 * PI * 115.0 * dt).sin();
                    let phase = 2.0 * PI * (6650.0 * dt + 7200.0 * dt * dt) + mod_term;
                    let env = (PI * u).sin().powf(0.5);
                    val += 0.66 * env * phase.sin();
                }

                *s = (val as f32).clamp(-1.0, 1.0);
            }

            WavRecording {
                source_label: "synth_passerine_nfc_48kHz.wav".to_string(),
                sample_rate_hz: sr,
                channels: 1,
                bits_per_sample: 16,
                samples_mono: samples,
                guano_lat_lon: Some((44.5646, -123.2620)), // Corvallis nocturnal roof station
            }
        }
    }
}

/// Computes Hann-windowed STFT power spectrogram (`fft_size` must be a power of two in `256..=4096`),
/// extracts bioacoustic call pulses (`Fc`, `Fmax`, `Fmin`, duration), computes Pieretti power-based ACI,
/// and classifies the vocalization guild.
///
/// For long multi-second ultrasonic recordings (`analysis_frames > 900`), the STFT retains overlapping
/// fine-resolution analysis frames (`hop <= n_fft / 2`) for pulse extraction and Pieretti ACI while
/// aggregating display columns via per-bin max-hold into `<= 900` columns so short bat pulses (`~2 ms`)
/// are never skipped in widened-hop blind gaps.
pub fn compute_stft_spectrogram(rec: &WavRecording, fft_size: usize) -> SpectrogramResult {
    let n_fft = fft_size.next_power_of_two().clamp(256, 4096);
    let base_hop = (n_fft / 4).max(64);
    let num_bins = n_fft / 2;
    let sr = rec.sample_rate_hz.max(1000) as f64;
    let bin_hz = sr / (n_fft as f64);

    let hann: Vec<f64> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * (k as f64) / ((n_fft - 1) as f64)).cos()))
        .collect();
    let win_coherent_gain: f64 = hann.iter().sum::<f64>() * 0.5;

    let total_samples = rec.samples_mono.len();
    let raw_frames = if total_samples >= n_fft {
        (total_samples - n_fft) / base_hop + 1
    } else {
        1
    };

    // Keep >=50% Hann overlap on ultra-long recordings so short bat echolocation pulses never fall
    // into inter-window blind gaps, while capping UI spectrogram columns at 900 via max-hold.
    let analysis_hop = if raw_frames > 24_000 {
        (n_fft / 2).max(base_hop)
    } else {
        base_hop
    };
    let analysis_frames = if total_samples >= n_fft {
        (total_samples - n_fft) / analysis_hop + 1
    } else {
        1
    };
    let display_frames = analysis_frames.min(900);

    let mut power_dbfs = vec![-105.0_f32; display_frames * num_bins];
    let mut prev_pow = vec![0.0_f64; num_bins];
    let mut diff_sum_per_bin = vec![0.0_f64; num_bins];
    let mut int_sum_per_bin = vec![0.0_f64; num_bins];
    let mut fine_peak_db = vec![-120.0_f32; analysis_frames];
    let mut fine_peak_bin = vec![0_usize; analysis_frames];

    let mut re = vec![0.0_f64; n_fft];
    let mut im = vec![0.0_f64; n_fft];

    let mut global_peak_db = -120.0_f32;
    for f_idx in 0..analysis_frames {
        let start = f_idx * analysis_hop;
        re.fill(0.0);
        im.fill(0.0);
        for k in 0..n_fft {
            if start + k < total_samples {
                re[k] = (rec.samples_mono[start + k] as f64) * hann[k];
            }
        }
        fft_radix2_inplace(&mut re, &mut im);

        let disp_col = if analysis_frames > display_frames {
            ((f_idx * display_frames) / analysis_frames).min(display_frames - 1)
        } else {
            f_idx
        };

        let mut frame_best_db = -120.0_f32;
        let mut frame_best_bin = 0_usize;

        for b in 1..num_bins {
            let mag = (re[b] * re[b] + im[b] * im[b]).sqrt() / win_coherent_gain.max(1.0);
            let pow = mag * mag;
            let db = (10.0 * (pow + 1e-18).log10()).clamp(-105.0, 0.0) as f32;
            let cell = &mut power_dbfs[disp_col * num_bins + b];
            if db > *cell {
                *cell = db;
            }
            if db > global_peak_db {
                global_peak_db = db;
            }
            if b >= 2 {
                if db > frame_best_db {
                    frame_best_db = db;
                    frame_best_bin = b;
                }
                int_sum_per_bin[b] += pow;
                if f_idx > 0 {
                    diff_sum_per_bin[b] += (pow - prev_pow[b]).abs();
                }
                prev_pow[b] = pow;
            }
        }

        fine_peak_db[f_idx] = frame_best_db;
        fine_peak_bin[f_idx] = frame_best_bin;
    }

    for col in 0..display_frames {
        power_dbfs[col * num_bins] = power_dbfs[col * num_bins + 1];
    }

    // Pieretti Acoustic Complexity Index (ACI): sum(|I_t - I_{t+1}|) / sum(I_t) using power I = mag^2
    // across all frames including the final frame (0..analysis_frames).
    let mut aci_total = 0.0_f64;
    if analysis_frames >= 2 {
        for b in 2..num_bins {
            let int_sum = int_sum_per_bin[b];
            if int_sum > 1e-12 {
                aci_total += diff_sum_per_bin[b] / int_sum;
            }
        }
    }

    // Extract active spectral ridge & discrete vocalization pulses (within 24 dB of peak & >= -58 dBFS)
    let gate_db = (global_peak_db - 24.0).max(-58.0);
    let mut fine_ridge_khz = vec![None; analysis_frames];
    for f_idx in 0..analysis_frames {
        if fine_peak_db[f_idx] > gate_db && fine_peak_bin[f_idx] >= 2 {
            fine_ridge_khz[f_idx] =
                Some(((fine_peak_bin[f_idx] as f64 * bin_hz) / 1000.0) as f32);
        }
    }

    let mut ridge_khz = vec![None; display_frames];
    for col in 0..display_frames {
        let mut best_db = gate_db;
        let mut best_bin = None;
        for b in 2..num_bins {
            let db = power_dbfs[col * num_bins + b];
            if db > best_db {
                best_db = db;
                best_bin = Some(b);
            }
        }
        if let Some(b) = best_bin {
            ridge_khz[col] = Some(((b as f64 * bin_hz) / 1000.0) as f32);
        }
    }

    let frame_dt_ms = (analysis_hop as f64 * 1000.0) / sr;
    let rec_dur_ms = rec.duration_ms();
    let mut pulses = Vec::new();
    let mut idx = 0usize;
    while idx < analysis_frames {
        if fine_ridge_khz[idx].is_some() {
            let p_start = idx;
            while idx < analysis_frames && fine_ridge_khz[idx].is_some() {
                idx += 1;
            }
            let p_end = idx;
            let slice_freqs: Vec<f64> = fine_ridge_khz[p_start..p_end]
                .iter()
                .filter_map(|x| x.map(|v| v as f64))
                .collect();
            if !slice_freqs.is_empty() {
                let f_max = slice_freqs.iter().copied().fold(0.0_f64, f64::max);
                let f_min = slice_freqs.iter().copied().fold(1e9_f64, f64::min);
                // Characteristic frequency Fc = lowest-slope tail segment (final third of pulse)
                let tail_start = (slice_freqs.len() * 2) / 3;
                let tail = &slice_freqs[tail_start..];
                let f_char = tail.iter().sum::<f64>() / (tail.len().max(1) as f64);

                let mut pk_db = -100.0_f64;
                for &db in &fine_peak_db[p_start..p_end] {
                    pk_db = pk_db.max(db as f64);
                }
                let start_ms = (p_start as f64 * frame_dt_ms).min(rec_dur_ms);
                let max_dur_ms = (rec_dur_ms - start_ms).max(0.0);
                let duration_ms = ((p_end - p_start) as f64 * frame_dt_ms)
                    .max(frame_dt_ms)
                    .min(max_dur_ms);
                pulses.push(CallPulse {
                    start_ms,
                    duration_ms,
                    f_max_khz: f_max,
                    f_min_khz: f_min,
                    f_char_khz: f_char,
                    peak_dbfs: pk_db,
                });
            }
        } else {
            idx += 1;
        }
    }

    let (fc, fmax, fmin, mean_dur) = if !pulses.is_empty() {
        let n_p = pulses.len() as f64;
        (
            pulses.iter().map(|p| p.f_char_khz).sum::<f64>() / n_p,
            pulses.iter().map(|p| p.f_max_khz).fold(0.0_f64, f64::max),
            pulses.iter().map(|p| p.f_min_khz).fold(1e9_f64, f64::min),
            pulses.iter().map(|p| p.duration_ms).sum::<f64>() / n_p,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let (classified_taxon, guild_label, scientific_hint) = classify_bioacoustic_guild(fc, fmax, fmin, mean_dur);

    SpectrogramResult {
        num_frames: display_frames,
        num_bins,
        fft_size: n_fft,
        hop_size: analysis_hop,
        sample_rate_hz: rec.sample_rate_hz,
        duration_ms: rec.duration_ms(),
        max_freq_khz: rec.nyquist_khz(),
        power_dbfs,
        ridge_khz,
        pulses,
        aci_index: aci_total,
        characteristic_freq_khz: fc,
        f_max_khz: fmax,
        f_min_khz: fmin,
        mean_duration_ms: mean_dur,
        classified_taxon,
        guild_label,
        scientific_hint,
    }
}

fn classify_bioacoustic_guild(
    fc_khz: f64,
    fmax_khz: f64,
    _fmin_khz: f64,
    dur_ms: f64,
) -> (Taxon, String, String) {
    if fc_khz >= 35.0 || fmax_khz >= 55.0 {
        (
            Taxon::Bat,
            "Myotis High-FM Echolocation Guild".to_string(),
            "Myotis lucifugus / M. yumanensis".to_string(),
        )
    } else if fc_khz >= 23.5 && fc_khz < 35.0 {
        (
            Taxon::Bat,
            "Medium-QCF Tree/Cavity Bat Guild".to_string(),
            "Lasionycteris noctivagans / Eptesicus fuscus".to_string(),
        )
    } else if fc_khz >= 15.0 && fc_khz < 23.5 {
        (
            Taxon::Bat,
            "Low-QCF Migratory Hoary Bat".to_string(),
            "Lasiurus cinereus".to_string(),
        )
    } else if fc_khz >= 5.0 && fc_khz < 12.0 {
        (
            Taxon::Bird,
            "High-Zeep Warbler Nocturnal Flight Call".to_string(),
            "Setophaga coronata / Parulidae".to_string(),
        )
    } else if fc_khz >= 1.2 && fc_khz < 5.0 && dur_ms >= 20.0 {
        (
            Taxon::Bird,
            "Passerine Thrush Nocturnal Flight Call".to_string(),
            "Catharus ustulatus / Turdidae".to_string(),
        )
    } else {
        (
            Taxon::Bird,
            "Broadband / Ambient Bioacoustic Signal".to_string(),
            "Unclassified Avian/Chiroptera".to_string(),
        )
    }
}

/// Iterative in-place Radix-2 Cooley-Tukey Complex FFT (`re.len()` must be a power of 2).
pub fn fft_radix2_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && n == im.len());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let ang = -2.0 * PI / (len as f64);
        let wlen_re = ang.cos();
        let wlen_im = ang.sin();

        let mut start = 0usize;
        while start < n {
            let mut w_re = 1.0_f64;
            let mut w_im = 0.0_f64;
            for k in 0..half {
                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + half] * w_re - im[start + k + half] * w_im;
                let v_im = re[start + k + half] * w_im + im[start + k + half] * w_re;

                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + half] = u_re - v_re;
                im[start + k + half] = u_im - v_im;

                let next_w_re = w_re * wlen_re - w_im * wlen_im;
                let next_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

pub fn magma_lut(t: f32) -> (u8, u8, u8) {
    if !t.is_finite() {
        return (8, 11, 24);
    }
    let stops: &[(f32, [f32; 3])] = &[
        (0.00, [8.0, 11.0, 24.0]),
        (0.25, [59.0, 15.0, 112.0]),
        (0.50, [140.0, 41.0, 129.0]),
        (0.75, [222.0, 73.0, 104.0]),
        (0.90, [254.0, 159.0, 109.0]),
        (1.00, [252.0, 253.0, 191.0]),
    ];
    let x = t.clamp(0.0, 1.0);
    for pair in stops.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if x >= t0 && x <= t1 {
            let u = (x - t0) / (t1 - t0).max(1e-6);
            return (
                (c0[0] + u * (c1[0] - c0[0])) as u8,
                (c0[1] + u * (c1[1] - c0[1])) as u8,
                (c0[2] + u * (c1[2] - c0[2])) as u8,
            );
        }
    }
    (252, 253, 191)
}

fn pseudo_noise(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1_u64 << 53) as f64)
}
