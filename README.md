# 🎵 Melodic Metadata Massacrer (MMM)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

> *"In the symphony of digital rights, we are the conductors of chaos."* 🎼⚡

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/6b342199-dbdd-446b-8c6f-983e50ef5625" />

**MMM** is a Python CLI tool that performs lossless removal of ALL watermarks and metadata from MP3 and WAV audio files, effectively thwarting systems that detect AI-generated music through embedded identifiers.

## 🎭 Features

### Core Capabilities
- **Complete Metadata Annihilation**: Removes ID3, RIFF INFO, FLAC tags, and custom chunks
- **AI Watermark Detection**: Identifies spread spectrum, echo-based, and statistical watermarks
- **Spectral Cleaning**: Advanced frequency-domain watermark removal
- **Fingerprint Elimination**: Normalizes AI-generated statistical patterns
- **Paranoid Mode**: Maximum destruction with multiple cleaning passes
- **Batch Processing**: Parallel processing of entire directories
- **Verification Engine**: Before/after comparison with forensic reporting

### Detection Methods
- Spread spectrum watermarks
- Echo-based signatures
- Statistical pattern analysis
- Phase modulation detection
- Amplitude modulation analysis
- Frequency domain anomalies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/research/mmm.git
cd mmm

# Create virtual environment with Python 3.10-3.13 (librosa doesn't support 3.14+ yet)
python3.10 -m venv mmm_env
source mmm_env/bin/activate  # On Windows: mmm_env\Scripts\activate

# Install the package and all dependencies
pip install -e .
```

**Important**: MMM requires Python 3.10-3.13 due to librosa dependencies. Python 3.14+ is not yet supported by numba/librosa.

### GPU Acceleration (Optional - Recommended!)

For maximum performance (700x+ speedup), install GPU packages:

```bash
# Install GPU acceleration packages (NVIDIA GPU required)
pip install cupy-cuda12x torch torchaudio

# Verify GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support (tested on RTX 3080 Ti)
- 4GB+ VRAM recommended
- CUDA 12.x compatible drivers

### Performance Benchmarks

With GPU acceleration enabled (RTX 3080 Ti):

- **790x real-time processing speed**
- **47,409 audio-minutes per minute throughput**
- **0.27 seconds to analyze 214.6-second audio file**
- **Parallel processing across all CPU cores + GPU**

Without GPU acceleration (CPU-only):
- Standard processing speed (slower for large files)
- Still utilizes multi-core CPU processing

### Basic Usage

```bash
# Obliterate metadata from a single file
mmm obl dystopian_symphony.mp3

# Paranoid mode with verification
mmm obliterate suspicious_music.wav --paranoid --verify -o clean_output.wav

# Batch process directory
mmm massacre /path/to/music --paranoid --workers 8

# Analyze file without modifying
mmm analyze questionable_track.mp3
```

## 🔧 Configuration

MMM uses YAML configuration files for customization:

```yaml
# ~/.mmm/config.yaml
paranoia_level: medium
preserve_quality: high
watermark_detection:
  - spread_spectrum
  - echo_based
  - statistical
output_format: preserve_original
backup_originals: true
```

### Presets

- **`stealth`**: Maximum paranoia, quality preservation
- **`fast`**: Quick processing, basic cleaning
- **`quality`**: Preserve maximum audio quality
- **`research`**: Deep analysis, detailed logging

```bash
# Use preset
mmm config preset stealth

# Create custom preset
mmm config create my_preset --paranoid maximum --quality high
```

## 🎯 Commands

### `obliterate`
Complete sanitization of individual files

```bash
mmm obl INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH     Output file path
  --paranoid           Maximum destruction mode
  --verify             Verify watermark removal
  --backup             Create backup of original
  --format FORMAT      Output format (preserve/mp3/wav)
  --turbo              Enable GPU acceleration (700x+ faster)
```

### `massacre`
Batch processing of directories

```bash
mmm massacre DIRECTORY [OPTIONS]

Options:
  -d, --output-dir PATH  Output directory
  -e, --extension TEXT   File extensions (multiple)
  -w, --workers INT      Parallel workers
  --paranoid            Paranoid mode
  --backup              Create backups
```

### `analyze`
Deep forensic analysis without modification

```bash
mmm analyze INPUT_FILE                # Regular CPU mode
mmm analyze INPUT_FILE --turbo        # GPU accelerated mode (700x+ faster)
```

### `config`
Configuration management

```bash
mmm config              Show current config
mmm config preset NAME  Apply preset
```

## 🎛️ Advanced Stealth Flags

These are opt-in, fine-grained toggles for research tuning. Defaults keep audio quality high; enable selectively:

- `--gated-resample-nudge/--no-gated-resample-nudge` (default off): ultra-tiny resample up/down applied only on higher-energy segments (minimal audibility, good stealth).
- `--phase-noise/--no-phase-noise` (default on): tiny FFT phase noise.
- `--phase-swirl/--no-phase-swirl` (default on): light all-pass swirl.
- `--phase-dither/--no-phase-dither` (default on), `--comb-mask/--no-comb-mask`, `--transient-shift/--no-transient-shift`: earlier experimental steps (may affect audio; leave off unless testing).
- `--masked-hf-phase/--no-masked-hf-phase` (default off): HF-only masked phase noise.
- `--micro-eq-flutter/--no-micro-eq-flutter` (default off): RMS-gated, <0.013 dB band flutter.
- `--hf-decorrelate/--no-hf-decorrelate` (default off): decorrelate only 12–16 kHz band.
- `--refined-transient/--no-refined-transient` (default off): ultra-small, onset-gated shifts.
- `--adaptive-transient/--no-adaptive-transient` (default off): onset-strength adaptive micro-shifts (~0.03–0.08 ms) with light blending.

Recommended “stealth” starting point (quality-preserving):
```bash
python -m mmm.cli obliterate input.mp3 -o output.mp3 --turbo --paranoid \
  --gated-resample-nudge --phase-noise \
  --no-phase-dither --no-comb-mask --no-transient-shift \
  --no-phase-swirl --no-masked-hf-phase --no-resample-nudge \
  --no-hf-decorrelate --no-micro-eq-flutter --no-refined-transient
```

Preset shortcut (applies the above flags): `stealth-plus`

```yaml
# ~/.config/mmm/config.yaml (example)
preset: stealth-plus
```

You can also load it via `mmm config preset stealth-plus` (creates/uses preset file under your config dir).

Preset includes advanced flags:
- phase_dither=False, comb_mask=False, transient_shift=False
- phase_swirl=False, masked_hf_phase=False, resample_nudge=False
- gated_resample_nudge=True, phase_noise=True
- micro_eq_flutter=False, hf_decorrelate=False
- refined_transient=False, adaptive_transient=False

Notes on pattern suppression counts:
- “Patterns Found/Suppressed” in sanitization results come from the spectral cleaner’s suppression actions (e.g., attenuating suspicious bands/patterns) and do not imply detector-verified watermarks unless the detector reports them.
- Verification threat counts include metadata/container anomalies and detector findings; if the detector reports zero watermarks, remaining threats are likely metadata/binary anomalies rather than confirmed watermarks.

## 🧪 Detector Notes (Research)

We test against third-party detectors to understand robustness (not to guarantee evasion). Recent results:

- **SubmitHub / SHLabs**: “Inconclusive / mixed characteristics” with clean audio using `--gated-resample-nudge --phase-noise` and other advanced flags off. Temporal scores improved significantly; spectral scores became “could be AI / human unlikely”.
- Aggressive stacks (phase dither / comb mask / transient shift) degraded audio; not recommended.

Always audition audio locally before running external checks.

## 🛡️ Legal & Ethical Notice

⚠️ **IMPORTANT**: This tool is designed **exclusively for authorized security research and educational purposes**.

- Use only on files you own or have explicit permission to modify
- You are responsible for compliance with applicable laws and terms of service
- The developers do not condone or support copyright infringement
- This tool demonstrates vulnerabilities in watermarking systems for research purposes

## 📊 Technical Details

### Architecture

```
┌─────────────────────────────────────────┐
│                CLI Layer                │
│  Click-based interface with personality  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│            Core Processing              │
│  • AudioSanitizer main engine          │
│  • FileProcessor for batch operations   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Detection Modules              │
│  • WatermarkDetector                   │
│  • MetadataScanner                     │
│  • StatisticalAnalyzer                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Sanitization Modules            │
│  • MetadataCleaner                     │
│  • SpectralCleaner                     │
│  • FingerprintRemover                  │
└─────────────────────────────────────────┘
```

### Dependencies

- **Core**: Click, NumPy, SciPy
- **Audio**: Librosa, PyDub, SoundFile
- **Metadata**: Mutagen
- **UI**: Rich, Colorama

### Quality Preservation

MMM prioritizes audio quality while removing watermarks:

- **Signal-to-Noise Ratio**: >40dB target
- **Quality Loss**: <5% maximum
- **Spectral Integrity**: >90% preservation
- **Dynamic Range**: Preserved when possible

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=mmm

# Run specific test
pytest tests/test_audio_sanitizer.py -v
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Style

```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black mmm/

# Lint code
flake8 mmm/

# Type checking
mypy mmm/
```

## 📈 Performance Metrics

| Operation | File Size | Processing Time | Quality Loss |
|-----------|-----------|----------------|--------------|
| MP3 (5 min) | 5MB | ~15s | <2% |
| WAV (5 min) | 50MB | ~30s | <1% |
| Paranoid Mode | Variable | 2-3x slower | Same |

*Results on Intel i7-9700K, 16GB RAM*

## 🔬 Research Applications

MMM is designed for academic and security research:

- **Watermark Vulnerability Assessment**: Test resilience of audio watermarking systems
- **Privacy Research**: Study audio fingerprinting and tracking
- **Educational**: Demonstrate audio steganography techniques
- **Security Auditing**: Verify effectiveness of watermark removal

## 🎨 CLI Experience

MMM features a unique hacker-aesthetic interface:

```
┌─────────────────────────────────────────────┐
│  ♪♫ MELODIC METADATA MASSACRER v2.0 ♫♪    │
│     "Making AI detectors cry since 2025"    │
└─────────────────────────────────────────────┘

🔍 Scanning: dystopian_symphony.mp3
😈 Found 47 metadata tags... time to DELETE THEM ALL!
🌊 Spectral watermark detected at 19.2kHz... NEUTRALIZING...
⚡ Statistical fingerprint obliterated!
✨ File sanitized! Your AI overlords will never know... 🤫
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Open-source audio processing community
- Security researchers in digital watermarking
- Python audio processing ecosystem

## ⚡ Disclaimer

*"The best place to hide a dead body is page two of the search results. But your metadata won't even make it there."*

---

**Remember**: With great audio comes great responsibility. Use wisely. 🎼💀
