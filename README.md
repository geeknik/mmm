# 🎵 Melodic Metadata Massacrer (MMM)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

> *"In the symphony of digital rights, we are the conductors of chaos."* 🎼⚡

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
mmm analyze INPUT_FILE
```

### `config`
Configuration management

```bash
mmm config              Show current config
mmm config preset NAME  Apply preset
```

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