"""
Core AudioSanitizer class - Main audio processing engine
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import librosa
import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile
from pydub import AudioSegment

from ..detection.watermark_detector import WatermarkDetector
from ..detection.metadata_scanner import MetadataScanner
from ..detection.statistical_analyzer import StatisticalAnalyzer
from ..sanitization.metadata_cleaner import MetadataCleaner
from ..sanitization.spectral_cleaner import SpectralCleaner
from ..sanitization.fingerprint_remover import FingerprintRemover


class AudioSanitizer:
    """
    Main audio sanitization engine that orchestrates all cleaning operations
    """

    def __init__(self, input_file: Path, output_file: Optional[Path] = None,
                 paranoid_mode: bool = False, config: Dict[str, Any] = None,
                 output_format: Optional[str] = None):
        self.input_file = input_file
        self.output_format = self._determine_output_format(output_file, output_format)
        self.output_file = self._resolve_output_file(output_file)
        self.paranoid_mode = paranoid_mode
        self.config = config or {}

        # Audio data
        self.audio_data = None
        self.sample_rate = None
        self.original_hash = None
        self.metadata_crimes = []

        # Initialize modules
        self.watermark_detector = WatermarkDetector(config)
        self.metadata_scanner = MetadataScanner()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.metadata_cleaner = MetadataCleaner()
        self.spectral_cleaner = SpectralCleaner(paranoid_mode)
        self.fingerprint_remover = FingerprintRemover(paranoid_mode)

        # Status tracking
        self.processing_stats = {
            'metadata_removed': 0,
            'watermarks_detected': 0,
            'watermarks_removed': 0,
            'quality_loss': 0.0,
            'processing_time': 0
        }

    def _generate_output_path(self, output_format: Optional[str] = None) -> Path:
        """Generate output file path if not specified"""
        stem = self.input_file.stem
        suffix = f".{output_format}" if output_format else self.input_file.suffix
        parent = self.input_file.parent

        # Add _clean suffix
        return parent / f"{stem}_clean{suffix}"

    def load_audio(self) -> bool:
        """
        Load audio file using multiple backends for robustness

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try librosa first (best for analysis)
            self.audio_data, self.sample_rate = librosa.load(
                str(self.input_file),
                sr=None,
                mono=True  # Load as mono for consistency
            )
            self.audio_data = self._ensure_channel_layout(self.audio_data)

            # Calculate original hash
            with open(self.input_file, 'rb') as f:
                self.original_hash = hashlib.sha256(f.read()).hexdigest()

            return True

        except Exception as e:
            # Fallback to pydub for problematic files
            try:
                audio = AudioSegment.from_file(str(self.input_file))
                self.sample_rate = audio.frame_rate

                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))

                self.audio_data = samples.astype(np.float32) / 32768.0
                self.audio_data = self._ensure_channel_layout(self.audio_data)

                # Calculate hash
                with open(self.input_file, 'rb') as f:
                    self.original_hash = hashlib.sha256(f.read()).hexdigest()

                return True

            except Exception:
                return False

    def analyze_file(self, deep: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of audio file

        Args:
            deep: Perform deep analysis including statistical patterns

        Returns:
            Dict containing analysis results
        """
        if not self.load_audio():
            raise Exception("Failed to load audio file")

        analysis = {
            'file_info': {
                'path': str(self.input_file),
                'size': self.input_file.stat().st_size,
                'hash': self.original_hash,
                'format': self.input_file.suffix.lower(),
                'duration': len(self.audio_data) / self.sample_rate if self.sample_rate else 0,
                'sample_rate': self.sample_rate,
                'channels': 1 if self.audio_data.ndim == 1 else self.audio_data.shape[1]
            }
        }

        # Metadata analysis
        metadata_analysis = self.metadata_scanner.scan_file(self.input_file)
        analysis['metadata'] = metadata_analysis

        # Watermark detection
        watermark_analysis = self.watermark_detector.detect_all(self.audio_data, self.sample_rate)
        analysis['watermarks'] = watermark_analysis

        # Statistical analysis (if deep scan requested)
        if deep:
            statistical_analysis = self.statistical_analyzer.analyze(self.audio_data, self.sample_rate)
            analysis['statistical'] = statistical_analysis

            # Calculate threat level
            threat_score = self._calculate_threat_level(analysis)
            analysis['threat_level'] = threat_score
        else:
            analysis['threat_level'] = 'UNKNOWN'

        # Count total threats
        analysis['threats_found'] = (
            len(analysis['metadata'].get('tags', [])) +
            len(analysis['watermarks'].get('detected', [])) +
            len(analysis.get('statistical', {}).get('anomalies', []))
        )

        return analysis

    def _calculate_threat_level(self, analysis: Dict[str, Any]) -> str:
        """
        Calculate overall threat level based on analysis results

        Args:
            analysis: Analysis results from analyze_file()

        Returns:
            String indicating threat level (LOW, MEDIUM, HIGH)
        """
        threat_score = 0

        # Metadata threats
        threat_score += len(analysis['metadata'].get('tags', [])) * 1
        threat_score += len(analysis['metadata'].get('suspicious_chunks', [])) * 2

        # Watermark threats
        threat_score += len(analysis['watermarks'].get('detected', [])) * 3

        # Statistical threats
        threat_score += len(analysis.get('statistical', {}).get('anomalies', [])) * 2

        # Determine threat level
        if threat_score >= 10:
            return 'HIGH'
        elif threat_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def sanitize_audio(self) -> Dict[str, Any]:
        """
        Perform complete audio sanitization process

        Returns:
            Dict containing sanitization results
        """
        import time
        start_time = time.time()

        try:
            # Load and analyze
            if not self.load_audio():
                raise Exception("Failed to load audio file")

            analysis = self.analyze_file(deep=True)

            # Initialize sanitized data
            sanitized_audio = self.audio_data.copy()

            # Phase 1: Metadata removal
            self._info("Phase 1: Metadata annihilation...")
            metadata_result = self.metadata_cleaner.clean_file(self.input_file, self.output_file)
            self.processing_stats['metadata_removed'] = metadata_result['tags_removed']

            # Reload clean audio for further processing
            if self.output_file.exists():
                sanitized_audio, self.sample_rate = librosa.load(str(self.output_file), sr=None, mono=False)
                sanitized_audio = self._ensure_channel_layout(sanitized_audio)

            # Phase 2: Spectral watermark removal
            self._info("Phase 2: Spectral watermark elimination...")
            spectral_result = self.spectral_cleaner.clean_watermarks(sanitized_audio, self.sample_rate)
            sanitized_audio = spectral_result['cleaned_audio']
            self.processing_stats['patterns_found'] = spectral_result['watermarks_found']
            self.processing_stats['patterns_suppressed'] = spectral_result['watermarks_removed']

            # Phase 3: Statistical fingerprint removal
            self._info("Phase 3: Statistical fingerprint destruction...")
            fingerprint_result = self.fingerprint_remover.remove_fingerprints(sanitized_audio, self.sample_rate)
            sanitized_audio = fingerprint_result['cleaned_audio']

            # Phase 4: Final cleanup and quality preservation
            self._info("Phase 4: Final sanitization...")
            if self.paranoid_mode:
                # Multiple passes in paranoid mode
                for i in range(3):
                    sanitized_audio = self._paranoid_pass(sanitized_audio)

            # Save final result
            self._save_audio(sanitized_audio)

            # Calculate quality metrics
            self.processing_stats['quality_loss'] = self._calculate_quality_loss()
            self.processing_stats['processing_time'] = time.time() - start_time

            # Verify output
            final_hash = self._calculate_file_hash(self.output_file)

            return {
                'success': True,
                'output_file': str(self.output_file),
                'original_hash': self.original_hash,
                'final_hash': final_hash,
                'stats': self.processing_stats,
                'analysis': analysis
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stats': self.processing_stats
            }

    def _paranoid_pass(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Additional paranoid sanitization pass

        Args:
            audio_data: Audio data to process

        Returns:
            Further sanitized audio data
        """
        # Add minimal noise to break any remaining patterns
        noise_level = 1e-6
        noise = np.random.normal(0, noise_level, audio_data.shape)

        # Apply subtle phase randomization
        if audio_data.ndim == 1:
            # Mono
            fft_data = np.fft.fft(audio_data)
            phases = np.angle(fft_data)
            # Randomize phase slightly
            phases += np.random.normal(0, 0.01, phases.shape)
            fft_data = np.abs(fft_data) * np.exp(1j * phases)
            result = np.fft.ifft(fft_data).real
        else:
            # Stereo
            result = np.zeros_like(audio_data)
            for channel in range(audio_data.shape[1]):
                fft_data = np.fft.fft(audio_data[:, channel])
                phases = np.angle(fft_data)
                phases += np.random.normal(0, 0.01, phases.shape)
                fft_data = np.abs(fft_data) * np.exp(1j * phases)
                result[:, channel] = np.fft.ifft(fft_data).real

        return result + noise

    def _save_audio(self, audio_data: np.ndarray):
        """Save processed audio to output file"""
        if self.sample_rate is None:
            raise ValueError("Sample rate is not set; cannot save audio")

        audio_data = np.real(self._ensure_channel_layout(audio_data))
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        # Prevent clipping by normalizing if needed
        max_val = float(np.max(np.abs(audio_data))) if audio_data.size else 1.0
        if max_val > 1.0:
            audio_data = audio_data / max_val

        target_format = (self.output_format or self.output_file.suffix.lstrip('.')).lower()
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)

        if target_format == 'mp3':
            channels = audio_int16.shape[1] if audio_int16.ndim > 1 else 1
            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=channels
            )
            segment.export(
                str(self.output_file),
                format='mp3',
                bitrate='320k',
                parameters=[
                    '-map_metadata', '-1',
                    '-write_xing', '0',
                    '-id3v2_version', '0',
                    '-write_id3v1', '0'
                ]  # ensure no metadata or encoder info is re-added
            )
        else:
            sf.write(
                str(self.output_file),
                audio_int16,
                self.sample_rate,
                format=target_format.upper()
            )

    def _calculate_quality_loss(self) -> float:
        """Calculate percentage of quality loss during processing"""
        # Simple SNR-based calculation
        try:
            original_audio, _ = librosa.load(str(self.input_file), sr=self.sample_rate)
            clean_audio, _ = librosa.load(str(self.output_file), sr=self.sample_rate)

            # Ensure same length
            min_len = min(len(original_audio), len(clean_audio))
            original_audio = original_audio[:min_len]
            clean_audio = clean_audio[:min_len]

            # Calculate SNR
            noise = original_audio - clean_audio
            signal_power = np.mean(original_audio ** 2)
            noise_power = np.mean(noise ** 2)

            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                # Convert to percentage (higher SNR = less quality loss)
                quality_loss = max(0, 100 - (snr_db + 100) / 2)
                return round(quality_loss, 2)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def verify_sanitization(self) -> Dict[str, Any]:
        """
        Verify that sanitization was successful by re-analyzing the output

        Returns:
            Dict containing verification results
        """
        try:
            # Re-analyze the sanitized file
            sanitizer = AudioSanitizer(self.output_file, paranoid_mode=self.paranoid_mode)
            new_analysis = sanitizer.analyze_file(deep=True)

            # Compare with original
            original_threats = self.processing_stats['metadata_removed'] + \
                             self.processing_stats['watermarks_detected']
            new_threats = new_analysis['threats_found']

            removal_effectiveness = 0
            if original_threats > 0:
                removal_effectiveness = ((original_threats - new_threats) / original_threats) * 100

            return {
                'success': True,
                'original_threats': original_threats,
                'remaining_threats': new_threats,
                'removal_effectiveness': round(removal_effectiveness, 2),
                'new_analysis': new_analysis,
                'hash_different': self.original_hash != self._calculate_file_hash(self.output_file)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def create_backup(self):
        """Create backup of original file"""
        backup_path = self.input_file.with_suffix(f'.backup{self.input_file.suffix}')
        shutil.copy2(self.input_file, backup_path)
        self.metadata_crimes.append(f"Created backup: {backup_path}")

    def _info(self, message: str):
        """Internal info logging"""
        self.metadata_crimes.append(message)

    def _determine_output_format(self, output_file: Optional[Path], output_format: Optional[str]) -> str:
        """
        Decide which audio format to use for saving.

        Priority: explicit output_format arg -> output_file suffix -> input file suffix.
        """
        normalized_format = (output_format or '').lower().lstrip('.')
        if normalized_format == 'preserve':
            normalized_format = ''

        if normalized_format:
            return normalized_format

        if output_file and output_file.suffix:
            return output_file.suffix.lstrip('.').lower()

        # Fallback to input file extension
        return self.input_file.suffix.lstrip('.').lower() or 'wav'

    def _resolve_output_file(self, output_file: Optional[Path]) -> Path:
        """Ensure the output path matches the chosen format."""
        target_suffix = f".{self.output_format}"
        if output_file:
            return output_file.with_suffix(target_suffix)
        return self._generate_output_path(self.output_format)

    def _ensure_channel_layout(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure audio arrays are shaped as (samples, channels).

        Librosa returns (channels, samples) when mono=False; most of our code
        expects the opposite, so transpose when the first dimension looks like
        the channel count.
        """
        if audio is None or audio.ndim == 1:
            return audio

        if audio.shape[0] < audio.shape[1]:
            return np.ascontiguousarray(audio.T)

        return audio
