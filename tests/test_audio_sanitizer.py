"""
Tests for AudioSanitizer class
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mmm.core.audio_sanitizer import AudioSanitizer
from mmm.config.config_manager import ConfigManager


class TestAudioSanitizer:
    """Test cases for AudioSanitizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = self.config_manager.get_config()

        # Create test audio data
        self.sample_rate = 44100
        self.duration = 2.0  # 2 seconds
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = 0.5 * np.sin(2 * np.pi * 440 * self.t)  # 440 Hz sine wave

    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_audio_file(self, filename: str, audio_data: np.ndarray = None) -> Path:
        """Create a test audio file"""
        import soundfile as sf

        if audio_data is None:
            audio_data = self.test_audio

        file_path = self.test_dir / filename
        sf.write(str(file_path), audio_data, self.sample_rate)
        return file_path

    def test_initialization(self):
        """Test AudioSanitizer initialization"""
        input_file = self.create_test_audio_file("test.wav")

        sanitizer = AudioSanitizer(
            input_file=input_file,
            config=self.config
        )

        assert sanitizer.input_file == input_file
        assert sanitizer.output_file == input_file.parent / "test_clean.wav"
        assert sanitizer.audio_data is None
        assert sanitizer.sample_rate is None
        assert sanitizer.original_hash is None

    def test_load_audio(self):
        """Test audio loading functionality"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Test loading
        result = sanitizer.load_audio()
        assert result is True
        assert sanitizer.audio_data is not None
        assert sanitizer.sample_rate == self.sample_rate
        assert sanitizer.original_hash is not None

    def test_analyze_file(self):
        """Test file analysis functionality"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Perform analysis
        analysis = sanitizer.analyze_file()

        # Check analysis structure
        assert 'file_info' in analysis
        assert 'metadata' in analysis
        assert 'watermarks' in analysis
        assert 'threats_found' in analysis
        assert analysis['file_info']['format'] == 'WAV'

    def test_calculate_threat_level(self):
        """Test threat level calculation"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Create mock analysis with various threat levels
        low_threat_analysis = {
            'metadata': {'tags': []},
            'watermarks': {'detected': []},
            'statistical': {'anomalies': []}
        }

        threat_level = sanitizer._calculate_threat_level(low_threat_analysis)
        assert threat_level == 'LOW'

        # Test medium threat
        medium_threat_analysis = {
            'metadata': {'tags': ['tag1', 'tag2', 'tag3']},
            'watermarks': {'detected': []},
            'statistical': {'anomalies': []}
        }

        threat_level = sanitizer._calculate_threat_level(medium_threat_analysis)
        assert threat_level == 'MEDIUM'

        # Test high threat
        high_threat_analysis = {
            'metadata': {'tags': ['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']},
            'watermarks': {'detected': [{'method': 'test'}]},
            'statistical': {'anomalies': ['anomaly1', 'anomaly2']}
        }

        threat_level = sanitizer._calculate_threat_level(high_threat_analysis)
        assert threat_level == 'HIGH'

    def test_sanitize_audio(self):
        """Test audio sanitization"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Perform sanitization
        result = sanitizer.sanitize_audio()

        # Check result structure
        assert 'success' in result
        assert 'output_file' in result
        assert 'original_hash' in result
        assert 'final_hash' in result
        assert 'stats' in result

        if result['success']:
            assert Path(result['output_file']).exists()
            assert result['original_hash'] != result['final_hash']

    def test_paranoid_mode(self):
        """Test paranoid mode functionality"""
        input_file = self.create_test_audio_file("test.wav")

        # Test with paranoid mode disabled
        normal_sanitizer = AudioSanitizer(
            input_file=input_file,
            paranoid_mode=False,
            config=self.config
        )
        normal_result = normal_sanitizer.sanitize_audio()

        # Test with paranoid mode enabled
        paranoid_sanitizer = AudioSanitizer(
            input_file=input_file,
            paranoid_mode=True,
            config=self.config
        )
        paranoid_result = paranoid_sanitizer.sanitize_audio()

        # Both should succeed
        assert normal_result.get('success', False)
        assert paranoid_result.get('success', False)

        # Paranoid mode should use more methods
        if normal_result['success'] and paranoid_result['success']:
            assert len(paranoid_result['stats']['processing_time']) > 0

    def test_verify_sanitization(self):
        """Test sanitization verification"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # First sanitize the file
        sanitize_result = sanitizer.sanitize_audio()
        assert sanitize_result['success']

        # Then verify the sanitization
        verification = sanitizer.verify_sanitization()

        assert 'success' in verification
        assert 'original_threats' in verification
        assert 'remaining_threats' in verification
        assert 'removal_effectiveness' in verification

    def test_calculate_quality_loss(self):
        """Test quality loss calculation"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Sanitize first to have an output file
        sanitizer.sanitize_audio()

        # Calculate quality loss
        quality_loss = sanitizer._calculate_quality_loss()

        assert isinstance(quality_loss, float)
        assert quality_loss >= 0.0

    def test_create_backup(self):
        """Test backup creation"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)

        # Create backup
        sanitizer.create_backup()

        # Check backup file exists
        backup_file = input_file.with_suffix('.backup.wav')
        assert backup_file.exists()
        assert backup_file.stat().st_size > 0

    def test_invalid_file_handling(self):
        """Test handling of invalid files"""
        # Create a non-audio file
        invalid_file = self.test_dir / "invalid.txt"
        invalid_file.write_text("This is not audio data")

        sanitizer = AudioSanitizer(input_file=invalid_file, config=self.config)

        # Should fail to load
        result = sanitizer.load_audio()
        assert result is False

    @pytest.mark.parametrize("paranoid_mode", [True, False])
    def test_different_modes(self, paranoid_mode):
        """Test different processing modes"""
        input_file = self.create_test_audio_file("test.wav")
        sanitizer = AudioSanitizer(
            input_file=input_file,
            paranoid_mode=paranoid_mode,
            config=self.config
        )

        result = sanitizer.sanitize_audio()
        assert result.get('success', False)

        if result['success']:
            assert Path(result['output_file']).exists()

    def test_stereo_audio(self):
        """Test processing of stereo audio"""
        # Create stereo test audio
        stereo_audio = np.column_stack([self.test_audio, self.test_audio * 0.8])
        input_file = self.create_test_audio_file("test_stereo.wav", stereo_audio)

        sanitizer = AudioSanitizer(input_file=input_file, config=self.config)
        result = sanitizer.sanitize_audio()

        assert result.get('success', False)

        if result['success']:
            # Verify output file exists
            assert Path(result['output_file']).exists()

            # Load and verify it's still stereo
            import soundfile as sf
            cleaned_audio, sr = sf.read(str(result['output_file']))
            assert cleaned_audio.ndim == 2  # Should be stereo
            assert cleaned_audio.shape[1] == 2  # Should have 2 channels