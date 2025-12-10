"""
Watermark detection modules for various AI watermarking techniques
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Any, Tuple


class WatermarkDetector:
    """
    Detects various watermarking techniques in audio files
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detection_methods = [
            'spread_spectrum',
            'echo_based',
            'statistical',
            'phase_modulation',
            'amplitude_modulation',
            'frequency_domain'
        ]

    def detect_all(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Run all watermark detection methods

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Dict containing all detection results
        """
        results = {
            'detected': [],
            'method_results': {},
            'confidence_scores': {},
            'watermark_count': 0
        }

        # Ensure stereo handling
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        # Check if audio data is too short for analysis
        min_samples = 4096  # Minimum samples for meaningful analysis
        if audio_data.shape[0] < min_samples:
            return {
                'detected': [],
                'method_results': {},
                'confidence_scores': {},
                'watermark_count': 0,
                'error': 'Audio too short for analysis'
            }

        # Run each detection method
        for method in self.detection_methods:
            try:
                if method == 'spread_spectrum':
                    result = self.detect_spread_spectrum(audio_data, sample_rate)
                elif method == 'echo_based':
                    result = self.detect_echo_signatures(audio_data, sample_rate)
                elif method == 'statistical':
                    result = self.analyze_statistical_anomalies(audio_data, sample_rate)
                elif method == 'phase_modulation':
                    result = self.detect_phase_modulation(audio_data, sample_rate)
                elif method == 'amplitude_modulation':
                    result = self.detect_amplitude_modulation(audio_data, sample_rate)
                elif method == 'frequency_domain':
                    result = self.detect_frequency_domain_watermarks(audio_data, sample_rate)
                else:
                    continue

                results['method_results'][method] = result

                if result['detected']:
                    results['detected'].append({
                        'method': method,
                        'confidence': result['confidence'],
                        'details': result['details']
                    })
                    results['confidence_scores'][method] = result['confidence']
                    results['watermark_count'] += 1

            except Exception as e:
                results['method_results'][method] = {
                    'detected': False,
                    'error': str(e),
                    'confidence': 0
                }

        # Calculate overall confidence
        if results['watermark_count'] > 0:
            results['overall_confidence'] = np.mean(list(results['confidence_scores'].values()))
        else:
            results['overall_confidence'] = 0.0

        return results

    def detect_spread_spectrum(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect spread spectrum watermarks using spectral analysis

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing detection results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        # Process each channel
        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Perform STFT with adaptive segment size
            nperseg = min(2048, len(channel_data) // 8)
            if nperseg < 256:
                nperseg = 256  # Minimum segment size
            f, t, Zxx = signal.stft(channel_data, fs=sample_rate, nperseg=nperseg)

            # Convert to magnitude
            magnitude = np.abs(Zxx)

            # Look for unusual patterns in high frequencies
            high_freq_mask = f > 15000  # Focus on high frequencies where watermarks often hide
            high_freq_data = magnitude[high_freq_mask, :]

            if high_freq_data.size == 0:
                continue

            # Statistical analysis of high frequency content
            mean_power = np.mean(high_freq_data)
            std_power = np.std(high_freq_data)

            # Check for suspiciously consistent patterns
            consistency_score = 1.0 - (std_power / (mean_power + 1e-10))

            # Look for periodic patterns
            correlation = np.correlate(high_freq_data.flatten(), high_freq_data.flatten(), mode='same')
            auto_corr_peaks = signal.find_peaks(correlation, height=np.max(correlation) * 0.8)[0]

            # Detection criteria
            if consistency_score > 0.7 or len(auto_corr_peaks) > 10:
                result['detected'] = True
                result['confidence'] = max(result['confidence'], consistency_score)
                result['details'].append({
                    'channel': channel_idx,
                    'consistency_score': consistency_score,
                    'periodic_peaks': len(auto_corr_peaks),
                    'type': 'high_frequency_pattern'
                })

            # Check for known watermark frequencies
            known_watermark_freqs = [18000, 19000, 20000, 21000]
            for wf in known_watermark_freqs:
                freq_idx = np.argmin(np.abs(f - wf))
                freq_power = np.mean(magnitude[freq_idx, :])

                if freq_power > mean_power + 3 * std_power:
                    result['detected'] = True
                    result['confidence'] = max(result['confidence'], 0.8)
                    result['details'].append({
                        'channel': channel_idx,
                        'frequency': wf,
                        'power_anomaly': freq_power / (mean_power + 1e-10),
                        'type': 'suspicious_frequency'
                    })

        return result

    def detect_echo_signatures(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect echo-based watermarks

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing detection results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Compute autocorrelation to find echo patterns
            autocorr = np.correlate(channel_data, channel_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr, height=0.1, distance=100)

            # Look for suspicious echo delays
            echo_delays = []
            echo_strengths = []

            for peak in peaks[1:10]:  # Check first 10 significant peaks
                delay_samples = peak
                delay_ms = (delay_samples / sample_rate) * 1000
                strength = autocorr[peak] / autocorr[0]

                # Echo watermarks typically have delays between 1-50ms
                if 1 <= delay_ms <= 50 and strength > 0.1:
                    echo_delays.append(delay_ms)
                    echo_strengths.append(strength)

            # Detection logic
            if len(echo_delays) >= 2:
                # Check if echoes form a pattern
                delay_diffs = np.diff(echo_delays)
                delay_consistency = 1.0 - (np.std(delay_diffs) / (np.mean(delay_diffs) + 1e-10))

                if delay_consistency > 0.8:
                    result['detected'] = True
                    result['confidence'] = max(result['confidence'], delay_consistency)
                    result['details'].append({
                        'channel': channel_idx,
                        'echo_count': len(echo_delays),
                        'delay_consistency': delay_consistency,
                        'avg_strength': np.mean(echo_strengths),
                        'type': 'patterned_echoes'
                    })

        return result

    def analyze_statistical_anomalies(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect statistical anomalies typical of AI-generated audio

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing analysis results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Statistical features
            features = {}

            # Distribution analysis
            features['skewness'] = float(self._skewness(channel_data))
            features['kurtosis'] = float(self._kurtosis(channel_data))
            features['entropy'] = float(entropy(np.histogram(channel_data, bins=100)[0] + 1e-10))

            # Temporal analysis
            diff_signal = np.diff(channel_data)
            features['diff_std'] = float(np.std(diff_signal))
            features['diff_entropy'] = float(entropy(np.histogram(diff_signal, bins=100)[0] + 1e-10))

            # Frequency domain statistics
            fft = np.fft.fft(channel_data)
            magnitude = np.abs(fft[:len(fft)//2])
            features['spectral_entropy'] = float(entropy(magnitude + 1e-10))
            features['spectral_centroid'] = float(np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10))

            # AI-generated audio often has unusual statistical properties
            anomalies = []

            # Check for unusually low entropy (too perfect)
            if features['entropy'] < 6.0:
                anomalies.append('low_entropy')
                result['detected'] = True
                result['confidence'] = max(result['confidence'], 0.7)

            # Check for unusual kurtosis
            if abs(features['kurtosis'] - 3.0) > 2.0:  # Normal distribution has kurtosis = 3
                anomalies.append('unusual_kurtosis')
                result['detected'] = True
                result['confidence'] = max(result['confidence'], 0.6)

            # Check for low spectral entropy
            if features['spectral_entropy'] < 8.0:
                anomalies.append('low_spectral_entropy')
                result['detected'] = True
                result['confidence'] = max(result['confidence'], 0.5)

            if anomalies:
                result['details'].append({
                    'channel': channel_idx,
                    'anomalies': anomalies,
                    'features': features,
                    'type': 'statistical_anomaly'
                })

        return result

    def detect_phase_modulation(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect phase modulation watermarks

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing detection results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Short-time Fourier transform for phase analysis
            f, t, Zxx = signal.stft(channel_data, fs=sample_rate, nperseg=2048)
            phase = np.angle(Zxx)

            # Unwrap phase to avoid 2π jumps
            unwrapped_phase = np.unwrap(phase, axis=1)

            # Calculate phase differences
            phase_diff = np.diff(unwrapped_phase, axis=1)

            # Look for suspicious phase patterns
            # Watermarks often create regular phase patterns
            phase_std = np.std(phase_diff, axis=1)
            phase_mean = np.mean(phase_std)

            # Check for unusually consistent phase patterns
            consistency_score = 1.0 - (phase_mean / (np.std(phase_std) + 1e-10))

            if consistency_score > 0.7:
                result['detected'] = True
                result['confidence'] = max(result['confidence'], consistency_score)
                result['details'].append({
                    'channel': channel_idx,
                    'phase_consistency': consistency_score,
                    'type': 'phase_pattern'
                })

        return result

    def detect_amplitude_modulation(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect amplitude modulation watermarks

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing detection results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Calculate envelope
            analytic_signal = signal.hilbert(channel_data)
            amplitude_envelope = np.abs(analytic_signal)

            # Look for amplitude modulation patterns
            # Compute FFT of envelope to find modulation frequencies
            envelope_fft = np.fft.fft(amplitude_envelope)
            envelope_freqs = np.fft.fftfreq(len(amplitude_envelope), 1/sample_rate)

            # Check for suspicious modulation frequencies (typically 1-100 Hz for watermarks)
            modulation_range = (envelope_freqs > 1) & (envelope_freqs < 100)
            modulation_power = np.abs(envelope_fft[modulation_range])

            if len(modulation_power) > 0:
                modulation_peaks, _ = signal.find_peaks(modulation_power, height=np.max(modulation_power) * 0.1)

                if len(modulation_peaks) > 5:
                    result['detected'] = True
                    result['confidence'] = max(result['confidence'], 0.6)
                    result['details'].append({
                        'channel': channel_idx,
                        'modulation_peaks': len(modulation_peaks),
                        'max_modulation_power': float(np.max(modulation_power)),
                        'type': 'amplitude_modulation'
                    })

        return result

    def detect_frequency_domain_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Detect frequency domain watermarks using advanced spectral analysis

        Args:
            audio_data: Audio data
            sample_rate: Sample rate

        Returns:
            Dict containing detection results
        """
        result = {'detected': False, 'confidence': 0.0, 'details': []}

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]

            # Multi-resolution analysis
            window_sizes = [512, 1024, 2048, 4096]

            for window_size in window_sizes:
                f, t, Zxx = signal.stft(channel_data, fs=sample_rate, nperseg=window_size)
                magnitude = np.abs(Zxx)

                # Look for watermark signatures across different resolutions
                # Watermarks often create consistent patterns across scales

                # Detect abnormal spectral flatness
                spectral_flatness = self._spectral_flatness(magnitude)
                avg_flatness = np.mean(spectral_flatness)

                if avg_flatness > 0.3:  # Suspiciously flat spectrum
                    result['detected'] = True
                    result['confidence'] = max(result['confidence'], 0.5)
                    result['details'].append({
                        'channel': channel_idx,
                        'window_size': window_size,
                        'spectral_flatness': avg_flatness,
                        'type': 'spectral_flatness_anomaly'
                    })

                # Check for suspicious spectral peaks
                spectral_peaks = []
                for time_bin in range(magnitude.shape[1]):
                    spectrum = magnitude[:, time_bin]
                    peaks, properties = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1)
                    spectral_peaks.append(len(peaks))

                peak_consistency = 1.0 - (np.std(spectral_peaks) / (np.mean(spectral_peaks) + 1e-10))

                if peak_consistency > 0.8:
                    result['detected'] = True
                    result['confidence'] = max(result['confidence'], peak_consistency * 0.7)
                    result['details'].append({
                        'channel': channel_idx,
                        'window_size': window_size,
                        'peak_consistency': peak_consistency,
                        'avg_peaks': np.mean(spectral_peaks),
                        'type': 'consistent_spectral_peaks'
                    })

        return result

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) if std > 0 else 0

    def _spectral_flatness(self, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral flatness for each time frame"""
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
        arithmetic_mean = np.mean(magnitude, axis=0)
        return geometric_mean / (arithmetic_mean + 1e-10)