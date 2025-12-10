"""
Spectral cleaner for removing frequency-domain watermarks
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Dict, List, Any, Tuple
import librosa


class SpectralCleaner:
    """
    Removes watermarks and anomalies from the frequency domain
    """

    def __init__(self, paranoid_mode: bool = False):
        self.paranoid_mode = paranoid_mode
        self.watermark_freq_bands = [
            (18000, 18500),  # Known AI watermark ranges
            (19000, 19500),
            (20000, 20500),
            (21000, 21500)
        ]
        self.suspicious_patterns = [
            'periodic_peaks',
            'constant_frequencies',
            'unnatural_harmonics',
            'synchronization_tones'
        ]

    def clean_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Remove spectral watermarks from audio

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate

        Returns:
            Dict containing cleaning results
        """
        result = {
            'cleaned_audio': audio_data.copy(),
            'watermarks_found': 0,
            'watermarks_removed': 0,
            'methods_used': [],
            'details': []
        }

        # Ensure stereo handling
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        cleaned_channels = []

        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]
            cleaned_channel = channel_data.copy()

            # Method 1: High-frequency watermark removal
            watermark_result = self._remove_high_frequency_watermarks(
                cleaned_channel, sample_rate
            )
            cleaned_channel = watermark_result['cleaned_data']
            result['watermarks_found'] += watermark_result['found']
            result['watermarks_removed'] += watermark_result['removed']
            result['methods_used'].append('high_freq_filter')

            # Method 2: Periodic pattern disruption
            pattern_result = self._disrupt_periodic_patterns(
                cleaned_channel, sample_rate
            )
            cleaned_channel = pattern_result['cleaned_data']
            result['methods_used'].append('pattern_disruption')

            # Method 3: Spectral smoothing
            if self.paranoid_mode:
                smooth_result = self._spectral_smoothing(cleaned_channel, sample_rate)
                cleaned_channel = smooth_result['cleaned_data']
                result['methods_used'].append('spectral_smoothing')

            # Method 4: Adaptive noise shaping
            if self.paranoid_mode:
                noise_result = self._adaptive_noise_shaping(cleaned_channel, sample_rate)
                cleaned_channel = noise_result['cleaned_data']
                result['methods_used'].append('noise_shaping')

            cleaned_channels.append(cleaned_channel)

        # Reconstruct multi-channel audio
        if len(cleaned_channels) == 1:
            result['cleaned_audio'] = cleaned_channels[0]
        else:
            result['cleaned_audio'] = np.column_stack(cleaned_channels)

        # Add verification details
        result['details'] = self._verify_cleaning(
            audio_data, result['cleaned_audio'], sample_rate
        )

        return result

    def _remove_high_frequency_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Remove watermarks in high frequency ranges"""
        result = {
            'cleaned_data': audio_data.copy(),
            'found': 0,
            'removed': 0,
            'frequencies_cleaned': []
        }

        # Perform FFT
        fft_data = fft(audio_data)
        freqs = fftfreq(len(audio_data), 1/sample_rate)

        # Work on a copy to satisfy static analyzers
        fft_mod = fft_data.copy()

        # Scan for suspicious high frequency content
        for freq_range in self.watermark_freq_bands:
            freq_min, freq_max = freq_range

            if freq_max > sample_rate / 2:
                continue  # Skip frequencies above Nyquist

            # Find frequencies in range
            freq_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) < freq_max)
            freq_power = np.abs(fft_mod[freq_mask])

            if len(freq_power) > 0:
                avg_power = np.mean(freq_power)
                noise_floor = np.median(np.abs(fft_mod))

                # Detect watermark if power is significantly above noise floor
                if avg_power > noise_floor * 5:
                    result['found'] += 1

                    # Remove or attenuate suspicious frequencies
                    attenuation_factor = 0.1  # Reduce by 90%
                    fft_mod[freq_mask] = fft_mod[freq_mask] * attenuation_factor

                    result['removed'] += 1
                    result['frequencies_cleaned'].append(freq_range)

        # Convert back to time domain
        result['cleaned_data'] = np.real(ifft(fft_mod))

        return result

    def _disrupt_periodic_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Disrupt periodic spectral patterns that indicate watermarks"""
        result = {
            'cleaned_data': audio_data.copy(),
            'patterns_disrupted': []
        }

        # Compute spectrogram
        nperseg = min(2048, len(audio_data) // 8)
        if nperseg < 256:
            nperseg = 256
        f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate, nperseg=nperseg)

        # Look for periodic patterns in frequency domain
        spectral_mean = np.mean(Sxx, axis=1)

        # Find peaks in spectral content
        peaks, properties = signal.find_peaks(spectral_mean, height=np.max(spectral_mean) * 0.1)

        # Check for suspicious periodicity in peaks
        if len(peaks) > 2:
            peak_spacing = np.diff(peaks)
            spacing_consistency = 1.0 - (np.std(peak_spacing) / (np.mean(peak_spacing) + 1e-10))

            if spacing_consistency > 0.8:  # Suspiciously consistent
                result['patterns_disrupted'].append({
                    'type': 'spectral_periodicity',
                    'consistency': spacing_consistency,
                    'peak_count': len(peaks)
                })

                # Apply subtle randomization to disrupt pattern
                noise_level = 1e-5
                phase_noise = np.random.normal(0, noise_level, len(audio_data))
                result['cleaned_data'] += phase_noise

        # Apply notch filters to suspicious frequencies
        for peak_idx in peaks:
            freq = f[peak_idx]

            # Skip fundamental frequencies (< 100 Hz) and very high frequencies
            if 100 <= freq <= sample_rate / 2 - 100:
                # Apply narrow notch filter
                result['cleaned_data'] = self._apply_notch_filter(
                    result['cleaned_data'], freq, sample_rate, q=30
                )

        return result

    def _apply_notch_filter(self, data: np.ndarray, freq: float, sample_rate: int, q: float = 30) -> np.ndarray:
        """Apply notch filter to remove specific frequency"""
        from scipy.signal import iirnotch, filtfilt

        # Design notch filter
        nyquist = sample_rate / 2
        normalized_freq = freq / nyquist

        # Ensure frequency is within valid range
        if normalized_freq >= 0.99:
            return data

        try:
            b, a = iirnotch(normalized_freq, q)
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except Exception:
            return data

    def _spectral_smoothing(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Apply spectral smoothing to hide watermark patterns"""
        result = {
            'cleaned_data': audio_data.copy(),
            'smoothing_applied': True
        }

        # Short-time Fourier Transform
        nperseg = min(2048, len(audio_data) // 8)
        if nperseg < 256:
            nperseg = 256
        stft = librosa.stft(audio_data, nperseg=nperseg)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Apply spectral smoothing
        window_size = 5
        smoothed_magnitude = np.zeros_like(magnitude)

        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                # Get local window
                i_start = max(0, i - window_size // 2)
                i_end = min(magnitude.shape[0], i + window_size // 2 + 1)
                j_start = max(0, j - window_size // 2)
                j_end = min(magnitude.shape[1], j + window_size // 2 + 1)

                # Apply weighted average
                local_window = magnitude[i_start:i_end, j_start:j_end]
                weights = np.exp(-0.5 * (np.arange(local_window.shape[0]) - window_size // 2) ** 2)
                weights = weights.reshape(-1, 1) * np.exp(-0.5 * (np.arange(local_window.shape[1]) - window_size // 2) ** 2)

                smoothed_magnitude[i, j] = np.sum(local_window * weights) / (np.sum(weights) + 1e-10)

        # Reconstruct signal
        smoothed_stft = smoothed_magnitude * np.exp(1j * phase)
        result['cleaned_data'] = librosa.istft(smoothed_stft, length=len(audio_data))

        return result

    def _adaptive_noise_shaping(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Apply adaptive noise shaping to mask watermarks"""
        result = {
            'cleaned_data': audio_data.copy(),
            'noise_shaping_applied': True
        }

        # Analyze spectral content
        fft_data = fft(audio_data)
        freqs = fftfreq(len(audio_data), 1/sample_rate)

        # Calculate spectral envelope
        magnitude = np.abs(fft_data)
        log_magnitude = np.log10(magnitude + 1e-10)

        # Smooth spectral envelope to detect anomalies
        envelope = signal.savgol_filter(log_magnitude, window_length=51, polyorder=3)

        # Identify spectral anomalies
        residuals = log_magnitude - envelope
        anomaly_threshold = np.std(residuals) * 2
        anomalies = np.abs(residuals) > anomaly_threshold

        # Add adaptive noise to mask anomalies
        noise_level = 1e-6
        adaptive_noise = np.random.normal(0, noise_level, len(audio_data))

        # Increase noise at anomaly frequencies
        anomaly_freq_indices = np.where(anomalies)[0]
        if len(anomaly_freq_indices) > 0:
            adaptive_noise[anomaly_freq_indices] *= 3

        # Apply noise in frequency domain
        noise_fft = fft(adaptive_noise)
        modified_fft = fft_data + noise_fft

        # Convert back to time domain
        result['cleaned_data'] = np.real(ifft(modified_fft))

        return result

    def _verify_cleaning(self, original: np.ndarray, cleaned: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Verify that cleaning was effective"""
        verification = []

        # Compute spectral comparison
        orig_fft = np.abs(fft(original, axis=0))
        clean_fft = np.abs(fft(cleaned, axis=0))

        # Calculate spectral difference
        spectral_diff = np.mean(np.abs(orig_fft - clean_fft), axis=0)

        # Check high frequency reduction
        freqs = fftfreq(len(original), 1/sample_rate)
        high_freq_mask = np.abs(freqs) > 15000

        if np.any(high_freq_mask):
            orig_hf_power = np.mean(orig_fft[high_freq_mask])
            clean_hf_power = np.mean(clean_fft[high_freq_mask])

            if clean_hf_power < orig_hf_power * 0.5:
                verification.append({
                    'metric': 'high_frequency_reduction',
                    'original_power': float(orig_hf_power),
                    'cleaned_power': float(clean_hf_power),
                    'reduction_percentage': float((1 - clean_hf_power / orig_hf_power) * 100)
                })

        # Check for pattern disruption
        orig_autocorr = np.correlate(original.flatten(), original.flatten(), mode='same')
        clean_autocorr = np.correlate(cleaned.flatten(), cleaned.flatten(), mode='same')

        orig_peaks = len(signal.find_peaks(orig_autocorr, height=np.max(orig_autocorr) * 0.8)[0])
        clean_peaks = len(signal.find_peaks(clean_autocorr, height=np.max(clean_autocorr) * 0.8)[0])

        if clean_peaks < orig_peaks:
            verification.append({
                'metric': 'pattern_disruption',
                'original_peaks': orig_peaks,
                'cleaned_peaks': clean_peaks,
                'reduction': orig_peaks - clean_peaks
            })

        return verification

    def remove_synchronization_tones(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove synchronization tones often used in watermarking"""
        cleaned_data = audio_data.copy()

        # Common synchronization tone frequencies
        sync_freqs = [1000, 2000, 3000, 4000, 5000, 10000, 15000]

        for sync_freq in sync_freqs:
            if sync_freq < sample_rate / 2:
                # Apply very narrow notch filter
                cleaned_data = self._apply_notch_filter(cleaned_data, sync_freq, sample_rate, q=50)

        return cleaned_data

    def spread_spectrum_watermark_removal(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Advanced spread spectrum watermark removal"""
        cleaned_data = audio_data.copy()

        # Multiple window sizes for comprehensive analysis
        window_sizes = [512, 1024, 2048, 4096]

        for window_size in window_sizes:
            # Compute STFT
            f, t, Zxx = signal.stft(cleaned_data, fs=sample_rate, nperseg=window_size)
            magnitude = np.abs(Zxx)

            # Detect and suppress suspicious patterns
            for time_idx in range(magnitude.shape[1]):
                spectrum = magnitude[:, time_idx]

                # Find unusual peaks that could be watermark carriers
                median_level = np.median(spectrum)
                peak_threshold = median_level * 5

                peaks = signal.find_peaks(spectrum, height=peak_threshold)[0]

                for peak_idx in peaks:
                    # Attenuate suspicious peaks
                    attenuation = 0.2
                    Zxx[peak_idx, time_idx] *= attenuation

            # Reconstruct signal
            _, cleaned_data = signal.istft(Zxx, fs=sample_rate, nperseg=window_size)

        return cleaned_data
