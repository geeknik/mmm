"""
Statistical analyzer for detecting AI-generated patterns in audio
"""

import numpy as np
import librosa
from scipy import stats
from scipy.stats import entropy
from typing import Dict, List, Any, Tuple


class StatisticalAnalyzer:
    """
    Analyzes statistical patterns typical of AI-generated audio
    """

    def __init__(self):
        self.human_audio_characteristics = {
            'entropy_range': (6.0, 10.0),
            'kurtosis_range': (1.5, 6.0),
            'skewness_range': (-0.5, 0.5),
            'rhythm_variance_range': (0.02, 0.2),
            'spectral_rolloff_variance': (0.01, 0.1)
        }

    def analyze(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis

        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Dict containing analysis results
        """
        results = {
            'anomalies': [],
            'statistical_features': {},
            'ai_probability': 0.0,
            'human_confidence': 0.0,
            'detailed_analysis': {}
        }

        # Ensure stereo handling
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=1)

        # Analyze each channel
        channel_results = []
        for channel_idx in range(audio_data.shape[1]):
            channel_data = audio_data[:, channel_idx]
            channel_analysis = self._analyze_channel(channel_data, sample_rate)
            channel_results.append(channel_analysis)

        # Combine channel results
        results['detailed_analysis']['channels'] = channel_results
        results['statistical_features'] = self._combine_channel_features(channel_results)

        # Detect anomalies
        results['anomalies'] = self._detect_anomalies(results['statistical_features'])

        # Calculate AI probability
        results['ai_probability'] = self._calculate_ai_probability(results['statistical_features'])
        results['human_confidence'] = 1.0 - results['ai_probability']

        # Add temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(audio_data, sample_rate)
        results['detailed_analysis']['temporal'] = temporal_analysis

        # Add spectral analysis
        spectral_analysis = self._analyze_spectral_patterns(audio_data, sample_rate)
        results['detailed_analysis']['spectral'] = spectral_analysis

        return results

    def _analyze_channel(self, channel_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze statistical features of a single channel"""
        features = {}

        # Basic statistical features
        features['mean'] = float(np.mean(channel_data))
        features['std'] = float(np.std(channel_data))
        features['min'] = float(np.min(channel_data))
        features['max'] = float(np.max(channel_data))
        features['range'] = features['max'] - features['min']

        # Distribution features
        features['skewness'] = float(stats.skew(channel_data))
        features['kurtosis'] = float(stats.kurtosis(channel_data))
        features['entropy'] = float(entropy(np.histogram(channel_data, bins=100)[0] + 1e-10))

        # Time-domain features
        features['zero_crossing_rate'] = float(librosa.feature.zero_crossing_rate(channel_data)[0].mean())
        features['rms_energy'] = float(np.sqrt(np.mean(channel_data ** 2)))

        # Spectral features
        features['spectral_centroid'] = float(librosa.feature.spectral_centroid(y=channel_data, sr=sample_rate)[0].mean())
        features['spectral_rolloff'] = float(librosa.feature.spectral_rolloff(y=channel_data, sr=sample_rate)[0].mean())
        features['spectral_bandwidth'] = float(librosa.feature.spectral_bandwidth(y=channel_data, sr=sample_rate)[0].mean())

        # MFCC features
        mfccs = librosa.feature.mfcc(y=channel_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))

        # Temporal variation features
        features['tempo'], beats = librosa.beat.beat_track(y=channel_data, sr=sample_rate)
        features['beat_consistency'] = self._calculate_beat_consistency(beats, sample_rate)

        return features

    def _combine_channel_features(self, channel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine features from multiple channels"""
        combined = {}

        for key in channel_results[0].keys():
            values = [ch[key] for ch in channel_results]
            combined[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }

        return combined

    def _detect_anomalies(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies indicating AI generation"""
        anomalies = []

        # Check entropy
        entropy_mean = features['entropy']['mean']
        if entropy_mean < self.human_audio_characteristics['entropy_range'][0]:
            anomalies.append({
                'type': 'low_entropy',
                'severity': 'high',
                'value': entropy_mean,
                'expected_range': self.human_audio_characteristics['entropy_range']
            })
        elif entropy_mean > self.human_audio_characteristics['entropy_range'][1]:
            anomalies.append({
                'type': 'high_entropy',
                'severity': 'medium',
                'value': entropy_mean,
                'expected_range': self.human_audio_characteristics['entropy_range']
            })

        # Check kurtosis
        kurtosis_mean = features['kurtosis']['mean']
        if not (self.human_audio_characteristics['kurtosis_range'][0] <= kurtosis_mean <=
                self.human_audio_characteristics['kurtosis_range'][1]):
            anomalies.append({
                'type': 'unusual_kurtosis',
                'severity': 'medium',
                'value': kurtosis_mean,
                'expected_range': self.human_audio_characteristics['kurtosis_range']
            })

        # Check skewness
        skewness_mean = abs(features['skewness']['mean'])
        if skewness_mean > self.human_audio_characteristics['skewness_range'][1]:
            anomalies.append({
                'type': 'unusual_skewness',
                'severity': 'medium',
                'value': skewness_mean,
                'expected_range': self.human_audio_characteristics['skewness_range']
            })

        # Check spectral features
        spectral_centroid_std = features['spectral_centroid']['std']
        if spectral_centroid_std < 0.01:  # Too consistent
            anomalies.append({
                'type': 'low_spectral_variation',
                'severity': 'medium',
                'value': spectral_centroid_std
            })

        # Check zero crossing rate
        zcr_mean = features['zero_crossing_rate']['mean']
        if zcr_mean < 0.01 or zcr_mean > 0.2:  # Unusual values
            anomalies.append({
                'type': 'unusual_zero_crossing_rate',
                'severity': 'low',
                'value': zcr_mean
            })

        return anomalies

    def _calculate_ai_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability that audio is AI-generated based on statistical features

        Returns:
            float: Probability between 0.0 and 1.0
        """
        ai_indicators = []

        # Entropy score
        entropy_score = self._score_feature(
            features['entropy']['mean'],
            self.human_audio_characteristics['entropy_range']
        )
        ai_indicators.append(entropy_score)

        # Kurtosis score
        kurtosis_score = self._score_feature(
            features['kurtosis']['mean'],
            self.human_audio_characteristics['kurtosis_range']
        )
        ai_indicators.append(kurtosis_score)

        # Skewness score
        skewness_score = self._score_feature(
            abs(features['skewness']['mean']),
            self.human_audio_characteristics['skewness_range']
        )
        ai_indicators.append(skewness_score)

        # Spectral consistency (AI often has too consistent spectra)
        spectral_consistency = 1.0 - min(1.0, features['spectral_centroid']['std'] * 100)
        ai_indicators.append(spectral_consistency)

        # Beat consistency (AI often has too perfect rhythm)
        beat_consistency = features['beat_consistency']['mean'] if 'beat_consistency' in features else 0.5
        beat_score = abs(beat_consistency - 0.5) * 2  # Far from 0.5 is suspicious
        ai_indicators.append(beat_score)

        # Weighted average
        weights = [0.2, 0.2, 0.15, 0.25, 0.2]
        weighted_score = sum(indicator * weight for indicator, weight in zip(ai_indicators, weights))

        return min(1.0, max(0.0, weighted_score))

    def _score_feature(self, value: float, expected_range: Tuple[float, float]) -> float:
        """
        Score how much a value deviates from expected range

        Returns:
            float: Score between 0.0 (normal) and 1.0 (very unusual)
        """
        min_val, max_val = expected_range

        if min_val <= value <= max_val:
            return 0.0
        elif value < min_val:
            # Exponential decay below minimum
            return min(1.0, np.exp(-(value / min_val)))
        else:
            # Exponential growth above maximum
            return min(1.0, 1.0 - np.exp(-(value / max_val - 1.0)))

    def _analyze_temporal_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze temporal patterns for AI generation indicators"""
        temporal_features = {}

        # Onset detection and analysis
        if audio_data.ndim == 1:
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
        else:
            # Use first channel for onset detection
            onset_frames = librosa.onset.onset_detect(y=audio_data[:, 0], sr=sample_rate)

        onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

        if len(onset_times) > 1:
            # Analyze onset intervals
            intervals = np.diff(onset_times)
            temporal_features['onset_interval_mean'] = float(np.mean(intervals))
            temporal_features['onset_interval_std'] = float(np.std(intervals))
            temporal_features['onset_regularity'] = 1.0 - (temporal_features['onset_interval_std'] / (temporal_features['onset_interval_mean'] + 1e-10))

            # AI often has too regular onsets
            if temporal_features['onset_regularity'] > 0.8:
                temporal_features['suspicious_regularity'] = True
            else:
                temporal_features['suspicious_regularity'] = False

        else:
            temporal_features['onset_interval_mean'] = 0.0
            temporal_features['onset_interval_std'] = 0.0
            temporal_features['onset_regularity'] = 0.0
            temporal_features['suspicious_regularity'] = False

        # Temporal entropy
        if audio_data.ndim == 1:
            temp_entropy = entropy(np.histogram(audio_data, bins=50)[0] + 1e-10)
        else:
            # Average entropy across channels
            entropies = [entropy(np.histogram(audio_data[:, ch], bins=50)[0] + 1e-10)
                        for ch in range(audio_data.shape[1])]
            temp_entropy = np.mean(entropies)

        temporal_features['temporal_entropy'] = float(temp_entropy)

        return temporal_features

    def _analyze_spectral_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze spectral patterns for AI generation indicators"""
        spectral_features = {}

        # Compute spectrogram
        if audio_data.ndim == 1:
            S = np.abs(librosa.stft(audio_data))
        else:
            # Use first channel for spectral analysis
            S = np.abs(librosa.stft(audio_data[:, 0]))

        # Convert to dB
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # Spectral features over time
        spectral_centroids = librosa.feature.spectral_centroid(S=S)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S)

        spectral_features['centroid_variance'] = float(np.var(spectral_centroids))
        spectral_features['rolloff_variance'] = float(np.var(spectral_rolloff))
        spectral_features['bandwidth_variance'] = float(np.var(spectral_bandwidth))

        # Check for unusual spectral consistency
        centroid_consistency = 1.0 / (spectral_features['centroid_variance'] + 1e-10)
        spectral_features['centroid_consistency_score'] = min(1.0, centroid_consistency / 1000)

        # Harmonic-percussive separation
        harmonic, percussive = librosa.decompose.hpss(S)
        harmonic_ratio = np.mean(harmonic) / (np.mean(harmonic) + np.mean(percussive) + 1e-10)
        spectral_features['harmonic_ratio'] = float(harmonic_ratio)

        # AI often has unusual harmonic characteristics
        if harmonic_ratio > 0.9 or harmonic_ratio < 0.1:
            spectral_features['unusual_harmonic_balance'] = True
        else:
            spectral_features['unusual_harmonic_balance'] = False

        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(S=S)
        spectral_features['flatness_mean'] = float(np.mean(spectral_flatness))
        spectral_features['flatness_variance'] = float(np.var(spectral_flatness))

        return spectral_features

    def _calculate_beat_consistency(self, beats: np.ndarray, sample_rate: int) -> float:
        """
        Calculate how consistent the beat timing is

        Returns:
            float: Consistency score (0.5 = normal, approaching 0 or 1 = suspicious)
        """
        if len(beats) < 2:
            return 0.5

        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)

        # Calculate intervals
        intervals = np.diff(beat_times)

        if len(intervals) == 0:
            return 0.5

        # Calculate coefficient of variation
        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)

        # Convert to consistency score (0.5 is normal)
        consistency = min(1.0, cv)

        return consistency