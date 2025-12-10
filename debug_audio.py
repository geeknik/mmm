#!/usr/bin/env python3
"""
Debug script to test audio loading
"""

import sys
sys.path.insert(0, '/home/geeknik/dev/mmm')

from pathlib import Path
import librosa
import numpy as np

def test_audio_loading():
    file_path = Path("Schizo Shaman.mp3")

    print("🔍 Testing audio loading...")
    print(f"File exists: {file_path.exists()}")

    # Test librosa loading with different settings
    print("\n1. Loading with librosa (default mono=True):")
    y1, sr1 = librosa.load(str(file_path), sr=None, mono=True)
    print(f"   Shape: {y1.shape}")
    print(f"   Sample rate: {sr1}")
    print(f"   Duration: {len(y1)/sr1:.1f} seconds")
    print(f"   Data type: {y1.dtype}")

    print("\n2. Loading with librosa (mono=False):")
    y2, sr2 = librosa.load(str(file_path), sr=None, mono=False)
    print(f"   Shape: {y2.shape}")
    print(f"   Sample rate: {sr2}")
    print(f"   Duration: {y2.shape[-1]/sr2:.1f} seconds")
    print(f"   Data type: {y2.dtype}")

    print("\n3. Testing watermark detector audio processing:")
    from mmm.detection.watermark_detector import WatermarkDetector

    detector = WatermarkDetector()

    # Test with mono audio
    print(f"\n   Mono audio shape: {y1.shape}")
    if y1.ndim == 1:
        y1_expanded = np.expand_dims(y1, axis=1)
        print(f"   Expanded shape: {y1_expanded.shape}")
        print(f"   Processing channels: {y1_expanded.shape[1]}")
        for channel_idx in range(y1_expanded.shape[1]):
            channel_data = y1_expanded[:, channel_idx]
            print(f"   Channel {channel_idx} shape: {channel_data.shape}")

    print("\n4. Testing with short sample:")
    sample = y1[:10000]  # First 10k samples
    print(f"   Sample shape: {sample.shape}")
    if sample.ndim == 1:
        sample_expanded = np.expand_dims(sample, axis=1)
        print(f"   Expanded sample shape: {sample_expanded.shape}")

        # Test detection
        result = detector.detect_spread_spectrum(sample_expanded, sr1)
        print(f"   Detection result: {result}")

if __name__ == "__main__":
    test_audio_loading()