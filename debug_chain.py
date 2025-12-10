#!/usr/bin/env python3
"""
Debug the full audio analysis chain
"""

import sys
sys.path.insert(0, '/home/geeknik/dev/mmm')

from pathlib import Path
from mmm.core.audio_sanitizer import AudioSanitizer
from mmm.config.config_manager import ConfigManager

def debug_analysis_chain():
    file_path = Path("Schizo Shaman.mp3")

    print("🔍 Debugging full analysis chain...")
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size/1024/1024:.1f} MB")

    # Test audio sanitizer
    print("\n1. Creating AudioSanitizer...")
    config = ConfigManager()
    sanitizer = AudioSanitizer(file_path, config=config.config)

    print("\n2. Loading audio...")
    if sanitizer.load_audio():
        print(f"✅ Audio loaded successfully!")
        print(f"   Shape: {sanitizer.audio_data.shape}")
        print(f"   Sample rate: {sanitizer.sample_rate}")
        print(f"   Duration: {len(sanitizer.audio_data.flatten())/sanitizer.sample_rate:.1f} seconds")
    else:
        print("❌ Failed to load audio")
        return

    print("\n3. Testing watermark detector...")
    watermark_result = sanitizer.watermark_detector.detect_all(
        sanitizer.audio_data, sanitizer.sample_rate
    )
    print(f"   Watermark result keys: {list(watermark_result.keys())}")
    print(f"   Watermarks detected: {watermark_result.get('watermark_count', 0)}")
    print(f"   Error: {watermark_result.get('error', 'None')}")

    print("\n4. Testing metadata scanner...")
    metadata_result = sanitizer.metadata_scanner.scan_file(file_path)
    print(f"   Metadata result keys: {list(metadata_result.keys())}")
    print(f"   Tags found: {len(metadata_result.get('tags', []))}")

    print("\n5. Testing statistical analyzer...")
    stat_result = sanitizer.statistical_analyzer.analyze(
        sanitizer.audio_data, sanitizer.sample_rate
    )
    print(f"   Statistical result keys: {list(stat_result.keys())}")
    print(f"   AI probability: {stat_result.get('ai_probability', 0):.1%}")

if __name__ == "__main__":
    debug_analysis_chain()