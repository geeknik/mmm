#!/usr/bin/env python3
"""
Fast analysis using all CPU cores and optimized processing
"""

import os
import librosa
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from mmm.detection.metadata_scanner import MetadataScanner
from mmm.detection.watermark_detector import WatermarkDetector
from mmm.detection.statistical_analyzer import StatisticalAnalyzer
import time

# Optimize for all CPU cores
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
os.environ['NUMBA_NUM_THREADS'] = str(mp.cpu_count())

def analyze_audio_chunk(args):
    """
    Analyze a chunk of audio data
    Args: (audio_chunk, sample_rate, chunk_id)
    """
    audio_chunk, sample_rate, chunk_id = args

    results = {
        'chunk_id': chunk_id,
        'watermarks': None,
        'statistics': None,
        'error': None
    }

    try:
        # Watermark detection
        detector = WatermarkDetector()
        results['watermarks'] = detector.detect_all(audio_chunk, sample_rate)

        # Statistical analysis
        analyzer = StatisticalAnalyzer()
        results['statistics'] = analyzer.analyze(audio_chunk, sample_rate)

    except Exception as e:
        results['error'] = str(e)

    return results

def fast_parallel_analysis(file_path, chunk_duration=10.0):
    """
    Fast parallel analysis using all CPU cores
    """
    print(f"🚀 Fast Parallel Analysis")
    print(f"   Using {mp.cpu_count()} CPU cores")
    print(f"   File: {file_path}")
    print(f"   Chunk duration: {chunk_duration}s")
    print()

    # Load audio
    print("🔍 Loading audio...")
    y, sr = librosa.load(str(file_path), sr=None, mono=True)

    chunk_samples = int(chunk_duration * sr)
    total_duration = len(y) / sr
    num_chunks = int(np.ceil(len(y) / chunk_samples))

    print(f"   Duration: {total_duration:.1f} seconds")
    print(f"   Processing in {num_chunks} chunks of {chunk_duration}s each")
    print()

    # Create chunks
    chunks = []
    chunk_positions = []
    for i in range(0, len(y), chunk_samples):
        end = min(i + chunk_samples, len(y))
        chunk = y[i:end]
        chunks.append(chunk)
        chunk_positions.append((i, end, i/sr, end/sr))

    # Prepare arguments for parallel processing
    args_list = [(chunk, sr, i) for i, chunk in enumerate(chunks)]

    # Process in parallel using all CPU cores
    print(f"🔬 Processing chunks in parallel...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        chunk_results = list(executor.map(analyze_audio_chunk, args_list))

    elapsed = time.time() - start_time

    print(f"✅ Completed in {elapsed:.2f} seconds")
    print(f"   Speed: {total_duration/elapsed:.1f}x real-time")

    # Aggregate results
    print("\n📊 Aggregating Results...")

    # Metadata analysis (single thread - fast)
    print("   🔍 Scanning metadata...")
    scanner = MetadataScanner()
    metadata = scanner.scan_file(file_path)

    # Aggregate watermark results
    total_watermarks = 0
    avg_confidence = 0
    for result in chunk_results:
        if result['watermarks'] and result['watermarks'].get('watermark_count', 0) > 0:
            total_watermarks += result['watermarks']['watermark_count']
            avg_confidence += result['watermarks']['overall_confidence']

    avg_confidence = avg_confidence / len(chunk_results) if chunk_results else 0

    # Aggregate statistics
    avg_ai_prob = 0
    valid_stats = [r['statistics'] for r in chunk_results if r['statistics'] is not None]
    if valid_stats:
        avg_ai_prob = np.mean([s['ai_probability'] for s in valid_stats])

    # Display results
    print("\n" + "="*60)
    print("🎯 ANALYSIS RESULTS")
    print("="*60)

    print(f"\n📁 File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size/1024/1024:.1f} MB")
    print(f"   Duration: {total_duration:.1f} seconds")

    print(f"\n📋 Metadata:")
    print(f"   Tags found: {len(metadata['tags'])}")
    print(f"   Suspicious chunks: {len(metadata['suspicious_chunks'])}")
    print(f"   Hidden patterns: {len(metadata['hidden_data'])}")

    if metadata['tags']:
        for tag in metadata['tags'][:3]:
            suspicious = "🚨" if tag['suspicious'] else "✅"
            print(f"      {suspicious} {tag['key']}")

    print(f"\n🌊 Watermarks:")
    print(f"   Watermarks detected: {total_watermarks}")
    print(f"   Average confidence: {avg_confidence:.1%}")

    print(f"\n📊 Statistics:")
    print(f"   AI probability: {avg_ai_prob:.1%}")
    print(f"   Human confidence: {(1-avg_ai_prob):.1%}")

    # Calculate threat level
    total_threats = (
        len(metadata['tags']) +
        len(metadata['suspicious_chunks']) +
        len(metadata['hidden_data']) +
        total_watermarks
    )

    print(f"\n🚨 THREAT LEVEL: ", end="")
    if total_threats > 20:
        print("🔴 VERY HIGH - Extensive AI markers detected")
    elif total_threats > 10:
        print("🟠 HIGH - Strong AI generation indicators")
    elif total_threats > 5:
        print("🟡 MEDIUM - Some AI traces detected")
    else:
        print("🟢 LOW")

    print(f"   Total threats: {total_threats}")

    # Performance summary
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Processing time: {elapsed:.2f} seconds")
    print(f"   Real-time factor: {total_duration/elapsed:.1f}x")
    print(f"   CPU cores used: {mp.cpu_count()}")
    print(f"   Throughput: {(total_duration/elapsed)*60:.1f} audio-minutes/min")

    return {
        'metadata': metadata,
        'watermarks': {
            'total_count': total_watermarks,
            'avg_confidence': avg_confidence
        },
        'statistics': {
            'ai_probability': avg_ai_prob,
            'chunk_count': num_chunks
        },
        'performance': {
            'processing_time': elapsed,
            'realtime_factor': total_duration/elapsed
        },
        'total_threats': total_threats
    }

def main():
    """Main function"""
    print("🚀 MMM Fast Parallel Analysis")
    print("🎵 Optimized for multi-core CPU processing")
    print("="*60)

    file_path = Path("Schizo Shaman.mp3")
    if not file_path.exists():
        print("❌ Error: Schizo Shaman.mp3 not found")
        return

    results = fast_parallel_analysis(file_path)

    print("\n💀 Analysis Complete!")
    print("   Successfully analyzed AI-generated audio with maximum CPU utilization")
    print("   Ready for sanitization with optimized processing!")

if __name__ == "__main__":
    main()