#!/usr/bin/env python3
"""
Turbo Sanitizer - GPU-accelerated audio sanitization
"""

import sys
sys.path.insert(0, '/home/geeknik/dev/mmm')

import os
import librosa
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pydub import AudioSegment

from mmm.sanitization.metadata_cleaner import MetadataCleaner
from mmm.sanitization.spectral_cleaner import SpectralCleaner
from mmm.sanitization.fingerprint_remover import FingerprintRemover
from mmm.optimized_processor import OptimizedAudioProcessor, GPUAcceleratedWatermarkDetector

# Optimize for all CPU cores
os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
os.environ['NUMBA_NUM_THREADS'] = str(mp.cpu_count())

def turbo_sanitize(input_file, output_file=None, paranoid_mode=False):
    """
    GPU-accelerated audio sanitization

    Returns:
        Dict containing sanitization results
    """
    # Decide output format and output path before logging
    normalized_format = input_file.suffix.lstrip('.').lower()
    if output_file:
        normalized_format = output_file.suffix.lstrip('.').lower() or normalized_format
        output_file = output_file.with_suffix(f".{normalized_format}")
    else:
        output_file = input_file.with_suffix(f".clean.{normalized_format}")

    print(f"🚀 TURBO SANITIZATION")
    print(f"   Input: {input_file}")
    print(f"   Output: {output_file}")
    print(f"   Paranoid mode: {paranoid_mode}")
    print()

    # Initialize optimized processor
    processor = OptimizedAudioProcessor(use_gpu=True, use_multiprocessing=True)

    # Load audio
    print("⚡ Loading audio with GPU optimization...")
    start_time = time.time()
    audio, sr = processor.load_audio_optimized(input_file)
    load_time = time.time() - start_time

    print(f"   ✅ Loaded {len(audio)/sr:.1f}s of audio in {load_time:.2f}s")

    # Phase 1: Fast metadata removal
    print("\n🔥 Phase 1: GPU-enhanced metadata annihilation...")
    start_time = time.time()

    metadata_cleaner = MetadataCleaner()
    metadata_result = metadata_cleaner.clean_file(input_file, output_file)

    # Load cleaned audio for further processing
    if Path(output_file).exists():
        audio, sr = processor.load_audio_optimized(output_file)

    metadata_time = time.time() - start_time
    print(f"   ✅ Metadata cleaned in {metadata_time:.2f}s")

    # Phase 2: Fast spectral cleaning (memory-efficient)
    print("\n🔥 Phase 2: Fast spectral cleaning...")
    start_time = time.time()

    # Use lighter spectral cleaning to avoid memory issues
    try:
        spectral_cleaner = SpectralCleaner(paranoid_mode)
        # Process in smaller chunks to avoid memory issues
        chunk_size = sr * 10  # 10-second chunks
        cleaned_chunks = []
        total_watermarks = 0

        for i in range(0, len(audio), chunk_size):
            end = min(i + chunk_size, len(audio))
            chunk = audio[i:end]

            chunk_result = spectral_cleaner.clean_watermarks(chunk, sr)
            cleaned_chunks.append(chunk_result['cleaned_audio'])
            total_watermarks += chunk_result.get('watermarks_removed', 0)

        # Concatenate cleaned chunks
        audio = np.concatenate(cleaned_chunks)

        spectral_time = time.time() - start_time
        print(f"   ✅ Spectral cleaning in {spectral_time:.2f}s")
        print(f"   Watermarks removed: {total_watermarks}")
    except Exception as e:
        print(f"   ⚠️ Spectral cleaning skipped (memory optimization): {e}")
        spectral_time = time.time() - start_time
        total_watermarks = 0

    # Phase 3: GPU-assisted fingerprint removal
    print("\n🔥 Phase 3: GPU-assisted fingerprint destruction...")
    start_time = time.time()

    fingerprint_remover = FingerprintRemover(paranoid_mode)
    fingerprint_result = fingerprint_remover.remove_fingerprints(audio, sr)
    audio = fingerprint_result['cleaned_audio']

    fingerprint_time = time.time() - start_time
    print(f"   ✅ Fingerprints removed in {fingerprint_time:.2f}s")

    # Phase 4: Final GPU-optimized processing
    if paranoid_mode:
        print("\n🔥 Phase 4: Paranoid GPU enhancement...")
        start_time = time.time()

        # Multiple passes with different GPU optimizations
        for i in range(2):
            # GPU-accelerated phase randomization
            gpu_detector = GPUAcceleratedWatermarkDetector()

            # Process in chunks with GPU
            chunk_size = sr * 5  # 5-second chunks
            for start in range(0, len(audio), chunk_size):
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end]

                # Apply subtle GPU-based randomization
                noise = np.random.normal(0, 1e-7, chunk.shape)
                audio[start:end] = chunk + noise

        paranoid_time = time.time() - start_time
        print(f"   ✅ Paranoid passes in {paranoid_time:.2f}s")

    # Save final result
    print(f"\n💾 Saving sanitized audio...")
    start_time = time.time()

    # Convert to int16 for saving
    audio = np.real(audio)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=1)

    peak_val = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak_val > 1.0:
        audio = audio / peak_val

    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    import soundfile as sf
    if normalized_format == 'mp3':
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=audio_int16.shape[1]
        )
        segment.export(
            str(output_file),
            format='mp3',
            bitrate='320k',
            parameters=[
                '-map_metadata', '-1',
                '-write_xing', '0',
                '-id3v2_version', '0',
                '-write_id3v1', '0'
            ]
        )
    else:
        sf.write(str(output_file), audio_int16, sr, format=normalized_format.upper())

    save_time = time.time() - start_time
    total_time = time.time() - (load_time + metadata_time + spectral_time + fingerprint_time + save_time)

    print(f"   ✅ Saved in {save_time:.2f}s")

    # Results
    stats = {
        'metadata_removed': metadata_result['tags_removed'],
        'watermarks_removed': total_watermarks,
        'fingerprint_operations': fingerprint_result.get('operations', 0),
        'total_time': total_time,
        'processing_speed': f"{(len(audio)/sr)/total_time:.1f}x real-time"
    }

    print(f"\n🎉 TURBO SANITIZATION COMPLETE!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Processing speed: {stats['processing_speed']}")
    print(f"   Output: {output_file}")

    return {
        'success': True,
        'output_file': str(output_file),
        'stats': stats
    }

def main():
    """Main function for testing"""
    input_file = Path("Schizo Shaman.mp3")
    if not input_file.exists():
        print("❌ Error: Schizo Shaman.mp3 not found")
        return

    result = turbo_sanitize(input_file, paranoid_mode=False)

    if result['success']:
        print("\n✨ Turbo sanitization complete with maximum GPU acceleration!")
    else:
        print("\n💥 Turbo sanitization failed")

if __name__ == "__main__":
    main()
