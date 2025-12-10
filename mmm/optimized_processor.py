"""
Optimized processor with CPU multi-threading and GPU acceleration
"""

import os
import numpy as np
import librosa
import cupy as cp  # GPU acceleration
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
from functools import partial

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("💻 GPU not available, using CPU-only mode")


class OptimizedAudioProcessor:
    """
    High-performance audio processor using multi-core CPU and GPU acceleration
    """

    def __init__(self, use_gpu: bool = True, use_multiprocessing: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_multiprocessing = use_multiprocessing
        self.num_cores = mp.cpu_count()

        print(f"🔧 Initialized processor:")
        print(f"   CPU cores: {self.num_cores}")
        print(f"   GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        print(f"   Multiprocessing: {'Enabled' if self.use_multiprocessing else 'Disabled'}")

    def load_audio_optimized(self, file_path: Path, sample_rate: int = None) -> Tuple[np.ndarray, int]:
        """
        Load audio with optimized parameters
        """
        # Use librosa's efficient loading
        y, sr = librosa.load(
            str(file_path),
            sr=sample_rate,
            mono=True,
            dtype=np.float32
        )

        return y, sr

    def process_in_chunks(self, audio_data: np.ndarray, sample_rate: int,
                           chunk_duration: float = 10.0,
                           chunk_overlap: float = 1.0,
                           process_func=None) -> List[Any]:
        """
        Process audio in parallel chunks using all available cores
        """
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(chunk_overlap * sample_rate)
        step_size = chunk_samples - overlap_samples

        # Create chunks
        chunks = []
        for start in range(0, len(audio_data) - chunk_samples + 1, step_size):
            end = min(start + chunk_samples, len(audio_data))
            chunks.append(audio_data[start:end])

        print(f"📊 Processing {len(chunks)} chunks ({chunk_duration}s each)")
        print(f"   Total audio: {len(audio_data)/sample_rate:.1f} seconds")

        # Process chunks in parallel
        if self.use_multiprocessing and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                # Prepare partial function with sample_rate
                func = partial(process_func, sample_rate=sample_rate)
                results = list(executor.map(func, chunks))
        else:
            # Sequential processing
            results = [process_func(chunk, sample_rate) for chunk in chunks]

        return results

    def detect_watermarks_parallel(self, audio_data: np.ndarray, sample_rate: int,
                                   chunk_duration: float = 15.0) -> Dict[str, Any]:
        """
        Parallel watermark detection using all CPU cores
        """
        from .detection.watermark_detector import WatermarkDetector

        detector = WatermarkDetector()
        chunk_samples = int(chunk_duration * sample_rate)

        # Create chunks
        chunks = []
        chunk_positions = []
        for start in range(0, len(audio_data), chunk_samples):
            end = min(start + chunk_samples, len(audio_data))
            chunks.append(audio_data[start:end])
            chunk_positions.append((start, end))

        print(f"🔍 Running parallel watermark detection on {len(chunks)} chunks")

        # Detection function for each chunk
        def detect_chunk(audio_chunk, chunk_idx):
            return detector.detect_all(audio_chunk, sample_rate)

        # Process in parallel
        if self.use_multiprocessing and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
                results = list(executor.map(detect_chunk, chunks))
        else:
            results = [detect_chunk(chunk, i) for i, chunk in enumerate(chunks)]

        # Aggregate results
        aggregated = {
            'detected': [],
            'method_results': {},
            'confidence_scores': {},
            'watermark_count': 0,
            'chunk_count': len(chunks)
        }

        total_confidence = 0
        for i, result in enumerate(results):
            if 'error' not in result:
                aggregated['detected'].extend(result.get('detected', []))
                aggregated['watermark_count'] += result.get('watermark_count', 0)
                total_confidence += result.get('overall_confidence', 0)

        aggregated['overall_confidence'] = total_confidence / len(results) if results else 0

        return aggregated

    def gpu_accelerated_stft(self, audio_data: np.ndarray,
                             n_fft: int = 2048,
                             hop_length: int = 512) -> np.ndarray:
        """
        GPU-accelerated STFT using CuPy
        """
        if not self.use_gpu:
            # Fallback to librosa CPU version
            return librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

        try:
            # Transfer to GPU
            audio_gpu = cp.asarray(audio_data)

            # GPU STFT
            stft_gpu = cp.fft.rfft(audio_gpu, n=n_fft)

            # Transfer back to CPU
            return cp.asnumpy(stft_gpu)
        except Exception as e:
            print(f"⚠️ GPU STFT failed, falling back to CPU: {e}")
            return librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)

    def optimize_librosa_performance(self):
        """
        Configure librosa for maximum performance
        """
        import librosa
        import threading

        # Use multiple threads for librosa operations
        librosa.set_audio_backend("soundfile")
        librosa.set_max_threads(self.num_cores)

        # Set thread affinity for better performance
        if hasattr(os, 'sched_setaffinity'):
            pid = os.getpid()
            os.sched_setaffinity(pid, range(self.num_cores))

        print(f"⚡ Optimized for {self.num_cores} CPU cores")


class GPUAcceleratedWatermarkDetector:
    """
    GPU-accelerated watermark detection for RTX 3080 Ti
    """

    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            import torch
            self.device = torch.device('cuda')
            print(f"🚀 GPU Watermark Detector Initialized on {torch.cuda.get_device_name()}")
        else:
            self.device = None
            print("💻 GPU not available, using CPU mode")

    def detect_spectral_patterns_gpu(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        GPU-accelerated spectral pattern detection
        """
        if not self.gpu_available:
            # Fallback to CPU implementation
            from .detection.watermark_detector import WatermarkDetector
            detector = WatermarkDetector()
            return detector.detect_spread_spectrum(audio_data, sample_rate)

        import torch

        # Convert to PyTorch tensor on GPU
        audio_tensor = torch.from_numpy(audio_data).float().to(self.device)

        # GPU FFT computation
        fft_tensor = torch.fft.fft(audio_tensor)
        magnitude = torch.abs(fft_tensor)

        # Create frequency array
        freqs = torch.fft.fftfreq(len(audio_tensor), 1/sample_rate)

        # GPU-based high frequency analysis
        high_freq_mask = torch.abs(freqs) > 15000
        high_freq_power = torch.mean(magnitude[high_freq_mask])

        # Check for suspicious high frequency content
        threshold = torch.median(magnitude) * 5
        suspicious = high_freq_power > threshold

        return {
            'detected': suspicious,
            'confidence': float(high_freq_power / threshold) if threshold > 0 else 0.0,
            'details': ['GPU-accelerated spectral analysis']
        }

    def batch_process_files(self, file_paths: List[Path],
                           chunk_duration: float = 30.0) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel using GPU batch processing
        """
        if not self.gpu_available:
            print("💻 GPU not available, using CPU batch processing")
            # Fallback implementation
            from concurrent.futures import ThreadPoolExecutor
            from .core.audio_sanitizer import AudioSanitizer
            from .config.config_manager import ConfigManager

            config_manager = ConfigManager()
            with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
                futures = []
                for file_path in file_paths:
                    sanitizer = AudioSanitizer(file_path, config=config_manager.config)
                    futures.append(executor.submit(self._analyze_single_file, sanitizer))

                results = [f.result() for f in futures]
                return results

        import torch

        # GPU batch processing
        batch_size = 4  # Adjust based on VRAM
        results = []

        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]

            # Load batch to GPU (small chunks)
            batch_audio = []
            for file_path in batch_files:
                y, sr = librosa.load(str(file_path), sr=48000, mono=True)
                chunk = y[:int(chunk_duration * sr)]
                batch_audio.append(chunk)

            # Pad to same length
            max_len = max(len(a) for a in batch_audio)
            padded_batch = []
            for audio in batch_audio:
                padded = np.zeros(max_len)
                padded[:len(audio)] = audio
                padded_batch.append(padded)

            # Convert to tensor batch
            batch_tensor = torch.from_numpy(np.array(padded_batch)).float().to(self.device)

            # GPU processing
            fft_batch = torch.fft.fft(batch_tensor, dim=1)
            magnitude_batch = torch.abs(fft_batch)

            # Analyze results
            for j, file_path in enumerate(batch_files):
                mag = magnitude_batch[j].cpu().numpy()
                results.append({
                    'file': str(file_path),
                    'gpu_processed': True,
                    'spectral_energy': np.mean(mag),
                    'high_freq_content': np.mean(mag[len(mag)//4:]),
                })

        return results

    def _analyze_single_file(self, sanitizer):
        """Helper for CPU fallback"""
        try:
            analysis = sanitizer.analyze_file(deep=False)  # Skip heavy statistical analysis
            return {
                'success': True,
                'file': str(sanitizer.input_file),
                'analysis': analysis
            }
        except Exception as e:
            return {
                'success': False,
                'file': str(sanitizer.input_file),
                'error': str(e)
            }


# Performance optimization utilities
def optimize_system():
    """
    Optimize system for maximum audio processing performance
    """
    import os

    # Set environment variables for better performance
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(mp.cpu_count())
    os.environ['NUMBA_NUM_THREADS'] = str(mp.cpu_count())

    # Optimize numpy
    import numpy as np
    np.set_printoptions(threshold=100)  # Less printing overhead

    print(f"⚡ System optimized for {mp.cpu_count()} CPU cores")

    # Check NVIDIA GPU details
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 NVIDIA GPU Status:")
            print(result.stdout)
    except:
        pass
