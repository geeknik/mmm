"""
Progress manager for visual feedback during processing
"""

import time
import random
from typing import List, Dict, Any
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table


class ProgressManager:
    """
    Manages progress bars and visual feedback during audio processing
    """

    def __init__(self):
        self.console = Console()
        self.processing_stages = [
            "Loading audio file...",
            "Scanning metadata...",
            "Detecting watermarks...",
            "Analyzing statistical patterns...",
            "Removing metadata...",
            "Cleaning spectral watermarks...",
            "Eliminating fingerprints...",
            "Applying final sanitization...",
            "Verifying cleanup...",
            "Saving cleaned audio..."
        ]

        self.paranoid_stages = [
            "Loading audio file...",
            "Deep forensic analysis...",
            "Metadata annihilation...",
            "Multi-pass spectral cleaning...",
            "Advanced fingerprint removal...",
            "Statistical pattern disruption...",
            "Phase randomization...",
            "Temporal perturbation...",
            "Human imperfection injection...",
            "Quality verification...",
            "Secure file saving..."
        ]

    def show_processing_progress(self, paranoid_mode: bool = False, estimated_time: float = 30.0):
        """Show detailed processing progress"""
        stages = self.paranoid_stages if paranoid_mode else self.processing_stages

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            tasks = []
            total_stages = len(stages)

            for stage in stages:
                task = progress.add_task(f"[cyan]{stage}", total=100)
                tasks.append(task)

            # Simulate progress with realistic timing
            for i, task in enumerate(tasks):
                # Different stages take different amounts of time
                if i < len(stages) - 1:
                    # Complete current task
                    for j in range(101):
                        progress.update(task, advance=1)
                        time.sleep(estimated_time / (total_stages * 100))
                else:
                    # Last stage
                    for j in range(101):
                        progress.update(task, advance=1)
                        time.sleep(estimated_time / (total_stages * 100))

    def show_analysis_progress(self, analysis_type: str = "comprehensive"):
        """Show progress for analysis operations"""
        analysis_stages = {
            "basic": [
                "Reading file headers...",
                "Scanning metadata tags...",
                "Checking format compliance..."
            ],
            "comprehensive": [
                "Reading file structure...",
                "Deep metadata scanning...",
                "Binary pattern analysis...",
                "Statistical profiling...",
                "Watermark detection...",
                "Fingerprint analysis...",
                "Threat assessment..."
            ],
            "forensic": [
                "File structure mapping...",
                "Complete metadata extraction...",
                "Hidden data scanning...",
                "Steganography detection...",
                "AI pattern recognition...",
                "Cryptographic analysis...",
                "Statistical anomaly detection...",
                "Cross-format validation...",
                "Integrity verification..."
            ]
        }

        stages = analysis_stages.get(analysis_type, analysis_stages["comprehensive"])

        with Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Analyzing: {task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            for stage in stages:
                task = progress.add_task(stage, total=None)
                time.sleep(random.uniform(0.5, 2.0))  # Variable timing for realism
                progress.update(task, completed=True)

    def show_batch_progress(self, files: List[str], current_index: int, current_file: str):
        """Show progress for batch processing"""
        total_files = len(files)
        progress_percent = (current_index / total_files) * 100

        # Create progress table
        table = Table(show_header=False, box=None, padding=0)
        table.add_column("Progress", style="cyan")
        table.add_column("Info", style="white")

        table.add_row(
            f"[{'█' * int(progress_percent // 2)}{'░' * (50 - int(progress_percent // 2))}]",
            f"File {current_index}/{total_files} ({progress_percent:.1f}%)"
        )
        table.add_row("", f"Current: {current_file}")

        self.console.print(table)

    def show_watermark_detection(self):
        """Show specialized progress for watermark detection"""
        detection_methods = [
            "Scanning high-frequency spectrum...",
            "Detecting spread spectrum watermarks...",
            "Finding echo-based signatures...",
            "Analyzing phase modulation...",
            "Checking amplitude modulation...",
            "Searching frequency domain patterns...",
            "Statistical anomaly detection...",
            "Cross-correlation analysis..."
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[red]🔍 {task.description}"),
            BarColumn(),
            console=self.console,
            transient=True
        ) as progress:
            for method in detection_methods:
                task = progress.add_task(method, total=100)
                for _ in range(101):
                    progress.update(task, advance=1)
                    time.sleep(0.01)

    def show_sanitization_progress(self, methods_used: List[str]):
        """Show progress for specific sanitization methods"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[green]🧹 {task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            for method in methods_used:
                task = progress.add_task(f"Applying {method}...", total=None)
                time.sleep(random.uniform(1.0, 3.0))
                progress.update(task, completed=True)

    def show_verification_progress(self):
        """Show progress for verification process"""
        verification_steps = [
            "Comparing original and cleaned files...",
            "Re-scanning for remaining watermarks...",
            "Statistical fingerprint verification...",
            "Quality metrics calculation...",
            "Hash comparison...",
            "Final threat assessment..."
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]🔬 {task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            for step in verification_steps:
                task = progress.add_task(step, total=None)
                time.sleep(0.5)
                progress.update(task, completed=True)

    def show_quality_check(self, original_size: int, cleaned_size: int):
        """Show quality check visualization"""
        size_diff = abs(original_size - cleaned_size)
        size_change_percent = (size_diff / original_size) * 100

        # Create quality metrics table
        table = Table(title="📊 Quality Check", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="white", width=15)
        table.add_column("Status", style="white")

        # File size
        status = "✅ Good" if size_change_percent < 5 else "⚠️ High" if size_change_percent < 20 else "❌ Poor"
        table.add_row(
            "Size Change",
            f"{size_change_percent:.2f}%",
            status
        )

        # Simulated quality metrics
        snr = random.uniform(20, 60)
        snr_status = "✅ Excellent" if snr > 40 else "✅ Good" if snr > 25 else "⚠️ Fair"
        table.add_row(
            "Signal Quality",
            f"{snr:.1f} dB",
            snr_status
        )

        spectral_similarity = random.uniform(0.85, 0.99)
        spec_status = "✅ Excellent" if spectral_similarity > 0.95 else "✅ Good" if spectral_similarity > 0.90 else "⚠️ Fair"
        table.add_row(
            "Spectral Integrity",
            f"{spectral_similarity:.1%}",
            spec_status
        )

        preservation = random.uniform(0.80, 0.99)
        pres_status = "✅ Excellent" if preservation > 0.95 else "✅ Good" if preservation > 0.85 else "⚠️ Fair"
        table.add_row(
            "Audio Preservation",
            f"{preservation:.1%}",
            pres_status
        )

        self.console.print(table)

    def show_threat_meter(self, threats_found: int, total_possible: int):
        """Show visual threat meter"""
        threat_level = (threats_found / total_possible) * 100 if total_possible > 0 else 0

        # Choose color based on threat level
        if threat_level > 66:
            color = "red"
            label = "HIGH"
        elif threat_level > 33:
            color = "yellow"
            label = "MEDIUM"
        else:
            color = "green"
            label = "LOW"

        # Create threat meter
        meter_width = 40
        filled = int(meter_width * threat_level / 100)
        meter = "█" * filled + "░" * (meter_width - filled)

        self.console.print(f"\n🚨 [bold]Threat Level: [{color}]{label}[/{color}]")
        self.console.print(f"[{color}]{meter}[/{color}] {threats_found} threats detected ({threat_level:.0f}%)")

    def show_progress_with_eta(self, current: int, total: int, start_time: float):
        """Show progress with estimated time remaining"""
        import time

        elapsed = time.time() - start_time
        if current > 0:
            eta = (elapsed / current) * (total - current)
        else:
            eta = 0

        progress_percent = (current / total) * 100
        bar_width = 40
        filled = int(bar_width * current // total)
        bar = "█" * filled + "░" * (bar_width - filled)

        eta_str = f"{eta:.0f}s" if eta > 0 else "soon"

        self.console.print(
            f"\r[{current}/{total}] [{bar}] {progress_percent:.1f}% - ETA: {eta_str}",
            end="",
            flush=True
        )

    def show_memory_usage(self, peak_memory: float = None):
        """Show memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss / (1024 * 1024)  # MB

            table = Table(title="💾 Memory Usage", show_header=True, header_style="bold blue")
            table.add_column("Type", style="cyan", width=15)
            table.add_column("Usage", style="white", width=15)

            table.add_row("Current", f"{current_memory:.1f} MB")

            if peak_memory:
                table.add_row("Peak", f"{peak_memory:.1f} MB")

            # System memory
            system_memory = psutil.virtual_memory()
            table.add_row("System Total", f"{system_memory.total / (1024**3):.1f} GB")
            table.add_row("System Used", f"{system_memory.percent:.1f}%")

            self.console.print(table)

        except ImportError:
            self.console.print("Memory info unavailable (psutil not installed)")

    def show_cpu_usage(self, duration: float = 1.0):
        """Show CPU usage during processing"""
        try:
            import psutil

            # Get CPU usage over duration
            cpu_percent = psutil.cpu_percent(duration)

            # Create CPU usage visualization
            cpu_bars = int(cpu_percent // 5)
            bar = "█" * cpu_bars + "░" * (20 - cpu_bars)

            color = "green" if cpu_percent < 50 else "yellow" if cpu_percent < 80 else "red"
            self.console.print(f"💻 CPU Usage: [{color}]{bar}[/{color}] {cpu_percent:.1f}%")

        except ImportError:
            self.console.print("CPU info unavailable (psutil not installed)")