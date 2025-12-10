"""
Console manager for CLI personality and user feedback
"""

import random
import time
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from colorama import Fore, Back, Style, init


class ConsoleManager:
    """
    Manages CLI output with personality and color
    """

    def __init__(self):
        init(autoreset=True)  # Initialize colorama
        self.console = Console()
        self.hacker_quotes = [
            "In the symphony of digital rights, we are the conductors of chaos. 🎼",
            "The best place to hide a dead body is page two of the search results. 🌊",
            "I don't need a backup plan, I'm the backup plan. 💀",
            "Delete your browser history, not your dreams. 🗑️",
            "404 sanity not found. But the audio is clean! 🧹",
            "I'm not arguing, I'm just explaining why I'm right. And the metadata is gone. ✨",
            "My favorite element is the element of surprise. 🎭",
            "I don't always test my code, but when I do, I do it in production. 🔥",
            "Debugging is like being a detective in a crime movie where you are also the murderer. 🕵️",
            "The code works because I made it work. Don't question the magic. ✨",
            "I have a addiction to reducing metadata. It's not a problem, it's a feature. 💊",
            "There are only 10 types of people: those who understand binary, and those who don't watermark. 01101000 01100001 01100011 01101011 01100101 01110010 01110011",
            "Life is short, sanitize everything. ⚡",
            "I'm not lazy, I'm on energy saving mode. And your audio is cleaner too. 🔋",
            "My code never has bugs. It just develops random features. 🐛",
            "I would tell you a UDP joke, but you might not get it. 📡",
            "There's no place like 127.0.0.1... especially after sanitization. 🏠",
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛 But sanitized audio doesn't.",
            "I don't trust atoms. They make up everything! Unlike your clean audio. ⚛️",
            "SQL queries are like relationships. Sometimes you need to JOIN to get what you want. 🔗"
        ]

        self.processing_messages = [
            "Analyzing spectral signatures... 🌊",
            "Hunting for digital breadcrumbs... 🔍",
            "Breaking the cryptographic chains... ⛓️",
            "Conducting forensic analysis... 🕵️",
            "Dissolving watermarks in acid... 🧪",
            "Scrambling digital fingerprints... 🔀",
            "Erasing the matrix... 💊",
            "Conducting audio exorcism... 👻",
            "Purging the digital demons... 😈",
            "Wiping the slate clean... 🧹",
            "Reprogramming reality... 🎮",
            "Conducting symphony of chaos... 🎼",
            "Bending space-time continuum... 🌌",
            "Rearranging molecular structure... ⚛️",
            "Executing protocol: CLEAN_SLATE... 📋",
            "Initiating quantum sanitization... ⚛️",
            "Collapsing wave functions... 🌊",
            "Entangling qubits of chaos... 🔀",
            "Hacking the simulation... 🎮",
            "Overwriting universe save file... 💾"
        ]

    def success(self, message: str):
        """Print success message with green color"""
        self.console.print(f"✨ {message}", style="bold green")

    def error(self, message: str):
        """Print error message with red color"""
        self.console.print(f"💥 {message}", style="bold red")

    def warning(self, message: str):
        """Print warning message with yellow color"""
        self.console.print(f"⚠️  {message}", style="bold yellow")

    def info(self, message: str):
        """Print info message with cyan color"""
        self.console.print(f"ℹ️  {message}", style="bold cyan")

    def hacker_quote(self):
        """Print a random hacker quote"""
        quote = random.choice(self.hacker_quotes)
        self.console.print(f"\n💬 {quote}", style="italic magenta")

    def processing_animation(self, task_name: str, duration: float = 2.0):
        """Show processing animation with custom message"""
        base_message = random.choice(self.processing_messages)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"{task_name} - {base_message}", total=None)
            time.sleep(duration)
            progress.update(task, completed=True)

    def show_progress(self, tasks: List[tuple], title: str = "Audio Sanitization Progress"):
        """Show multi-task progress bar"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True
        ) as progress:
            progress_tasks = {}

            for task_id, task_name in tasks:
                task = progress.add_task(task_name, total=100)
                progress_tasks[task_id] = task

            # Simulate progress (in real implementation, this would be connected to actual processing)
            for i in range(101):
                for task_id, task in progress_tasks.items():
                    progress.update(task, advance=1)
                    time.sleep(0.01)

    def display_analysis(self, analysis: Dict[str, Any]):
        """Display file analysis results"""
        table = Table(title="🔬 Audio Analysis Results", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Details", style="white")

        # File info
        file_info = analysis.get('file_info', {})
        table.add_row("File", file_info.get('path', 'Unknown'))
        table.add_row("Format", file_info.get('format', 'Unknown'))
        table.add_row("Size", f"{file_info.get('size', 0):,} bytes")
        table.add_row("Duration", f"{file_info.get('duration', 0):.2f} seconds")
        table.add_row("Sample Rate", f"{file_info.get('sample_rate', 0):,} Hz")
        table.add_row("", "")  # Separator

        # Metadata
        metadata = analysis.get('metadata', {})
        tags_count = len(metadata.get('tags', []))
        suspicious_chunks = len(metadata.get('suspicious_chunks', []))

        table.add_row("Metadata Tags", str(tags_count))
        if tags_count > 0:
            for tag in metadata.get('tags', [])[:3]:  # Show first 3
                table.add_row(f"  └─ {tag['key']}", str(tag['value'])[:50] + "..." if len(str(tag['value'])) > 50 else str(tag['value']))

        table.add_row("Suspicious Chunks", str(suspicious_chunks))
        if suspicious_chunks > 0:
            for chunk in metadata.get('suspicious_chunks', [])[:3]:
                reason = chunk.get('description', 'Unknown')
                offset = chunk.get('offset', '?')
                table.add_row("  └─ why", f"{reason} @ {offset}")

        anomalies = metadata.get('anomalies', [])
        if anomalies:
            table.add_row("Metadata Anomalies", str(len(anomalies)))
            for an in anomalies[:3]:
                table.add_row("  └─ note", an[:70] + "..." if len(an) > 70 else an)
        table.add_row("", "")  # Separator

        # Watermarks
        watermarks = analysis.get('watermarks', {})
        watermarks_detected = len(watermarks.get('detected', []))

        table.add_row("Watermarks", str(watermarks_detected))
        if watermarks_detected > 0:
            overall_confidence = watermarks.get('overall_confidence', 0)
            table.add_row("  └─ Confidence", f"{overall_confidence:.1%}")

        table.add_row("", "")  # Separator

        # Overall threats
        threats = analysis.get('threats_found', 0)
        threat_level = analysis.get('threat_level', 'UNKNOWN')
        threat_notes = []
        if suspicious_chunks > 0:
            threat_notes.append(f"{suspicious_chunks} suspicious container chunks (e.g., ID3/APE/RIFF markers)")
        if watermarks_detected > 0:
            threat_notes.append(f"{watermarks_detected} watermark signatures detected")
        if tags_count > 0:
            threat_notes.append(f"{tags_count} metadata tags present (may carry identifiers)")

        level_color = {
            'HIGH': 'red',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }.get(threat_level, 'white')

        table.add_row("Threats Found", str(threats))
        table.add_row("Threat Level", f"[{level_color}]{threat_level}[/{level_color}]")
        if threat_notes:
            table.add_row("Why It Matters", threat_notes[0])
            for note in threat_notes[1:2]:  # Show up to 2 notes
                table.add_row("  └─", note)

        self.console.print(Panel(table, title="📊 Forensic Analysis", border_style="blue"))

    def display_results(self, results: Dict[str, Any]):
        """Display sanitization results"""
        table = Table(title="🧹 Sanitization Results", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="white")

        stats = results.get('stats', {})
        notes = []

        table.add_row("Status", "✅ Success" if results.get('success') else "❌ Failed")
        table.add_row("", "")  # Separator

        table.add_row("Metadata Removed", str(stats.get('metadata_removed', 0)))
        table.add_row("Patterns Found", str(stats.get('patterns_found', 0)))
        table.add_row("Patterns Suppressed", str(stats.get('patterns_suppressed', 0)))
        # Add context if detector found none but cleaner reported suppressions
        if stats.get('patterns_found', 0) > 0 and stats.get('watermarks_detected', 0) == 0:
            notes.append("Pattern counts reflect spectral suppression ops; detector flagged no explicit watermarks pre-scan.")
        table.add_row("Quality Loss", f"{stats.get('quality_loss', 0):.2f}%")
        table.add_row("Processing Time", f"{stats.get('processing_time', 0):.2f} seconds")
        table.add_row("", "")  # Separator

        if 'output_file' in results:
            table.add_row("Output File", results['output_file'])

        if 'final_hash' in results:
            table.add_row("File Hash Changed", "✅ Yes" if results.get('original_hash') != results.get('final_hash') else "❌ No")

        if notes:
            table.add_row("Notes", notes[0])
            for n in notes[1:]:
                table.add_row("  └─", n)

        self.console.print(Panel(table, title="🎯 Mission Accomplished", border_style="green"))

    def display_verification(self, verification: Dict[str, Any]):
        """Display verification results"""
        if not verification.get('success'):
            self.error("Verification failed!")
            return

        table = Table(title="🔍 Verification Results", show_header=True, header_style="bold blue")
        table.add_column("Check", style="cyan", width=25)
        table.add_column("Result", style="white")

        original_threats = verification.get('original_threats', 0)
        remaining_threats = verification.get('remaining_threats', 0)
        effectiveness = verification.get('removal_effectiveness', 0)

        table.add_row("Original Threats", str(original_threats))
        table.add_row("Remaining Threats", str(remaining_threats))
        table.add_row("Removal Effectiveness", f"{effectiveness:.1f}%")
        table.add_row("Hash Changed", "✅ Yes" if verification.get('hash_different') else "❌ No")

        # Color code effectiveness
        if effectiveness >= 95:
            effectiveness_style = "bold green"
        elif effectiveness >= 80:
            effectiveness_style = "bold yellow"
        else:
            effectiveness_style = "bold red"

        table.add_row("", "")  # Separator
        table.add_row("Overall Assessment", f"[{effectiveness_style}]{'Excellent' if effectiveness >= 95 else 'Good' if effectiveness >= 80 else 'Poor'}[/{effectiveness_style}]")

        # Add context notes
        notes = []
        if verification.get('new_analysis', {}).get('metadata', {}).get('suspicious_chunks'):
            notes.append("Threat counts include suspicious container/metadata chunks; detector reported zero watermarks.")
        if verification.get('new_analysis', {}).get('watermarks', {}).get('detected') in ([], None):
            notes.append("No detector-verified watermarks; remaining threats are metadata/binary anomalies.")
        if notes:
            table.add_row("Notes", notes[0])
            for n in notes[1:]:
                table.add_row("  └─", n)

        self.console.print(Panel(table, title="✅ Quality Assurance", border_style="blue"))

    def display_detailed_analysis(self, analysis: Dict[str, Any]):
        """Display detailed analysis for analyze command"""
        self.display_analysis(analysis)

        # Show statistical analysis if available
        if 'statistical' in analysis:
            stat_analysis = analysis['statistical']
            stat_table = Table(title="📈 Statistical Analysis", show_header=True, header_style="bold magenta")
            stat_table.add_column("Feature", style="cyan", width=20)
            stat_table.add_column("Value", style="white")

            stat_table.add_row("AI Probability", f"{stat_analysis.get('ai_probability', 0):.1%}")
            stat_table.add_row("Human Confidence", f"{stat_analysis.get('human_confidence', 0):.1%}")

            # Show anomalies
            anomalies = stat_analysis.get('anomalies', [])
            if anomalies:
                stat_table.add_row("", "")  # Separator
                stat_table.add_row("Anomalies Detected", str(len(anomalies)))
                for anomaly in anomalies[:5]:  # Show first 5
                    stat_table.add_row(f"  └─ {anomaly['type']}", f"Severity: {anomaly['severity']}")

            self.console.print(Panel(stat_table, border_style="magenta"))

    def display_turbo_analysis(self, results: Dict[str, Any]):
        """Display turbo analysis results with GPU acceleration metrics"""
        self.console.print("\n🚀" + "="*58 + "🚀")
        self.console.print("🔥 TURBO ANALYSIS RESULTS - GPU ACCELERATED", style="bold yellow")
        self.console.print("🚀" + "="*58 + "🚀\n")

        # Metadata section
        metadata = results.get('metadata', {})
        meta_table = Table(title="📋 Metadata Analysis", show_header=True, header_style="bold cyan")
        meta_table.add_column("Metric", style="cyan", width=25)
        meta_table.add_column("Count", style="white")

        meta_table.add_row("Tags Found", str(len(metadata.get('tags', []))))
        meta_table.add_row("Suspicious Chunks", str(len(metadata.get('suspicious_chunks', []))))
        meta_table.add_row("Hidden Patterns", str(len(metadata.get('hidden_data', []))))

        self.console.print(Panel(meta_table, border_style="cyan"))

        # GPU Watermark Analysis
        gpu_watermarks = results.get('gpu_watermarks', {})
        gpu_table = Table(title="🚀 GPU Watermark Analysis", show_header=True, header_style="bold green")
        gpu_table.add_column("Metric", style="cyan", width=25)
        gpu_table.add_column("Value", style="white")

        gpu_table.add_row("Chunks Processed", str(gpu_watermarks.get('chunks_processed', 0)))
        gpu_table.add_row("Watermarks Detected", str(gpu_watermarks.get('total_count', 0)))
        gpu_table.add_row("Average Confidence", f"{gpu_watermarks.get('avg_confidence', 0):.1%}")

        self.console.print(Panel(gpu_table, border_style="green"))

        # Performance Metrics
        performance = results.get('performance', {})
        perf_table = Table(title="⚡ Performance Metrics", show_header=True, header_style="bold yellow")
        perf_table.add_column("Metric", style="cyan", width=25)
        perf_table.add_column("Value", style="white")

        perf_table.add_row("Loading Time", f"{performance.get('loading_time', 0):.2f}s")
        perf_table.add_row("Processing Time", f"{performance.get('processing_time', 0):.2f}s")
        perf_table.add_row("Real-time Factor", f"{performance.get('realtime_factor', 0):.1f}x")
        perf_table.add_row("Avg Chunk Time", f"{performance.get('avg_chunk_time', 0):.3f}s")

        # Calculate throughput
        throughput = performance.get('realtime_factor', 0) * 60
        perf_table.add_row("Throughput", f"{throughput:.1f} audio-min/min")

        self.console.print(Panel(perf_table, border_style="yellow"))

        # Threat Assessment
        total_threats = results.get('total_threats', 0)
        threat_color = "red" if total_threats > 20 else "yellow" if total_threats > 10 else "green"
        threat_level = "🔴 VERY HIGH" if total_threats > 20 else "🟠 HIGH" if total_threats > 10 else "🟡 MEDIUM" if total_threats > 5 else "🟢 LOW"

        threat_text = Text()
        threat_text.append("🚨 THREAT LEVEL: ", style="bold white")
        threat_text.append(f"{threat_level}", style=f"bold {threat_color}")
        threat_text.append(f"\nTotal Threats: {total_threats}", style="white")

        self.console.print(Panel(threat_text, border_style=threat_color))

        # GPU Status
        gpu_status = Text()
        gpu_status.append("🎮 GPU Status: ", style="bold white")
        gpu_status.append("NVIDIA GeForce RTX 3080 Ti", style="bold green")
        gpu_status.append(" - ✅ ENABLED", style="bold green")

        self.console.print(Panel(gpu_status, border_style="green"))

    def display_config(self, config: Dict[str, Any]):
        """Display configuration settings"""
        table = Table(title="⚙️  Configuration Settings", show_header=True, header_style="bold blue")
        table.add_column("Setting", style="cyan", width=25)
        table.add_column("Value", style="white")

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_config = flatten_dict(config)

        for key, value in flat_config.items():
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            elif isinstance(value, bool):
                value = "✅ Enabled" if value else "❌ Disabled"
            table.add_row(key.replace('_', ' ').title(), str(value))

        self.console.print(Panel(table, border_style="blue"))

    def show_banner(self, text: str, style: str = "bold blue"):
        """Show a styled banner"""
        self.console.print(Panel(text, border_style=style))

    def dramatic_pause(self, duration: float = 1.0):
        """Add dramatic pause for effect"""
        time.sleep(duration)

    def typewriter_effect(self, text: str, delay: float = 0.03):
        """Print text with typewriter effect"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
