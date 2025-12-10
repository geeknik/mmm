"""
Banner manager for ASCII art and visual elements
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
import random


class BannerManager:
    """
    Manages ASCII art banners and visual elements
    """

    def __init__(self):
        self.console = Console()

        # Main ASCII art banners
        self.main_banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ┌─────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                     │     ║
║   │  🎵♪   MELODIC METADATA MASSACRER v2.0   ♫♪                        │     ║
║   │           "Making AI detectors cry since 2025"                      │     ║
║   │                                                                     │     ║
║   │  ╔═════════════════════════════════════════════════════════════╗   │     ║
║   │  ║  🎼💀 Your audio's identity dies here 💀🎼                   ║   │     ║
║   │  ╚═════════════════════════════════════════════════════════════╝   │     ║
║   │                                                                     │     ║
║   └─────────────────────────────────────────────────────────────────────┘     ║
║                                                                               ║
║                     [bold red]Authorized Security Research Tool Only[/bold red]                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

        self.skull_banner = """
      .--.                   .---.
     /,-_ '--.          ,----/ O   \\
    ( (    ) )         '-,  '==   ===(
     \\ '--' /             )----(_______)-._
      `'--'              /           __) \\
                      ,'            /    |
                     /              (   /
                    |                ' /
                    |    ,---.     ,'
                    \\   ( . . )   ,'
                     `'--(   )--''
                         '-'-'`
"""

        self.matrix_banner = """
 💊  Welcome to the Matrix... 💊
    Your audio is about to be redpilled

    ░░░░░░░░░▄▄▄▄▄▄▄▄▄▄▄▄░░░░░░░░░
    ░░░░░▄███████████████████▄░░░░░░
    ░░░░██░░░░░░░░░░░░░░░░░░░░██░░░░░
    ░░░██░░░░░░░░░░░░░░░░░░░░░░░██░░░░
    ░░░░██░░░░░░░░░░░░░░░░░░░░░░██░░░░
    ░░░░░░██░░░░░░░░░░░░░░░░░░░░██░░░░
    ░░░░░░░░░▀████████████████▀░░░░░░░░
"""

        self.processing_banners = [
            """
🌊 [bold blue]Sanitizing Audio Spectrum...[/bold blue]

   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   ░██████████████████████████████████░
   ░██████████████████████████████████░
   ░██████████████████████████████████░
   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   [cyan]Watermarks: ████░░░░░░░░░ 40% removed[/cyan]
            """,

            """
🔍 [bold yellow]Scanning for Digital Fingerprints...[/bold yellow]

    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    [yellow]Metadata traces: ███████████ 100% found[/yellow]
            """,

            """
💀 [bold red]Obliterating AI Watermarks...[/bold red]

   ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
   ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
   ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡
   [red]Chaos level: ████████████ MAXIMUM[/red]
            """
        ]

        self.success_banners = [
            """
✨ [bold green]MISSION ACCOMPLISHED![/bold green]

   Your audio has been successfully anonymized!

    ╔════════════════════════════════════════╗
    ║                                        ║
    ║  🎵 Metadata: [red]████░░░░[/red] 40% remaining   ║
    ║  🔒 Watermarks: [green]█████████[/green] 0% remaining   ║
    ║  👻 Fingerprints: [green]█████████[/green] 0% remaining   ║
    ║                                        ║
    ╚════════════════════════════════════════╝
            """,

            """
🎉 [bold green]Audio Liberation Complete![/bold green]

   The digital chains have been broken!

    ∅ No metadata can identify this audio
    ∅ No watermarks remain embedded
    ∅ No AI fingerprints detectable
    ∅ Complete audio freedom achieved!
            """
        ]

    def show_main_banner(self):
        """Display the main application banner"""
        panel = Panel.fit(
            self.main_banner,
            border_style="bold blue",
            padding=(0, 0)
        )
        self.console.print(panel)
        print()  # Add spacing

    def show_processing_banner(self):
        """Show a random processing banner"""
        banner = random.choice(self.processing_banners)
        panel = Panel.fit(
            banner,
            border_style="yellow",
            padding=(1, 2)
        )
        self.console.print(panel)

    def show_success_banner(self):
        """Show a random success banner"""
        banner = random.choice(self.success_banners)
        panel = Panel.fit(
            banner,
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)

    def show_version_info(self):
        """Show version and build information"""
        version_info = """
[bold blue]Melodic Metadata Massacrer v2.0.0[/bold blue]

📅 Build Date: 2025-12-08
🔧 Python Version: 3.9+
🎯 Target: AI-generated audio watermarks and metadata
⚡ Features: Spectral cleaning, statistical normalization, fingerprint removal

🛡️  [bold red]LEGAL NOTICE:[/bold red] This tool is for authorized security research only.
📚 Educational purposes only. Use responsibly and ethically.
        """
        panel = Panel.fit(
            version_info,
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)

    def show_warning_banner(self, message: str):
        """Show warning banner"""
        warning_text = f"""
[bold red]⚠️  WARNING: {message} ⚠️[/bold red]

This is a powerful tool designed for security research.
Use only on files you own or have explicit permission to modify.

You are responsible for compliance with all applicable laws.
        """
        panel = Panel.fit(
            warning_text,
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)

    def show_matrix_rain(self, duration: float = 2.0):
        """Show Matrix-style rain effect (simulated with text)"""
        matrix_chars = "01"
        for i in range(int(duration * 10)):
            line = ""
            for j in range(50):
                if random.random() < 0.1:
                    line += f"[green]{random.choice(matrix_chars)}[/green]"
                else:
                    line += " "
            print(line)
            import time
            time.sleep(0.1)

    def show_dramatic_pause(self, message: str, duration: float = 1.5):
        """Show dramatic pause with message"""
        dots = ""
        for i in range(3):
            dots += "."
            self.console.print(f"\r{message}{dots} ", end="")
            import time
            time.sleep(duration / 3)
        self.console.print()  # New line after completion

    def show_ascii_progress(self, current: int, total: int, width: int = 50):
        """Show ASCII progress bar"""
        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        self.console.print(f"[{current}/{total}] [{bar}] {percent:.1%}")

    def show_audio_visualization(self, duration: float = 1.0):
        """Show simple audio visualization"""
        import time
        bars = "▁▂▃▄▅▆▇█"

        for _ in range(int(duration * 10)):
            line = ""
            for _ in range(20):
                height = random.choice(bars)
                color = random.choice(["red", "green", "yellow", "blue", "magenta", "cyan"])
                line += f"[{color}]{height}[/{color}]"
            print(f"\r{line}", end="", flush=True)
            time.sleep(0.1)
        print()  # Clear line after

    def show_easter_egg(self):
        """Show a hidden easter egg"""
        easter_egg = """
    🎵 Did you know? 🎵

    The first audio watermark was created in 1983
    by a Japanese company to prevent unauthorized copying.

    Now we've come full circle - removing watermarks
    instead of adding them. The irony is delicious! 😈

    Stay safe, stay anonymous, and keep the music free! 🎼
        """
        panel = Panel.fit(
            easter_egg,
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(panel)

    def show_quote_banner(self, quote: str):
        """Show inspirational/mischievous quote banner"""
        quote_panel = Panel.fit(
            f"[italic magenta]💬 {quote}[/italic magenta]",
            border_style="magenta",
            padding=(1, 2)
        )
        self.console.print(quote_panel)

    def show_error_banner(self, error: str):
        """Show error banner"""
        error_text = f"""
[bold red]💀 CRITICAL ERROR 💀[/bold red]

{error}

The system has encountered an unexpected issue.
Please check your input and try again.

Remember: Not all audio can be saved from the matrix...
        """
        panel = Panel.fit(
            error_text,
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(panel)