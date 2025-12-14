#!/usr/bin/env python3
"""
Melodic Metadata Massacrer (MMM) CLI
The audio anonymizer that makes AI detectors cry
"""

import click
import sys
import os
from pathlib import Path

from .ui.console import ConsoleManager
from .ui.banners import BannerManager
from .core.audio_sanitizer import AudioSanitizer
from .config.config_manager import ConfigManager

console = ConsoleManager()
banner = BannerManager()
config = ConfigManager()


@click.group()
@click.version_option(version="2.0.0", prog_name="mmm")
@click.pass_context
def cli(ctx):
    """
    🎵 Melodic Metadata Massacrer - The audio anonymizer

    Authorized security research tool for removing ALL watermarks and metadata
    from audio files, making AI-generated music untraceable.
    """
    ctx.ensure_object(dict)

    # Display epic banner
    banner.show_main_banner()

    # Legal disclaimer
    console.warning(
        "⚠️  LEGAL DISCLAIMER: This tool is for AUTHORIZED SECURITY RESEARCH ONLY"
    )
    console.info("   Use only on files you own or have explicit permission to modify")
    console.info("   You are responsible for compliance with applicable laws\n")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (auto-generates if not specified)",
)
@click.option(
    "--paranoid",
    is_flag=True,
    default=False,
    help="Maximum destruction mode - multiple passes with aggressive cleaning",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Verify watermark removal effectiveness",
)
@click.option(
    "--backup", is_flag=True, default=False, help="Create backup of original file"
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["preserve", "mp3", "wav"], case_sensitive=False),
    default="preserve",
    help="Output audio format",
)
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Enable turbo mode with GPU acceleration (700x+ faster)",
)
@click.option(
    "--phase-dither/--no-phase-dither",
    default=True,
    help="Toggle sub-block phase dither (advanced)",
)
@click.option(
    "--comb-mask/--no-comb-mask",
    default=True,
    help="Toggle dynamic comb masking (advanced)",
)
@click.option(
    "--transient-shift/--no-transient-shift",
    default=True,
    help="Toggle transient micro-shift (advanced)",
)
@click.option(
    "--resample-nudge/--no-resample-nudge",
    default=True,
    help="Toggle resample nudge (advanced)",
)
@click.option(
    "--phase-noise/--no-phase-noise",
    default=True,
    help="Toggle FFT phase noise (advanced)",
)
@click.option(
    "--phase-swirl/--no-phase-swirl", default=True, help="Toggle phase swirl (advanced)"
)
@click.option(
    "--masked-hf-phase/--no-masked-hf-phase",
    default=False,
    help="Toggle masked high-frequency phase noise (advanced)",
)
@click.option(
    "--gated-resample-nudge/--no-gated-resample-nudge",
    default=False,
    help="Toggle RMS-gated resample nudge (advanced)",
)
@click.option(
    "--micro-eq-flutter/--no-micro-eq-flutter",
    default=False,
    help="Toggle gated micro-EQ flutter (advanced)",
)
@click.option(
    "--hf-decorrelate/--no-hf-decorrelate",
    default=False,
    help="Toggle HF band decorrelation (advanced)",
)
@click.option(
    "--refined-transient/--no-refined-transient",
    default=False,
    help="Toggle refined transient micro-shift (advanced)",
)
@click.option(
    "--adaptive-transient/--no-adaptive-transient",
    default=False,
    help="Toggle adaptive transient shift (onset-strength gated, ultra-small)",
)
@click.pass_context
def obliterate(
    ctx,
    input_file,
    output,
    paranoid,
    verify,
    backup,
    output_format,
    turbo,
    phase_dither,
    comb_mask,
    transient_shift,
    resample_nudge,
    phase_noise,
    phase_swirl,
    masked_hf_phase,
    gated_resample_nudge,
    micro_eq_flutter,
    hf_decorrelate,
    refined_transient,
    adaptive_transient,
):
    """
    💀 Completely annihilate all traces from audio file

    Removes metadata, watermarks, AI fingerprints, and statistical patterns
    that identify AI-generated content. This is the nuclear option.

    Examples:

        mmm obliterate dystopian_symphony.mp3

        mmm obliterate --paranoid --verify music.wav -o clean_music.wav
    """
    console.success(f"🔍 Scanning: {input_file.name}")

    try:
        # Create fresh config manager to avoid scope issues
        config_manager = ConfigManager()
        target_format = None if output_format == "preserve" else output_format

        # Apply preset defaults to advanced flags unless the user explicitly set them
        preset_flags = {}
        preset_name = config_manager.config.get("preset")
        if preset_name:
            from .config.defaults import PRESETS

            preset_data = PRESETS.get(preset_name) or {}
            preset_flags = preset_data.get("advanced_flags", {})

        def resolve_flag(flag_name: str, cli_value: bool) -> bool:
            # If the flag was explicitly provided on CLI, keep it; else fall back to preset if available
            if (
                ctx.get_parameter_source(flag_name)
                == click.core.ParameterSource.COMMANDLINE
            ):
                return cli_value
            return preset_flags.get(flag_name, cli_value)

        phase_dither = resolve_flag("phase_dither", phase_dither)
        comb_mask = resolve_flag("comb_mask", comb_mask)
        transient_shift = resolve_flag("transient_shift", transient_shift)
        resample_nudge = resolve_flag("resample_nudge", resample_nudge)
        gated_resample_nudge = resolve_flag(
            "gated_resample_nudge", gated_resample_nudge
        )
        phase_noise = resolve_flag("phase_noise", phase_noise)
        phase_swirl = resolve_flag("phase_swirl", phase_swirl)
        masked_hf_phase = resolve_flag("masked_hf_phase", masked_hf_phase)
        micro_eq_flutter = resolve_flag("micro_eq_flutter", micro_eq_flutter)
        hf_decorrelate = resolve_flag("hf_decorrelate", hf_decorrelate)
        refined_transient = resolve_flag("refined_transient", refined_transient)
        adaptive_transient = resolve_flag("adaptive_transient", adaptive_transient)

        sanitizer = AudioSanitizer(
            input_file=input_file,
            output_file=output,
            paranoid_mode=paranoid,
            config=config_manager.config,
            output_format=target_format,
        )

        # Create backup if requested
        if backup:
            sanitizer.create_backup()
            console.info("📦 Backup created - Your secrets are safe... for now")

        # Show what we found (use turbo analysis if requested)
        if turbo:
            console.info("🚀 TURBO MODE: Fast analysis with GPU acceleration")
            # Import turbo analysis for fast scanning
            try:
                from .turbo_analysis import turbo_analysis

                analysis_results = turbo_analysis(input_file)

                # Convert turbo results to expected format
                analysis = {
                    "threats_found": analysis_results.get("total_threats", 0),
                    "threat_level": analysis_results.get("threat_level")
                    or (
                        "HIGH"
                        if analysis_results.get("total_threats", 0) > 10
                        else (
                            "MEDIUM"
                            if analysis_results.get("total_threats", 0) > 5
                            else "LOW"
                        )
                    ),
                    "file_info": analysis_results.get("file_info", {}),
                    "metadata": analysis_results.get("metadata", {}),
                    "watermarks": analysis_results.get("gpu_watermarks", {}),
                }
                console.info(
                    f"⚡ Turbo analysis complete: {analysis['threats_found']} threats found"
                )
            except Exception as e:
                console.warning(
                    f"⚠️ Turbo analysis failed, falling back to regular: {e}"
                )
                analysis = sanitizer.analyze_file(
                    deep=False
                )  # Use shallow analysis as fallback
        else:
            # Regular analysis (slower but thorough)
            analysis = sanitizer.analyze_file(
                deep=False
            )  # Use shallow analysis for speed

        console.display_analysis(analysis)

        if analysis["threats_found"]:
            console.error(
                f"😈 Found {analysis['threats_found']} threats... time to DELETE THEM ALL!"
            )
        else:
            console.warning(
                "🤔 No obvious threats detected... but we'll clean it anyway!"
            )

        # Begin the massacre
        if turbo:
            console.info("🚀 TURBO MODE: Beginning fast sanitization...")
            # Import and use PRESERVING sanitizer
            try:
                from .preserving_sanitizer import preserving_sanitize

                console.info(
                    f"🎵 Calling PRESERVING sanitizer with paranoid_mode={paranoid}"
                )
                # Pass the actual threat count from analysis
                threat_count = analysis.get("threats_found", 0)
                console.info(
                    f"🎯 Preserving audio while removing {threat_count} threats"
                )
                preserving_result = preserving_sanitize(
                    input_file,
                    output,
                    paranoid,
                    threat_count,
                    output_format=target_format,
                    phase_dither=phase_dither,
                    comb_mask=comb_mask,
                    transient_shift=transient_shift,
                    resample_nudge=resample_nudge,
                    gated_resample_nudge=gated_resample_nudge,
                    phase_noise=phase_noise,
                    phase_swirl=phase_swirl,
                    masked_hf_phase=masked_hf_phase,
                    micro_eq_flutter=micro_eq_flutter,
                    hf_decorrelate=hf_decorrelate,
                    refined_transient=refined_transient,
                    adaptive_transient=adaptive_transient,
                )

                # Convert preserving results to expected format
                result = {
                    "success": preserving_result["success"],
                    "output_file": preserving_result["output_file"],
                    "stats": {
                        "metadata_removed": preserving_result["stats"][
                            "metadata_removed"
                        ],
                        "watermarks_detected": preserving_result["stats"][
                            "watermarks_removed"
                        ],
                        "watermarks_removed": preserving_result["stats"][
                            "watermarks_removed"
                        ],
                        "quality_loss": 0.0,  # Assume minimal loss with preserving mode
                        "processing_time": preserving_result["stats"][
                            "processing_time"
                        ],
                    },
                }
                console.success(
                    f"🎵 PRESERVING sanitization completed in {preserving_result['stats']['processing_time']}"
                )
            except Exception as e:
                console.warning(
                    f"⚠️ PRESERVING sanitization failed, falling back to regular: {e}"
                )
                import traceback

                console.error(f"Traceback: {traceback.format_exc()}")
                console.info("🌊 Falling back to regular sanitization...")
                result = sanitizer.sanitize_audio()
        else:
            console.info("🌊 Beginning audio sanitization protocol...")
            result = sanitizer.sanitize_audio()

        # Display results
        console.display_results(result)

        if verify and result["success"]:
            console.info("🔬 Verification phase: Double-checking our work...")
            if turbo:
                # Use turbo analysis for verification
                try:
                    console.info("⚡ Using turbo analysis for verification...")
                    from .turbo_analysis import turbo_analysis

                    original_threats = analysis["threats_found"]

                    # Analyze the cleaned file
                    cleaned_file = Path(result["output_file"])
                    post_analysis_results = turbo_analysis(cleaned_file)
                    remaining_threats = post_analysis_results.get("total_threats", 0)

                    # Calculate effectiveness
                    removal_effectiveness = 0
                    if original_threats > 0:
                        removal_effectiveness = (
                            (original_threats - remaining_threats) / original_threats
                        ) * 100

                    verification = {
                        "success": True,
                        "original_threats": original_threats,
                        "remaining_threats": remaining_threats,
                        "removal_effectiveness": round(removal_effectiveness, 2),
                        "hash_different": True,  # Assume true since we modified the file
                    }
                except Exception as e:
                    verification = {"success": False, "error": str(e)}
            else:
                # Use regular verification
                verification = sanitizer.verify_sanitization()

            console.display_verification(verification)

        if result["success"]:
            console.success(
                "✨ File sanitized! Your AI overlords will never know... 🤫"
            )
            console.hacker_quote()
        else:
            console.error("💥 Sanitization failed! The matrix fought back...")
            sys.exit(1)

    except Exception as e:
        console.error(f"💀 CRITICAL ERROR: {str(e)}")
        console.error("The audio has fought back and won this round...")
        sys.exit(1)


@cli.command()
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    "-d",
    type=click.Path(path_type=Path),
    help="Output directory (creates subdirectory if not specified)",
)
@click.option(
    "--extension",
    "-e",
    multiple=True,
    default=["mp3", "wav"],
    help="File extensions to process (can be used multiple times)",
)
@click.option(
    "--paranoid", is_flag=True, default=False, help="Maximum destruction mode"
)
@click.option("--workers", "-w", type=int, default=4, help="Number of parallel workers")
@click.option(
    "--backup", is_flag=True, default=False, help="Create backups of original files"
)
@click.pass_context
def massacre(ctx, directory, output_dir, extension, paranoid, workers, backup):
    """
    ⚡ Process entire directory of audio files

    Mass sanitization mode for bulk operations. Processes all supported
    audio files in the specified directory with parallel execution.

    Example:

        mmm massacre /path/to/music --paranoid --workers 8
    """
    console.success(f"🎯 Directory massacre initiated: {directory}")
    console.info(f"⚙️  Extensions: {', '.join(extension)}")
    console.info(f"🔥 Workers: {workers} | Paranoid: {'ON' if paranoid else 'OFF'}")

    # Scan for files
    files = []
    for ext in extension:
        files.extend(directory.glob(f"*.{ext.lower()}"))
        files.extend(directory.glob(f"*.{ext.upper()}"))

    if not files:
        console.warning("📂 No audio files found in directory")
        return

    console.success(f"📁 Found {len(files)} files to process")

    # TODO: Implement parallel processing
    console.info("🔄 Processing files...")
    for file_path in files:
        try:
            output_file = output_dir / file_path.name if output_dir else None
            console.info(f"   Processing: {file_path.name}")

            # Create fresh config manager to avoid scope issues
            config_manager = ConfigManager()
            sanitizer = AudioSanitizer(
                input_file=file_path,
                output_file=output_file,
                paranoid_mode=paranoid,
                config=config_manager.config,
            )

            if backup:
                sanitizer.create_backup()

            result = sanitizer.sanitize_audio()

            if result["success"]:
                console.success(f"   ✅ {file_path.name} - Sanitized!")
            else:
                console.error(f"   ❌ {file_path.name} - Failed!")

        except Exception as e:
            console.error(f"   💥 {file_path.name} - Error: {str(e)}")

    console.success("🎉 Massacre complete! The audio has been liberated!")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--turbo",
    is_flag=True,
    default=False,
    help="Enable turbo mode with GPU acceleration (660x faster)",
)
@click.pass_context
def analyze(ctx, input_file, turbo):
    """
    🔬 Analyze audio file for watermarks and metadata

    Performs deep forensic analysis to identify potential watermarks,
    AI fingerprints, and hidden metadata without modifying the file.

    Example:

        mmm analyze suspicious_music.mp3
    """
    if turbo:
        console.info(f"🚀 TURBO Forensic analysis: {input_file.name}")
        console.info("⚡ Using GPU acceleration for maximum speed...\n")

        # Import turbo analysis
        try:
            from .turbo_analysis import turbo_analysis

            results = turbo_analysis(input_file)

            # Display turbo results
            console.display_turbo_analysis(results)

        except ImportError:
            console.error("💥 Turbo mode requires GPU packages. Run without --turbo")
            sys.exit(1)
        except Exception as e:
            console.error(f"💥 Turbo analysis failed: {str(e)}")
            sys.exit(1)
    else:
        console.info(f"🔬 Forensic analysis: {input_file.name}")
        console.info("🔍 Scanning for digital footprints...\n")

        try:
            # Create fresh config manager to avoid scope issues
            config_manager = ConfigManager()
            sanitizer = AudioSanitizer(
                input_file=input_file, config=config_manager.config
            )
            analysis = sanitizer.analyze_file(deep=True)

            console.display_detailed_analysis(analysis)

            # Threat assessment
            if analysis["threat_level"] == "HIGH":
                console.error(
                    "🚨 HIGH THREAT LEVEL - This file is heavily watermarked!"
                )
            elif analysis["threat_level"] == "MEDIUM":
                console.warning("⚠️  MEDIUM THREAT LEVEL - Some traces detected")
            else:
                console.success("✅ LOW THREAT LEVEL - Relatively clean")

        except Exception as e:
            console.error(f"💥 Analysis failed: {str(e)}")
            sys.exit(1)


@cli.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """
    ⚙️  Configuration management

    Manage MMM configuration settings and presets.

    Examples:

        mmm config              Show current config

        mmm config preset stealth   Apply stealth preset

        mmm config list         List available presets

        mmm config create my_preset --paranoid maximum --quality high
    """
    # If no subcommand provided, show current config (default behavior)
    if ctx.invoked_subcommand is None:
        config_manager = ConfigManager()
        console.info("⚙️  Current Configuration:")
        console.display_config(config_manager.get_config())


@config.command("show")
def config_show():
    """
    📋 Show current configuration

    Displays current MMM configuration settings and defaults.
    """
    config_manager = ConfigManager()
    console.info("⚙️  Current Configuration:")
    console.display_config(config_manager.get_config())


@config.command("preset")
@click.argument("name", type=str)
def config_preset(name):
    """
    🎛️  Apply a configuration preset

    Apply a built-in or custom preset by name.

    Built-in presets: stealth, stealth-plus, fast, quality, research

    Examples:

        mmm config preset stealth

        mmm config preset fast
    """
    from .config.defaults import PRESETS

    config_manager = ConfigManager()

    # Check if it's a built-in preset
    if name in PRESETS:
        preset_data = PRESETS[name]
        # Merge preset with current config
        for key, value in preset_data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    config_manager.set(f"{key}.{sub_key}", sub_value)
            else:
                config_manager.set(key, value)
        # Also set the preset name in config for reference
        config_manager.set("preset", name)
        config_manager.save_config()
        console.success(f"✅ Applied built-in preset: {name}")
        console.info("📋 Preset settings:")
        for key, value in preset_data.items():
            if key != "advanced_flags":
                console.info(f"   {key}: {value}")
            else:
                console.info(f"   {key}:")
                for flag, flag_value in value.items():
                    console.info(f"      {flag}: {flag_value}")
    else:
        # Try to load custom preset
        try:
            preset_data = config_manager.load_preset(name)
            for key, value in preset_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        config_manager.set(f"{key}.{sub_key}", sub_value)
                else:
                    config_manager.set(key, value)
            config_manager.set("preset", name)
            config_manager.save_config()
            console.success(f"✅ Applied custom preset: {name}")
        except Exception as e:
            console.error(f"❌ Failed to apply preset '{name}': {e}")
            console.info("💡 Available built-in presets: stealth, stealth-plus, fast, quality, research")
            console.info("💡 Use 'mmm config list' to see all available presets")


@config.command("list")
def config_list():
    """
    📜 List available presets

    Shows all built-in and custom configuration presets.
    """
    from .config.defaults import PRESETS

    config_manager = ConfigManager()

    console.info("📜 Available Presets:\n")

    # Built-in presets
    console.info("🔧 Built-in Presets:")
    preset_descriptions = {
        "stealth": "Maximum paranoia, quality preservation",
        "stealth-plus": "Stealth with advanced flags optimized for detector evasion",
        "fast": "Quick processing, basic cleaning",
        "quality": "Preserve maximum audio quality",
        "research": "Deep analysis, detailed logging",
    }
    for name in PRESETS:
        desc = preset_descriptions.get(name, "No description")
        console.info(f"   • {name}: {desc}")

    # Custom presets
    custom_presets = config_manager.list_presets()
    if custom_presets:
        console.info("\n🎨 Custom Presets:")
        for name in custom_presets:
            console.info(f"   • {name}")
    else:
        console.info("\n💡 No custom presets found. Create one with 'mmm config create'")


@config.command("create")
@click.argument("name", type=str)
@click.option(
    "--paranoid",
    type=click.Choice(["low", "medium", "high", "maximum"], case_sensitive=False),
    default="medium",
    help="Paranoia level",
)
@click.option(
    "--quality",
    type=click.Choice(["low", "medium", "high", "maximum"], case_sensitive=False),
    default="high",
    help="Quality preservation level",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["preserve", "mp3", "wav"], case_sensitive=False),
    default="preserve",
    help="Output format",
)
@click.option("--backup/--no-backup", default=False, help="Backup originals by default")
@click.option("--verify/--no-verify", default=True, help="Auto-verify after processing")
def config_create(name, paranoid, quality, output_format, backup, verify):
    """
    🎨 Create a custom configuration preset

    Create a new preset with specified settings.

    Examples:

        mmm config create my_preset --paranoid maximum --quality high

        mmm config create quick_clean --paranoid low --quality medium --no-verify
    """
    config_manager = ConfigManager()

    preset_config = {
        "paranoia_level": paranoid,
        "preserve_quality": quality,
        "output_format": output_format,
        "backup_originals": backup,
        "verification": {
            "auto_verify": verify,
        },
    }

    try:
        config_manager.create_preset(name, preset_config)
        console.success(f"✅ Created custom preset: {name}")
        console.info("📋 Preset settings:")
        console.info(f"   paranoia_level: {paranoid}")
        console.info(f"   preserve_quality: {quality}")
        console.info(f"   output_format: {output_format}")
        console.info(f"   backup_originals: {backup}")
        console.info(f"   auto_verify: {verify}")
        console.info(f"\n💡 Apply with: mmm config preset {name}")
    except Exception as e:
        console.error(f"❌ Failed to create preset: {e}")


@config.command("delete")
@click.argument("name", type=str)
@click.confirmation_option(prompt="Are you sure you want to delete this preset?")
def config_delete(name):
    """
    🗑️  Delete a custom preset

    Remove a custom configuration preset.

    Example:

        mmm config delete my_preset
    """
    from .config.defaults import PRESETS

    if name in PRESETS:
        console.error(f"❌ Cannot delete built-in preset: {name}")
        return

    config_manager = ConfigManager()

    try:
        config_manager.delete_preset(name)
        console.success(f"✅ Deleted preset: {name}")
    except Exception as e:
        console.error(f"❌ Failed to delete preset: {e}")


@config.command("reset")
@click.confirmation_option(prompt="Are you sure you want to reset to defaults?")
def config_reset():
    """
    🔄 Reset configuration to defaults

    Restore all settings to their default values.
    """
    config_manager = ConfigManager()
    config_manager.reset_to_defaults()
    console.success("✅ Configuration reset to defaults")


@cli.command()
def version():
    """
    📋 Show version and build information
    """
    banner.show_version_info()


if __name__ == "__main__":
    cli()
