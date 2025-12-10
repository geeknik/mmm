"""
File processor for handling batch operations and file management
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from .audio_sanitizer import AudioSanitizer
from ..config.config_manager import ConfigManager
from ..ui.console import ConsoleManager


class FileProcessor:
    """
    Handles batch processing of multiple audio files
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.console = ConsoleManager()

    def process_directory(
        self,
        directory: Path,
        output_dir: Optional[Path] = None,
        extensions: List[str] = None,
        paranoid_mode: bool = False,
        workers: int = 4,
        continue_on_error: bool = None
    ) -> Dict[str, Any]:
        """
        Process all audio files in a directory

        Args:
            directory: Input directory
            output_dir: Output directory (creates subdirectory if None)
            extensions: File extensions to process
            paranoid_mode: Enable paranoid mode
            workers: Number of parallel workers
            continue_on_error: Continue processing even if some files fail

        Returns:
            Dict containing processing results
        """
        if extensions is None:
            extensions = ['mp3', 'wav']

        if continue_on_error is None:
            continue_on_error = self.config.get('batch_processing.continue_on_error', False)

        # Find audio files
        files = self._find_audio_files(directory, extensions)

        if not files:
            self.console.warning("No audio files found in directory")
            return {
                'success': True,
                'files_processed': 0,
                'files_failed': 0,
                'results': []
            }

        # Set up output directory
        if output_dir is None:
            batch_output_dir = directory / 'cleaned'
        else:
            batch_output_dir = output_dir

        batch_output_dir.mkdir(parents=True, exist_ok=True)

        # Process files
        self.console.info(f"Processing {len(files)} files with {workers} workers...")

        results = {
            'success': True,
            'files_processed': 0,
            'files_failed': 0,
            'results': [],
            'processing_time': 0
        }

        start_time = time.time()

        if workers > 1:
            results = self._process_files_parallel(
                files, batch_output_dir, paranoid_mode, workers, continue_on_error
            )
        else:
            results = self._process_files_sequential(
                files, batch_output_dir, paranoid_mode, continue_on_error
            )

        results['processing_time'] = time.time() - start_time

        # Display summary
        self._display_summary(results)

        return results

    def _find_audio_files(self, directory: Path, extensions: List[str]) -> List[Path]:
        """Find all audio files with specified extensions"""
        files = []

        for ext in extensions:
            # Find files with both lowercase and uppercase extensions
            files.extend(directory.glob(f"*.{ext.lower()}"))
            files.extend(directory.glob(f"*.{ext.upper()}"))

        # Remove duplicates and sort
        files = sorted(list(set(files)))
        return files

    def _process_files_parallel(
        self,
        files: List[Path],
        output_dir: Path,
        paranoid_mode: bool,
        workers: int,
        continue_on_error: bool
    ) -> Dict[str, Any]:
        """Process files using multiple workers"""
        results = {
            'success': True,
            'files_processed': 0,
            'files_failed': 0,
            'results': []
        }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path, output_dir, paranoid_mode): file_path
                for file_path in files
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]

                try:
                    result = future.result()

                    if result['success']:
                        results['files_processed'] += 1
                        self.console.success(f"✅ {file_path.name} - Sanitized!")
                    else:
                        results['files_failed'] += 1
                        self.console.error(f"❌ {file_path.name} - Failed: {result.get('error', 'Unknown error')}")

                        if not continue_on_error:
                            results['success'] = False
                            # Cancel remaining tasks
                            for remaining_future in future_to_file:
                                remaining_future.cancel()
                            break

                    results['results'].append({
                        'file': str(file_path),
                        'result': result
                    })

                except Exception as e:
                    results['files_failed'] += 1
                    error_msg = f"❌ {file_path.name} - Exception: {str(e)}"
                    self.console.error(error_msg)

                    results['results'].append({
                        'file': str(file_path),
                        'result': {'success': False, 'error': str(e)}
                    })

                    if not continue_on_error:
                        results['success'] = False
                        break

                # Show progress
                progress = ((i + 1) / len(files)) * 100
                self.console.show_progress_with_eta(i + 1, len(files), time.time())

        return results

    def _process_files_sequential(
        self,
        files: List[Path],
        output_dir: Path,
        paranoid_mode: bool,
        continue_on_error: bool
    ) -> Dict[str, Any]:
        """Process files sequentially"""
        results = {
            'success': True,
            'files_processed': 0,
            'files_failed': 0,
            'results': []
        }

        for i, file_path in enumerate(files):
            try:
                result = self._process_single_file(file_path, output_dir, paranoid_mode)

                if result['success']:
                    results['files_processed'] += 1
                    self.console.success(f"✅ {file_path.name} - Sanitized!")
                else:
                    results['files_failed'] += 1
                    self.console.error(f"❌ {file_path.name} - Failed: {result.get('error', 'Unknown error')}")

                    if not continue_on_error:
                        results['success'] = False
                        break

                results['results'].append({
                    'file': str(file_path),
                    'result': result
                })

            except Exception as e:
                results['files_failed'] += 1
                error_msg = f"❌ {file_path.name} - Exception: {str(e)}"
                self.console.error(error_msg)

                results['results'].append({
                    'file': str(file_path),
                    'result': {'success': False, 'error': str(e)}
                })

                if not continue_on_error:
                    results['success'] = False
                    break

            # Show progress
            self.console.show_progress_with_eta(i + 1, len(files), time.time())

        return results

    def _process_single_file(
        self,
        input_file: Path,
        output_dir: Path,
        paranoid_mode: bool
    ) -> Dict[str, Any]:
        """Process a single audio file"""
        try:
            # Generate output filename
            output_file = self._generate_output_path(input_file, output_dir)

            # Create sanitizer
            sanitizer = AudioSanitizer(
                input_file=input_file,
                output_file=output_file,
                paranoid_mode=paranoid_mode,
                config=self.config
            )

            # Process file
            result = sanitizer.sanitize_audio()

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_output_path(self, input_file: Path, output_dir: Path) -> Path:
        """Generate output file path based on configuration"""
        naming_pattern = self.config.get('batch_processing.naming_pattern', '{name}_clean{ext}')

        # Extract components
        name = input_file.stem
        ext = input_file.suffix
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Apply naming pattern
        output_name = naming_pattern.format(
            name=name,
            ext=ext,
            timestamp=timestamp
        )

        return output_dir / output_name

    def _display_summary(self, results: Dict[str, Any]):
        """Display processing summary"""
        total_files = results['files_processed'] + results['files_failed']
        success_rate = (results['files_processed'] / total_files * 100) if total_files > 0 else 0

        summary = f"""
📊 [bold]Batch Processing Summary[/bold]

Files Processed: {results['files_processed']}
Files Failed: {results['files_failed']}
Success Rate: {success_rate:.1f}%
Processing Time: {results['processing_time']:.2f} seconds
        """

        if success_rate == 100:
            self.console.success("🎉 All files processed successfully!")
        elif success_rate >= 80:
            self.console.warning(f"⚠️  Processing completed with {results['files_failed']} failures")
        else:
            self.console.error(f"💀 Processing failed for {results['files_failed']} files")

    def create_backup_batch(self, directory: Path, backup_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Create backup of all audio files in directory"""
        if backup_dir is None:
            backup_dir = directory.parent / f"{directory.name}_backup_{int(time.time())}"

        backup_dir.mkdir(parents=True, exist_ok=True)

        # Find audio files
        audio_files = self._find_audio_files(directory, ['mp3', 'wav', 'flac', 'ogg'])

        if not audio_files:
            self.console.warning("No audio files found for backup")
            return {'success': True, 'files_backed_up': 0}

        self.console.info(f"Creating backup of {len(audio_files)} files...")

        backed_up = 0
        for file_path in audio_files:
            try:
                # Preserve directory structure
                relative_path = file_path.relative_to(directory)
                backup_path = backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file_path, backup_path)
                backed_up += 1

            except Exception as e:
                self.console.error(f"Failed to backup {file_path.name}: {str(e)}")

        self.console.success(f"✅ Backup created: {backup_dir}")
        self.console.info(f"Backed up {backed_up} files")

        return {
            'success': True,
            'backup_dir': str(backup_dir),
            'files_backed_up': backed_up
        }

    def verify_batch_integrity(self, original_dir: Path, cleaned_dir: Path) -> Dict[str, Any]:
        """Verify integrity of batch processing"""
        # Get all cleaned files
        cleaned_files = self._find_audio_files(cleaned_dir, ['mp3', 'wav'])

        verification_results = {
            'success': True,
            'files_verified': 0,
            'files_failed': 0,
            'details': []
        }

        for cleaned_file in cleaned_files:
            try:
                # Find corresponding original file
                original_name = cleaned_file.name.replace('_clean', '').replace('_sanitized', '')
                original_file = original_dir / original_name

                if not original_file.exists():
                    # Try to find file with different naming pattern
                    original_stem = cleaned_file.stem.replace('_clean', '').replace('_sanitized', '')
                    for ext in ['.mp3', '.wav']:
                        potential_original = original_dir / (original_stem + ext)
                        if potential_original.exists():
                            original_file = potential_original
                            break

                if original_file.exists():
                    # Verify sanitization
                    sanitizer = AudioSanitizer(original_file, cleaned_file)
                    verification = sanitizer.verify_sanitization()

                    if verification.get('success', False):
                        verification_results['files_verified'] += 1
                    else:
                        verification_results['files_failed'] += 1
                        verification_results['success'] = False

                    verification_results['details'].append({
                        'file': str(cleaned_file),
                        'verification': verification
                    })
                else:
                    verification_results['files_failed'] += 1
                    verification_results['success'] = False

            except Exception as e:
                verification_results['files_failed'] += 1
                verification_results['success'] = False
                verification_results['details'].append({
                    'file': str(cleaned_file),
                    'error': str(e)
                })

        return verification_results