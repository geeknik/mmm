"""
Metadata cleaner for removing all traces of file metadata
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from mutagen import File as MutagenFile
from mutagen.id3 import ID3, ID3NoHeaderError, TXXX, APIC, GEOB, PRIV
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from pydub import AudioSegment


class MetadataCleaner:
    """
    Performs nuclear-level metadata removal from audio files
    """

    def __init__(self):
        self.aggressive_tags = [
            'TXXX', 'TPE1', 'TIT2', 'TALB', 'TDRC', 'TRCK', 'TCON',
            'TPOS', 'TPE2', 'TPE3', 'TPE4', 'TCOM', 'TENC', 'TSSE',
            'TSRC', 'TLAN', 'TPUB', 'TOWN', 'COPY', 'COMM', 'APIC',
            'GEOB', 'PRIV', 'UFID', 'MCDI', 'ETCO', 'MLLT', 'SYTC',
            'USLT', 'SYLT', 'RVA2', 'EQU2', 'RVRB', 'PCNT', 'POPM',
            'RBUF', 'AENC', 'GRID', 'LINK', 'POSS', 'OWNE', 'COMR',
            'ENCR', 'BUF', 'CNT', 'POP', 'CRM', 'RVA', 'EQU'
        ]

    def clean_file(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """
        Clean all metadata from audio file

        Args:
            input_file: Input file path
            output_file: Output file path

        Returns:
            Dict containing cleaning results
        """
        result = {
            'success': False,
            'tags_removed': 0,
            'chunks_removed': 0,
            'methods_used': [],
            'errors': []
        }

        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Determine file format
            file_extension = input_file.suffix.lower()

            if file_extension == '.mp3':
                result = self._clean_mp3(input_file, output_file)
            elif file_extension == '.wav':
                result = self._clean_wav(input_file, output_file)
            else:
                # Generic cleaning for other formats
                result = self._clean_generic(input_file, output_file)

        except Exception as e:
            result['errors'].append(f'Cleaning failed: {str(e)}')
            result['success'] = False

        return result

    def _clean_mp3(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Clean metadata from MP3 file"""
        result = {
            'success': False,
            'tags_removed': 0,
            'chunks_removed': 0,
            'methods_used': [],
            'errors': []
        }

        try:
            # Method 1: Remove metadata using mutagen
            try:
                mp3_file = MP3(input_file)

                if mp3_file.tags:
                    tags_count = len(mp3_file.tags)
                    mp3_file.tags.clear()
                    mp3_file.save()
                    result['tags_removed'] += tags_count
                    result['methods_used'].append('mutagen_clear')

                # Remove all ID3 headers completely
                mp3_file.delete()
                result['methods_used'].append('id3_delete')

            except ID3NoHeaderError:
                # No ID3 header - already clean
                pass
            except Exception as e:
                result['errors'].append(f'Mutagen cleaning: {str(e)}')

            # Method 2: Binary cleaning - strip all ID3 headers
            try:
                with open(input_file, 'rb') as infile:
                    data = bytearray(infile.read())

                # Remove ID3v2 header at beginning
                if data.startswith(b'ID3'):
                    # Find end of ID3v2 tag
                    id3_size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
                    id3_total_size = id3_size + 10
                    data = data[id3_total_size:]
                    result['chunks_removed'] += 1

                # Remove ID3v1 tag at end
                if len(data) >= 128 and data[-128:-125] == b'TAG':
                    data = data[:-128]
                    result['chunks_removed'] += 1

                # Remove APE tag if present
                ape_footer = b'APETAGEX'
                if ape_footer in data:
                    ape_start = data.rfind(ape_footer)
                    if ape_start > 0:
                        data = data[:ape_start]
                        result['chunks_removed'] += 1

                # Write cleaned data
                with open(output_file, 'wb') as outfile:
                    outfile.write(data)

                result['methods_used'].append('binary_strip')
                result['success'] = True

            except Exception as e:
                result['errors'].append(f'Binary cleaning: {str(e)}')

            # Method 3: Audio re-encoding (most aggressive)
            if not result['success'] or self._verify_metadata_present(output_file):
                try:
                    # Load with pydub and re-export as clean MP3
                    audio = AudioSegment.from_mp3(input_file)

                    # Export without any metadata
                    audio.export(output_file, format='mp3', parameters=[
                        '-write_id3v1', '0',
                        '-write_id3v2', '0',
                        '-write_xing', '0'
                    ])

                    result['methods_used'].append('reencode_clean')
                    result['success'] = True

                except Exception as e:
                    result['errors'].append(f'Re-encoding: {str(e)}')

        except Exception as e:
            result['errors'].append(f'MP3 cleaning failed: {str(e)}')

        return result

    def _clean_wav(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Clean metadata from WAV file"""
        result = {
            'success': False,
            'tags_removed': 0,
            'chunks_removed': 0,
            'methods_used': [],
            'errors': []
        }

        try:
            # Method 1: Clean with mutagen
            try:
                wav_file = WAVE(input_file)

                if wav_file.tags:
                    tags_count = len(wav_file.tags)
                    wav_file.tags.clear()
                    wav_file.save()
                    result['tags_removed'] += tags_count
                    result['methods_used'].append('mutagen_clear')

            except Exception as e:
                result['errors'].append(f'Mutagen cleaning: {str(e)}')

            # Method 2: RIFF chunk cleaning
            try:
                with open(input_file, 'rb') as infile:
                    data = bytearray(infile.read())

                if len(data) < 12 or not data.startswith(b'RIFF'):
                    raise ValueError('Not a valid RIFF/WAV file')

                # Parse RIFF structure
                chunk_positions = []
                pos = 12  # Skip RIFF header

                while pos < len(data) - 8:
                    chunk_id = data[pos:pos+4]
                    chunk_size = int.from_bytes(data[pos+4:pos+8], 'little')

                    if chunk_id == b'data ':
                        break  # Stop at data chunk

                    chunk_positions.append((pos, chunk_size))
                    pos += 8 + chunk_size + (chunk_size % 2)  # Skip padding

                # Remove all chunks before data chunk (metadata)
                if chunk_positions:
                    # Keep RIFF header and data chunk only
                    riff_header = data[:12]
                    data_start = chunk_positions[-1][0] + chunk_positions[-1][1] + (chunk_positions[-1][1] % 2)

                    # Find data chunk
                    data_chunk_start = data.find(b'data ', data_start)
                    if data_chunk_start > 0:
                        data_chunk_size = int.from_bytes(data[data_chunk_start+4:data_chunk_start+8], 'little')
                        data_chunk_end = data_chunk_start + 8 + data_chunk_size + (data_chunk_size % 2)

                        # Rebuild minimal WAV file
                        clean_data = bytearray()
                        clean_data.extend(riff_header)

                        # Update RIFF size
                        file_size = data_chunk_end - 12
                        clean_data[4:8] = file_size.to_bytes(4, 'little')

                        # Add fmt and data chunks only
                        fmt_start = data.find(b'fmt ', 12)
                        if fmt_start > 0:
                            fmt_size = int.from_bytes(data[fmt_start+4:fmt_start+8], 'little')
                            fmt_end = fmt_start + 8 + fmt_size + (fmt_size % 2)
                            clean_data.extend(data[fmt_start:fmt_end])
                            result['chunks_removed'] = len(chunk_positions) - 1  # Keep fmt chunk

                        clean_data.extend(data[data_chunk_start:data_chunk_end])

                        with open(output_file, 'wb') as outfile:
                            outfile.write(clean_data)

                        result['methods_used'].append('riff_clean')
                        result['success'] = True

            except Exception as e:
                result['errors'].append(f'RIFF cleaning: {str(e)}')

            # Method 3: Audio re-encoding
            if not result['success']:
                try:
                    audio = AudioSegment.from_wav(input_file)
                    audio.export(output_file, format='wav', parameters=[
                        '-write_id3v1', '0',
                        '-write_id3v2', '0'
                    ])

                    result['methods_used'].append('reencode_clean')
                    result['success'] = True

                except Exception as e:
                    result['errors'].append(f'Re-encoding: {str(e)}')

        except Exception as e:
            result['errors'].append(f'WAV cleaning failed: {str(e)}')

        return result

    def _clean_generic(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """Clean metadata from generic audio file"""
        result = {
            'success': False,
            'tags_removed': 0,
            'chunks_removed': 0,
            'methods_used': [],
            'errors': []
        }

        try:
            # Try to load with pydub and re-export
            audio = AudioSegment.from_file(str(input_file))

            # Determine format
            file_format = input_file.suffix.lower().lstrip('.')

            # Export clean
            audio.export(str(output_file), format=file_format)

            result['methods_used'].append('generic_reencode')
            result['success'] = True

        except Exception as e:
            result['errors'].append(f'Generic cleaning failed: {str(e)}')

            # Fallback: simple copy if all else fails
            try:
                shutil.copy2(input_file, output_file)
                result['methods_used'].append('copy_fallback')
                result['success'] = True
            except Exception as e2:
                result['errors'].append(f'Copy fallback failed: {str(e2)}')

        return result

    def _verify_metadata_present(self, file_path: Path) -> bool:
        """Check if file still contains metadata"""
        try:
            audio_file = MutagenFile(file_path)
            if audio_file and hasattr(audio_file, 'tags') and audio_file.tags:
                return len(audio_file.tags) > 0

            # Also check for binary signatures
            with open(file_path, 'rb') as f:
                header = f.read(100)
                return b'ID3' in header or b'TAG' in header[-128:]

        except Exception:
            return False

    def strip_all_binary_metadata(self, file_data: bytes) -> bytes:
        """
        Strip all known metadata from binary audio data

        Args:
            file_data: Raw file data

        Returns:
            Cleaned binary data
        """
        data = bytearray(file_data)

        # Remove ID3v2 header
        if data.startswith(b'ID3'):
            id3_size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
            id3_total_size = id3_size + 10
            if id3_total_size < len(data):
                data = data[id3_total_size:]

        # Remove ID3v1 footer
        if len(data) >= 128 and data[-128:-125] == b'TAG':
            data = data[:-128]

        # Remove APE tags
        ape_patterns = [b'APETAGEX', b'TAG', b'ID3']
        for pattern in ape_patterns:
            while pattern in data:
                pos = data.rfind(pattern)
                if pos > 0:
                    data = data[:pos]
                else:
                    break

        return bytes(data)

    def clean_custom_chunks(self, file_data: bytes, allowed_chunks: List[bytes]) -> bytes:
        """
        Remove all RIFF chunks except those explicitly allowed

        Args:
            file_data: Raw WAV file data
            allowed_chunks: List of 4-byte chunk IDs to keep

        Returns:
            Cleaned binary data
        """
        if not file_data.startswith(b'RIFF'):
            return file_data  # Not a RIFF file

        data = bytearray(file_data)
        result = bytearray()

        # Copy RIFF header
        result.extend(data[:12])
        pos = 12

        # Process chunks
        while pos < len(data) - 8:
            chunk_id = data[pos:pos+4]
            chunk_size = int.from_bytes(data[pos+4:pos+8], 'little')
            chunk_data = data[pos:pos+8+chunk_size]

            if chunk_id in allowed_chunks:
                result.extend(chunk_data)
                if chunk_size % 2 == 1:
                    result.append(0)  # Padding byte

            pos += 8 + chunk_size
            if chunk_size % 2 == 1:
                pos += 1  # Skip padding

        # Update RIFF size
        if len(result) > 4:
            file_size = len(result) - 8
            result[4:8] = file_size.to_bytes(4, 'little')

        return bytes(result)
