"""
Metadata scanner for deep forensic analysis of audio file metadata
"""

import struct
import binascii
from pathlib import Path
from typing import Dict, List, Any, Optional
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import MP3


class MetadataScanner:
    """
    Performs deep forensic analysis of audio file metadata
    """

    def __init__(self):
        self.suspicious_tags = [
            'TXXX', 'TPE1', 'TIT2', 'TRCK', 'TCON',
            'COMENT', 'APIC', 'GEOB', 'PRIV'
        ]

        self.suspicious_patterns = [
            b'ai', b'generated', b'synthetic', b'watermark',
            b'fingerprint', b'identifier', b'trace',
            b'sun', b'microsoft', b'google', b'openai'
        ]

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Perform comprehensive metadata scan

        Args:
            file_path: Path to audio file

        Returns:
            Dict containing metadata analysis
        """
        results = {
            'tags': [],
            'suspicious_chunks': [],
            'hidden_data': [],
            'anomalies': [],
            'format_info': {}
        }

        # Get format information
        results['format_info'] = self._get_format_info(file_path)

        # Standard metadata scan
        standard_metadata = self._scan_standard_metadata(file_path)
        results['tags'] = standard_metadata['tags']
        results['anomalies'].extend(standard_metadata['anomalies'])

        # Deep binary scan
        deep_scan = self._deep_binary_scan(file_path)
        results['suspicious_chunks'] = deep_scan['suspicious_chunks']
        results['hidden_data'] = deep_scan['hidden_data']
        results['anomalies'].extend(deep_scan['anomalies'])

        # Format-specific scans
        if file_path.suffix.lower() == '.mp3':
            mp3_scan = self._scan_mp3_details(file_path)
            results.update(mp3_scan)
        elif file_path.suffix.lower() == '.wav':
            wav_scan = self._scan_wav_details(file_path)
            results.update(wav_scan)

        return results

    def _get_format_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic format information"""
        try:
            audio_file = MutagenFile(file_path)
            if audio_file is None:
                return {'error': 'Unable to read file'}

            info = audio_file.info if hasattr(audio_file, 'info') else None

            format_info = {
                'format': str(file_path.suffix).upper(),
                'file_size': file_path.stat().st_size,
                'readable': True
            }

            if info:
                format_info.update({
                    'length': getattr(info, 'length', 0),
                    'bitrate': getattr(info, 'bitrate', 0),
                    'sample_rate': getattr(info, 'sample_rate', 0),
                    'channels': getattr(info, 'channels', 0),
                    'mode': getattr(info, 'mode', 'unknown'),
                    'layer': getattr(info, 'layer', None),
                    'version': getattr(info, 'version', None)
                })

            return format_info

        except Exception as e:
            return {
                'error': f'Failed to read format info: {str(e)}',
                'readable': False
            }

    def _scan_standard_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Scan standard metadata tags"""
        results = {
            'tags': [],
            'anomalies': []
        }

        try:
            audio_file = MutagenFile(file_path)

            if audio_file is None:
                results['anomalies'].append('No readable metadata found')
                return results

            if hasattr(audio_file, 'tags') and audio_file.tags:
                for key, value in audio_file.tags.items():
                    tag_info = {
                        'key': key,
                        'value': str(value)[:100],  # Limit length
                        'raw_value': str(value),
                        'suspicious': self._is_tag_suspicious(key, str(value))
                    }

                    results['tags'].append(tag_info)

                    if tag_info['suspicious']:
                        results['anomalies'].append(f'Suspicious tag: {key}')

            else:
                results['anomalies'].append('No metadata tags found')

        except ID3NoHeaderError:
            results['anomalies'].append('No ID3 header found (possibly clean)')
        except Exception as e:
            results['anomalies'].append(f'Metadata scan error: {str(e)}')

        return results

    def _deep_binary_scan(self, file_path: Path) -> Dict[str, Any]:
        """Perform deep binary scan for hidden data"""
        results = {
            'suspicious_chunks': [],
            'hidden_data': [],
            'anomalies': []
        }

        try:
            with open(file_path, 'rb') as f:
                data = f.read()

            # Scan for suspicious patterns
            for pattern in self.suspicious_patterns:
                occurrences = self._find_pattern(data, pattern)
                if occurrences:
                    for occ in occurrences[:5]:  # Limit to first 5 occurrences
                        results['hidden_data'].append({
                            'pattern': pattern.decode('utf-8', errors='ignore'),
                            'offset': occ['offset'],
                            'context': occ['context']
                        })

            # Look for suspicious chunks/headers
            chunk_signatures = {
                b'RIFF': 'RIFF container',
                b'ID3': 'ID3 metadata',
                b'LIST': 'LIST chunk',
                b'APETAGEX': 'APE tag',
                b'TAG': 'ID3v1 tag',
                b'\xff\xfb': 'MP3 frame sync',
                b'\xff\xf3': 'MP3 frame sync',
                b'\xff\xf2': 'MP3 frame sync'
            }

            for sig, description in chunk_signatures.items():
                occurrences = self._find_pattern(data, sig)
                for occ in occurrences:
                    results['suspicious_chunks'].append({
                        'signature': sig.decode('utf-8', errors='ignore'),
                        'description': description,
                        'offset': occ['offset'],
                        'size': self._get_chunk_size(data, occ['offset'], sig)
                    })

            # Check for unusual patterns
            if self._has_repeating_patterns(data):
                results['anomalies'].append('Suspicious repeating patterns detected')

            if self._has_unusual_entropy(data):
                results['anomalies'].append('Unusual entropy patterns detected')

        except Exception as e:
            results['anomalies'].append(f'Binary scan error: {str(e)}')

        return results

    def _is_tag_suspicious(self, key: str, value: str) -> bool:
        """Check if a metadata tag is suspicious"""
        suspicious_indicators = [
            'ai', 'generated', 'synthetic', 'watermark',
            'fingerprint', 'identifier', 'trace', 'uuid',
            'hash', 'signature', 'auth', 'verify'
        ]

        key_lower = key.lower()
        value_lower = value.lower()

        for indicator in suspicious_indicators:
            if indicator in key_lower or indicator in value_lower:
                return True

        # Check for very long tag values (might contain hidden data)
        if len(value) > 1000:
            return True

        # Check for hexadecimal strings (might be hashes/IDs)
        if self._looks_like_hex(value):
            return True

        return False

    def _find_pattern(self, data: bytes, pattern: bytes) -> List[Dict[str, Any]]:
        """Find all occurrences of a pattern in binary data"""
        occurrences = []
        offset = 0

        while True:
            pos = data.find(pattern, offset)
            if pos == -1:
                break

            # Get context around the pattern
            start = max(0, pos - 20)
            end = min(len(data), pos + len(pattern) + 20)
            context = data[start:end]

            occurrences.append({
                'offset': pos,
                'context': binascii.hexlify(context).decode('ascii')
            })

            offset = pos + len(pattern)

            # Limit to avoid infinite loops
            if len(occurrences) > 100:
                break

        return occurrences

    def _get_chunk_size(self, data: bytes, offset: int, signature: bytes) -> int:
        """Try to determine the size of a chunk"""
        # This is a simplified implementation
        # Real implementation would need format-specific logic

        if signature == b'RIFF' and offset + 8 < len(data):
            # RIFF chunks have 4-byte size after signature
            size = struct.unpack('<I', data[offset + 4:offset + 8])[0]
            return size + 8  # Include header

        return len(signature)

    def _has_repeating_patterns(self, data: bytes) -> bool:
        """Check for suspicious repeating patterns"""
        # Look for repeated byte sequences
        sequence_lengths = [4, 8, 16, 32]

        for length in sequence_lengths:
            seen_sequences = {}
            for i in range(len(data) - length):
                seq = data[i:i + length]
                if seq in seen_sequences:
                    seen_sequences[seq] += 1
                    if seen_sequences[seq] > 10:  # Threshold for suspicion
                        return True
                else:
                    seen_sequences[seq] = 1

        return False

    def _has_unusual_entropy(self, data: bytes) -> bool:
        """Check for unusual entropy patterns"""
        # Calculate entropy in chunks
        chunk_size = 1024
        entropies = []

        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            if len(chunk) > 0:
                # Calculate byte frequency
                byte_counts = [0] * 256
                for byte in chunk:
                    byte_counts[byte] += 1

                # Calculate entropy
                total = sum(byte_counts)
                if total > 0:
                    probs = [count / total for count in byte_counts if count > 0]
                    from math import log2
                    entropy = -sum(p * log2(p) for p in probs)
                    entropies.append(entropy)

        if entropies:
            avg_entropy = sum(entropies) / len(entropies)
            # Very low or very high entropy can be suspicious
            return avg_entropy < 3.0 or avg_entropy > 7.8

        return False

    def _looks_like_hex(self, string: str) -> bool:
        """Check if a string looks like hexadecimal"""
        # Remove common hex prefixes and separators
        cleaned = string.replace('0x', '').replace('-', '').replace(' ', '')

        # Check if it's a reasonable length and consists mostly of hex digits
        if len(cleaned) < 8 or len(cleaned) > 256:
            return False

        hex_chars = set('0123456789abcdefABCDEF')
        return sum(c in hex_chars for c in cleaned) / len(cleaned) > 0.9

    def _scan_mp3_details(self, file_path: Path) -> Dict[str, Any]:
        """MP3-specific metadata scanning"""
        results = {
            'mp3_details': {},
            'id3v1': None,
            'id3v2': None,
            'anomalies': []
        }

        try:
            # Try to read with mutagen's MP3 class
            mp3_file = MP3(file_path)

            # Check ID3v1 tag
            if hasattr(mp3_file, 'tags') and mp3_file.tags:
                id3v1_tags = {}
                for key in ['TPE1', 'TIT2', 'TALB', 'TDRC', 'TRCK']:
                    if key in mp3_file.tags:
                        id3v1_tags[key] = str(mp3_file.tags[key])

                if id3v1_tags:
                    results['id3v1'] = id3v1_tags

            # Look for ID3v2 header
            with open(file_path, 'rb') as f:
                header = f.read(10)
                if header.startswith(b'ID3'):
                    id3_version = header[3]
                    id3_flags = header[5]
                    id3_size = (header[6] << 21) | (header[7] << 14) | (header[8] << 7) | header[9]

                    results['id3v2'] = {
                        'version': f'2.{id3_version}',
                        'flags': id3_flags,
                        'size': id3_size
                    }

                    # Check for unusual ID3 flags
                    if id3_flags & 0x80:  # Unknown flag set
                        results['anomalies'].append('Unusual ID3v2 flags detected')

        except Exception as e:
            results['anomalies'].append(f'MP3 scan error: {str(e)}')

        return results

    def _scan_wav_details(self, file_path: Path) -> Dict[str, Any]:
        """WAV-specific metadata scanning"""
        results = {
            'wav_details': {},
            'chunks': [],
            'anomalies': []
        }

        try:
            wav_file = WAVE(file_path)

            # Scan for all RIFF chunks
            with open(file_path, 'rb') as f:
                # Skip RIFF header
                f.seek(12)

                chunk_positions = []
                while True:
                    # Read chunk header
                    chunk_header = f.read(8)
                    if len(chunk_header) < 8:
                        break

                    chunk_id = chunk_header[:4]
                    chunk_size = struct.unpack('<I', chunk_header[4:8])[0]

                    # Store chunk info
                    chunk_info = {
                        'id': chunk_id.decode('ascii', errors='ignore'),
                        'size': chunk_size,
                        'position': f.tell() - 8
                    }
                    results['chunks'].append(chunk_info)
                    chunk_positions.append(f.tell() - 8)

                    # Skip chunk data
                    f.seek(chunk_size, 1)
                    # Skip padding byte if chunk size is odd
                    if chunk_size % 2 == 1:
                        f.seek(1, 1)

                    # Safety check
                    if f.tell() > file_path.stat().st_size:
                        break

                # Check for suspicious chunks
                suspicious_chunk_ids = ['fact', 'bext', 'cart', 'list', 'data ']
                for chunk in results['chunks']:
                    if chunk['id'].lower() not in suspicious_chunk_ids:
                        results['anomalies'].append(f'Unusual WAV chunk: {chunk["id"]}')

                # Check chunk ordering
                if len(results['chunks']) > 1:
                    data_chunk_idx = next((i for i, c in enumerate(results['chunks']) if c['id'] == 'data'), None)
                    if data_chunk_idx and data_chunk_idx > 0:
                        # fmt chunk should come before data
                        fmt_chunk_idx = next((i for i, c in enumerate(results['chunks']) if c['id'] == 'fmt '), None)
                        if fmt_chunk_idx is None or fmt_chunk_idx > data_chunk_idx:
                            results['anomalies'].append('Unusual WAV chunk ordering')

        except Exception as e:
            results['anomalies'].append(f'WAV scan error: {str(e)}')

        return results