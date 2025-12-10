"""
Configuration manager for MMM settings and preferences
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .defaults import DEFAULT_CONFIG


class ConfigManager:
    """
    Manages configuration settings for the MMM application
    """

    def __init__(self, config_file: Optional[Path] = None):
        self.config_dir = self._get_config_dir()
        self.config_file = config_file or self.config_dir / 'config.yaml'
        self.config = DEFAULT_CONFIG.copy()

        # Load existing config if present
        self.load_config()

    def _get_config_dir(self) -> Path:
        """Get platform-specific configuration directory"""
        home = Path.home()

        # Platform-specific config directories
        if os.name == 'nt':  # Windows
            config_dir = home / 'AppData' / 'Local' / 'mmm'
        elif os.name == 'posix':
            if os.uname().sysname == 'Darwin':  # macOS
                config_dir = home / 'Library' / 'Application Support' / 'mmm'
            else:  # Linux/Unix
                config_dir = home / '.config' / 'mmm'
        else:
            config_dir = home / '.mmm'

        # Create directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)

        return config_dir

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f)

                # Merge with defaults
                self.config = self._merge_configs(DEFAULT_CONFIG, user_config)
            else:
                # Create default config file
                self.save_config()

        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            self.config = DEFAULT_CONFIG.copy()

        return self.config

    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config file: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get specific configuration value using dot notation"""
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set specific configuration value using dot notation"""
        keys = key.split('.')
        config_section = self.config

        # Navigate to parent section
        for k in keys[:-1]:
            if k not in config_section:
                config_section[k] = {}
            config_section = config_section[k]

        # Set the value
        config_section[keys[-1]] = value

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = DEFAULT_CONFIG.copy()
        self.save_config()

    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with defaults"""
        result = default.copy()

        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def create_preset(self, name: str, config_subset: Dict[str, Any]):
        """Create a configuration preset"""
        preset_file = self.config_dir / f'preset_{name}.yaml'

        try:
            with open(preset_file, 'w') as f:
                yaml.dump(config_subset, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to create preset: {e}")

    def load_preset(self, name: str) -> Dict[str, Any]:
        """Load a configuration preset"""
        preset_file = self.config_dir / f'preset_{name}.yaml'

        if not preset_file.exists():
            raise Exception(f"Preset '{name}' not found")

        try:
            with open(preset_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"Failed to load preset: {e}")

    def list_presets(self) -> list:
        """List available presets"""
        presets = []
        for file in self.config_dir.glob('preset_*.yaml'):
            preset_name = file.stem.replace('preset_', '')
            presets.append(preset_name)
        return presets

    def delete_preset(self, name: str):
        """Delete a configuration preset"""
        preset_file = self.config_dir / f'preset_{name}.yaml'

        if preset_file.exists():
            preset_file.unlink()
        else:
            raise Exception(f"Preset '{name}' not found")

    def validate_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        if config is None:
            config = self.config

        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate paranoia_level
        paranoia_levels = ['low', 'medium', 'high', 'maximum']
        paranoia = config.get('paranoia_level', 'medium')
        if paranoia not in paranoia_levels:
            validation['errors'].append(f"Invalid paranoia_level: {paranoia}. Must be one of: {paranoia_levels}")
            validation['valid'] = False

        # Validate preserve_quality
        quality_levels = ['low', 'medium', 'high', 'maximum']
        quality = config.get('preserve_quality', 'high')
        if quality not in quality_levels:
            validation['errors'].append(f"Invalid preserve_quality: {quality}. Must be one of: {quality_levels}")
            validation['valid'] = False

        # Validate output_format
        output_formats = ['preserve', 'mp3', 'wav']
        output_format = config.get('output_format', 'preserve')
        if output_format not in output_formats:
            validation['errors'].append(f"Invalid output_format: {output_format}. Must be one of: {output_formats}")
            validation['valid'] = False

        # Validate watermark_detection methods
        valid_methods = ['spread_spectrum', 'echo_based', 'statistical', 'phase_modulation', 'amplitude_modulation', 'frequency_domain']
        detection_methods = config.get('watermark_detection', [])
        for method in detection_methods:
            if method not in valid_methods:
                validation['warnings'].append(f"Unknown watermark detection method: {method}")

        # Validate numeric ranges
        numeric_ranges = {
            'batch_processing.workers': (1, 32),
            'audio_processing.sample_rate': (8000, 192000),
            'audio_processing.bit_depth': (16, 32)
        }

        for key, (min_val, max_val) in numeric_ranges.items():
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                validation['errors'].append(f"{key} must be between {min_val} and {max_val}")
                validation['valid'] = False

        return validation

    def get_effective_config(self, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get effective configuration with applied overrides"""
        effective = self.get_config()

        if overrides:
            effective = self._merge_configs(effective, overrides)

        return effective

    def export_config(self, file_path: Path, include_defaults: bool = False):
        """Export configuration to file"""
        if include_defaults:
            config_to_export = self.config
        else:
            # Export only user-defined settings
            config_to_export = self._get_user_settings()

        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_to_export, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise Exception(f"Failed to export config: {e}")

    def import_config(self, file_path: Path, merge: bool = True):
        """Import configuration from file"""
        try:
            with open(file_path, 'r') as f:
                imported_config = yaml.safe_load(f)

            if merge:
                self.config = self._merge_configs(self.config, imported_config)
            else:
                self.config = self._merge_configs(DEFAULT_CONFIG, imported_config)

            # Validate imported config
            validation = self.validate_config()
            if not validation['valid']:
                raise Exception(f"Invalid imported config: {'; '.join(validation['errors'])}")

            self.save_config()

        except Exception as e:
            raise Exception(f"Failed to import config: {e}")

    def _get_user_settings(self) -> Dict[str, Any]:
        """Get only user-defined settings (different from defaults)"""
        user_settings = {}

        def extract_user_diff(current: Dict[str, Any], default: Dict[str, Any], path: str = ""):
            for key, value in current.items():
                current_path = f"{path}.{key}" if path else key

                if key not in default:
                    user_settings[current_path] = value
                elif isinstance(value, dict) and isinstance(default.get(key), dict):
                    extract_user_diff(value, default[key], current_path)
                elif value != default.get(key):
                    user_settings[current_path] = value

        extract_user_diff(self.config, DEFAULT_CONFIG)

        # Rebuild nested structure from dot notation
        result = {}
        for key, value in user_settings.items():
            keys = key.split('.')
            current = result

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return result

    def update_config_schema(self):
        """Update configuration file when schema changes"""
        # This would be called when upgrading MMM versions
        # to ensure config file matches expected schema

        # Get current config version
        current_version = self.get('version', '1.0.0')
        target_version = DEFAULT_CONFIG.get('version', '2.0.0')

        if current_version != target_version:
            # Perform migration logic here
            print(f"Migrating config from version {current_version} to {target_version}")

            # Update version
            self.set('version', target_version)
            self.save_config()