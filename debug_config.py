#!/usr/bin/env python3
"""
Debug the config issue in CLI
"""

import sys
sys.path.insert(0, '/home/geeknik/dev/mmm')

from mmm.config.config_manager import ConfigManager

def test_config():
    print("Testing config manager...")

    # Test global config creation
    global_config = ConfigManager()
    print(f"1. Global config created: {type(global_config)}")
    print(f"   Has config attribute: {hasattr(global_config, 'config')}")
    print(f"   Config type: {type(global_config.config)}")

    # Test accessing config
    try:
        config_dict = global_config.config
        print(f"2. Config dict accessed successfully: {type(config_dict)}")
        print(f"   Config keys: {list(config_dict.keys())[:5]}")
    except Exception as e:
        print(f"2. Error accessing config: {e}")

    # Test creating AudioSanitizer with this config
    try:
        from mmm.core.audio_sanitizer import AudioSanitizer
        from pathlib import Path

        sanitizer = AudioSanitizer(
            input_file=Path("Schizo Shaman.mp3"),
            config=config_dict
        )
        print("3. AudioSanitizer created successfully")
    except Exception as e:
        print(f"3. Error creating AudioSanitizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()