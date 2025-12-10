"""
Default configuration settings for MMM
"""

DEFAULT_CONFIG = {
    # Core settings
    'version': '2.0.0',
    'paranoia_level': 'medium',  # low, medium, high, maximum
    'preserve_quality': 'high',   # low, medium, high, maximum
    'output_format': 'preserve',  # preserve, mp3, wav
    'backup_originals': False,

    # Audio processing settings
    'audio_processing': {
        'sample_rate': None,      # Auto-detect if None
        'bit_depth': None,        # Auto-detect if None
        'channels': 'preserve',   # preserve, mono, stereo
        'normalize': True,
        'dithering': True
    },

    # Watermark detection methods
    'watermark_detection': [
        'spread_spectrum',
        'echo_based',
        'statistical',
        'phase_modulation',
        'amplitude_modulation',
        'frequency_domain'
    ],

    # Spectral cleaning settings
    'spectral_cleaning': {
        'high_freq_cutoff': 15000,  # Hz
        'notch_filter_q': 30,
        'smoothing_window': 5,
        'adaptive_noise': True
    },

    # Metadata cleaning settings
    'metadata_cleaning': {
        'aggressive_mode': False,
        'preserve_date': False,
        'preserve_technical': False,  # Preserve sample rate, bit depth, etc.
        'strip_binary_chunks': True,
        'remove_id3v1': True,
        'remove_id3v2': True,
        'remove_ape_tags': True
    },

    # Fingerprint removal settings
    'fingerprint_removal': {
        'statistical_normalization': True,
        'temporal_randomization': True,
        'phase_randomization': False,      # Only in paranoid mode
        'micro_timing_perturbation': True,
        'human_imperfections': False      # Only in paranoid mode
    },

    # Quality preservation
    'quality_preservation': {
        'target_snr': 40,         # dB minimum
        'max_quality_loss': 5,    # percentage
        'preserve_dynamics': True,
        'preserve_frequency_response': True
    },

    # Batch processing
    'batch_processing': {
        'workers': 4,             # Parallel workers
        'progress_updates': True,
        'continue_on_error': False,
        'output_directory': None,
        'naming_pattern': '{name}_clean{ext}'  # {name}, {ext}, {timestamp}
    },

    # Verification settings
    'verification': {
        'auto_verify': True,
        'deep_analysis': False,
        'compare_spectra': True,
        'check_watermarks': True,
        'calculate_metrics': True
    },

    # Logging settings
    'logging': {
        'level': 'INFO',          # DEBUG, INFO, WARNING, ERROR
        'file': None,             # Log to file if specified
        'console': True,
        'detailed': False
    },

    # Security settings
    'security': {
        'memory_cleanup': True,   # Clear memory after processing
        'temp_cleanup': True,     # Clean temporary files
        'hash_verification': True,
        'integrity_checks': True
    },

    # Advanced settings
    'advanced': {
        'custom_filters': [],
        'exclude_methods': [],
        'include_experimental': False,
        'research_mode': False,
        'debug_mode': False
    },

    # UI settings
    'ui': {
        'color_output': True,
        'unicode_symbols': True,
        'progress_bars': True,
        'detailed_output': False,
        'show_quotes': True,
        'ascii_art': True
    },

    # Performance settings
    'performance': {
        'memory_limit': 1024,     # MB
        'temp_dir': None,         # System default if None
        'cache_results': False,
        'optimize_for_speed': False
    },

    # Format-specific settings
    'formats': {
        'mp3': {
            'bitrate': 'preserve',  # preserve, 128, 192, 256, 320
            'quality': 2,           # VBR quality (0-9)
            'joint_stereo': True
        },
        'wav': {
            'bit_depth': 'preserve',  # preserve, 16, 24, 32
            'sample_format': 'pcm'   # pcm, float
        }
    },

    # Experimental features
    'experimental': {
        'ml_detection': False,      # Use ML for watermark detection
        'gpu_acceleration': False,  # GPU processing if available
        'real_time_processing': False,
        'advanced_steganography_detection': False
    }
}

# Preset configurations
PRESETS = {
    'stealth-plus': {
        'paranoia_level': 'maximum',
        'preserve_quality': 'maximum',
        'advanced_flags': {
            'phase_dither': False,
            'comb_mask': False,
            'transient_shift': False,
            'phase_swirl': False,
            'masked_hf_phase': False,
            'resample_nudge': False,
            'gated_resample_nudge': True,
            'phase_noise': True,
            'micro_eq_flutter': False,
            'hf_decorrelate': False,
            'refined_transient': False,
            'adaptive_transient': False
        },
        'verification': {
            'deep_analysis': True,
            'auto_verify': True
        },
        'performance': {
            'optimize_for_speed': True
        }
    },
    'stealth': {
        'paranoia_level': 'maximum',
        'preserve_quality': 'maximum',
        'fingerprint_removal': {
            'statistical_normalization': True,
            'temporal_randomization': True,
            'phase_randomization': True,
            'micro_timing_perturbation': True,
            'human_imperfections': True
        },
        'spectral_cleaning': {
            'adaptive_noise': True,
            'smoothing_window': 3
        },
        'verification': {
            'deep_analysis': True,
            'auto_verify': True
        }
    },

    'fast': {
        'paranoia_level': 'low',
        'preserve_quality': 'medium',
        'watermark_detection': [
            'spread_spectrum',
            'statistical'
        ],
        'spectral_cleaning': {
            'adaptive_noise': False
        },
        'verification': {
            'auto_verify': False,
            'deep_analysis': False
        },
        'performance': {
            'optimize_for_speed': True
        }
    },

    'quality': {
        'paranoia_level': 'medium',
        'preserve_quality': 'maximum',
        'metadata_cleaning': {
            'preserve_technical': True,
            'aggressive_mode': False
        },
        'quality_preservation': {
            'target_snr': 50,
            'max_quality_loss': 2,
            'preserve_dynamics': True,
            'preserve_frequency_response': True
        }
    },

    'research': {
        'paranoia_level': 'high',
        'preserve_quality': 'high',
        'watermark_detection': [
            'spread_spectrum',
            'echo_based',
            'statistical',
            'phase_modulation',
            'amplitude_modulation',
            'frequency_domain'
        ],
        'verification': {
            'deep_analysis': True,
            'compare_spectra': True,
            'check_watermarks': True,
            'calculate_metrics': True
        },
        'logging': {
            'level': 'DEBUG',
            'detailed': True
        },
        'advanced': {
            'include_experimental': True,
            'research_mode': True,
            'debug_mode': True
        }
    }
}

# Paranoia level configurations
PARANOIA_CONFIGS = {
    'low': {
        'watermark_detection': [
            'spread_spectrum',
            'statistical'
        ],
        'fingerprint_removal': {
            'statistical_normalization': True,
            'temporal_randomization': False,
            'phase_randomization': False,
            'micro_timing_perturbation': False,
            'human_imperfections': False
        },
        'spectral_cleaning': {
            'adaptive_noise': False
        },
        'verification': {
            'auto_verify': False,
            'deep_analysis': False
        }
    },

    'medium': {
        'watermark_detection': [
            'spread_spectrum',
            'echo_based',
            'statistical'
        ],
        'fingerprint_removal': {
            'statistical_normalization': True,
            'temporal_randomization': True,
            'phase_randomization': False,
            'micro_timing_perturbation': True,
            'human_imperfections': False
        },
        'spectral_cleaning': {
            'adaptive_noise': True
        },
        'verification': {
            'auto_verify': True,
            'deep_analysis': False
        }
    },

    'high': {
        'watermark_detection': [
            'spread_spectrum',
            'echo_based',
            'statistical',
            'phase_modulation',
            'amplitude_modulation'
        ],
        'fingerprint_removal': {
            'statistical_normalization': True,
            'temporal_randomization': True,
            'phase_randomization': True,
            'micro_timing_perturbation': True,
            'human_imperfections': False
        },
        'spectral_cleaning': {
            'adaptive_noise': True,
            'smoothing_window': 3
        },
        'verification': {
            'auto_verify': True,
            'deep_analysis': True
        }
    },

    'maximum': {
        'watermark_detection': [
            'spread_spectrum',
            'echo_based',
            'statistical',
            'phase_modulation',
            'amplitude_modulation',
            'frequency_domain'
        ],
        'fingerprint_removal': {
            'statistical_normalization': True,
            'temporal_randomization': True,
            'phase_randomization': True,
            'micro_timing_perturbation': True,
            'human_imperfections': True
        },
        'spectral_cleaning': {
            'adaptive_noise': True,
            'smoothing_window': 5
        },
        'metadata_cleaning': {
            'aggressive_mode': True,
            'strip_binary_chunks': True
        },
        'verification': {
            'auto_verify': True,
            'deep_analysis': True,
            'compare_spectra': True
        },
        'advanced': {
            'include_experimental': True
        }
    }
}
