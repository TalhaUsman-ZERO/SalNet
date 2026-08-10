#!/usr/bin/python3
# coding=utf-8

"""
TRSNet: Salient Object Detection
Version: 1.0
Status: Active Development

Main exports:
- TRSNet: Main model class
- get_model, get_model_lightweight, get_model_performance: Model factories
- ModelConfig: Configuration system
"""

try:
    from TRSNet import TRSNet
    from get_model import get_model, get_model_lightweight, get_model_performance
    from config import ModelConfig, get_lightweight_config, get_performance_config, get_custom_config
    
    __version__ = "1.0.0"
    __all__ = [
        'TRSNet',
        'get_model',
        'get_model_lightweight',
        'get_model_performance',
        'ModelConfig',
        'get_lightweight_config',
        'get_performance_config',
        'get_custom_config'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import TRSNet modules: {e}")
    __version__ = "1.0.0"

