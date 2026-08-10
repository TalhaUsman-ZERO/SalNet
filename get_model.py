"""
Model factory for TRSNet

This module provides utilities to instantiate TRSNet with various configurations
and compute model statistics (parameters, FLOPs, FPS).
"""

import torch
import time
from typing import Optional
from TRSNet import TRSNet
from config import ModelConfig, get_lightweight_config, get_performance_config

try:
    from ptflops import get_model_complexity_info
    HAS_PTFLOPS = True
except ImportError:
    HAS_PTFLOPS = False
    print("Warning: ptflops not installed. FLOPs calculation will be skipped.")


def get_model(cfg: Optional[ModelConfig] = None, 
              backbone_type: str = 'pvt_b5',
              checkpoint_path: Optional[str] = None,
              verbose: bool = True) -> TRSNet:
    """
    Create and initialize TRSNet model
    
    Args:
        cfg: ModelConfig object. If None, uses performance config
        backbone_type: 'pvt_b0' for lightweight or 'pvt_b5' for performance
        checkpoint_path: Optional path to pretrained backbone weights
        verbose: Whether to print model statistics
        
    Returns:
        Initialized TRSNet model on GPU
        
    Example:
        # Lightweight model
        model = get_model(backbone_type='pvt_b0')
        
        # Custom config
        from config import get_custom_config
        cfg = get_custom_config(backbone_type='pvt_b5', batch_size=16)
        model = get_model(cfg=cfg)
    """
    
    # Use provided config or create default
    if cfg is None:
        cfg = get_performance_config() if backbone_type == 'pvt_b5' else get_lightweight_config()
    
    # Create model
    model = TRSNet(cfg=cfg, backbone_type=backbone_type, checkpoint_path=checkpoint_path)
    model = model.cuda()
    
    if verbose:
        print_model_statistics(model, backbone_type)
    
    return model


def get_model_lightweight(cfg: Optional[ModelConfig] = None, verbose: bool = True) -> TRSNet:
    """
    Create lightweight TRSNet model (PVT-B0)
    
    Args:
        cfg: Optional custom config. If None, uses lightweight config
        verbose: Whether to print model statistics
        
    Returns:
        Lightweight TRSNet model on GPU
    """
    if cfg is None:
        cfg = get_lightweight_config()
    return get_model(cfg=cfg, backbone_type='pvt_b0', verbose=verbose)


def get_model_performance(cfg: Optional[ModelConfig] = None, verbose: bool = True) -> TRSNet:
    """
    Create high-performance TRSNet model (PVT-B5)
    
    Args:
        cfg: Optional custom config. If None, uses performance config
        verbose: Whether to print model statistics
        
    Returns:
        High-performance TRSNet model on GPU
    """
    if cfg is None:
        cfg = get_performance_config()
    return get_model(cfg=cfg, backbone_type='pvt_b5', verbose=verbose)


def print_model_statistics(model: TRSNet, backbone_type: str = 'pvt_b5') -> None:
    """
    Print model statistics including parameters, FLOPs, and FPS
    
    Args:
        model: TRSNet model instance
        backbone_type: Backbone type for logging
    """
    print(f"\n{'='*60}")
    print(f"TRSNet Model Statistics - Backbone: {backbone_type.upper()}")
    print(f"{'='*60}")
    
    # Parameter count
    param_count = sum(x.numel() for x in model.parameters())
    print(f"Total Parameters: {param_count / 1e6:.2f}M")
    
    # Trainable vs non-trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
    
    # FLOPs computation (requires ptflops)
    if HAS_PTFLOPS:
        try:
            input_size = (3, 384, 384)
            flops, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=True,
                print_per_layer_stat=False
            )
            print(f"FLOPs: {flops}")
            print(f"MACs: {params}")
        except Exception as e:
            print(f"Could not compute FLOPs: {e}")
    
    # Throughput (FPS)
    try:
        fps = compute_fps(model, num_iterations=100)
        print(f"Throughput: {fps:.2f} FPS (input: 384x384)")
    except Exception as e:
        print(f"Could not compute FPS: {e}")
    
    print(f"{'='*60}\n")


def compute_fps(model: TRSNet, num_iterations: int = 100, input_size: tuple = (1, 3, 384, 384)) -> float:
    """
    Compute frames per second (FPS) for the model
    
    Args:
        model: TRSNet model instance (should be on GPU)
        num_iterations: Number of forward passes for averaging
        input_size: Input tensor shape (B, C, H, W)
        
    Returns:
        FPS value
    """
    model.eval()
    dummy_input = torch.randn(*input_size).cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
    
    fps = num_iterations / elapsed_time
    return fps


if __name__ == '__main__':
    print("Testing TRSNet model instantiation...\n")
    
    # Test lightweight model
    print("Creating lightweight model (PVT-B0)...")
    model_light = get_model_lightweight()
    print("✓ Lightweight model created successfully\n")
    
    # Test performance model
    print("Creating high-performance model (PVT-B5)...")
    model_perf = get_model_performance()
    print("✓ High-performance model created successfully\n")
    
    # Test custom configuration
    print("Creating custom model...")
    from config import get_custom_config
    custom_cfg = get_custom_config(backbone_type='pvt_b0', batch_size=16)
    model_custom = get_model(cfg=custom_cfg)
    print("✓ Custom model created successfully\n")

