"""
TRSNet Configuration File
Configurable parameters for model architecture, training, and evaluation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal


@dataclass
class BackboneConfig:
    """Configuration for PVT v2 backbone"""
    backbone_type: Literal['pvt_b0', 'pvt_b5'] = 'pvt_b5'
    checkpoint_path: str = 'checkpoint/Backbone/PVTv2/'
    
    # PVT backbone channel configurations
    backbone_channels: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set channel configuration based on backbone type"""
        if not self.backbone_channels:
            if self.backbone_type == 'pvt_b0':
                # PVT-B0 channels for each stage
                self.backbone_channels = {
                    'stage1': 64,
                    'stage2': 128,
                    'stage3': 320,
                    'stage4': 512,
                }
            elif self.backbone_type == 'pvt_b5':
                # PVT-B5 channels for each stage
                self.backbone_channels = {
                    'stage1': 64,
                    'stage2': 128,
                    'stage3': 320,
                    'stage4': 512,
                }
    
    @property
    def checkpoint_file(self) -> str:
        """Get the full checkpoint file path"""
        return f"{self.checkpoint_path}pvt_v2_{self.backbone_type.split('_')[-1]}.pth"


@dataclass
class DecoderConfig:
    """Configuration for decoder modules (CFRB, ASCA, CCG, FAMHA)"""
    # CFRB (Cross-scale Feature Refinement Block)
    cfrb_out_channels: int = 64
    cfrb_dilation_rates: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    
    # ASCA (Adaptive Spatial-Channel Attention)
    asca_enabled: bool = True
    
    # CCG (Cross-Channel Gating)
    ccg_max_kernel: int = 3
    
    # FAMHA (Feature Augmentation with Multi-Head Attention)
    famha_enabled: bool = True
    famha_d_model: int = 320
    famha_num_heads: int = 8
    famha_ratio: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 0.0001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    
    # Learning rate scheduling
    lr_scheduler: Literal['cosine', 'step', 'linear'] = 'linear'
    lr_decay_step: int = 25
    lr_decay_gamma: float = 0.1
    
    # Loss weights
    bce_weight: float = 1.0
    iou_weight: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Device
    device: str = 'cuda'
    num_workers: int = 6


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters"""
    # Input resolution
    image_size: int = 384
    
    # Data augmentation
    enable_random_crop: bool = True
    enable_random_flip: bool = True
    enable_random_rotate: bool = True
    rotation_angle_range: int = 10
    
    # Normalization
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)
    
    # Supported datasets
    supported_datasets: List[str] = field(default_factory=lambda: [
        'DUTS',
        'ECSSD',
        'HKU-IS',
        'PASCAL-S',
        'SOD',
        'DUT-OMRON'
    ])


@dataclass
class ModelConfig:
    """Main configuration class combining all components"""
    model_name: str = 'TRSNet'
    
    # Subconfigurations
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Model input/output
    input_channels: int = 3
    output_channels: int = 1
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
TRSNet Configuration
====================
Model: {self.model_name}
Backbone: {self.backbone.backbone_type}
Checkpoint: {self.backbone.checkpoint_file}
Input Size: {self.dataset.image_size}x{self.dataset.image_size}
Batch Size: {self.training.batch_size}
Learning Rate: {self.training.learning_rate}
Epochs: {self.training.epochs}
Device: {self.training.device}
"""


# Predefined configurations for quick setup

def get_lightweight_config() -> ModelConfig:
    """Get lightweight configuration (PVT-B0)"""
    config = ModelConfig()
    config.backbone.backbone_type = 'pvt_b0'
    config.training.batch_size = 16  # Can use larger batch with smaller model
    return config


def get_performance_config() -> ModelConfig:
    """Get high-performance configuration (PVT-B5)"""
    config = ModelConfig()
    config.backbone.backbone_type = 'pvt_b5'
    config.training.batch_size = 8  # Smaller batch for larger model
    return config


def get_custom_config(
    backbone_type: str = 'pvt_b5',
    batch_size: int = 8,
    epochs: int = 100,
    learning_rate: float = 0.0001,
    image_size: int = 384
) -> ModelConfig:
    """Get custom configuration with specified parameters"""
    config = ModelConfig()
    config.backbone.backbone_type = backbone_type
    config.training.batch_size = batch_size
    config.training.epochs = epochs
    config.training.learning_rate = learning_rate
    config.dataset.image_size = image_size
    return config


if __name__ == '__main__':
    # Example usage
    print("Lightweight Configuration:")
    print(get_lightweight_config())
    
    print("\nPerformance Configuration:")
    print(get_performance_config())
