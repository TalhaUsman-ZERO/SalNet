"""
TRSNet: Trapezoidal Attention Network for Salient Object Detection

This module implements the TRSNet architecture with configurable PVT backbone (B0/B5).
The model supports lightweight (B0) and high-performance (B5) configurations.

Paper: TRSNet - Submitted to EAAI 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Optional, Tuple

from pvtv2_encoder import pvt_v2_b0, pvt_v2_b5
from modules import CFRB, CCG, ASCA, FAMHA


def weight_init(module):
    """Initialize weights using Kaiming Normal initialization"""
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, 
                           nn.AdaptiveAvgPool1d, nn.Identity)):
            pass
        else:
            if hasattr(m, 'initialize'):
                m.initialize()


class BasicConv2d(nn.Module):
    """Basic Convolution block with BatchNorm"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        return self.conv_bn(x)

class TRSNet(nn.Module):
    """
    TRSNet: Trapezoidal Attention Network
    Configurable architecture supporting both lightweight (PVT-B0) and 
    high-performance (PVT-B5) backbones.
    
    Args:
        cfg: Configuration object containing model parameters
        backbone_type: 'pvt_b0' for lightweight or 'pvt_b5' for performance
    """
    
    def __init__(self, cfg=None, backbone_type: str = 'pvt_b5', checkpoint_path: Optional[str] = None):
        super(TRSNet, self).__init__()
        
        self.cfg = cfg
        self.backbone_type = backbone_type
        self.model_name = 'TRSNet'
        
        # Initialize backbone
        self.encoder = self._build_encoder(backbone_type, checkpoint_path)
        
        # CFRB modules for each encoder stage
        self.cfrb_conv1 = CFRB(feature_layer='1')
        self.cfrb_conv2 = CFRB(feature_layer='2')
        self.cfrb_conv3 = CFRB(feature_layer='3')
        self.cfrb_conv4 = CFRB(feature_layer='4')
        
        # Attention modules
        self.asca_stage1 = ASCA(inp=64, oup=64)
        self.asca_stage2 = ASCA(inp=320, oup=320)
        self.ccg = CCG(max_kernel=3)
        
        # Multi-Head Self Attention modules
        self.famha_f2 = FAMHA(d_model=320, d_k=320, d_v=320, h=8, H=48, W=48, 
                              ratio=2, apply_transform=True)
        self.famha_f3 = FAMHA(d_model=320, d_k=320, d_v=320, h=8, H=24, W=24, 
                              ratio=2, apply_transform=True)
        
        # Feature fusion modules
        self.encoder_merge1_2 = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU()
        )
        
        self.encoder_merge3_4 = nn.Sequential(
            nn.BatchNorm2d(832),
            nn.ConvTranspose2d(832, 416, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU()
        )
        
        self.encoder_merge1234 = nn.Sequential(
            nn.BatchNorm2d(384),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU()
        )
        
        self.encoder_mergeall = nn.Sequential(
            nn.BatchNorm2d(832),
            nn.ConvTranspose2d(832, 416, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU()
        )
        
        # Transition and output modules
        self.trans_conv = nn.ConvTranspose2d(in_channels=416, out_channels=192, 
                                             kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        self.ff_conv_1 = nn.ConvTranspose2d(192, 1, kernel_size=3, padding=1)
        
        self.initialize()
    
    def _build_encoder(self, backbone_type: str, checkpoint_path: Optional[str] = None) -> nn.Module:
        """
        Build PVT encoder based on backbone type
        
        Args:
            backbone_type: 'pvt_b0' or 'pvt_b5'
            checkpoint_path: Path to pretrained weights
            
        Returns:
            Initialized encoder module
        """
        if backbone_type == 'pvt_b0':
            encoder = pvt_v2_b0()
            default_checkpoint = 'checkpoint/Backbone/PVTv2/pvt_v2_b0.pth'
        elif backbone_type == 'pvt_b5':
            encoder = pvt_v2_b5()
            default_checkpoint = 'checkpoint/Backbone/PVTv2/pvt_v2_b5.pth'
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. "
                           f"Choose 'pvt_b0' or 'pvt_b5'")
        
        # Load pretrained weights
        ckpt_path = checkpoint_path or default_checkpoint
        if os.path.exists(ckpt_path):
            pretrained_dict = torch.load(ckpt_path, map_location='cpu')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in encoder.state_dict()}
            encoder.load_state_dict(pretrained_dict)
            print(f"Loaded pretrained weights from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. "
                  f"Using random initialization.")
        
        return encoder
    
    def forward(self, x: torch.Tensor, shape: Optional[Tuple] = None, 
                name: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass of TRSNet
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            shape: Optional shape for output resizing
            name: Optional name for processing
            
        Returns:
            Output saliency map of shape (B, 1, H, W)
        """
        batch_size = x.size(0)
        
        # Encoder forward pass - returns features from 4 stages
        # shapes: x4 (B, 512, H/32, W/32), x3 (B, 320, H/16, W/16), 
        #         x2 (B, 128, H/8, W/8), x1 (B, 64, H/4, W/4)
        features = self.encoder(x)
        x4, x3, x2, x1 = features[0], features[1], features[2], features[3]
        
        # CFRB processing for each stage
        conv1_cfrb_feats = self.cfrb_conv1(x1)  # (B, 320, H/4, W/4)
        conv2_cfrb_feats = self.cfrb_conv2(x2)  # (B, 320, H/8, W/8)
        conv3_cfrb_feats = self.cfrb_conv3(x3)  # (B, 320, H/16, W/16)
        conv4_cfrb_feats = self.cfrb_conv4(x4)  # (B, 320, H/32, W/32)
        
        # Attention modules
        f1 = self.asca_stage1(x1)                    # (B, 64, H/4, W/4)
        f2 = self.asca_stage2(conv2_cfrb_feats)      # (B, 320, H/8, W/8)
        f3 = self.ccg(conv3_cfrb_feats)              # (B, 320, H/16, W/16)
        f4 = self.ccg(x4)                            # (B, 512, H/32, W/32)
        
        # Multi-head attention for f2
        f2_reshaped = f2.permute(0, 2, 3, 1).reshape(batch_size, -1, 320)
        f2_mhsa = self.famha_f2(f2_reshaped, f2_reshaped, f2_reshaped)
        f2_mhsa = f2_mhsa.reshape(batch_size, 48, 48, 320).permute(0, 3, 1, 2)
        f2_mhsa = F.interpolate(f2_mhsa, size=(96, 96), mode='bilinear', align_corners=False)
        
        # Merge f1 and f2
        f12_feat = self.encoder_merge1_2(torch.cat([f1, f2_mhsa], dim=1))
        
        # Multi-head attention for f3
        f3_reshaped = f3.permute(0, 2, 3, 1).reshape(batch_size, -1, 320)
        f3_mhsa = self.famha_f3(f3_reshaped, f3_reshaped, f3_reshaped)
        f3_mhsa = f3_mhsa.reshape(batch_size, 24, 24, 320).permute(0, 3, 1, 2)
        
        # Merge f3 and f4
        f4_upsampled = F.interpolate(f4, size=(24, 24), mode='bilinear', align_corners=False)
        f34_feat = self.encoder_merge3_4(torch.cat([f3_mhsa, f4_upsampled], dim=1))
        
        # Merge all features
        f34_feat = F.interpolate(f34_feat, size=(96, 96), mode='bilinear', align_corners=False)
        f34_feat = self.trans_conv(f34_feat)
        f1234_feat = self.encoder_merge1234(torch.cat([f12_feat, f34_feat], dim=1))
        
        # Final fusion
        conv4_cpfe_feats_upsampled = F.interpolate(conv4_cfrb_feats, size=(96, 96), 
                                                   mode='bilinear', align_corners=False)
        fused_feat = self.encoder_mergeall(
            torch.cat([f1234_feat, conv1_cfrb_feats, conv4_cpfe_feats_upsampled], dim=1)
        )
        
        fused_feat = self.trans_conv(fused_feat)
        fused_feat = F.interpolate(fused_feat, size=(112, 112), mode='bilinear', align_corners=False)
        fused_feat = self.dropout(fused_feat)
        
        # Final output
        fused_feat = F.interpolate(fused_feat, size=(384, 384), mode='bilinear', align_corners=False)
        fused_feat = self.dropout(fused_feat)
        output = self.ff_conv_1(fused_feat)
        
        return output
    
    def initialize(self):
        """Initialize model weights"""
        if self.cfg is not None and hasattr(self.cfg, 'snapshot') and self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
            print(f"Loaded checkpoint from {self.cfg.snapshot}")
        else:
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    weight_init(module)
