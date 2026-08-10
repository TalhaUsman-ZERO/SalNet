"""
Training script for TRSNet Salient Object Detection

Supports configurable PVT backbones (B0 for lightweight, B5 for performance)
Uses mixed precision training with gradient scaling for efficiency
"""

import os
import datetime
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import dataset
import argparse
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from get_model import get_model
from config import ModelConfig, get_custom_config


def iou_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Intersection over Union (IoU) Loss
    
    Args:
        pred: Predicted logits (B, 1, H, W)
        mask: Ground truth mask (B, 1, H, W)
        
    Returns:
        Mean IoU loss
    """
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def train(Dataset, parser):
    """
    Main training loop
    
    Args:
        Dataset: Dataset module
        parser: Argument parser with configuration
    """
    
    args = parser.parse_args()
    
    # Parse arguments
    backbone_type = args.backbone
    dataset_path = args.dataset
    learning_rate = args.lr
    weight_decay = args.decay
    momentum = args.momen
    batch_size = args.batchsize
    num_epochs = args.epoch
    loss_type = args.loss
    save_path = args.savepath
    validate = args.valid
    
    print(f"\n{'='*60}")
    print("TRSNet Training Configuration")
    print(f"{'='*60}")
    print(f"Backbone: {backbone_type}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Save Path: {save_path}")
    print(f"{'='*60}\n")
    
    # Create config
    cfg = get_custom_config(
        backbone_type=backbone_type,
        batch_size=batch_size,
        epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    # Dataset configuration
    dataset_cfg = Dataset.Config(
        datapath=dataset_path,
        savepath=save_path,
        mode='train',
        batch=batch_size,
        lr=learning_rate,
        momen=momentum,
        decay=weight_decay,
        epoch=num_epochs
    )
    
    # Create data loader
    data = Dataset.Data(dataset_cfg, 'TRSNet')
    loader = DataLoader(
        data,
        collate_fn=data.collate,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=6
    )
    
    # Initialize model
    net = get_model(cfg=cfg, backbone_type=backbone_type, verbose=True)
    net.train(True)
    net.cuda()
    
    # Separate parameters for different learning rates
    base_params = []
    head_params = []
    
    for name, param in net.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            # Freeze first layer
            param.requires_grad = False
        elif 'encoder' in name:
            base_params.append(param)
        else:
            head_params.append(param)
    
    # Optimizer with parameter groups
    optimizer = torch.optim.SGD(
        [
            {'params': base_params, 'lr': learning_rate * 0.1},
            {'params': head_params, 'lr': learning_rate}
        ],
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # TensorBoard writer
    writer = SummaryWriter(save_path)
    global_step = 0
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(num_epochs):
        
        # Linear learning rate schedule
        lr_factor = 1 - abs((epoch + 1) / (num_epochs + 1) * 2 - 1)
        optimizer.param_groups[0]['lr'] = lr_factor * learning_rate * 0.1
        optimizer.param_groups[1]['lr'] = lr_factor * learning_rate
        
        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda(), mask.cuda()
            
            # Forward pass with mixed precision
            with autocast():
                image = image.to(dtype=torch.float32)
                output = net(image)
                loss_bce = F.binary_cross_entropy_with_logits(output, mask)
                loss_iou = iou_loss(output, mask)
                loss = loss_bce + loss_iou
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            
            # Logging
            global_step += 1
            writer.add_scalar('lr/base', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('lr/head', optimizer.param_groups[1]['lr'], global_step)
            writer.add_scalar('loss/total', loss.item(), global_step)
            writer.add_scalar('loss/bce', loss_bce.item(), global_step)
            writer.add_scalar('loss/iou', loss_iou.item(), global_step)
            
            if step % 10 == 0:
                print(f'{datetime.datetime.now()} | '
                      f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Step: {step}/{len(loader)} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                      f'Loss: {loss.item():.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_path, f'TRSNet-{backbone_type}-epoch{epoch + 1}.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Validation
        if validate:
            print(f"Running validation at epoch {epoch + 1}...")
            for val_path in ['datasets/ECSSD/Test', 'datasets/PASCAL-S/Test']:
                if os.path.exists(val_path):
                    v = Valid(dataset, val_path, epoch, backbone_type, save_path, "SOD")
                    v.save()
    
    writer.close()
    print("Training completed!")


class Valid(object):
    """Validation class for periodic evaluation during training"""
    
    def __init__(self, Dataset, path: str, epoch: int, backbone_type: str, 
                 checkpoint_path: str, task: str = 'SOD'):
        """
        Initialize validation instance
        
        Args:
            Dataset: Dataset module
            path: Validation dataset path
            epoch: Current epoch number
            backbone_type: 'pvt_b0' or 'pvt_b5'
            checkpoint_path: Path where checkpoints are saved
            task: Task type
        """
        # Dataset config
        snapshot = os.path.join(checkpoint_path, f'TRSNet-{backbone_type}-epoch{epoch + 1}.pth')
        self.cfg = Dataset.Config(datapath=path, snapshot=snapshot, mode='test')
        self.data = Dataset.Data(self.cfg, 'TRSNet')
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=4)
        
        # Model
        self.net = get_model(cfg=self.cfg, backbone_type=backbone_type, verbose=False)
        self.net.train(False)
        self.net.cuda()
        self.epoch = epoch
        self.backbone_type = backbone_type
        self.task = task
    
    def save(self):
        """Run validation and save predictions"""
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image = image.cuda().float()
                output = self.net(image)
                pred = torch.sigmoid(output[0, 0]).cpu().numpy() * 255
                
                # Determine output directory
                dataset_name = self.cfg.datapath.split('/')[-2]
                head = f'Prediction/TRSNet-{self.backbone_type}-valid-epoch{self.epoch + 1}/{dataset_name}'
                
                os.makedirs(head, exist_ok=True)
                cv2.imwrite(os.path.join(head, name[0] + '.png'), np.round(pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRSNet Training')
    
    # Model configuration
    parser.add_argument('--backbone', default='pvt_b5', choices=['pvt_b0', 'pvt_b5'],
                       help='Backbone type: pvt_b0 (lightweight) or pvt_b5 (performance)')
    
    # Training configuration
    parser.add_argument('--dataset', default='datasets/DUTS/Train',
                       help='Path to training dataset')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate')
    parser.add_argument('--momen', type=float, default=0.9,
                       help='Momentum for SGD')
    parser.add_argument('--decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--batchsize', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epoch', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--loss', default='BCE+IoU',
                       help='Loss function type')
    parser.add_argument('--savepath', default='checkpoint/TRSNet',
                       help='Path to save checkpoints')
    parser.add_argument('--valid', type=bool, default=True,
                       help='Perform validation during training')
    
    args = parser.parse_args()
    
    train(dataset, parser)

