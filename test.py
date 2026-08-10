#!/usr/bin/python3
# coding=utf-8

"""
Test script for TRSNet Salient Object Detection model

Supports evaluation on multiple datasets:
- SOD (Salient Object Detection): ECSSD, PASCAL-S, DUTS, HKU-IS, DUT-OMRON, SOD
- SOC (Salient Object in Clutter): Various SOC attribute datasets
- COD (Camouflaged Object Detection): CHAMELEON, CAMO, COD10K, CPD1K
"""

import os
import sys
sys.path.insert(0, './')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import numpy as np
import torch
import argparse
import dataset
import datetime
import time
from torch.utils.data import DataLoader
from get_model import get_model
from config import ModelConfig


class Test(object):
    """Inference class for testing TRSNet on various datasets"""
    
    def __init__(self, Dataset, path: str, backbone_type: str = 'pvt_b5', 
                 checkpoint: str = None, task: str = 'SOD'):
        """
        Initialize test instance
        
        Args:
            Dataset: Dataset module
            path: Path to test dataset
            backbone_type: 'pvt_b0' or 'pvt_b5'
            checkpoint: Path to model checkpoint
            task: Task type (SOD, SOC, COD, FPS)
        """
        self.task = task
        self.backbone_type = backbone_type
        
        # Dataset configuration
        self.cfg = Dataset.Config(datapath=path, snapshot=checkpoint, mode='test')
        self.data = Dataset.Data(self.cfg, 'TRSNet')
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=4)
        
        # Initialize model
        self.net = get_model(cfg=self.cfg, backbone_type=backbone_type, verbose=False)
        if checkpoint and os.path.exists(checkpoint):
            self.net.load_state_dict(torch.load(checkpoint), strict=False)
            print(f"Loaded checkpoint: {checkpoint}")
        self.net.train(False)
        self.net.cuda()
    
    def save(self):
        """Run inference and save predictions"""
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image = image.cuda().float()
                out = self.net(image)
                pred = torch.sigmoid(out[0, 0]).cpu().numpy() * 255
                
                # Determine output directory
                if self.task == "SOC":
                    head = f'util/evaltool/Prediction/TRSNet-{self.backbone_type.upper()}/SOC/{self.cfg.datapath.split("/")[-1]}'
                else:
                    head = f'util/evaltool/Prediction/TRSNet-{self.backbone_type.upper()}/{self.cfg.datapath.split("/")[-2]}'
                
                if not os.path.exists(head):
                    os.makedirs(head)
                    print(f"Created directory: {head}")
                
                cv2.imwrite(os.path.join(head, name[0] + '.png'), np.round(pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRSNet Inference')
    
    # Model configuration
    parser.add_argument('--backbone', default='pvt_b5', choices=['pvt_b0', 'pvt_b5'],
                       help='Backbone type: pvt_b0 (lightweight) or pvt_b5 (performance)')
    
    # Task configuration
    parser.add_argument('--task', default='SOD', choices=['SOD', 'SOC', 'COD', 'FPS'],
                       help='Task: SOD, SOC-Attr, COD, or FPS measurement')
    
    # Checkpoint
    parser.add_argument('--checkpoint', default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"TRSNet Inference")
    print(f"{'='*60}")
    print(f"Backbone: {args.backbone}")
    print(f"Task: {args.task}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    backbone = args.backbone
    task = args.task
    checkpoint = args.checkpoint
    
    # SOD datasets
    if task == "SOD":
        sod_datasets = [
            'datasets/ECSSD/Test',
            'datasets/PASCAL-S/Test',
            'datasets/DUTS/Test',
            'datasets/HKU-IS/Test',
            'datasets/DUT-OMRON/Test',
            'datasets/SOD/Test'
        ]
        for path in sod_datasets:
            if os.path.exists(path):
                print(f"Processing {path}...")
                t = Test(dataset, path, backbone_type=backbone, checkpoint=checkpoint, task=task)
                t.save()
            else:
                print(f"Skipping {path} (not found)")
    
    # SOC datasets
    elif task == "SOC":
        soc_datasets = [
            'datasets/SOC/SOC-AC', 'datasets/SOC/SOC-BO', 'datasets/SOC/SOC-CL',
            'datasets/SOC/SOC-HO', 'datasets/SOC/SOC-MB', 'datasets/SOC/SOC-OC',
            'datasets/SOC/SOC-OV', 'datasets/SOC/SOC-SC', 'datasets/SOC/SOC-SO'
        ]
        for path in soc_datasets:
            if os.path.exists(path):
                print(f"Processing {path}...")
                t = Test(dataset, path, backbone_type=backbone, checkpoint=checkpoint, task=task)
                t.save()
            else:
                print(f"Skipping {path} (not found)")
    
    # COD datasets
    elif task == "COD":
        cod_datasets = [
            'datasets/CHAMELEON/Test',
            'datasets/CAMO/Test',
            'datasets/COD10K/Test',
            'datasets/CPD1K/Test'
        ]
        for path in cod_datasets:
            if os.path.exists(path):
                print(f"Processing {path}...")
                t = Test(dataset, path, backbone_type=backbone, checkpoint=checkpoint, task=task)
                t.save()
            else:
                print(f"Skipping {path} (not found)")
    
    # FPS benchmark
    else:
        print("Running FPS benchmark...")
        inf_time = 0
        num_images = 0
        for path in ['datasets/SOD/Test/']:
            if os.path.exists(path):
                start_time = time.time()
                t = Test(dataset, path, backbone_type=backbone, checkpoint=checkpoint, task=task)
                t.save()
                end_time = time.time()
                inf_time += (end_time - start_time)
                num_images = len(t.loader)
        
        if num_images > 0:
            inf_per_image = inf_time / num_images
            fps = 1.0 / inf_per_image
            print(f"Time per image: {inf_per_image:.4f}s")
            print(f"FPS: {fps:.2f}")

