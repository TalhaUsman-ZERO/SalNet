# TRSNet: Lightweight Transformer-Driven Multi-Scale Trapezoidal Attention Network for Saliency Detection

Official implementation of TRSNet, published in **Engineering Applications of Artificial Intelligence (EAAI), 2025**.

**Paper:** [https://doi.org/10.1016/j.engappai.2025.110917](https://doi.org/10.1016/j.engappai.2025.110917)

---

## Overview

TRSNet is a scalable salient object detection (SOD) network built on a PVTv2 backbone. It introduces a novel **Trapezoidal Attention Module (TAM)** that assigns scale-appropriate attention operations across the feature hierarchy, combined with a progressive multi-scale decoder. The architecture supports two variants:

- **TRSNet** (PVT-B5): High-performance variant — 98.73M parameters, 105.84 GFLOPs
- **TRSNet-Lite** (PVT-B0): Lightweight deployment variant — 5.96M parameters, 9.44 GFLOPs, **126 FPS**

Both variants surpass 26 state-of-the-art SOD methods across 6 benchmark datasets.

---

## Architecture

TRSNet consists of four main components:

**1. PVTv2 Backbone**
Extracts multi-scale features F1–F4 at four hierarchical stages from a 384×384 input image.

**2. Contextual Feature Refinement Block (CFRB)**
Applied to all four feature stages. Uses parallel dilated convolutions (rates 3, 5, 7, 9) to capture rich contextual information at each scale before attention processing.

**3. Trapezoidal Attention Module (TAM)**
Assigns scale-appropriate operations across the feature hierarchy:
- **F1, F2** (low-level): processed by **ASCA** (Adaptive Spatial Coordinate Attention) to preserve fine-grained spatial detail
- **F3, F4** (high-level): processed by **CCG** (Compact Channel Gate) to recalibrate channel-wise semantic responses
- **F2, F3** (mid-level): further refined by **FAMHA** (Feature-Aware Multi-Head Attention) to model long-range dependencies — a deliberate design choice since mid-level transition features are routinely neglected in existing architectures

**4. Progressive Decoder**
Fuses multi-scale features step by step via upsampling and concatenation, preserving information across scales without loss.

---

## Results

Evaluated on six standard SOD benchmarks: ECSSD, HKU-IS, PASCAL-S, DUTS-TE, DUT-OMRON, and SOD.

| Variant | Params | GFLOPs | Speed |
|---|---|---|---|
| TRSNet (PVT-B5) | 98.73M | 105.84G | — |
| TRSNet-Lite (PVT-B0) | 5.96M | 9.44G | 126 FPS |

Both variants outperform 26 SOTA SOD methods. TRSNet-Lite achieves 126 FPS on a single GPU, making it suitable for real-time deployment.

---

## Installation

```bash
pip install torch torchvision timm opencv-python tensorboard tqdm
```

Download PVTv2 backbone weights and place in `checkpoint/Backbone/PVTv2/`:
- PVT-B0: https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth
- PVT-B5: https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth

---

## Training

```bash
# TRSNet (PVT-B5)
python train.py --backbone pvt_b5 --dataset datasets/DUTS/Train --batchsize 8 --epoch 100 --lr 0.05

# TRSNet-Lite (PVT-B0)
python train.py --backbone pvt_b0 --dataset datasets/DUTS/Train --batchsize 16 --epoch 100 --lr 0.05
```

---

## Testing

```bash
# Evaluate on SOD benchmarks
python test.py --backbone pvt_b5 --task SOD --checkpoint path/to/checkpoint.pth

# FPS benchmark
python test.py --backbone pvt_b0 --task FPS --checkpoint path/to/checkpoint.pth
```

---

## Supported Datasets

**Training:** DUTS-TR

**Testing:** ECSSD, HKU-IS, PASCAL-S, DUTS-TE, DUT-OMRON, SOD

---

## Project Structure

```
TRSNet/
├── TRSNet.py          # Main model architecture
├── modules.py         # CFRB, ASCA, CCG, FAMHA modules
├── pvtv2_encoder.py   # PVTv2 backbone
├── config.py          # Configuration system
├── get_model.py       # Model factory
├── train.py           # Training script
├── test.py            # Testing and inference
├── dataset.py         # Dataset handling
├── metrics.py         # Evaluation metrics (Fβ, MAE, Sα, Eξ)
├── checkpoint/        # Backbone weights and saved models
├── datasets/          # Dataset directory
└── Prediction/        # Output saliency maps
```

---

## Citation

```bibtex
@article{usman2025trsnet,
  title={Lightweight transformer-driven multi-scale trapezoidal attention network for saliency detection},
  author={Usman, Muhammad Talha and Khan, Habib and Rida, Imad and Koo, JaKeoung},
  journal={Engineering Applications of Artificial Intelligence},
  volume={155},
  pages={110917},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.engappai.2025.110917}
}
```
