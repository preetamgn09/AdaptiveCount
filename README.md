# AdaptiveCount: Efficiency-Aware Crowd Counting

Official implementation of:

> **"Efficiency-Aware Crowd Counting: A Framework for Adaptive Model Selection and Deployment"**  
> Preetam Giridhar Nadoni, Prital Rajkumar Nyamagoud, Rohini R, Nivedana J, Shobana T.S., Rashmi K.B.  
> Department of Information Science, B.M.S. College of Engineering, Bengaluru 560019, India  
> *Submitted to Scientific Reports*

---

## Overview

AdaptiveCount routes each crowd scene to the most appropriate counting model based on predicted density level and available hardware resources. A lightweight ResNet-18 density classifier (94.3% accuracy, <10 ms overhead) selects among four portfolio models:

| Model | Params | MAE (SHA) | Target Density | GPU FPS | RPi4 FPS |
|-------|--------|-----------|----------------|---------|----------|
| MCNN | 130 K | 236.08 | Low (0–100) | 83.3 | 8.3 |
| EdgeCrowdNet | 340 K | 297.64 | Medium (100–500) | 66.7 | 6.7 |
| EfficientCSRNet | 5.0 M | 195.39 | High (500–1,000) | 31.3 | 0.43 |
| CSRNet | 16.3 M | 145.73 | Extreme (1,000+) | 22.2 | OOM |
| **AdaptiveCount** | **Variable** | **187.6** | **All** | **52.1** | **4.2** |

AdaptiveCount is the **only evaluated method** achieving MAE ≤ 200 with CPU FPS ≥ 2.0 and memory ≤ 300 MB simultaneously (Pareto-frontier result).

---

## Repository Structure

```
AdaptiveCount/
├── models/
│   ├── mcnn.py               # Multi-Column CNN
│   ├── csrnet.py             # CSRNet (VGG-16 + dilated conv)
│   ├── efficient_csrnet.py   # EfficientNet-B0 + CSRNet backend
│   └── edge_crowd_net.py     # MobileNetV3 + uncertainty estimation
├── training/
│   ├── dataset.py            # ShanghaiTech dataset loader
│   ├── dal_loss.py           # Density-Adaptive Loss (DAL)
│   └── train.py              # Training script (all models)
├── evaluation/
│   └── evaluate.py           # Evaluation script
├── notebooks/
│   └── MLmodel.ipynb         # Full Colab training notebook
├── results/                  # Figures from the paper
├── weights/
│   └── README.md             # Download link for pretrained weights
├── requirements.txt
└── README.md
```

---

## Quick Start (Google Colab)

Open the full training notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmsce-ise/AdaptiveCount/blob/main/notebooks/MLmodel.ipynb)

---

## Install

```bash
git clone https://github.com/bmsce-ise/AdaptiveCount.git
cd AdaptiveCount
pip install -r requirements.txt
```

---

## Training

Mount Google Drive first in Colab (so weights survive disconnections):

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then train any model:

```bash
# MCNN (~45 min on Colab T4)
python training/train.py --model mcnn \
    --dataset_root /content/ShanghaiTech \
    --save_dir /content/drive/MyDrive/AdaptiveCount/weights

# CSRNet (~18 h — resume if session disconnects)
python training/train.py --model csrnet \
    --dataset_root /content/ShanghaiTech \
    --save_dir /content/drive/MyDrive/AdaptiveCount/weights \
    --resume /content/drive/MyDrive/AdaptiveCount/weights/csrnet_epoch5_mae167.34.pth
```

Available `--model` values: `mcnn`, `csrnet`, `efficient_csrnet`, `edge_crowd_net`

---

## Evaluation

```bash
python evaluation/evaluate.py \
    --model csrnet \
    --checkpoint /path/to/csrnet_best.pth \
    --dataset_root /content/ShanghaiTech \
    --part A
```

---

## Pretrained Weights

Pretrained model weights are available from the corresponding author upon request:  
📧 **shobanats.ise@bmsce.ac.in**

---

## Datasets

| Dataset | Link |
|---------|------|
| ShanghaiTech | https://github.com/desenzhou/ShanghaiTechDataset |
| UCF-QNRF | https://www.crcv.ucf.edu/data/ucf-qnrf/ |
| NWPU-Crowd | https://www.crowdbenchmark.com/nwpucrowd.html |

Place the ShanghaiTech dataset so the folder structure is:
```
ShanghaiTech/
├── part_A/
│   ├── train_data/
│   │   ├── images/          ← .jpg files
│   │   └── ground-truth/    ← GT_*.mat files
│   └── test_data/
│       ├── images/
│       └── ground-truth/
└── part_B/
    └── ...
```

---

## Key Results (ShanghaiTech Part A)

| Configuration | MAE | CPU FPS | Memory |
|---------------|-----|---------|--------|
| Always CSRNet | 145.73 | 0.43 | 1247 MB |
| Always MCNN | 236.08 | 6.67 | 28 MB |
| Random selection | 228.7 | 3.84 | — |
| Density classifier + MSE | 203.4 | 3.18 | — |
| **AdaptiveCount (DAL)** | **187.6** | **3.21** | **187 MB** |
| Oracle selection | 162.3 | 3.45 | — |

---

## Citation

If you use this code, please cite:

```
@article{nadoni2025adaptivecount,
  title   = {Efficiency-Aware Crowd Counting: A Framework for Adaptive 
             Model Selection and Deployment},
  author  = {Nadoni, Preetam Giridhar and Nyamagoud, Prital Rajkumar and 
             R, Rohini and J, Nivedana and T.S., Shobana and K.B., Rashmi},
  journal = {Scientific Reports},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see LICENSE file.
