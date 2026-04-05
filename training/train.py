"""
train.py — Train any of the four AdaptiveCount portfolio models.

Usage (Google Colab or local):
    python train.py --model mcnn --dataset_root /content/ShanghaiTech --part A
    python train.py --model csrnet --dataset_root /content/ShanghaiTech --part A
    python train.py --model efficient_csrnet --dataset_root /content/ShanghaiTech
    python train.py --model edge_crowd_net --dataset_root /content/ShanghaiTech

Weights are saved to --save_dir after every epoch that improves validation MAE.
Mount Google Drive first in Colab and set --save_dir to a Drive path so weights
survive session disconnections.
"""

import os
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# ── Local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.mcnn           import MCNN
from models.csrnet         import CSRNet
from models.efficient_csrnet import EfficientCSRNet
from models.edge_crowd_net import EdgeCrowdNet, gaussian_nll_loss
from training.dataset      import ShanghaiTechDataset, ShanghaiTechDatasetAdaptive
from training.dal_loss     import density_adaptive_loss


# ── Hyperparameters (from paper Table 7) ───────────────────────────────────────
CONFIGS = {
    'mcnn': dict(
        lr=1e-4, epochs=30, img_size=256,
        dataset_cls=ShanghaiTechDataset,
    ),
    'edge_crowd_net': dict(
        lr=1e-4, epochs=30, img_size=256,
        dataset_cls=ShanghaiTechDataset,
    ),
    'csrnet': dict(
        lr=1e-5, epochs=50, img_size=512,
        dataset_cls=ShanghaiTechDatasetAdaptive,
    ),
    'efficient_csrnet': dict(
        lr=1e-5, epochs=50, img_size=512,
        dataset_cls=ShanghaiTechDatasetAdaptive,
    ),
}

MODEL_MAP = {
    'mcnn':             MCNN,
    'csrnet':           CSRNet,
    'efficient_csrnet': EfficientCSRNet,
    'edge_crowd_net':   EdgeCrowdNet,
}


# ── Training / validation loops ────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, model_name):
    model.train()
    total_loss, total_mae, total_mse = 0.0, 0.0, 0.0

    for images, density_maps, counts in tqdm(loader, desc='Train', leave=False):
        images      = images.to(device)
        density_maps = density_maps.to(device)
        counts      = torch.FloatTensor(list(counts)).to(device)

        optimizer.zero_grad()

        if model_name == 'edge_crowd_net':
            pred_mean, pred_log_var = model(images)
            loss = gaussian_nll_loss(pred_mean, pred_log_var, density_maps)
            pred = pred_mean
        else:
            pred = model(images)
            loss = density_adaptive_loss(pred, density_maps)

        loss.backward()
        optimizer.step()

        if pred.shape != density_maps.shape:
            pred = F.interpolate(pred, size=density_maps.shape[2:],
                                 mode='bilinear', align_corners=False)

        pred_counts = pred.sum(dim=[1, 2, 3])
        mae = torch.abs(pred_counts - counts).sum().item()
        mse = ((pred_counts - counts) ** 2).sum().item()

        total_loss += loss.item()
        total_mae  += mae
        total_mse  += mse

    n = len(loader.dataset)
    return total_loss / len(loader), total_mae / n, (total_mse / n) ** 0.5


def validate(model, loader, device, model_name):
    model.eval()
    total_mae, total_mse = 0.0, 0.0

    with torch.no_grad():
        for images, _, counts in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            counts = torch.FloatTensor(list(counts)).to(device)

            if model_name == 'edge_crowd_net':
                pred, _ = model(images)
            else:
                pred = model(images)

            pred_counts = pred.sum(dim=[1, 2, 3])
            total_mae += torch.abs(pred_counts - counts).sum().item()
            total_mse += ((pred_counts - counts) ** 2).sum().item()

    n = len(loader.dataset)
    return total_mae / n, (total_mse / n) ** 0.5


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = CONFIGS[args.model]
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────
    DatasetCls = cfg['dataset_cls']
    train_ds = DatasetCls(args.dataset_root, args.part, 'train',
                          transform, cfg['img_size'])
    val_ds   = DatasetCls(args.dataset_root, args.part, 'test',
                          transform, cfg['img_size'])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────
    model = MODEL_MAP[args.model]().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr']
    )

    # ── Resume from checkpoint if provided ────────────────────────
    start_epoch = 0
    best_mae = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f"Resumed from {args.resume} (epoch {start_epoch}, MAE {best_mae:.2f})")

    # ── Training loop ─────────────────────────────────────────────
    print(f"\nTraining {args.model.upper()} for {cfg['epochs']} epochs\n")

    for epoch in range(start_epoch, cfg['epochs']):
        t0 = time.time()
        tr_loss, tr_mae, tr_rmse = train_epoch(
            model, train_loader, optimizer, device, args.model)
        val_mae, val_rmse = validate(
            model, val_loader, device, args.model)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Loss {tr_loss:.4f} | Train MAE {tr_mae:.2f} | "
              f"Val MAE {val_mae:.2f} | RMSE {val_rmse:.2f} | "
              f"{elapsed:.0f}s")

        # Save every epoch to Drive
        ckpt_path = os.path.join(
            args.save_dir,
            f"{args.model}_epoch{epoch+1}_mae{val_mae:.2f}.pth"
        )
        torch.save({
            'epoch':    epoch,
            'model':    model.state_dict(),
            'best_mae': best_mae,
            'val_mae':  val_mae,
        }, ckpt_path)

        # Save best separately
        if val_mae < best_mae:
            best_mae = val_mae
            best_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'best_mae': best_mae,
                'val_mae':  val_mae,
            }, best_path)
            print(f"  *** Best model saved: MAE {best_mae:.2f} → {best_path}")

    print(f"\nTraining complete. Best MAE: {best_mae:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AdaptiveCount models')
    parser.add_argument('--model', choices=list(CONFIGS.keys()),
                        required=True, help='Which model to train')
    parser.add_argument('--dataset_root', required=True,
                        help='Path to ShanghaiTech root folder')
    parser.add_argument('--part', default='A', choices=['A', 'B'],
                        help='ShanghaiTech part (default: A)')
    parser.add_argument('--save_dir',
                        default='/content/drive/MyDrive/AdaptiveCount/weights',
                        help='Directory to save checkpoints (use Drive path in Colab)')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
