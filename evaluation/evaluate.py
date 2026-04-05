"""
evaluate.py — Evaluate a trained model on ShanghaiTech test set.

Usage:
    python evaluate.py --model mcnn \
                       --checkpoint /path/to/mcnn_best.pth \
                       --dataset_root /content/ShanghaiTech \
                       --part A
"""

import os, argparse, time
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.mcnn             import MCNN
from models.csrnet           import CSRNet
from models.efficient_csrnet import EfficientCSRNet
from models.edge_crowd_net   import EdgeCrowdNet
from training.dataset        import ShanghaiTechDataset, ShanghaiTechDatasetAdaptive

MODEL_MAP = {
    'mcnn':             (MCNN,             ShanghaiTechDataset,         256),
    'edge_crowd_net':   (EdgeCrowdNet,     ShanghaiTechDataset,         256),
    'csrnet':           (CSRNet,           ShanghaiTechDatasetAdaptive, 512),
    'efficient_csrnet': (EfficientCSRNet,  ShanghaiTechDatasetAdaptive, 512),
}


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ModelCls, DatasetCls, img_size = MODEL_MAP[args.model]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = DatasetCls(args.dataset_root, args.part, 'test',
                         transform, img_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load model
    model = ModelCls().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    total_mae, total_mse = 0.0, 0.0
    latencies = []

    with torch.no_grad():
        for images, _, counts in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            counts = torch.FloatTensor(list(counts)).to(device)

            t0 = time.perf_counter()
            if args.model == 'edge_crowd_net':
                pred, _ = model(images)
            else:
                pred = model(images)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

            pred_counts = pred.sum(dim=[1, 2, 3])
            total_mae  += torch.abs(pred_counts - counts).sum().item()
            total_mse  += ((pred_counts - counts) ** 2).sum().item()

    n = len(dataset)
    mae  = total_mae / n
    rmse = (total_mse / n) ** 0.5
    fps  = 1000.0 / np.median(latencies)

    print(f"\n{'='*40}")
    print(f"Model : {args.model.upper()}")
    print(f"Part  : ShanghaiTech Part {args.part}")
    print(f"Images: {n}")
    print(f"{'─'*40}")
    print(f"MAE   : {mae:.2f}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"Median latency: {np.median(latencies):.1f} ms  ({fps:.1f} FPS)")
    print(f"{'='*40}\n")

    return mae, rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        required=True, choices=list(MODEL_MAP))
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--part',         default='A', choices=['A', 'B'])
    evaluate(parser.parse_args())
