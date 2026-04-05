"""
ShanghaiTech Dataset Loader.
Handles Part A (dense, mean=501) and Part B (sparse, mean=123).
Ground-truth .mat files use the 'ground-truth' folder convention.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


class ShanghaiTechDataset(Dataset):
    """
    Args:
        root_dir   : path to ShanghaiTech root (contains part_A/, part_B/)
        part       : 'A' or 'B'
        phase      : 'train' or 'test'
        transform  : torchvision transforms applied to the image
        img_size   : square resize target for image and density map (pixels)
    """

    def __init__(self, root_dir, part='A', phase='train',
                 transform=None, img_size=256):
        self.root_dir = root_dir
        self.part = part
        self.phase = phase
        self.transform = transform
        self.img_size = img_size

        self.img_dir = os.path.join(root_dir, f'part_{part}',
                                    f'{phase}_data', 'images')
        self.gt_dir = os.path.join(root_dir, f'part_{part}',
                                   f'{phase}_data', 'ground-truth')

        self.img_files = sorted(
            [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        )
        print(f"Loaded {len(self.img_files)} images — "
              f"Part {part} / {phase}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # ── Image ──────────────────────────────────────────────────
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # ── Ground truth ────────────────────────────────────────────
        gt_name = img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, f'GT_{gt_name}')
        points = sio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        count = len(points)

        # ── Transform image ─────────────────────────────────────────
        if self.transform:
            image = self.transform(image)

        # ── Density map ─────────────────────────────────────────────
        density_map = self._make_density_map(orig_w, orig_h, points, count)

        return image, density_map, count

    def _make_density_map(self, orig_w, orig_h, points, count):
        """
        Create a density map at (img_size × img_size).
        Points are scaled proportionally; a Gaussian filter is applied;
        the map is re-normalised so its sum equals the true head count.
        """
        h = w = self.img_size
        density = np.zeros((h, w), dtype=np.float32)

        if count == 0:
            return torch.from_numpy(density).unsqueeze(0).float()

        scale_w = w / orig_w
        scale_h = h / orig_h

        for pt in points:
            x = int(np.clip(pt[0] * scale_w, 0, w - 1))
            y = int(np.clip(pt[1] * scale_h, 0, h - 1))
            density[y, x] += 1.0

        density = gaussian_filter(density, sigma=2.0)

        # Re-normalise: sum must equal count after smoothing
        if density.sum() > 0:
            density = density * (count / density.sum())

        return torch.from_numpy(density).unsqueeze(0).float()


class ShanghaiTechDatasetAdaptive(ShanghaiTechDataset):
    """
    Variant used for CSRNet / EfficientCSRNet (img_size=512).
    Density map is at 1/8 resolution (CSRNet output stride = 8).
    Uses an adaptive Gaussian kernel following the CSRNet paper:
    sigma_i = beta * mean(d_i, k=3 nearest neighbours).
    """

    BETA = 0.3
    K_NEIGHBOURS = 3

    def _make_density_map(self, orig_w, orig_h, points, count):
        out_h = self.img_size // 8
        out_w = self.img_size // 8
        density = np.zeros((out_h, out_w), dtype=np.float32)

        if count == 0:
            return torch.from_numpy(density).unsqueeze(0).float()

        scale_w = out_w / orig_w
        scale_h = out_h / orig_h

        scaled = []
        for pt in points:
            x = pt[0] * scale_w
            y = pt[1] * scale_h
            if 0 <= x < out_w and 0 <= y < out_h:
                scaled.append([x, y])

        if not scaled:
            return torch.from_numpy(density).unsqueeze(0).float()

        scaled = np.array(scaled)
        dists = cdist(scaled, scaled, 'euclidean')

        for i, (x, y) in enumerate(scaled):
            # k nearest neighbours (excluding self)
            nn_dists = np.sort(dists[i])[1: self.K_NEIGHBOURS + 1]
            sigma = self.BETA * nn_dists.mean() if len(nn_dists) > 0 else 1.0
            sigma = max(sigma, 1.0)

            xi, yi = int(x), int(y)
            r = int(3 * sigma)
            x0, x1 = max(0, xi - r), min(out_w, xi + r + 1)
            y0, y1 = max(0, yi - r), min(out_h, yi + r + 1)

            for row in range(y0, y1):
                for col in range(x0, x1):
                    density[row, col] += np.exp(
                        -((row - yi) ** 2 + (col - xi) ** 2) /
                        (2 * sigma ** 2)
                    )

        if density.sum() > 0:
            density = density * (count / density.sum())

        return torch.from_numpy(density).unsqueeze(0).float()
