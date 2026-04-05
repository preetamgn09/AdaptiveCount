"""
MCNN — Multi-Column Convolutional Neural Network for crowd counting.
Three parallel columns with large (9x9), medium (7x7), and small (5x5)
receptive fields, merged into a single density map.

Reference: Zhang et al., CVPR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        # Column 1 — large filters (9x9)
        self.column_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

        # Column 2 — medium filters (7x7)
        self.column_2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # Column 3 — small filters (5x5)
        self.column_3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Merge: 30 channels → 1 density map
        self.merge = nn.Sequential(
            nn.Conv2d(30, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.column_1(x)
        c2 = self.column_2(x)
        c3 = self.column_3(x)
        x = torch.cat((c1, c2, c3), dim=1)
        return self.merge(x)


if __name__ == "__main__":
    model = MCNN()
    x = torch.rand(1, 3, 256, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {params:,}")
