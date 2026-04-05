"""
EdgeCrowdNet — MobileNetV3-Small backbone + Lite Attention decoder
with dual output heads (density map + uncertainty map).
Designed for edge deployment: 340K parameters, 15 ms GPU latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LiteAttention(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, channels, reduction_ratio=16):
        super(LiteAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EdgeCrowdNet(nn.Module):
    """
    MobileNetV3-Small backbone with Lite Attention decoder.
    Outputs:
        pred_mean    — density map  (B, 1, H, W)
        pred_log_var — log-variance map for uncertainty estimation (B, 1, H, W)
    """
    def __init__(self, pretrained=True):
        super(EdgeCrowdNet, self).__init__()

        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        self.backbone = mobilenet.features[:9]   # 48-channel output

        self.decoder = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LiteAttention(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LiteAttention(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            LiteAttention(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Density head
        self.head_mean = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # Uncertainty head
        self.head_log_var = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return self.head_mean(x), self.head_log_var(x)


def gaussian_nll_loss(pred_mean, pred_log_var, y_true):
    """Gaussian Negative Log-Likelihood loss for uncertainty-aware training."""
    pred_var = torch.exp(pred_log_var)
    precision = 1.0 / (pred_var + 1e-8)
    loss = 0.5 * (pred_log_var + (y_true - pred_mean) ** 2 * precision)
    return torch.mean(loss)


if __name__ == "__main__":
    model = EdgeCrowdNet(pretrained=False)
    x = torch.rand(1, 3, 256, 256)
    mean, log_var = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:       {x.shape}")
    print(f"Density map: {mean.shape}")
    print(f"Log-var map: {log_var.shape}")
    print(f"Params:      {params:,}")
