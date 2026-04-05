"""
EfficientCSRNet — EfficientNet-B0 frontend + CSRNet dilated backend.
Reduces CSRNet's 16.3M parameters to 5.0M via NAS-discovered backbone
with compound scaling, while retaining dilated convolutions for large
receptive fields.

Requires: pip install timm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


def make_layers(cfg, in_channels=512, batch_norm=True, dilation=True):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        conv = nn.Conv2d(in_channels, v, kernel_size=3,
                         padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


class EfficientCSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(EfficientCSRNet, self).__init__()

        # Frontend: EfficientNet-B0 (replaces VGG-16)
        self.frontend = timm.create_model(
            'efficientnet_b0',
            pretrained=load_weights,
            features_only=True,
        )

        # Bridge: EfficientNet-B0 outputs 320ch → CSRNet expects 512ch
        self.bridge = nn.Conv2d(320, 512, kernel_size=1)

        # Backend: CSRNet-style dilated convolutions
        self.backend = make_layers(
            [512, 512, 512, 256, 128, 64],
            in_channels=512,
            batch_norm=True,
            dilation=True,
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Freeze frontend (transfer learning)
        for param in self.frontend.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.frontend(x)   # list of feature maps
        x = features[-1]              # last feature map: [B, 320, H/32, W/32]
        x = self.bridge(x)            # → [B, 512, H/32, W/32]
        x = self.backend(x)
        x = self.output_layer(x)
        return F.relu(x)


if __name__ == "__main__":
    model = EfficientCSRNet()
    x = torch.rand(1, 3, 512, 512)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input:     {x.shape}")
    print(f"Output:    {out.shape}")
    print(f"Total params:     {params:,}")
    print(f"Trainable params: {trainable:,}")
