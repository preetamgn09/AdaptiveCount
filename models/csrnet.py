"""
CSRNet — Congested Scene Recognition Network.
VGG-16 backend (first 10 layers) + dilated convolutional frontend.

Reference: Li et al., CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3,
                             padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # Backend: VGG-16 layers 1-10 (1/8 downsampling)
        self.backend_feat = [64, 64, 'M', 128, 128, 'M',
                             256, 256, 256, 'M', 512, 512, 512]
        self.backend = make_layers(self.backend_feat, batch_norm=True)

        # Frontend: dilated convolutions
        self.frontend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, batch_norm=True,
                                    in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights()
            self._load_vgg_weights()

    def forward(self, x):
        x = self.backend(x)
        x = self.frontend(x)
        x = self.output_layer(x)
        return F.relu(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_vgg_weights(self):
        try:
            vgg16 = models.vgg16_bn(pretrained=True)
            backend_dict = self.backend.state_dict()
            vgg_dict = vgg16.features.state_dict()
            pretrained = {k: v for k, v in vgg_dict.items() if k in backend_dict}
            backend_dict.update(pretrained)
            self.backend.load_state_dict(backend_dict)
            print(f"Loaded {len(pretrained)} VGG-16 layers.")
        except Exception as e:
            print(f"Could not load VGG weights: {e}")


if __name__ == "__main__":
    model = CSRNet()
    x = torch.rand(1, 3, 512, 512)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {params:,}")
