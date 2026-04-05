# Pretrained Weights

Pretrained model weights for all four portfolio models are available
from the corresponding author upon reasonable request.

Contact: shobanats.ise@bmsce.ac.in

## Expected weight files

| File | Model | ShanghaiTech Part A MAE |
|------|-------|------------------------|
| `mcnn_best.pth` | MCNN | 236.08 |
| `csrnet_best.pth` | CSRNet | 145.73 |
| `efficient_csrnet_best.pth` | EfficientCSRNet | 195.39 |
| `edgecrowdnet_best.pth` | EdgeCrowdNet | 297.64 |

## Loading a checkpoint

```python
import torch
from models.csrnet import CSRNet

model = CSRNet()
ckpt  = torch.load('weights/csrnet_best.pth', map_location='cpu')
state = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state)
model.eval()
```
