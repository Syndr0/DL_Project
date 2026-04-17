"""EfficientNet encoder for image retrieval (timm backend).

Supported variants : 'b0' (feat_dim=1280, input 224×224)
                     'b3' (feat_dim=1536, input 300×300)
Supported pool_mode: 'gap' (Global Average Pooling)
                     'gem' (Generalized Mean Pooling)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import timm

from baseline.utils.pooling import GeM

SUPPORTED_VARIANTS   = ('b0', 'b3')
SUPPORTED_POOL_MODES = ('gap', 'gem')

_INPUT_SIZE = {'b0': 224, 'b3': 300}
_FEAT_DIM   = {'b0': 1280, 'b3': 1536}


def _make_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(input_size + 32),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_encoder(variant: str = 'b3', pool_mode: str = 'gap'):
    """Build an EfficientNet image encoder.

    Args:
        variant:   'b0' or 'b3'.
        pool_mode: 'gap' or 'gem'.

    Returns:
        model:     EfficientNet backbone module (fine-tune in-place).
        fwd:       Differentiable forward: Tensor(N,3,H,W) → Tensor(N,D).
        feat_dim:  Feature dimension (1280 for b0, 1536 for b3).
        transform: Preprocessing transform (size matches variant).
    """
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_sz  = _INPUT_SIZE[variant]
    feat_dim  = _FEAT_DIM[variant]
    transform = _make_transform(input_sz)

    # timm with num_classes=0 removes the classifier and returns pooled features
    model = timm.create_model(f'efficientnet_{variant}', pretrained=True, num_classes=0)
    model = model.to(device).eval()

    if pool_mode == 'gem':
        gem_pool = GeM().to(device)

        def fwd(x: torch.Tensor) -> torch.Tensor:
            feat_map = model.forward_features(x)   # (N, C, H, W)
            return gem_pool(feat_map)              # (N, D)
    else:
        def fwd(x: torch.Tensor) -> torch.Tensor:
            return model(x)                        # (N, D) — timm GAP output

    return model, fwd, feat_dim, transform
