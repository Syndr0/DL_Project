"""ResNet encoder for image retrieval.

Supported variants : '34' (feat_dim=512), '50'/'101'/'152' (feat_dim=2048)
Supported pool_mode: 'gap' (Global Average Pooling)
                     'gem' (Generalized Mean Pooling)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

from baseline.utils.pooling import GeM

SUPPORTED_VARIANTS   = ('34', '50', '101', '152')
SUPPORTED_POOL_MODES = ('gap', 'gem')

_WEIGHTS = {
    '34':  tvm.ResNet34_Weights.DEFAULT,
    '50':  tvm.ResNet50_Weights.DEFAULT,
    '101': tvm.ResNet101_Weights.DEFAULT,
    '152': tvm.ResNet152_Weights.DEFAULT,
}

cnn_transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_encoder(variant: str = '50', pool_mode: str = 'gap'):
    """Build a ResNet image encoder.

    Args:
        variant:   '34', '50', '101', or '152'.
        pool_mode: 'gap' or 'gem'.

    Returns:
        model:     ResNet backbone (fc removed).
        fwd:       Differentiable forward: Tensor(N,3,224,224) → Tensor(N,D).
        feat_dim:  Feature dimension (512 for ResNet-34, 2048 otherwise).
        transform: Standard ImageNet preprocessing.
    """
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    model    = getattr(tvm, f'resnet{variant}')(weights=_WEIGHTS[variant])
    feat_dim = model.fc.in_features

    if pool_mode == 'gem':
        model.avgpool = GeM()
    model.fc = nn.Identity()
    model    = model.to(device).eval()

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return model(x).flatten(1)

    return model, fwd, feat_dim, cnn_transform
