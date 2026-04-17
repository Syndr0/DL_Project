"""GoogLeNet encoder for image retrieval.

Supported variants : 'base' (feat_dim=1024)
Supported pool_mode: 'gap' (Global Average Pooling)
                     'gem' (Generalized Mean Pooling)

Note: aux_logits=False so forward() returns a plain tensor in both
train and eval mode, avoiding the GoogLeNetOutputs named-tuple.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

from baseline.utils.pooling import GeM

SUPPORTED_VARIANTS   = ('base',)
SUPPORTED_POOL_MODES = ('gap', 'gem')

cnn_transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_encoder(variant: str = 'base', pool_mode: str = 'gap'):
    """Build a GoogLeNet image encoder.

    Args:
        variant:   Only 'base' is supported.
        pool_mode: 'gap' or 'gem'.

    Returns:
        model:     GoogLeNet backbone (fc removed, aux_logits disabled).
        fwd:       Differentiable forward: Tensor(N,3,224,224) → Tensor(N,D).
        feat_dim:  1024.
        transform: Standard ImageNet preprocessing.
    """
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load with default aux_logits=True, then disable to get plain tensor output
    model    = tvm.googlenet(weights=tvm.GoogLeNet_Weights.DEFAULT)
    model.aux_logits = False
    feat_dim = model.fc.in_features   # 1024

    model.avgpool = GeM() if pool_mode == 'gem' else nn.AdaptiveAvgPool2d((1, 1))
    model.fc      = nn.Identity()
    model         = model.to(device).eval()

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return model(x).flatten(1)

    return model, fwd, feat_dim, cnn_transform
