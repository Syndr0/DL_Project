"""CLIP encoder for image retrieval.

Supported variants: 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'
Pooling: internal (CLIP's own projection head; pool_mode is ignored).

Install: pip install git+https://github.com/openai/CLIP.git
"""

import torch
import clip

SUPPORTED_VARIANTS = ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')


def build_encoder(variant: str = 'ViT-B/32', pool_mode: str = None):
    """Build a CLIP image encoder.

    Args:
        variant:   CLIP model variant.  One of SUPPORTED_VARIANTS.
        pool_mode: Ignored — CLIP uses its own internal projection.

    Returns:
        model:     The CLIP model (can be fine-tuned in-place).
        fwd:       Differentiable forward: Tensor(N,3,H,W) → Tensor(N,D)
                   un-normalised float32 features.
        feat_dim:  Output feature dimension.
        transform: Preprocessing transform (CLIP's own preprocess).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, transform = clip.load(variant, device=device)
    model.eval()
    feat_dim = model.visual.output_dim

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return model.encode_image(x).float()

    return model, fwd, feat_dim, transform
