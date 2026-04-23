"""ResNet-50 with projection head for SOP metric learning (contrastive fine-tuning)."""
import torch.nn as nn
from torchvision import models, transforms


_EVAL_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_TRAIN_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ResNetSOP(nn.Module):
    """ResNet-50 backbone + 2-layer projection head → embed_dim-d L2-normalised vector."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


def build_encoder(variant: str = '50', pool_mode: str = 'gap', embed_dim: int = 128):
    """Return (model, fwd, feat_dim, eval_tfm) for retrieval_engine.build_encoder.

    Args:
        variant:   ignored (always ResNet-50).
        pool_mode: ignored (projection head replaces pooling).
        embed_dim: output embedding dimension (default 128).
    """
    model = ResNetSOP(embed_dim=embed_dim)

    def fwd(x):
        return model(x)

    return model, fwd, embed_dim, _EVAL_TFM
