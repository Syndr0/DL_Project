"""Retrieval engine — orchestrates the baseline Top-K retrieval pipeline.

Typical usage (from baseline_colab.ipynb):
    from retrieval_engine import (
        build_encoder, fine_tune, extract_all, save_outputs,
        save_metrics_json, submit
    )
    from baseline.utils.dataset import build_dataset_split
    from baseline.utils.metrics import evaluate, precision_at_k

    train_paths, train_labels, query_paths, query_labels, classes = \
        build_dataset_split('/content/data/animals/animals')

    model, encode, feat_dim, tfm, fwd = build_encoder('clip', 'ViT-B/32')
    gallery_embs = extract_all(train_paths, encode)
    query_embs   = extract_all(query_paths, encode)
    scores       = evaluate(query_embs, gallery_embs, query_labels, train_labels)
    save_outputs('outputs/clip_ViT-B-32_gap', ...)
"""

import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline.models.clip_retrieval         import build_encoder as _clip
from baseline.models.efficientnet_retrieval import build_encoder as _efficientnet
from baseline.models.resnet_retrieval        import build_encoder as _resnet
from baseline.models.googlenet_retrieval     import build_encoder as _googlenet
from baseline.utils.dataset import _ImgDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_BUILDER = {
    'clip':         _clip,
    'efficientnet': _efficientnet,
    'resnet':       _resnet,
    'googlenet':    _googlenet,
}


# ── Encoder ───────────────────────────────────────────────────────────────────

def build_encoder(backbone: str, variant: str, pool_mode: str = 'gap'):
    """Unified encoder dispatcher.

    Args:
        backbone:  'clip' | 'efficientnet' | 'resnet' | 'googlenet'
        variant:   backbone-specific string (e.g. 'ViT-B/32', '50', 'b3')
        pool_mode: 'gap' | 'gem'  (ignored for clip)

    Returns:
        model:      backbone nn.Module (modifiable, fine-tune in-place)
        encode_fn:  List[PIL.Image] → Tensor(N,D) L2-normalised, no_grad
        feat_dim:   int — feature vector length
        tfm:        preprocessing transform
        fwd:        Tensor(N,...) → Tensor(N,D) un-normalised, differentiable
                    (used by fine_tune — no backbone name needed downstream)
    """
    model, fwd, feat_dim, tfm = _BUILDER[backbone](variant, pool_mode)

    def encode_fn(imgs: list) -> torch.Tensor:
        x = torch.stack([tfm(i) for i in imgs]).to(device)
        with torch.no_grad():
            return F.normalize(fwd(x), dim=-1).cpu()

    return model, encode_fn, feat_dim, tfm, fwd


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def _fine_tune_resnet(model, fwd, feat_dim, num_classes, paths, labels, tfm,
                      epochs, lr, batch_size):
    """ResNet-specific fine-tune: freeze all → unfreeze layer4 → MLP head."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True

    head = nn.Sequential(
        nn.Linear(feat_dim, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    ).to(device)

    loader    = DataLoader(_ImgDataset(paths, labels, tfm),
                           batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train(); head.train()
        total = 0.0
        for imgs, lbls in tqdm(loader, desc=f'Epoch {ep+1}/{epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats = F.normalize(fwd(imgs), dim=-1)
            loss  = criterion(head(feats), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f'  Epoch {ep+1}/{epochs}  loss {total/len(loader):.4f}')

    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _fine_tune_last_layer(model, fwd, feat_dim, num_classes, paths, labels, tfm,
                           epochs, lr, batch_size):
    """Fine-tune only the last layer of the model (GoogLeNet strategy)."""
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze parameters added at the end (avgpool onwards)
    unfreeze = False
    for name, p in model.named_parameters():
        if 'inception5' in name or 'avgpool' in name:
            unfreeze = True
        if unfreeze:
            p.requires_grad = True

    head      = nn.Linear(feat_dim, num_classes).to(device)
    loader    = DataLoader(_ImgDataset(paths, labels, tfm),
                           batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    optimizer = optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters()))
        + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train(); head.train()
        total = 0.0
        for imgs, lbls in tqdm(loader, desc=f'Epoch {ep+1}/{epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats = F.normalize(fwd(imgs), dim=-1)
            loss  = criterion(head(feats), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f'  Epoch {ep+1}/{epochs}  loss {total/len(loader):.4f}')

    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def fine_tune(
    model,
    fwd,
    feat_dim:    int,
    num_classes: int,
    paths:       Sequence,
    labels:      Sequence,
    tfm,
    epochs:      int   = 10,
    lr:          float = 5e-5,
    batch_size:  int   = 32,
    backbone:    str   = 'generic',
    last_layer_only: bool = False,
):
    """Proxy classification fine-tuning. Updates model in-place.

    Adds a temporary Linear head, trains with CrossEntropyLoss, then
    removes the head. Because encode_fn closes over model, it
    automatically uses the fine-tuned weights after this call.

    Args:
        model:       backbone returned by build_encoder.
        fwd:         differentiable forward fn returned by build_encoder.
        feat_dim:    feature dimension.
        num_classes: number of classes (e.g. 90 for the animal dataset).
        paths:       training image paths.
        labels:      integer class labels.
        tfm:         preprocessing transform.
        epochs:      training epochs.
        lr:          Adam learning rate.
        batch_size:  DataLoader batch size.
        backbone:    'resnet' uses layer4-only strategy; 'googlenet' with
                     last_layer_only=True uses last-layer strategy; others
                     unfreeze all parameters.
        last_layer_only: For GoogLeNet: only train last inception block.
    """
    if backbone == 'resnet':
        _fine_tune_resnet(model, fwd, feat_dim, num_classes, paths, labels,
                          tfm, epochs, lr, batch_size)
        return

    if backbone == 'googlenet' and last_layer_only:
        _fine_tune_last_layer(model, fwd, feat_dim, num_classes, paths, labels,
                              tfm, epochs, lr, batch_size)
        return

    # Generic: unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True

    head      = nn.Linear(feat_dim, num_classes).to(device)
    loader    = DataLoader(
        _ImgDataset(paths, labels, tfm),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    optimizer = optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train(); head.train()
        total = 0.0
        for imgs, lbls in tqdm(loader, desc=f'Epoch {ep+1}/{epochs}', leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            feats = F.normalize(fwd(imgs), dim=-1)
            loss  = criterion(head(feats), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f'  Epoch {ep+1}/{epochs}  loss {total/len(loader):.4f}')

    for p in model.parameters():
        p.requires_grad = False
    model.eval()


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_all(paths: Sequence, encode_fn, batch_size: int = 64) -> torch.Tensor:
    """Extract L2-normalised embeddings for all images.

    Args:
        paths:      iterable of image file paths.
        encode_fn:  function returned by build_encoder.
        batch_size: images per forward pass.

    Returns:
        Tensor of shape (N, D).
    """
    chunks = []
    paths  = list(paths)
    for i in tqdm(range(0, len(paths), batch_size), desc='Extracting embeddings'):
        imgs = [Image.open(p).convert('RGB') for p in paths[i:i + batch_size]]
        chunks.append(encode_fn(imgs))
    return torch.cat(chunks)


# ── Save metrics JSON ─────────────────────────────────────────────────────────

def save_metrics_json(
    model_name:   str,
    recall_scores: dict,
    precision_scores: dict = None,
    pooling_type: str = 'gap',
    extra:        dict = None,
    out_dir:      str = 'results_animals',
):
    """Save evaluation metrics to a timestamped JSON file.

    Args:
        model_name:       Human-readable model identifier (e.g. 'clip_ViT-B32').
        recall_scores:    Dict from evaluate() — {K: Recall@K}.
        precision_scores: Dict from precision_at_k() — {K: Precision@K}.
        pooling_type:     'gap' or 'gem'.
        extra:            Any additional key-value pairs to include.
        out_dir:          Directory to write JSON files (created if missing).

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts   = time.strftime('%Y%m%d_%H%M%S')
    name = model_name.replace('/', '-').replace(' ', '_')
    path = out_dir / f'{name}_metrics_{ts}.json'

    data = {
        'model':        model_name,
        'pooling_type': pooling_type,
        'timestamp':    ts,
        'recall':       {f'@{k}': round(v, 4) for k, v in recall_scores.items()},
    }
    if precision_scores:
        data['precision'] = {f'@{k}': round(v, 4) for k, v in precision_scores.items()}
    if extra:
        data.update(extra)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f'Metrics saved → {path}')
    return path


# ── Submission helpers ────────────────────────────────────────────────────────

def build_submission(
    query_paths:    Sequence,
    gallery_paths:  Sequence,
    topk_indices:   np.ndarray,
) -> dict:
    """Build a submission dict in the evaluation server format.

    Format: {query_filename: [retrieved_filename_1, ..., retrieved_filename_k]}

    Args:
        query_paths:   list of query image paths.
        gallery_paths: list of gallery image paths.
        topk_indices:  (N_query, k) array of gallery indices.

    Returns:
        dict suitable for json.dump() or passing to submit().
    """
    return {
        Path(q).name: [Path(gallery_paths[idx]).name for idx in row]
        for q, row in zip(query_paths, topk_indices)
    }


def submit(
    query_paths:    Sequence,
    gallery_paths:  Sequence,
    topk_indices:   np.ndarray,
    url:            str = 'http://65.108.245.177:3001/retrieval/',
    timeout:        int = 30,
):
    """Submit retrieval results to the evaluation server.

    Args:
        query_paths:   list of query image paths.
        gallery_paths: list of gallery image paths.
        topk_indices:  (N_query, k) array of gallery indices.
        url:           evaluation server endpoint.
        timeout:       request timeout in seconds.

    Returns:
        Server response dict.
    """
    payload  = build_submission(query_paths, gallery_paths, topk_indices)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    print('Server response:', result)
    return result


# ── Save outputs for Part 2 ───────────────────────────────────────────────────

def save_outputs(
    out_dir,
    gallery_embs:   torch.Tensor,
    query_embs:     torch.Tensor,
    gallery_paths:  Sequence,
    query_paths:    Sequence,
    gallery_labels: np.ndarray,
    query_labels:   np.ndarray,
    classes:        list,
    topk_k:         int  = 20,
    eval_scores:    dict = None,
    precision_scores: dict = None,
    metadata:       dict = None,
):
    """Save all retrieval artifacts consumed by Part 2 (CNN reranking).

    Computes Top-K internally from the embeddings, then writes:
        gallery_embeddings.pt   (N_gallery, D)  L2-normalised
        query_embeddings.pt     (N_query,   D)
        gallery_labels.npy      (N_gallery,)
        query_labels.npy        (N_query,)
        gallery_paths.json      list[str]
        query_paths.json        list[str]
        topk_indices.npy        (N_query, topk_k)  gallery indices
        topk_scores.npy         (N_query, topk_k)  cosine similarities
        classes.json            {class_name: int_idx}
        metadata.json           encoder config + Recall scores

    Part 2 usage pattern:
        for i, q_path in enumerate(query_paths):
            for j, g_idx in enumerate(topk_indices[i]):
                pair_label = int(query_labels[i] == gallery_labels[g_idx])
                # construct 6-channel input: concat(query_img, gallery_img[g_idx])

    Args:
        out_dir:          directory to write artifacts (created if missing).
        gallery_embs:     (N, D) gallery embeddings.
        query_embs:       (M, D) query embeddings.
        gallery_paths:    gallery image paths.
        query_paths:      query image paths.
        gallery_labels:   integer class labels for gallery.
        query_labels:     integer class labels for query.
        classes:          list of class name strings; index == label.
        topk_k:           how many candidates to store per query (default 20).
        eval_scores:      dict from evaluate() e.g. {1: 0.72, 5: 0.91, 10: 0.95}.
        precision_scores: dict from precision_at_k().
        metadata:         extra key-value pairs merged into metadata.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Top-K ────────────────────────────────────────────────────────────────
    sim_matrix   = query_embs @ gallery_embs.T                   # (M, N)
    topk_vals, topk_idx = torch.topk(sim_matrix, topk_k, dim=1)

    # ── Tensors ───────────────────────────────────────────────────────────────
    torch.save(gallery_embs,  out_dir / 'gallery_embeddings.pt')
    torch.save(query_embs,    out_dir / 'query_embeddings.pt')

    # ── NumPy arrays ──────────────────────────────────────────────────────────
    np.save(out_dir / 'gallery_labels.npy',  gallery_labels)
    np.save(out_dir / 'query_labels.npy',    query_labels)
    np.save(out_dir / 'topk_indices.npy',    topk_idx.numpy())
    np.save(out_dir / 'topk_scores.npy',     topk_vals.numpy())

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(out_dir / 'gallery_paths.json', 'w') as f:
        json.dump([str(p) for p in gallery_paths], f)
    with open(out_dir / 'query_paths.json', 'w') as f:
        json.dump([str(p) for p in query_paths], f)
    with open(out_dir / 'classes.json', 'w') as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, indent=2)

    meta = {
        'n_gallery':  int(gallery_embs.shape[0]),
        'n_query':    int(query_embs.shape[0]),
        'feat_dim':   int(gallery_embs.shape[1]),
        'topk_k':     topk_k,
        'n_classes':  len(classes),
    }
    if eval_scores:
        meta['recall'] = {f'@{k}': round(v, 4) for k, v in eval_scores.items()}
    if precision_scores:
        meta['precision'] = {f'@{k}': round(v, 4) for k, v in precision_scores.items()}
    if metadata:
        meta.update(metadata)

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Outputs saved to {out_dir}/')
    print(f'  gallery_embeddings : {gallery_embs.shape}')
    print(f'  query_embeddings   : {query_embs.shape}')
    print(f'  topk_indices       : {topk_idx.shape}  (Top-{topk_k})')
    if eval_scores:
        for k, v in sorted(eval_scores.items()):
            print(f'  Recall@{k:2d}          : {v:.4f}')
    if precision_scores:
        for k, v in sorted(precision_scores.items()):
            print(f'  Precision@{k:2d}       : {v:.4f}')
