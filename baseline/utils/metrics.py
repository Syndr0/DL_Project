from typing import Sequence

import torch
import numpy as np


def _topk_labels(query_embs, gallery_embs, gallery_labels, max_k):
    """Shared top-k retrieval for metric functions."""
    sim_matrix  = query_embs @ gallery_embs.T                       # (M, N)
    topk_idx    = sim_matrix.argsort(dim=1, descending=True)[:, :max_k]
    return torch.tensor(gallery_labels)[topk_idx]                   # (M, max_k)


def evaluate(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10),
) -> dict:
    """Compute Recall@K for each K in k_list.

    Fully vectorised: builds the (M x N) similarity matrix once,
    then checks top-k labels in parallel — no Python loop over queries.

    Both embedding tensors must be L2-normalised so that
    dot product == cosine similarity.

    Args:
        query_embs:    (M, D) L2-normalised query embeddings.
        gallery_embs:  (N, D) L2-normalised gallery embeddings.
        query_labels:  (M,)   integer class labels.
        gallery_labels:(N,)   integer class labels.
        k_list:        Which K values to evaluate.

    Returns:
        dict mapping each K → Recall@K  (float in [0, 1]).
    """
    topk_labels = _topk_labels(query_embs, gallery_embs, gallery_labels, max(k_list))
    q_labels    = torch.tensor(query_labels).unsqueeze(1)           # (M, 1)

    return {
        k: (topk_labels[:, :k] == q_labels).any(dim=1).float().mean().item()
        for k in k_list
    }


def precision_at_k(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10),
) -> dict:
    """Compute Precision@K — average fraction of correct matches in top-k.

    Unlike Recall@K (any correct?), Precision@K measures how many of the
    k retrieved images share the query's class label.

    Args:
        query_embs:    (M, D) L2-normalised query embeddings.
        gallery_embs:  (N, D) L2-normalised gallery embeddings.
        query_labels:  (M,)   integer class labels.
        gallery_labels:(N,)   integer class labels.
        k_list:        Which K values to evaluate.

    Returns:
        dict mapping each K → Precision@K  (float in [0, 1]).
    """
    topk_labels = _topk_labels(query_embs, gallery_embs, gallery_labels, max(k_list))
    q_labels    = torch.tensor(query_labels).unsqueeze(1)           # (M, 1)

    return {
        k: (topk_labels[:, :k] == q_labels).float().mean().item()
        for k in k_list
    }


def top_k_accuracy(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k:             int = 5,
) -> float:
    """Proportion of queries with at least one correct match in top-k.

    Equivalent to Recall@K for a single k value.
    """
    topk_labels = _topk_labels(query_embs, gallery_embs, gallery_labels, k)
    q_labels    = torch.tensor(query_labels).unsqueeze(1)
    return (topk_labels == q_labels).any(dim=1).float().mean().item()
