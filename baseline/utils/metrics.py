from typing import Sequence

import torch
import numpy as np
from tqdm import tqdm


def _topk_labels(query_embs, gallery_embs, gallery_labels, max_k):
    """Shared top-k retrieval for metric functions."""
    sim_matrix  = query_embs @ gallery_embs.T                       # (M, N)
    topk_idx    = sim_matrix.argsort(dim=1, descending=True)[:, :max_k]
    return torch.tensor(gallery_labels)[topk_idx]                   # (M, max_k)



def retrieve_top_k_cosine(
    query_embs:   torch.Tensor,
    gallery_embs: torch.Tensor,
    k:            int = 50,
):
    """Top-k by cosine similarity (dot product on L2-normalised vectors).

    Returns:
        indices : np.ndarray (M, k)
        scores  : np.ndarray (M, k)  cosine similarity in [-1, 1]
    """
    sim = query_embs @ gallery_embs.T          # (M, N)
    vals, idx = torch.topk(sim, k, dim=1)
    return idx.numpy(), vals.numpy()


def retrieve_top_k_l2(
    query_embs:   torch.Tensor,
    gallery_embs: torch.Tensor,
    k:            int = 50,
    chunk_size:   int = 256,
):
    """Top-k by L2 distance (ascending), chunked to support large galleries.

    Returns:
        indices   : np.ndarray (M, k)
        distances : np.ndarray (M, k)  L2 distance (lower = more similar)
    """
    M = query_embs.shape[0]
    topk_idx  = torch.zeros(M, k, dtype=torch.long)
    topk_dist = torch.zeros(M, k)

    for start in tqdm(range(0, M, chunk_size), desc='L2 retrieval', leave=False):
        end  = min(start + chunk_size, M)
        q    = query_embs[start:end]
        dist = torch.cdist(q, gallery_embs)    # (B, N)
        vals, idx = torch.topk(dist, k, dim=1, largest=False)
        topk_idx[start:end]  = idx
        topk_dist[start:end] = vals

    return topk_idx.numpy(), topk_dist.numpy()


def evaluate_l2(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10, 50),
) -> dict:
    """Recall@K using L2 distance ranking."""
    topk_idx, _ = retrieve_top_k_l2(query_embs, gallery_embs, max(k_list))
    topk_labels = torch.tensor(gallery_labels)[topk_idx]
    q_labels    = torch.tensor(query_labels).unsqueeze(1)
    return {
        k: (topk_labels[:, :k] == q_labels).any(dim=1).float().mean().item()
        for k in k_list
    }


def precision_at_k_l2(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10, 50),
) -> dict:
    """Precision@K using L2 distance ranking."""
    topk_idx, _ = retrieve_top_k_l2(query_embs, gallery_embs, max(k_list))
    topk_labels = torch.tensor(gallery_labels)[topk_idx]
    q_labels    = torch.tensor(query_labels).unsqueeze(1)
    return {
        k: (topk_labels[:, :k] == q_labels).float().mean().item()
        for k in k_list
    }


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


def _map_from_topk(topk_idx, query_labels, gallery_labels, k_list):
    """Shared mAP computation given top-k indices."""
    from collections import Counter
    topk_labels = torch.tensor(gallery_labels)[topk_idx]       # (M, max_k)
    q_labels    = torch.tensor(query_labels).unsqueeze(1)       # (M, 1)
    rel         = (topk_labels == q_labels).float()             # (M, max_k)

    gallery_counts = Counter(
        gallery_labels.tolist() if hasattr(gallery_labels, 'tolist') else list(gallery_labels)
    )
    n_rel = torch.tensor(
        [gallery_counts[int(q)] for q in query_labels], dtype=torch.float
    )

    results = {}
    for k in k_list:
        rel_k = rel[:, :k]
        cum   = rel_k.cumsum(dim=1)
        ranks = torch.arange(1, k + 1, dtype=torch.float).unsqueeze(0)
        prec  = cum / ranks
        denom = n_rel.clamp(max=k, min=1e-6)
        ap_k  = (prec * rel_k).sum(dim=1) / denom
        results[k] = ap_k.mean().item()
    return results


def mean_average_precision(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10),
) -> dict:
    """Mean Average Precision at K using cosine similarity.

    AP@K = sum_{i=1}^{K} [P(i) * rel(i)] / min(R, K)
    where R = number of relevant gallery items for that query.

    Returns:
        dict mapping each K → mAP@K  (float in [0, 1]).
    """
    topk_idx, _ = retrieve_top_k_cosine(query_embs, gallery_embs, max(k_list))
    return _map_from_topk(topk_idx, query_labels, gallery_labels, k_list)


def mean_average_precision_l2(
    query_embs:    torch.Tensor,
    gallery_embs:  torch.Tensor,
    query_labels:  np.ndarray,
    gallery_labels: np.ndarray,
    k_list:        Sequence[int] = (1, 5, 10),
    chunk_size:    int = 256,
) -> dict:
    """Mean Average Precision at K using L2 distance ranking.

    Returns:
        dict mapping each K → mAP@K  (float in [0, 1]).
    """
    topk_idx, _ = retrieve_top_k_l2(query_embs, gallery_embs, max(k_list), chunk_size)
    return _map_from_topk(topk_idx, query_labels, gallery_labels, k_list)


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
