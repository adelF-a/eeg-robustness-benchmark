"""Explanation-stability metrics between two channel-importance vectors.

Importance vectors must share the same channel ordering before comparison — comparing
misaligned vectors is meaningless, so callers must guarantee alignment (we assert equal
length here as a minimal guard).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def top_k_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-k channels by importance magnitude."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    k = min(k, len(a))
    ta = set(np.argsort(-np.abs(a))[:k].tolist())
    tb = set(np.argsort(-np.abs(b))[:k].tolist())
    union = ta | tb
    return float(len(ta & tb) / len(union)) if union else float("nan")


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of channels where the two vectors share the same sign."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.sign(a) == np.sign(b)))


def compare_importances(a: np.ndarray, b: np.ndarray, k: int = 8) -> dict[str, float]:
    """All stability metrics for one pair of aligned importance vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"Importance vectors must align; got {a.shape} vs {b.shape}")
    return {
        "spearman": spearman(a, b),
        "cosine_similarity": cosine_similarity(a, b),
        "top_k_jaccard": top_k_jaccard(a, b, k),
        "sign_agreement": sign_agreement(a, b),
    }
