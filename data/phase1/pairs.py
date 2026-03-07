"""
data/pairs.py
─────────────
DeletionLinePair, Batch, combine_pairs_to_batches — pairwise training
primitives for Phase 1 ranking.
"""

import random
from typing import List

from data.phase1.minigraph import MiniGraph


class DeletionLinePair:
    """
    Pairwise training example.

    prob = 1.0  →  x should rank higher than y  (x is rootcause)
    prob = 0.0  →  y should rank higher than x  (y is rootcause)
    prob = 0.5  →  tied (both same class)
    """

    def __init__(self, minig1: MiniGraph, minig2: MiniGraph, prob: float) -> None:
        self.x    = minig1
        self.y    = minig2
        self.prob = prob


class Batch:
    """Thin wrapper around a list of DeletionLinePairs."""

    def __init__(self, pairs: List[DeletionLinePair]) -> None:
        self.pairs = pairs
        self.size  = len(pairs)

    def __len__(self) -> int:
        return self.size


def combine_pairs_to_batches(
    pairs: List[DeletionLinePair], batch_size: int = 128
) -> List[Batch]:
    """Slice ``pairs`` into fixed-size Batch objects."""
    return [
        Batch(pairs[i : i + batch_size])
        for i in range(0, len(pairs), batch_size)
    ]


def build_pairs(
    graphs: List[MiniGraph], max_pairs: int = 50
) -> List[DeletionLinePair]:
    """
    Generate pairwise training examples from a list of MiniGraphs.

    - rootcause vs. non-rootcause  → prob 1.0 / 0.0
    - same class vs. same class    → prob 0.5 (tie)

    Capped at ``max_pairs`` via random sampling.
    """
    pos = [g for g in graphs if g.rootcause]
    neg = [g for g in graphs if not g.rootcause]

    pairs: List[DeletionLinePair] = []
    for rg in pos:
        for ng in neg:
            pairs.append(DeletionLinePair(rg, ng, 1.0))
            pairs.append(DeletionLinePair(ng, rg, 0.0))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            pairs.append(DeletionLinePair(pos[i], pos[j], 0.5))
    for i in range(len(neg)):
        for j in range(i + 1, len(neg)):
            pairs.append(DeletionLinePair(neg[i], neg[j], 0.5))

    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    return pairs
