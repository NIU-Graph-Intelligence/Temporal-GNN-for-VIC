"""
data/pairs.py
─────────────
DeletionLinePair, Batch, combine_pairs_to_batches — pairwise training
primitives for Phase 1 ranking.
"""

import random
from dataclasses import dataclass, field
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


# class Batch:
#     """Thin wrapper around a list of DeletionLinePairs."""

#     def __init__(self, pairs: List[DeletionLinePair]) -> None:
#         self.pairs = pairs
#         self.size  = len(pairs)

#     def __len__(self) -> int:
#         return self.size


# def combine_pairs_to_batches(
#     pairs: List[DeletionLinePair], batch_size: int = 128
# ) -> List[Batch]:
#     """Slice ``pairs`` into fixed-size Batch objects."""
#     return [
#         Batch(pairs[i : i + batch_size])
#         for i in range(0, len(pairs), batch_size)
#     ]

@dataclass
class TestCaseBatch:
    test_cases: List[str] = field(default_factory=list)
    mini_graphs: List[MiniGraph] = field(default_factory=list)
    pairs: List[DeletionLinePair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

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


def combine_testcases_to_batches(
    dataset,                          # DeletionLineDataset
    cases: List[str],
    max_pairs:            int = 50,
    max_graphs_per_batch: int = 9500,   # VRAM control: max graphs encoded at once
) -> List[TestCaseBatch]:
    """
    Group test cases into TestCaseBatches capped by ``max_graphs_per_batch``.
    Each test case contributes len(mini_graphs[name]) graphs. Once adding
    the next test case would exceed ``max_graphs_per_batch``, the current
    batch is sealed and a new one starts.

    A single test case that exceeds ``max_graphs_per_batch`` on its own is
    placed in a batch by itself (handled gracefully — the encoder will just
    have a large-ish batch for that one case).

    Parameters
    ----------
    dataset              : DeletionLineDataset (provides mini_graphs dict)
    cases                : ordered list of test-case names to batch
    max_pairs            : passed to build_pairs for within-testcase pair cap
    max_graphs_per_batch : maximum number of graphs encoded in one forward pass
    """
    batches: List[TestCaseBatch] = []
    current_graphs: List[MiniGraph]        = []
    current_cases:  List[str]              = []
    current_pairs:  List[DeletionLinePair] = []

    for name in cases:
        mgs = dataset.mini_graphs.get(name, [])
        if not mgs:
            continue

        # Seal current batch if adding this test case would exceed VRAM limit
        if current_graphs and len(current_graphs) + len(mgs) > max_graphs_per_batch:
            batches.append(TestCaseBatch(
                test_cases=current_cases,
                mini_graphs=current_graphs,
                pairs=current_pairs,
            ))
            current_graphs = []
            current_cases  = []
            current_pairs  = []

        current_cases.append(name)
        current_graphs.extend(mgs)
        current_pairs.extend(build_pairs(mgs, max_pairs))

    # Don't forget the last batch
    if current_graphs:
        batches.append(TestCaseBatch(
            test_cases=current_cases,
            mini_graphs=current_graphs,
            pairs=current_pairs,
        ))

    return batches