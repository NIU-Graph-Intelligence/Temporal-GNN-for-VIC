"""
data/phase1/pairs.py

DeletionLinePair, TestCaseBatch, combine_testcases_to_batches — pairwise
training primitives for Phase 1 ranking.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List

from data.phase1.minigraph import MiniGraph

@dataclass
class DeletionLinePair:
    """
    Pairwise training example.

    prob = 1.0  →  x should rank higher than y  (x is rootcause)
    prob = 0.0  →  y should rank higher than x  (y is rootcause)
    prob = 0.5  →  tied (both same class)
    """
    x:    MiniGraph
    y:    MiniGraph
    prob: float


@dataclass
class TestCaseBatch:
    test_cases: List[str] = field(default_factory=list)
    mini_graphs: List[MiniGraph] = field(default_factory=list)
    pairs: List[DeletionLinePair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

def build_pairs(
    graphs: List[MiniGraph], max_pairs: int = 100
) -> List[DeletionLinePair]:
    """
    Generate pairwise training examples from a list of MiniGraphs.

    - rootcause vs. non-rootcause  → prob 1.0 / 0.0
    - same class vs. same class    → prob 0.5 (tie)

    Capped at ``max_pairs`` via random sampling.
    """
    pos = [g for g in graphs if g.rootcause]
    neg = [g for g in graphs if not g.rootcause]

    graphs_ordered = pos + neg
    pairs = []

    for i in range(len(graphs_ordered)):
        for j in range(i+1, len(graphs_ordered)):
            g1, g2 = graphs_ordered[i], graphs_ordered[j]

            if g1.rootcause == g2.rootcause:
                prob = 0.5
            elif g1.rootcause:
                prob = 1.0
            else:
                prob = 0.0

            pairs.append(DeletionLinePair(g1, g2, prob))

            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def combine_testcases_to_batches(
    dataset,                          # DeletionLineDataset
    cases: List[str],
    pairs_cache: Dict[str, List[DeletionLinePair]],
    max_graphs_per_batch: int = 40,   # VRAM control: max graphs encoded at once
) -> List[TestCaseBatch]:
    """
    Group test cases into TestCaseBatches capped by ``max_graphs_per_batch``.

    Parameters
    ----------
    dataset              : DeletionLineDataset (provides mini_graphs dict)
    cases                : ordered list of test-case names to batch
    pairs_cache          : pre-built {test_name: List[DeletionLinePair]}
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
        current_pairs.extend(pairs_cache.get(name, []))

    # Don't forget the last batch
    if current_graphs:
        batches.append(TestCaseBatch(
            test_cases=current_cases,
            mini_graphs=current_graphs,
            pairs=current_pairs,
        ))

    return batches