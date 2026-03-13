"""
data
────
Public API for both training phases.

  Phase 1 graph structures  →  data.phase1 (MiniGraph, pairs, processing)
  Dataset classes            →  data.dataset (DeletionLineDataset, CommitRankingDataset)
"""

from config import EDGE_TYPES, NUM_EDGE_TYPES

# Phase 1 graph structures
from .phase1 import (
    MiniGraph,
    DeletionLinePair,
    TestCaseBatch,
    combine_testcases_to_batches,
    build_pairs,
    build_full_graph_structure,
)

# Dataset classes (both phases)
from .dataset import (
    DeletionLineDataset,
    CommitRankingDataset,
    collate_commit_ranking,
)

__all__ = [
    "EDGE_TYPES",
    "NUM_EDGE_TYPES",
    # phase 1 graph structures
    "MiniGraph",
    "DeletionLinePair",
    "TestCaseBatch",
    "combine_testcases_to_batches",
    "build_pairs",
    "build_full_graph_structure",
    # datasets
    "DeletionLineDataset",
    "CommitRankingDataset",
    "collate_commit_ranking",
]
