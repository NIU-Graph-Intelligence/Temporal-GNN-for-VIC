"""
data
────
Public API for both training phases.

  Phase 1 (deletion-line ranking)  →  data.phase1
  Phase 2 (commit-graph ranking)   →  data.phase2
"""

from config import EDGE_TYPES, NUM_EDGE_TYPES

# Phase 1
from .phase1 import (
    MiniGraph,
    DeletionLinePair,
    Batch,
    combine_pairs_to_batches,
    build_pairs,
    DeletionLineDataset,
    build_full_graph_structure,
)

# Phase 2
from .phase2 import (
    build_fix_commit_pyg,
    score_and_cache_top_embeddings,
    precompute_phase2_embeddings,
    Phase2EmbeddingDataset,
    collate_phase2,
)

__all__ = [
    "EDGE_TYPES",
    "NUM_EDGE_TYPES",
    # phase 1
    "MiniGraph",
    "DeletionLinePair",
    "Batch",
    "combine_pairs_to_batches",
    "build_pairs",
    "DeletionLineDataset",
    "build_full_graph_structure",
    # phase 2
    "build_fix_commit_pyg",
    "score_and_cache_top_embeddings",
    "precompute_phase2_embeddings",
    "Phase2EmbeddingDataset",
    "collate_phase2",
]
