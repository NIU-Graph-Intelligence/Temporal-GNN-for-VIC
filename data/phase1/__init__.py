"""data.phase1 — deletion-line ranking data layer (Phase 1)."""

from .minigraph import MiniGraph
# from .pairs import DeletionLinePair, Batch, combine_pairs_to_batches, build_pairs
from .pairs import DeletionLinePair, TestCaseBatch, combine_testcases_to_batches, build_pairs
from .processing import build_full_graph_structure
from dataclasses import dataclass, field

__all__ = [
    "MiniGraph",
    "DeletionLinePair",
    "TestCaseBatch",
    "combine_testcases_to_batches",
    "build_pairs",
    "build_full_graph_structure",
]
