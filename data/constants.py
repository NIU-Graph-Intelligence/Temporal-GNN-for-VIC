"""
data/constants.py
─────────────────
Graph edge-type vocabulary shared by data, model, and training layers.
"""

from enum import IntEnum

class EdgeType(IntEnum):
    CFG_FWD = 0
    CFG_BWD = 1
    DFG_FWD = 2
    DFG_BWD = 3
    LINEMAP = 4
    TEMPORAL_FWD = 5
    TEMPORAL_BWD = 6

NUM_EDGE_TYPES = len(EdgeType)

