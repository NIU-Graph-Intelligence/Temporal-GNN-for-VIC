"""
data/phase1/minigraph.py
────────────────────────
MiniGraph — one deletion line with its full-graph PyG context.
"""

from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data


class MiniGraph:
    """
    One deletion line with its full-graph context.

    The PyG Data object is held in memory (eager).  On the first run graphs
    are embedded with CodeBERT and cached as ``.pt`` files; subsequent runs
    load directly from the cache.

    Attributes
    ----------
    g                : raw graph data list (may be empty when loaded from .pt cache)
    pyg              : PyG Data object (node features + edges)
    tp_to_commit     : {int(temporal_pos): str(12-char commit SHA)}
    inducing_commits : set[str] — ground-truth inducing SHAs from info.json
    rootcause        : bool — whether this deletion line is a true root cause
    score            : float — filled in during evaluation
    history_chains   : list[dict] — V-SZZ history chains for this deletion line
                       (used by Phase 2 to build temporal edges)
    del_line_beg     : int — lineBeg of the deletion line in the fix commit
    del_code         : str — code text of the deletion line
    """

    def __init__(
        self,
        graph_data: List[Dict],
        pyg_data: Optional[Data] = None,
        test_name: str = "",
        del_idx: int = 0,
    ) -> None:
        # Raw graph data list — may be empty when loaded from .pt cache
        self.g                = graph_data
        self.pyg              = pyg_data
        self.test_name        = test_name
        self.del_idx          = torch.tensor([del_idx], dtype=torch.long)
        self.score            = 0.0
        self.rootcause        = (
            graph_data[del_idx]["rootcause"]
            if graph_data and len(graph_data) > del_idx
            else False
        )
        self.tp_to_commit:     Dict[int, str] = {}
        self.inducing_commits: set            = set()
        self.history_chains:   List[Dict]     = []
        self.del_line_beg:     int            = 0
        self.del_code:         str            = ""

    def __repr__(self) -> str:
        return (
            f"MiniGraph(test={self.test_name!r}, "
            f"del_idx={self.del_idx.item()}, "
            f"rootcause={self.rootcause})"
        )
