"""
models/phase1_model.py
───────────────────────
Phase 1 model: deletion line ranking.

  DeletionLineRanker      — RankNet-style MLP scoring head
  DeletionLineRankingModel — full model: SharedEncoder + ranker head

Architecture
────────────
token_ids [N, seq_len]
    ↓  SharedEncoder  (CodeBERT → project → temporal PE → GraphTransformer)
[N, hidden_dim]
    ↓  h[del_idx]           extract the deletion-line node embedding
[1, hidden_dim]
    ↓  DeletionLineRanker   linear stack → scalar score
scalar

For pairwise training (forward):
    score(graph_x, del_x) - score(graph_y, del_y)  →  sigmoid  →  P(x > y)

For inference (predict):
    returns the raw ranker score for a single graph.
"""

from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from config import NUM_EDGE_TYPES
from .shared_encoder import SharedEncoder, EMB_DIM


class DeletionLineRanker(nn.Module):
    """
    RankNet-style linear scoring head.
    """

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Pairwise forward for training.

        Args:
            emb1 : [B, hidden_dim]
            emb2 : [B, hidden_dim]
        Returns:
            prob : [B]  sigmoid(score(emb1) - score(emb2))
        """
        return self.output(self.model(emb1) - self.model(emb2)).squeeze(-1)

    def score(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Raw scalar score for a single embedding (inference).

        Args:
            emb : [hidden_dim] or [1, hidden_dim]
        Returns:
            scalar tensor
        """
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return self.model(emb).squeeze()


class DeletionLineRankingModel(nn.Module):
    """
    Full Phase 1 model: SharedEncoder + DeletionLineRanker.

    When include_bert=True  (default) PyG data carries token_ids +
    attention_mask; CodeBERT is fine-tuned jointly.
    When include_bert=False PyG data carries pre-computed x embeddings.
    """

    def __init__(self, input_dim: int = EMB_DIM, 
                hidden_dim: int = 1536,
                num_gt_layers: int = 4, 
                num_heads: int = 8,
                num_edge_types: int = NUM_EDGE_TYPES,
                dropout: float = 0.2,
                include_bert: bool = True,
                num_bert_layers_freeze: int = 8,
                bert_chunk: int = 256):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_bert = include_bert

        self.encoder = SharedEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_gt_layers=num_gt_layers, 
            num_heads=num_heads,
            num_edge_types=num_edge_types, 
            dropout=dropout,
            include_bert=include_bert,
            num_bert_layers_freeze=num_bert_layers_freeze,
            bert_chunk=bert_chunk,
        )
        self.ranker = DeletionLineRanker(hidden_dim)


    def predict(self, pyg_data, del_idx: int = 0) -> torch.Tensor:
        """
        Score a single deletion line for inference.

        Returns:
            scalar ranking score
        """
        if isinstance(del_idx, torch.Tensor):
            del_idx = del_idx.item() if del_idx.numel() == 1 else del_idx[0].item()
        h = self.encoder.encode_pyg(pyg_data)
        if del_idx >= h.size(0):
            raise ValueError(
                f"del_idx {del_idx} out of bounds for graph with {h.size(0)} nodes")
        return self.ranker.score(h[del_idx])

    def forward(
        self,
        mini_graphs,        # List[MiniGraph]  — all graphs in this batch
        pairs,              # List[DeletionLinePair] — all pairs for this batch
        device: torch.device,
        max_nodes: int = 9500,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
        """
        Encode every graph in mini_graphs exactly once, then extract
        deletion-line embeddings for every pair by lookup.

        Each deletion line is always node 0 in its subgraph (del_idx=0),
        so we extract h[ptr[i]] — the first node of each graph.

        Parameters
        ----------
        mini_graphs : all MiniGraph objects for the test cases in this batch
        pairs       : all DeletionLinePair combinations derived from those graphs
        device      : torch device
        max_nodes   : graphs with more nodes than this are skipped

        Returns
        -------
        emb_x      : [N, hidden_dim] left-side embeddings for valid pairs
        emb_y      : [N, hidden_dim] right-side embeddings for valid pairs
        valid_mask : indices into ``pairs`` that were successfully processed
                     (use to align with pair.prob in the caller)
        Returns (None, None, []) when no valid pairs exist.
        """

        # ── Step 1: filter oversized graphs, build id→embedding map ──────
        valid_pygs:   List           = []
        valid_gids:   List[int]      = []
        skipped_gids: set            = set()

        seen_gids: set = set()
        for mg in mini_graphs:
            gid = id(mg.pyg)
            if gid in seen_gids:
                continue                          # deduplicate (shouldn't happen but safe)
            seen_gids.add(gid)

            if mg.pyg is None:
                skipped_gids.add(gid)
                continue
            if mg.pyg.num_nodes > max_nodes:
                skipped_gids.add(gid)
                continue

            valid_pygs.append(mg.pyg.to(device))
            valid_gids.append(gid)

        if not valid_pygs:
            return None, None, []

        # ── Step 2: single batched encoder forward pass ───────────────────
        batched = Batch.from_data_list(valid_pygs)
        h_all   = self.encoder.encode_pyg(batched)   # [total_nodes, hidden_dim]

        # Map graph id → deletion-line embedding (node 0 of each graph)
        # ptr[i] is the global start index of graph i in h_all
        gid_to_emb: Dict[int, torch.Tensor] = {}
        for i, gid in enumerate(valid_gids):
            node_start = batched.ptr[i].item()
            # Deletion line is always node 0 → global index = node_start + 0
            emb = h_all[node_start]              # [hidden_dim]
            # Keep gradient connection during training; detach during eval
            gid_to_emb[gid] = emb if self.training else emb.detach()

        # ── Step 3: pair lookup — zero additional encoding ────────────────
        emb_x_list:  List[torch.Tensor] = []
        emb_y_list:  List[torch.Tensor] = []
        valid_mask:  List[int]          = []

        for i, pair in enumerate(pairs):
            gx = id(pair.x.pyg)
            gy = id(pair.y.pyg)

            if gx in skipped_gids or gy in skipped_gids:
                continue
            if gx not in gid_to_emb or gy not in gid_to_emb:
                continue
            emb_x_list.append(gid_to_emb[gx])   # [hidden_dim]
            emb_y_list.append(gid_to_emb[gy])   # [hidden_dim]
            valid_mask.append(i)

        if not emb_x_list:
            return None, None, []

        emb_x = torch.stack(emb_x_list)          # [N, hidden_dim]
        emb_y = torch.stack(emb_y_list)          # [N, hidden_dim]

        return emb_x, emb_y, valid_mask

