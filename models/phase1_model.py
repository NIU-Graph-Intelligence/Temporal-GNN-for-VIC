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

from typing import Dict, List, Tuple
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

    
        self._embedding_cache: dict = {}
        self._cache_enabled: bool = False


    def set_cache(self, enabled: bool) -> None:
        """Enable or disable embedding cache, always clears on change."""
        self._cache_enabled = enabled
        self._embedding_cache.clear()    

    @contextmanager
    def cache_context(self):
        """
        Usage (per-batch during training, per-epoch during validation):
        """
        self.set_cache(True)
        try:
            yield
        finally:
            self.set_cache(False)

    def _cache_tensor(self, h: torch.Tensor) -> torch.Tensor:
        """
        During training: keep tensor connected to computation graph so all
        pairs sharing this graph contribute gradients back to BERT.
        During eval: detach to free memory since no backward is needed.
        """
        return h if self.training else h.detach()

    def _encode_graph(self, pyg) -> torch.Tensor:
        """Encode one graph, returning cached result if available."""
        key = id(pyg)
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        h = self.encoder.encode_pyg(pyg)
        if self._cache_enabled:
            self._embedding_cache[key] = h
        return h

    def _encode_line(self, pyg, del_idx: int) -> torch.Tensor:
        """Encode one graph and extract the deletion-line node: [1, hidden_dim]."""
        h = self._encode_graph(pyg)
        return h[del_idx].unsqueeze(0)

    

    def predict(self, pyg_data, del_idx: int = 0) -> torch.Tensor:
        """
        Score a single deletion line for inference.

        Returns:
            scalar ranking score
        """
        if isinstance(del_idx, torch.Tensor):
            del_idx = del_idx.item() if del_idx.numel() == 1 else del_idx[0].item()
        h = self._encode_graph(pyg_data)
        if del_idx >= h.size(0):
            raise ValueError(
                f"del_idx {del_idx} out of bounds for graph with {h.size(0)} nodes")
        return self.ranker.score(h[del_idx])

    def forward(
        self,
        pair_specs: List[Tuple],
        device: torch.device,
        max_nodes: int = 9500,
    ) -> List[Tuple[torch.Tensor, bool]]:
        """
        Batch-encode all unique graphs and return stacked deletion line
        embeddings for all valid pairs — ready for a single ranker call.

        Returns
        -------
        emb_x       : [N, hidden_dim] embeddings for left side of each valid pair
        emb_y       : [N, hidden_dim] embeddings for right side of each valid pair
        valid_mask  : list of indices into pair_specs that succeeded
        (use this to align with pair_probs in the caller)

        If no valid pairs exist, returns (None, None, []).
        """

        # ── 1. Collect unique graphs (dedup by object id) ──
        graph_map: Dict[int, object] = {}  # id(pyg) → pyg
        for pyg_x, _, pyg_y, _ in pair_specs:
            if pyg_x is not None:
                graph_map[id(pyg_x)] = pyg_x
            if pyg_y is not None:
                graph_map[id(pyg_y)] = pyg_y

        # 2. Encode uncached graphs
        uncached = [(gid, pyg) for gid, pyg in graph_map.items()
                    if gid not in self._embedding_cache]

        # FIX: define these BEFORE the if block so they always exist
        # to_batch  = [(gid, pyg) for gid, pyg in uncached
        #              if pyg.num_nodes <= max_nodes]
        # to_single = [(gid, pyg) for gid, pyg in uncached
        #              if pyg.num_nodes > max_nodes]

        # Batch encode graphs that fit within max_nodes
        if uncached:
            pyg_list = [pyg.to(device) for _, pyg in uncached]
            batched  = Batch.from_data_list(pyg_list)
            h_all    = self.encoder.encode_pyg(batched)
            # ── KEY FIX: use ptr tensor for correct graph boundaries ──
            # batched.ptr = [0, n0, n0+n1, ...] cumulative node counts
            # This is safer than manual ptr arithmetic and handles
            # variable-length graphs (Del+commits) correctly
            for i, (gid, _) in enumerate(uncached):
                start = batched.ptr[i].item()
                end   = batched.ptr[i + 1].item()
                # h_slice = h_all[start:end]
                self._embedding_cache[gid] = self._cache_tensor(h_all[start:end])

            # except Exception:
            #     # Fallback: encode individually
            #     for gid, pyg in to_batch:
            #         try:
            #             h = self.encoder.encode_pyg(pyg.to(device))
            #             self._embedding_cache[gid] = (
            #                 h if self.training else h.detach()
            #             ) 
            #         except Exception:
            #             pass

        # Encode oversized graphs one at a time
        # for gid, pyg in to_single:
        #     try:
        #         h = self.encoder.encode_pyg(pyg.to(device))
        #         self._embedding_cache[gid] = (
        #             h if self.training else h.detach()
        #         )   
        #     except Exception:
        #         pass


        # ── 3. Extract deletion line embeddings for each pair ──────────────
        # Collect valid pairs first, then stack — avoids N separate ranker calls
        emb_x_list:  List[torch.Tensor] = []
        emb_y_list:  List[torch.Tensor] = []
        valid_mask:  List[int]          = []

        for i, (pyg_x, del_x, pyg_y, del_y) in enumerate(pair_specs):
            hx = self._embedding_cache.get(id(pyg_x))
            hy = self._embedding_cache.get(id(pyg_y))

            if hx is None or hy is None:
                continue
            if del_x >= hx.size(0) or del_y >= hy.size(0):
                continue

            # del_x/del_y are LOCAL indices within each graph's embedding
            # (already sliced back correctly via ptr above)
            emb_x_list.append(hx[del_x])   # [hidden_dim]
            emb_y_list.append(hy[del_y])   # [hidden_dim]
            valid_mask.append(i)

        if not emb_x_list:
            return None, None, []

        # Stack into [N, hidden_dim] — single tensor for batched ranker call
        emb_x = torch.stack(emb_x_list)   # [N, hidden_dim]
        emb_y = torch.stack(emb_y_list)   # [N, hidden_dim]

        return emb_x, emb_y, valid_mask