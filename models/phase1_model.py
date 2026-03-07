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

import torch
import torch.nn as nn

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

    def clear_embedding_cache(self) -> None:
        """Drop all cached encoder outputs.  Call at every batch/epoch boundary."""
        self._embedding_cache.clear()

    def enable_embedding_cache(self) -> None:
        """Enable caching (call before each batch during training)."""
        self._embedding_cache.clear()
        self._cache_enabled = True

    def disable_embedding_cache(self) -> None:
        """Disable and clear cache."""
        self._embedding_cache.clear()
        self._cache_enabled = False

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

    def forward(self, pyg1, del_idx1: int, pyg2, del_idx2: int) -> torch.Tensor:
        """
        Pairwise forward for training.

        Encodes each graph independently so only one graph's activations
        are held in memory at a time.

        Returns:
            prob : scalar P(line1 > line2)
        """
        return self.ranker(
            self._encode_line(pyg1, del_idx1),
            self._encode_line(pyg2, del_idx2),
        ).squeeze(0)

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