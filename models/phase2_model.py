"""
models/phase2_model.py
───────────────────────
Phase 2 model: commit ranking head.

  CommitRankingModule  — trainable head: node embeddings → commit embeddings → scores

Architecture
────────────
Pre-computed node embeddings [N, hidden_dim]
    (produced by the frozen Phase 1 encoder; temporal PE already applied)
    ↓  multi-head attention pooling     (nodes per commit → one vector)
[C, hidden_dim]
    ↓  TransformerEncoder               (commit-level sequence modelling)
    ↓  RankingHead  (Linear → GELU → Dropout → Linear)
scores [C]
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommitRankingModule(nn.Module):
    """
    Trainable commit ranking head operating on pre-computed node embeddings.
    """

    def __init__(self, hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_commit_transformer_layers: int = 2,
                 max_commits: int = 100, dropout: float = 0.2):

        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Multi-head attention pooling (nodes → one vector per commit)
        self.commit_queries = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm_pool = nn.LayerNorm(hidden_dim)

        # Commit-level Transformer
        self.commit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu",
            ),
            num_layers=num_commit_transformer_layers,
        )

        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_embeddings: torch.Tensor,
                commit_indices: torch.Tensor,
                num_commits: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            node_embeddings : [N, hidden_dim]  pre-computed by frozen encoder
            commit_indices  : [N]              which commit each node belongs to
            num_commits     : int              total number of commits (C)
        Returns:
            scores           : [C]
            attention_weights: list of per-commit attention tensors
        """
        device = node_embeddings.device

        # Batch the linear projections once over ALL nodes
        N = node_embeddings.size(0)
        K_all = self.k_proj(node_embeddings).view(N, self.num_heads, self.head_dim)
        V_all = self.v_proj(node_embeddings).view(N, self.num_heads, self.head_dim)

        # Group nodes by commit for the per-commit attention pooling
        commit_embeddings, attention_weights = [], []

        for c in range(num_commits):
            mask = commit_indices == c
            if not mask.any():
                commit_embeddings.append(torch.zeros(self.hidden_dim, device=device))
                attention_weights.append(torch.tensor([], device=device))
                continue
            K = K_all[mask]          # [N_c, H, D_h]
            V = V_all[mask]          # [N_c, H, D_h]
            attn = F.softmax(
                torch.einsum("hd,nhd->hn", self.commit_queries, K) * self.scale,
                dim=-1,
            )
            emb = self.out_proj_pool(
                torch.einsum("hn,nhd->hd", attn, V).reshape(-1))
            commit_embeddings.append(self.norm_pool(emb))
            attention_weights.append(attn)

        x = torch.stack(commit_embeddings).unsqueeze(1)   # [C, 1, D]
        x = self.commit_transformer(x).squeeze(1)          # [C, D]
        return self.ranking_head(x).squeeze(-1), attention_weights
