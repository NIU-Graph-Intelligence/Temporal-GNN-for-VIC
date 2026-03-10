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

    def __init__(self, 
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_commit_transformer_layers: int = 1,
        dropout: float = 0.3
    ):

        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # input_dim=768 (Phase 1 output) → hidden_dim=256 (Phase 2 internal)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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
                commit_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings : [N, hidden_dim]  pre-computed by frozen encoder
            commit_indices  : [N]              which commit each node belongs to
        Returns:
            scores           : [C]
        """
        device = node_embeddings.device

        num_commits = commit_indices.max().item() + 1

        # node_embeddings: [N, 768] → [N, 256]
        node_embeddings = self.input_proj(node_embeddings)

        # Batch the linear projections once over ALL nodes
        N = node_embeddings.size(0)
        K_all = self.k_proj(node_embeddings).view(N, self.num_heads, self.head_dim)
        V_all = self.v_proj(node_embeddings).view(N, self.num_heads, self.head_dim)

        # sort nodes by commit once (O(N log N)), then split into per-commit chunks
        sorted_order   = torch.argsort(commit_indices)
        sorted_commits = commit_indices[sorted_order]
        K_sorted       = K_all[sorted_order]   # [N, H, D_h]
        V_sorted       = V_all[sorted_order]   # [N, H, D_h]

        # sizes[c] = number of nodes belonging to commit c
        sizes   = torch.bincount(commit_indices, minlength=num_commits).tolist()
        K_split = torch.split(K_sorted, sizes)  
        V_split = torch.split(V_sorted, sizes) 

        # Group nodes by commit for the per-commit attention pooling
        commit_embeddings = []

        for c in range(num_commits):
            K = K_split[c]  # [N_c, H, D_h]
            V = V_split[c]  # [N_c, H, D_h]

            attn = F.softmax(
                torch.einsum("hd,nhd->hn", self.commit_queries, K) * self.scale,
                dim=-1,
            )
            emb = self.out_proj_pool(
                torch.einsum("hn,nhd->hd", attn, V).reshape(-1))
            commit_embeddings.append(self.norm_pool(emb))
            

        x = torch.stack(commit_embeddings).unsqueeze(1)   # [C, 1, D]
        x = self.commit_transformer(x).squeeze(1)          # [C, D]
        return self.ranking_head(x).squeeze(-1)
