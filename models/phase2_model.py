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
from models.shared_encoder import sinusoidal_pe
import math


class CommitRankingModule(nn.Module):
    """
    Trainable commit ranking head operating on pre-computed node embeddings.
    """

    def __init__(self, 
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_commit_transformer_layers: int = 1,
        dropout: float = 0.3,
        max_temporal_dist: int = 300,
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

        self.temporal_query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.general_query = nn.Parameter(torch.randn(num_heads, self.head_dim))


        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm_pool = nn.LayerNorm(hidden_dim)

        # self.distance_embedding = nn.Embedding(max_temporal_dist, hidden_dim)


        # Commit-level Transformer
        self.commit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu",
            ),
            num_layers=num_commit_transformer_layers,
        )
        

        # Final Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )



    def forward(self, 
                node_embeddings: torch.Tensor,
                commit_indices: torch.Tensor,
                is_temporal_node: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            node_embeddings : [N, hidden_dim]  pre-computed by frozen encoder
            commit_indices  : [N]              which commit each node belongs to
        Returns:
            scores           : [C]
        """
        device = node_embeddings.device
        N = node_embeddings.size(0)
        num_commits = commit_indices.max().item() + 1

        # node_embeddings: [N, input_dim] → [N, hidden_dim]
        node_embeddings = self.input_proj(node_embeddings)
       
        # Project all nodes to keys and values in one pass
        K_all = self.k_proj(node_embeddings).view(N, self.num_heads, self.head_dim) # [N, H, D_h]
        V_all = self.v_proj(node_embeddings).view(N, self.num_heads, self.head_dim) # [N, H, D_h]


        # temporal_query: [H, D_h], general_query: [H, D_h]
        # is_temporal_node: [N] bool -> expand to [N, H, D_h]
        mask = is_temporal_node.view(N,1,1).expand(N, self.num_heads, self.head_dim)
        queries = torch.where(mask, self.temporal_query.unsqueeze(0).expand(N, -1, -1),
                                self.general_query.unsqueeze(0).expand(N, -1, -1)
                            ) # [N, H, D_h]

        # Attention logits using per-node query
        logits = (queries * K_all).sum(dim=-1) * self.scale # [N, H]
 
        idx = commit_indices.unsqueeze(1).expand_as(logits)  # [N, H]

        logits_stable = logits - logits.max().detach()        
        exp_logits = torch.exp(logits_stable)   # [N, H]

        # per-commit sum of exp — shape [C, H]
        exp_sum = torch.zeros(num_commits, self.num_heads, device=device)
        exp_sum.scatter_add_(0, idx, exp_logits)

        # normalize: divide each node's exp by its commit's sum
        # exp_sum[commit_indices] broadcasts sum back to each node
        attn_weights = exp_logits / (exp_sum[commit_indices] + 1e-9)  # [N, H]

        # Step 3 — scatter weighted sum of values
        # attn_weights: [N, H], V_all: [N, H, D_h]
        # weighted values: [N, H, D_h]
        weighted_V = attn_weights.unsqueeze(-1) * V_all   # [N, H, D_h]

        # sum weighted values within each commit → [C, H, D_h]
        idx_v = commit_indices.view(N, 1, 1).expand_as(weighted_V)

        pooled = torch.zeros(num_commits, self.num_heads, self.head_dim, device=device)
        pooled.scatter_add_(0, idx_v, weighted_V)   # [C, H, D_h]

        # Step 4 — project and normalize
        # reshape [C, H, D_h] → [C, H*D_h] = [C, hidden_dim]
        # then linear projection + LayerNorm
        commit_embeddings = self.norm_pool(self.out_proj_pool(pooled.reshape(num_commits, -1))) # [C, hidden_dim]

        # Transformer + ranking head
        x = commit_embeddings.unsqueeze(1)         # [C, 1, D] (seq, batch, dim)
        x = self.commit_transformer(x).squeeze(1)  # [C, D]
        return self.ranking_head(x).squeeze(-1)     # [C]