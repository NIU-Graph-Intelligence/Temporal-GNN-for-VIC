#!/usr/bin/env python3
"""
Unified Two-Phase Training: Deletion Line Ranking → Commit Ranking

Architecture:
  Phase 1 (Deletion Line Ranking):
    - Train SharedEncoder + DeletionLineRanker with pairwise ranking loss
    - Identical to train_deletion_line_ranking.py
    - Saves per-fold checkpoints

  Phase 2 (Commit Ranking with Frozen Phase 1 Encoder):
    - Loads the Phase 1 checkpoint and FREEZES the SharedEncoder
    - For each test case, builds a unified temporal graph (same as
      TemporalGraphBuilder in temporal_graph_transformer.py)
    - Projects CodeBERT embeddings through the FROZEN Phase 1 encoder
      to get enriched node representations
    - Feeds enriched representations to a NEW commit-ranking head:
        HierarchicalAggregation → CommitTransformer → RankingHead
    - Trains ONLY the commit-ranking head with listwise ranking loss

Why this works (and why previous attempts failed):
  - Previous naive combination: trained BOTH heads simultaneously,
    causing conflicting gradient updates to the shared encoder
  - This approach: Phase 1 encoder is frozen in Phase 2, so commit
    ranking gradients never corrupt the line-ranking representations
  - Phase 2 benefits from Phase 1's enriched representations (the
    encoder has already learned to identify bug-relevant code patterns)
    while the commit-ranking head is free to learn its own task

K-Fold Leak-Free Design:
  - Both phases use the SAME fold splits (same seed, same K)
  - Phase 1 trains fold i's encoder on fold i's training data
  - Phase 2 uses fold i's frozen encoder when training on fold i's data
  - Evaluation is on each fold's validation set (never seen during
    either phase of training)
"""

import gc
import json
import math
import os
import sys
import time
import random
import argparse
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data

# Phase 1 imports
from model_architecture import (
    CodeBERTEmbedder,
    DeletionLineRankingModel,
    SharedEncoder,
    DEVICE,
    EMB_DIM,
    NUM_EDGE_TYPES,
)
from dataset_and_graph import DeletionLineDataset, DeletionLinePair, MiniGraph, combine_pairs_to_batches
from loss_and_metrics import PairwiseRankingLoss, evaluate_top1_metrics, load_true_commit_map, print_metrics

# Phase 2 imports
from temporal_graph_transformer import (
    TemporalGraphBuilder,
    TemporalGraphDataset,
    get_chronological_order,
    EDGE_TYPES,
)


# ==========================================================================
#  Configuration
# ==========================================================================
CONFIG = {
    # Data paths
    "data_path": "/mnt/data/NeuralSZZ/replication/replication/trainData",
    "test_cases_file": "successfultestcase661foundGT.json",
    "prebuilt_dir": "/mnt/data/NeuralSZZ/replication/replication/temporal_graph",
    "graph_mode": "full_graph",
    "save_dir": "/mnt/data/NeuralSZZ/replication/replication/checkpoints_unified_two_phase",

    # Phase 1 (Deletion Line Ranking)
    "phase1_epochs": 50,
    "phase1_lr": 5e-6,  # fallback when differential LRs not used
    "phase1_bert_lr": 2e-5,   # CodeBERT: small updates to preserve pre-trained knowledge
    "phase1_rest_lr": 1e-4,   # Graph layers + ranker: larger updates (random init)
    "phase1_bert_freeze_bottom_layers": 12,  # Freeze bottom 8 layers + embeddings; fine-tune top 4 only (~1.8x faster backward, much less VRAM)
    "phase1_batch_size": 8,   # increased from 4 — two 4090s can handle larger batches
    "phase1_patience": 10,
    "phase1_max_pairs_per_test": 100,

    # Phase 2 (Commit Ranking)
    "phase2_epochs": 100,
    "phase2_lr": 5e-5,
    "phase2_weight_decay": 0.05,
    "phase2_batch_size": 4,   # increased from 1 — commit graphs are small, 4090 can fit 4
    "phase2_patience": 15,
    "phase2_gradient_accumulation_steps": 2,  # reduced from 4 (larger batch absorbs the need)
    "phase2_temperature": 1.0,
    "phase2_label_smoothing": 0.1,
    "phase2_top_k_lines": 1,  # Extract commits from top-K deletion lines

    # Shared model architecture (matches baseline NeuralSZZ dimensions)
    # Baseline: HANConv(768→1536, heads=2) = 1 layer.
    "hidden_dim": 1536,
    "num_gt_layers": 1,
    "num_heads": 2,               # Phase 1 encoder heads (must match saved checkpoints)
    "num_edge_types": NUM_EDGE_TYPES,
    "dropout": 0.1,
    "include_bert": True,
    "max_nodes_per_graph": 9500,  # Keep all graphs — 300 lost 55% of rootcause graphs
    "bert_chunk": 256,          # nodes per BERT forward pass (was 16, now 256 for speed)

    # Phase 2 commit-ranking head
    # num_heads must evenly divide hidden_dim (1536).
    # 8 heads → head_dim=192: finer-grained attention than 4 (384) or 2 (768),
    # letting the commit transformer attend to more diverse sub-spaces.
    "phase2_num_heads": 8,
    "num_commit_transformer_layers": 2,
    "max_commits": 100,

    # Cross-validation
    "n_folds": 10,
    "seed": 42,
    "stratify_by_commits": True,

    # Multi-GPU: use all available CUDA devices via DataParallel
    # Set to None to auto-detect, or e.g. [0, 1] to pin specific GPUs
    # [1, 0] → primary=cuda:1 (GT+ranker), BERT model-parallel on cuda:0
    "gpu_ids": [1, 0],

    # DataLoader workers (set >0 for faster data loading on multi-core CPU)
    "num_workers": 4,

    # Logging
    "log_interval": 20,
}


# ==========================================================================
#  Phase 2 Model: Commit Ranking Head (trained on top of frozen encoder)
# ==========================================================================

class CommitRankingModule(nn.Module):
    """
    Commit ranking module that operates on enriched node representations
    from a frozen Phase 1 SharedEncoder.

    Architecture:
        Frozen Phase 1 Encoder (SharedEncoder)
              ↓
        enriched node embeddings [N, hidden_dim]
              ↓
        Fixed Sinusoidal Temporal PE  ← replaces learned nn.Embedding
              ↓
        HierarchicalAggregation (nodes → commit embeddings)
              ↓
        CommitTransformer (commit-level sequence modeling)
              ↓
        RankingHead (commit embedding → scalar score)

    Only the aggregation, transformer, and ranking head are trainable.

    WHY SINUSOIDAL (not learned) PE
    --------------------------------
    The previous learned nn.Embedding(max_commits, hidden_dim) lets each
    temporal slot train an independent vector. Because the fix commit is
    always appended at position nc-1, slot nc-2 (the commit just before
    the fix) consistently receives high-gradient positive signals during
    training and the model memorises "slot nc-2 → high score" regardless
    of code content — producing the second-to-last temporal position bias
    (81/121 Phase 2 losses).

    Sinusoidal PE uses a deterministic mathematical formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    No slot has a special learnable identity. Relative-distance information
    is still preserved (dot-products decay smoothly with distance), so the
    model can still tell "older" from "newer" — but cannot memorise
    "position 1 in a 3-commit sequence → inducer".
    """

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Build a [max_len, d_model] fixed sinusoidal positional table."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [max_len, d_model]

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_commit_transformer_layers: int = 2,
        max_commits: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        # Fixed sinusoidal temporal positional encoding (NOT learned).
        # Registered as a buffer so it moves to the right device automatically
        # and is saved/loaded with the state_dict (as a non-parameter tensor).
        self.register_buffer(
            "temporal_pe",
            self._build_sinusoidal_pe(max_commits, hidden_dim),
        )

        # Hierarchical aggregation: nodes → commit embeddings
        # Multi-head attention pooling (same design as TemporalGraphTransformer)
        self.commit_queries = nn.Parameter(
            torch.randn(num_heads, self.head_dim))
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm_pool = nn.LayerNorm(hidden_dim)
        self.scale = self.head_dim ** -0.5

        # Commit-level Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
        )
        self.commit_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_commit_transformer_layers,
        )

        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool_commit(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Multi-head attention pooling: aggregate nodes into a single
        commit embedding.

        Args:
            node_embeddings: [num_nodes, hidden_dim]

        Returns:
            commit_emb: [hidden_dim]
        """
        num_nodes = node_embeddings.size(0)
        K = self.k_proj(node_embeddings).view(
            num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(node_embeddings).view(
            num_nodes, self.num_heads, self.head_dim)
        Q = self.commit_queries  # [H, D]

        scores = torch.einsum('hd,nhd->hn', Q, K) * self.scale
        attn = F.softmax(scores, dim=-1)  # [H, N]

        head_out = torch.einsum('hn,nhd->hd', attn, V)  # [H, D]
        commit_emb = self.out_proj_pool(head_out.reshape(-1))
        commit_emb = self.norm_pool(commit_emb)
        return commit_emb, attn

    def forward(
        self,
        node_embeddings: torch.Tensor,
        commit_indices: torch.Tensor,
        num_commits: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass: enriched nodes → commit scores.

        Args:
            node_embeddings: [N, hidden_dim] from frozen Phase 1 encoder
            commit_indices: [N] which commit each node belongs to
            num_commits: total number of commits

        Returns:
            scores: [num_commits] ranking score per commit
            attention_weights: per-commit attention weights for interpretability
        """
        # Add fixed sinusoidal positional encoding by commit index.
        # temporal_pe[i] is the same deterministic vector for position i
        # regardless of training iteration — no slot can be memorised.
        clamped_indices = commit_indices.clamp(0, self.temporal_pe.size(0) - 1)
        node_embeddings = node_embeddings + self.temporal_pe[clamped_indices]

        # Pool nodes → commit embeddings
        commit_embeddings = []
        attention_weights = []

        for c in range(num_commits):
            mask = (commit_indices == c)
            nodes = node_embeddings[mask]

            if nodes.size(0) == 0:
                commit_embeddings.append(
                    torch.zeros(self.hidden_dim, device=node_embeddings.device))
                attention_weights.append(
                    torch.tensor([], device=node_embeddings.device))
                continue

            commit_emb, attn = self._pool_commit(nodes)
            commit_embeddings.append(commit_emb)
            attention_weights.append(attn)

        commit_embeddings = torch.stack(commit_embeddings, dim=0)  # [C, D]

        # Commit-level Transformer
        x = commit_embeddings.unsqueeze(1)  # [C, 1, D]
        x = self.commit_transformer(x)       # [C, 1, D]
        x = x.squeeze(1)                     # [C, D]

        # Ranking scores
        scores = self.ranking_head(x).squeeze(-1)  # [C]

        return scores, attention_weights


# ==========================================================================
#  Unified Model: Frozen Encoder + Commit Ranking Head
# ==========================================================================

class UnifiedPhase2Model(nn.Module):
    """
    Wraps the frozen Phase 1 SharedEncoder + trainable CommitRankingModule.

    The encoder processes the unified temporal graph (CodeBERT embeddings
    + edge structure) to produce enriched node representations.
    The commit ranking module aggregates and ranks commits.
    """

    def __init__(
        self,
        encoder: SharedEncoder,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_commit_transformer_layers: int = 2,
        max_commits: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Frozen encoder from Phase 1
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Trainable commit ranking head
        self.commit_ranker = CommitRankingModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_commit_transformer_layers=num_commit_transformer_layers,
            max_commits=max_commits,
            dropout=dropout,
        )

    def train(self, mode=True):
        """Override to keep encoder frozen even in train mode."""
        super().train(mode)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        return self

    def forward(
        self,
        pyg_data,
        commit_indices: torch.Tensor,
        num_commits: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through frozen encoder → commit ranking.

        Args:
            pyg_data: PyG Data with token_ids+attention_mask or x
            commit_indices: [N] commit membership
            num_commits: total commits

        Returns:
            scores: [num_commits]
            attention_weights: per-commit attention maps
        """
        kw = {
            "edge_index": pyg_data.edge_index,
            "edge_type":  pyg_data.edge_type,
            "temporal_pos": getattr(pyg_data, "temporal_pos", None),
        }
        if hasattr(pyg_data, "token_ids"):
            kw["token_ids"] = pyg_data.token_ids
            kw["attention_mask"] = pyg_data.attention_mask
        else:
            kw["x"] = pyg_data.x

        # Move encoder inputs to the encoder's device (may differ from
        # commit_ranker's device when using pipeline parallelism).
        encoder_device = next(self.encoder.parameters()).device
        kw = {
            k: (v.to(encoder_device) if isinstance(v, torch.Tensor) else v)
            for k, v in kw.items()
        }

        with torch.no_grad():
            h = self.encoder(**kw)
        # Detach and move to the commit_ranker's device.
        # When pipeline-parallel (encoder on GPU 1, ranker on GPU 0),
        # this cross-device copy is the only inter-GPU transfer per sample.
        ranker_device = next(self.commit_ranker.parameters()).device
        h = h.detach().to(ranker_device)
        commit_indices = commit_indices.to(ranker_device)

        scores, attention_weights = self.commit_ranker(
            h, commit_indices, num_commits)

        return scores, attention_weights


# ==========================================================================
#  Loss Functions
# ==========================================================================

class LabelSmoothingRankingLoss(nn.Module):
    """
    Listwise ranking loss with label smoothing (same as train_temporal_gt_kfold.py).
    
    Instead of hard targets (1 for ground truth, 0 for others),
    uses soft targets: (1 - smoothing)/num_gt for GT, 
    smoothing/max(num_commits - num_gt, 1) for others, then normalized.
    """
    def __init__(self, temperature=1.0, margin=1.0, smoothing=0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.smoothing = smoothing

    def forward(self, scores, ground_truth_positions):
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        num_commits = len(scores)
        num_gt = len(ground_truth_positions)
        
        if num_gt == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Create soft targets with label smoothing (matching original)
        # GT positions get (1 - smoothing)/num_gt, others get smoothing/max(num_commits - num_gt, 1)
        soft_targets = torch.full_like(scores, self.smoothing / max(num_commits - num_gt, 1))
        for gt_pos in ground_truth_positions:
            if 0 <= gt_pos < num_commits:
                soft_targets[gt_pos] = (1.0 - self.smoothing) / num_gt
        
        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum()

        # Cross-entropy with soft targets
        log_probs = F.log_softmax(scores / self.temperature, dim=0)
        ce_loss = -torch.sum(soft_targets * log_probs)

        # Margin-based ranking loss (matching original's per-pair margin)
        margin_loss = torch.tensor(0.0, device=scores.device)
        for gt_pos in ground_truth_positions:
            if 0 <= gt_pos < num_commits:
                gt_score = scores[gt_pos]
                for i in range(num_commits):
                    if i not in ground_truth_positions:
                        margin_loss += F.relu(self.margin - (gt_score - scores[i]))
        
        if num_gt > 0 and num_commits > num_gt:
            margin_loss = margin_loss / (num_gt * (num_commits - num_gt))

        return ce_loss + 0.5 * margin_loss


# ==========================================================================
#  Utility Functions
# ==========================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_devices(gpu_ids=None):
    """
    Configure primary DEVICE and optional DataParallel GPU list.

    Returns
    -------
    primary : torch.device   — device used for .to() calls
    dp_ids  : list[int]|None — GPU ids for nn.DataParallel (None = no DP)
    """
    if not torch.cuda.is_available():
        return torch.device("cpu"), None

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return torch.device("cpu"), None

    if gpu_ids is None:
        gpu_ids = list(range(n_gpus))

    gpu_ids = [g for g in gpu_ids if g < n_gpus]
    if not gpu_ids:
        return torch.device("cpu"), None

    primary = torch.device(f"cuda:{gpu_ids[0]}")
    dp_ids = gpu_ids if len(gpu_ids) > 1 else None
    return primary, dp_ids


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


def compute_metrics(scores, ground_truth_positions, k_values=[1, 2, 3, 5]):
    """Compute ranking metrics — global aggregation style."""
    if len(ground_truth_positions) == 0:
        empty = {}
        for k in k_values:
            empty[f"tp@{k}"] = 0
            empty[f"fp@{k}"] = 0
        empty["num_gt"] = 0
        empty["mrr"] = 0.0
        empty["first_rank"] = float('inf')
        return empty

    sorted_indices = torch.argsort(scores, descending=True).tolist()
    gt_set = set(ground_truth_positions)
    num_gt = len(gt_set)
    metrics = {}

    for k in k_values:
        top_k = set(sorted_indices[:k])
        hits = len(top_k & gt_set)
        metrics[f"tp@{k}"] = 1 if hits > 0 else 0
        metrics[f"fp@{k}"] = 1 if hits == 0 else 0

    metrics["num_gt"] = num_gt

    first_rank = float('inf')
    for rank, idx in enumerate(sorted_indices, 1):
        if idx in gt_set:
            first_rank = rank
            metrics["mrr"] = 1.0 / rank
            break
    else:
        metrics["mrr"] = 0.0
    metrics["first_rank"] = first_rank if first_rank != float('inf') else len(sorted_indices) + 1

    return metrics


def aggregate_global_metrics(all_metrics, k_values=[1, 2, 3, 5]):
    """Aggregate raw counts into global precision/recall/F1."""
    result = {}
    total_gt = sum(all_metrics.get("num_gt", [0]))

    for k in k_values:
        tp_total = sum(all_metrics.get(f"tp@{k}", [0]))
        fp_total = sum(all_metrics.get(f"fp@{k}", [0]))
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        recall = tp_total / total_gt if total_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        result[f"precision@{k}"] = precision
        result[f"recall@{k}"] = recall
        result[f"f1@{k}"] = f1
        result[f"success@{k}"] = precision

    mrr_values = all_metrics.get("mrr", [0])
    result["mrr"] = float(np.mean(mrr_values)) if mrr_values else 0.0
    result["accuracy"] = result.get("precision@1", 0.0)

    fr_values = all_metrics.get("first_rank", [])
    result["first_rank"] = float(np.mean(fr_values)) if fr_values else 0.0

    result["total_gt"] = total_gt
    for k in k_values:
        result[f"tp_total@{k}"] = sum(all_metrics.get(f"tp@{k}", [0]))
        result[f"fp_total@{k}"] = sum(all_metrics.get(f"fp@{k}", [0]))

    return result


# ==========================================================================
#  Phase 1: Deletion Line Ranking (same as train_deletion_line_ranking.py)
# ==========================================================================

def get_phase1_optimizer_param_groups(model, config):
    """
    Differential learning rates for Phase 1:
      - CodeBERT (125M params): small LR to preserve pre-trained knowledge (e.g. 2e-5).
      - Graph layers + ranker (random init): larger LR for faster learning (e.g. 1e-4).

    Returns a list of param groups for torch.optim.Adam(param_groups).
    """
    include_bert = config.get("include_bert", True)
    bert_lr = config.get("phase1_bert_lr")
    rest_lr = config.get("phase1_rest_lr")
    fallback_lr = config["phase1_lr"]

    if not include_bert or (bert_lr is None and rest_lr is None):
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": fallback_lr}]

    # BERT parameters (only those with requires_grad; bottom layers may be frozen)
    bert_params = [p for p in model.encoder.bert_model.parameters() if p.requires_grad] if include_bert else []
    # Rest: input_proj, gt_layers, ranker (temporal_pos_embedding is a fixed buffer, not a parameter)
    rest_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "bert_model" not in name
    ]

    groups = []
    if bert_params:
        groups.append({"params": bert_params, "lr": bert_lr if bert_lr is not None else fallback_lr})
    if rest_params:
        groups.append({"params": rest_params, "lr": rest_lr if rest_lr is not None else fallback_lr})
    return groups if groups else [{"params": [p for p in model.parameters() if p.requires_grad], "lr": fallback_lr}]


def train_epoch_phase1(model, pairs, optimizer, loss_fn, device, batch_size=4, verbose=False):
    """
    Train one epoch of Phase 1 (pairwise deletion line ranking).
    
    NOTE: This is the per-pair iteration version (kept for reference).
    The main training now uses _train_epoch_phase1_batched() which
    uses combine_pairs_to_batches() to match the original exactly.
    """
    model.train()
    random.shuffle(pairs)

    total_loss = 0.0
    total_pairs = 0
    skipped_pairs = 0
    error_counts = defaultdict(int)
    optimizer.zero_grad(set_to_none=True)

    for i, pair in enumerate(pairs):
        try:
            gd_x = pair.x.pyg.to(device)
            gd_y = pair.y.pyg.to(device)
            del_idx_x = pair.x.del_idx
            del_idx_y = pair.y.del_idx

            if isinstance(del_idx_x, torch.Tensor):
                del_idx_x = del_idx_x.item() if del_idx_x.numel() == 1 else del_idx_x[0].item()
            if isinstance(del_idx_y, torch.Tensor):
                del_idx_y = del_idx_y.item() if del_idx_y.numel() == 1 else del_idx_y[0].item()

            if del_idx_x >= gd_x.num_nodes or del_idx_y >= gd_y.num_nodes:
                skipped_pairs += 1
                continue

            prob_pred = model(gd_x, del_idx_x, gd_y, del_idx_y)

            target = torch.tensor([pair.prob], dtype=torch.float32, device=device)
            loss = loss_fn(prob_pred.view(1), target) / batch_size

            if not torch.isnan(loss):
                loss.backward()
                total_loss += loss.item() * batch_size
                total_pairs += 1
            else:
                error_counts["nan_loss"] += 1

            if (i + 1) % batch_size == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        except Exception as e:
            error_type = type(e).__name__
            error_counts[error_type] += 1
            if verbose and error_counts[error_type] <= 3:
                print(f"      Warning: {error_type}: {str(e)[:100]}")
            continue
        finally:
            pair.x.release_pyg()
            pair.y.release_pyg()
            gc.collect()

    # Final gradient step
    if total_pairs % batch_size != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if verbose or total_pairs == 0:
        print(f"      Phase 1 Train: {total_pairs} valid pairs, {skipped_pairs} skipped, errors: {dict(error_counts)}")

    return total_loss / max(total_pairs, 1)


def _train_epoch_phase1_batched(model, train_pairs, criterion, optimizer,
                                batch_size=128, max_nodes=9500):
    """
    Train for one epoch — line ranking head only.

    Per-pair gradient accumulation within each batch.
    Graphs exceeding ``max_nodes`` are skipped to prevent OOM.

    Performance notes:
    - gc.collect() + empty_cache() are called per-BATCH, not per-pair.
      Per-pair calls added ~28s overhead/epoch and caused CUDA sync stalls.
    - release_pyg() alone is sufficient to free graph memory between pairs.
    """
    model.train()
    random.shuffle(train_pairs)
    batches = combine_pairs_to_batches(train_pairs, batch_size)
    loss_sum = 0.0
    error_counts = defaultdict(int)
    total_valid = 0
    skipped_large = 0
    for batch in batches:
        optimizer.zero_grad()
        batch_loss = 0.0
        n_valid = 0
        for pair in batch.pairs:
            try:
                x_graph = pair.x.pyg
                y_graph = pair.y.pyg
                if x_graph is None or y_graph is None:
                    continue
                if x_graph.num_nodes > max_nodes or y_graph.num_nodes > max_nodes:
                    skipped_large += 1
                    continue
                x_graph = x_graph.to(DEVICE)
                y_graph = y_graph.to(DEVICE)
                x_del_idx = pair.x.del_idx
                if isinstance(x_del_idx, torch.Tensor):
                    x_del_idx = x_del_idx.item() if x_del_idx.numel() == 1 else x_del_idx[0].item()
                y_del_idx = pair.y.del_idx
                if isinstance(y_del_idx, torch.Tensor):
                    y_del_idx = y_del_idx.item() if y_del_idx.numel() == 1 else y_del_idx[0].item()
                if x_del_idx >= x_graph.num_nodes:
                    continue
                if y_del_idx >= y_graph.num_nodes:
                    continue
                line_prob = model(x_graph, x_del_idx, y_graph, y_del_idx)
                tgt = torch.tensor([pair.prob], dtype=torch.float32, device=DEVICE)
                pair_loss = criterion(line_prob.view(1), tgt)
                (pair_loss / len(batch.pairs)).backward()
                batch_loss += pair_loss.item()
                n_valid += 1
            except Exception as e:
                etype = type(e).__name__
                error_counts[etype] += 1
                if error_counts[etype] <= 3:
                    print(f"      [Train] {etype}: {str(e)[:120]}")
                if "CUDA" in str(e) or "out of memory" in str(e):
                    torch.cuda.empty_cache()
                continue
            finally:
                pair.x.release_pyg()
                pair.y.release_pyg()

        if n_valid > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_valid += n_valid
        loss_sum += batch_loss / max(n_valid, 1)
        # Free CUDA cache once per batch (not per-pair)
        torch.cuda.empty_cache()

    if error_counts or skipped_large:
        print(f"      [Train] valid: {total_valid}/{len(train_pairs)}, "
              f"skipped_large: {skipped_large}, errors: {dict(error_counts)}")

    return loss_sum / max(len(batches), 1)


def _validate_epoch_phase1_batched(model, val_pairs, criterion, batch_size=128,
                                   max_nodes=9500):
    """Validate for one epoch — line ranking head only."""
    model.eval()
    batches = combine_pairs_to_batches(val_pairs, batch_size)
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in batches:
            batch_loss = 0.0
            n_valid = 0
            for pair in batch.pairs:
                try:
                    x_graph = pair.x.pyg
                    y_graph = pair.y.pyg
                    if x_graph is None or y_graph is None:
                        continue
                    if x_graph.num_nodes > max_nodes or y_graph.num_nodes > max_nodes:
                        continue
                    x_graph = x_graph.to(DEVICE)
                    y_graph = y_graph.to(DEVICE)
                    x_del_idx = pair.x.del_idx
                    if isinstance(x_del_idx, torch.Tensor):
                        x_del_idx = x_del_idx.item() if x_del_idx.numel() == 1 else x_del_idx[0].item()
                    y_del_idx = pair.y.del_idx
                    if isinstance(y_del_idx, torch.Tensor):
                        y_del_idx = y_del_idx.item() if y_del_idx.numel() == 1 else y_del_idx[0].item()
                    if x_del_idx >= x_graph.num_nodes:
                        continue
                    if y_del_idx >= y_graph.num_nodes:
                        continue
                    line_prob = model(x_graph, x_del_idx, y_graph, y_del_idx)
                    tgt = torch.tensor([pair.prob], dtype=torch.float32, device=DEVICE)
                    pair_loss = criterion(line_prob.view(1), tgt)
                    batch_loss += pair_loss.item()
                    n_valid += 1
                except Exception:
                    continue
                finally:
                    pair.x.release_pyg()
                    pair.y.release_pyg()

            if n_valid > 0:
                loss_sum += batch_loss / n_valid
                n_batches += 1
            torch.cuda.empty_cache()  # per-batch, not per-pair

    return loss_sum / max(n_batches, 1)


def validate_epoch_phase1(model, pairs, loss_fn, device):
    """
    Validate Phase 1 (legacy per-pair version, kept for reference).
    The main training now uses _validate_epoch_phase1_batched().
    """
    model.eval()
    total_loss = 0.0
    total_pairs = 0

    with torch.no_grad():
        for pair in pairs:
            try:
                gd_x = pair.x.pyg.to(device)
                gd_y = pair.y.pyg.to(device)
                del_idx_x = pair.x.del_idx
                del_idx_y = pair.y.del_idx

                if isinstance(del_idx_x, torch.Tensor):
                    del_idx_x = del_idx_x.item() if del_idx_x.numel() == 1 else del_idx_x[0].item()
                if isinstance(del_idx_y, torch.Tensor):
                    del_idx_y = del_idx_y.item() if del_idx_y.numel() == 1 else del_idx_y[0].item()

                if del_idx_x >= gd_x.num_nodes or del_idx_y >= gd_y.num_nodes:
                    continue

                prob_pred = model(gd_x, del_idx_x, gd_y, del_idx_y)

                target = torch.tensor([pair.prob], dtype=torch.float32, device=device)
                loss = loss_fn(prob_pred.view(1), target)

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    total_pairs += 1

            except Exception:
                continue
            finally:
                pair.x.release_pyg()
                pair.y.release_pyg()

    return total_loss / max(total_pairs, 1)


def evaluate_phase1_ranking(model, dataset, test_cases, device, data_path=None):
    """
    Evaluate Phase 1 line ranking: score each deletion line, compute
    precision/recall/F1 using the SAME evaluation as train_deletion_line_ranking.py.
    
    Uses evaluate_top1_metrics from loss_and_metrics.py which reads info.json
    for ground truth inducing commits, matching NeuralSZZ's eval_top semantics.
    """
    if data_path is None:
        data_path = CONFIG["data_path"]
    
    model.eval()
    mini_graphs_dict = dataset.get_mini_graphs_dict()

    # Load ground truth commit map
    true_cid_map = load_true_commit_map(test_cases, data_path)
    
    test_cases_graphs = {}

    with torch.no_grad():
        for test_name in test_cases:
            if test_name not in mini_graphs_dict:
                continue
            graphs = mini_graphs_dict[test_name]
            if not graphs:
                continue

            for mg in graphs:
                try:
                    gd = mg.pyg.to(device)
                    del_idx = mg.del_idx
                    if isinstance(del_idx, torch.Tensor):
                        del_idx = del_idx.item() if del_idx.numel() == 1 else del_idx[0].item()
                    if del_idx >= gd.num_nodes:
                        mg.score = 0.0
                        continue
                    score = model.predict(gd, del_idx)
                    mg.score = score.item()
                except Exception:
                    mg.score = 0.0
                    continue
                finally:
                    mg.release_pyg()

            graphs.sort(key=lambda g: g.score, reverse=True)
            test_cases_graphs[test_name] = graphs
            if hasattr(dataset, "release_test_case"):
                dataset.release_test_case(test_name)

    metrics = evaluate_top1_metrics(test_cases_graphs, true_cid_map, data_path)
    return metrics


def extract_phase1_commits(
    model, dataset, test_cases, data_path, top_k=2
):
    """
    Extract commits from top-K ranked deletion lines (in-memory,
    no file I/O). Returns dict {test_name: {fix_commit, selected_commits, ...}}.
    """
    model.eval()
    mini_graphs_dict = dataset.get_mini_graphs_dict()
    results = OrderedDict()

    with torch.no_grad():
        for test_name in test_cases:
            if test_name not in mini_graphs_dict:
                continue
            graphs = mini_graphs_dict[test_name]
            if not graphs:
                continue

            scored = []
            for mg in graphs:
                try:
                    gd = mg.pyg.to(DEVICE)
                    del_idx = mg.del_idx
                    if isinstance(del_idx, torch.Tensor):
                        del_idx = del_idx.item() if del_idx.numel() == 1 else del_idx[0].item()
                    if del_idx >= gd.num_nodes:
                        continue
                    score = model.predict(gd, del_idx).item()
                    scored.append((score, mg))
                except Exception:
                    continue
                finally:
                    mg.release_pyg()

            if not scored:
                continue

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k_lines = scored[:top_k]

            # Collect union of commits from top-K lines
            all_shas_12 = OrderedDict()
            for score, mg in top_k_lines:
                if mg.tp_to_commit:
                    for tp, sha in mg.tp_to_commit.items():
                        if sha and sha not in all_shas_12:
                            all_shas_12[sha] = True

            selected_shas_12 = list(all_shas_12.keys())
            if not selected_shas_12:
                continue

            # Load commits.json for fix_commit and ground_truth
            commits_path = Path(data_path) / test_name / "commits.json"
            fix_commit = ""
            ground_truth = []
            if commits_path.exists():
                with open(commits_path) as f:
                    cdata = json.load(f)
                fix_commit = cdata.get("fix_commit", "")
                ground_truth = cdata.get("ground_truth", [])

            # Resolve short SHAs to full SHAs
            test_dir = Path(data_path) / test_name
            full_shas = _resolve_full_shas(test_dir, selected_shas_12, fix_commit)

            results[test_name] = {
                "fix_commit": fix_commit,
                "selected_commits": full_shas,
                "ground_truth": ground_truth,
            }

    return results


def _resolve_full_shas(test_dir, short_shas, fix_commit):
    """Resolve 12-char SHAs to full SHAs from directories on disk."""
    if not test_dir.exists():
        return short_shas

    short_to_full = {}
    for d in test_dir.iterdir():
        if d.is_dir() and len(d.name) == 40:
            short_to_full[d.name[:12]] = d.name
    if fix_commit:
        short_to_full[fix_commit[:12]] = fix_commit

    resolved = []
    for sha12 in short_shas:
        full = short_to_full.get(sha12, sha12)
        if full not in resolved:
            resolved.append(full)
    return resolved


# ==========================================================================
#  Phase 1 Training for a Single Fold
# ==========================================================================

def train_phase1_fold(
    fold_idx, train_test_cases, val_test_cases, embedder, config
):
    """
    Train Phase 1 (deletion line ranking) for one fold.
    
    Mirrors train_deletion_line_ranking.py exactly:
      - Separate train/val DeletionLineDatasets (not shared)
      - Uses dataset.get_all_pairs() for pair generation (per-test-case limit)
      - Uses combine_pairs_to_batches() for batching
      - Uses Adam optimizer with ReduceLROnPlateau scheduler
      - Uses evaluate_top1_metrics (NeuralSZZ eval_top) for evaluation
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 1 — Fold {fold_idx + 1}: Deletion Line Ranking")
    print(f"{'─'*60}")

    # Load ground truth for validation
    val_true_cid_map = load_true_commit_map(val_test_cases, config["data_path"])
    total_inducing = sum(len(c) for c in val_true_cid_map.values())
    print(f"  Val inducing commits: {total_inducing}")

    prebuilt_dir = config["prebuilt_dir"]
    source = "pre-built JSON" if prebuilt_dir else "on-the-fly"
    print(f"\n  Loading training dataset ({config['graph_mode']}, {source})...")
    train_dataset = DeletionLineDataset(
        data_path=config["data_path"],
        test_cases=train_test_cases,
        embedder=embedder,
        graph_mode=config["graph_mode"],
        prebuilt_dir=prebuilt_dir,
        lazy=True,
    )
    print(f"  Loading validation dataset ({config['graph_mode']}, {source})...")
    val_dataset = DeletionLineDataset(
        data_path=config["data_path"],
        test_cases=val_test_cases,
        embedder=embedder,
        graph_mode=config["graph_mode"],
        prebuilt_dir=prebuilt_dir,
        lazy=True,
    )

    print("  Generating deletion line pairs...")
    train_pairs = train_dataset.get_all_pairs(
        max_pairs_per_test=config["phase1_max_pairs_per_test"])
    val_pairs = val_dataset.get_all_pairs(
        max_pairs_per_test=config["phase1_max_pairs_per_test"])
    print(f"  Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs")

    if not train_pairs:
        print(f"  WARNING: No training pairs for fold {fold_idx + 1}")
        return None

    # Analyze pair distribution
    prob_counts = defaultdict(int)
    for pair in train_pairs:
        prob_counts[pair.prob] += 1
    
    print(f"    Pair distribution (train): "
          f"prob=1.0: {prob_counts.get(1.0, 0)}, "
          f"prob=0.0: {prob_counts.get(0.0, 0)}, "
          f"prob=0.5: {prob_counts.get(0.5, 0)}")

    # Model
    bert_chunk = config.get("bert_chunk", 256)
    model = DeletionLineRankingModel(
        input_dim=EMB_DIM,
        hidden_dim=config["hidden_dim"],
        num_gt_layers=config["num_gt_layers"],
        num_heads=config["num_heads"],
        num_edge_types=config["num_edge_types"],
        dropout=config["dropout"],
        include_bert=config.get("include_bert", True),
        num_bert_layers_freeze=config.get("phase1_bert_freeze_bottom_layers", 0),
        bert_chunk=bert_chunk,
    ).to(DEVICE)

    # Model-parallel: move CodeBERT to GPU 1 if available
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2 and config.get("include_bert", True):
        bert_dev = torch.device("cuda:1") if DEVICE == torch.device("cuda:0") else torch.device("cuda:0")
        model.set_bert_device(bert_dev)
        print(f"    Model-parallel: BERT on {bert_dev}, "
              f"GT+ranker on {DEVICE}  (BERT_CHUNK={bert_chunk})")
    else:
        print(f"    Single GPU: {DEVICE}  (BERT_CHUNK={bert_chunk})")

    n_freeze = config.get("phase1_bert_freeze_bottom_layers", 0)
    if n_freeze > 0 and config.get("include_bert", True) and hasattr(model.encoder, "bert_model"):
        n_total = len(model.encoder.bert_model.encoder.layer)
        print(f"    BERT: freezing bottom {n_freeze} layers + embeddings; fine-tuning top {n_total - n_freeze} layers")
    param_groups = get_phase1_optimizer_param_groups(model, config)
    optimizer = torch.optim.Adam(param_groups)
    for i, g in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in g["params"])
        print(f"    Optimizer group {i}: lr={g['lr']:.2e}, params={n:,}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True)
    criterion = PairwiseRankingLoss()

    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config["phase1_epochs"] + 1):
        gc.collect()
        torch.cuda.empty_cache()

        # Train one epoch using combine_pairs_to_batches (matching original)
        train_loss = _train_epoch_phase1_batched(
            model, train_pairs, criterion, optimizer,
            batch_size=config["phase1_batch_size"],
            max_nodes=config.get("max_nodes_per_graph", 9500))
        
        val_loss = _validate_epoch_phase1_batched(
            model, val_pairs, criterion,
            batch_size=config["phase1_batch_size"],
            max_nodes=config.get("max_nodes_per_graph", 9500))

        # Evaluate ranking on validation set (NeuralSZZ eval_top style)
        val_metrics = evaluate_phase1_ranking(
            model, val_dataset, val_test_cases, DEVICE, config["data_path"])

        # evaluate_top1_metrics returns "F1@1" (uppercase F)
        current_f1 = val_metrics.get("F1@1", 0.0)
        
        scheduler.step(current_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val P@1={val_metrics.get('precision@1', 0):.4f}, "
                  f"R@1={val_metrics.get('recall@1', 0):.4f}, "
                  f"F1@1={current_f1:.4f}")

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            print(f"    ✓ New best Phase 1 "
                  f"(F1@1: {current_f1:.4f}, epoch {epoch})")
        else:
            patience_counter += 1

        if patience_counter >= config["phase1_patience"]:
            print(f"    ⚠️  Phase 1 early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    final_metrics = evaluate_phase1_ranking(
        model, val_dataset, val_test_cases, DEVICE, config["data_path"])

    final_f1 = final_metrics.get("F1@1", 0.0)
    print(f"\n  Phase 1 Fold {fold_idx + 1} Best (Epoch {best_epoch}):")
    print(f"    P@1={final_metrics.get('precision@1', 0):.4f}, "
          f"R@1={final_metrics.get('recall@1', 0):.4f}, "
          f"F1@1={final_f1:.4f}")

    return {
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "metrics": final_metrics,
    }


# ==========================================================================
#  Phase 2: Commit Ranking Training for a Single Fold
# ==========================================================================

def collate_fn_phase2(batch):
    """Filter out invalid samples."""
    return [item for item in batch if item.get("valid", False)]


def train_epoch_phase2(
    model, dataloader, optimizer, loss_fn, device, epoch,
    log_interval=20, gradient_accumulation_steps=4
):
    """Train one epoch of Phase 2 (commit ranking)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_metrics = defaultdict(list)

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        if not batch:
            continue

        batch_loss = 0.0
        batch_count = 0

        for sample in batch:
            try:
                data = sample["data"].to(device)
                commit_indices = sample["commit_indices"].to(device)
                gt_positions = sample["ground_truth_positions"]
                num_commits = sample["num_commits"]

                scores, _ = model(data, commit_indices, num_commits)

                loss = loss_fn(scores, gt_positions) / gradient_accumulation_steps

                if loss.requires_grad and not torch.isnan(loss):
                    loss.backward()
                    batch_loss += loss.item() * gradient_accumulation_steps
                    batch_count += 1

                with torch.no_grad():
                    metrics = compute_metrics(scores, gt_positions)
                    for k, v in metrics.items():
                        all_metrics[k].append(v)

            except Exception as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                continue

        # Light cleanup after each batch (don't gc.collect every batch — too slow)
        if (batch_idx + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            has_grads = any(
                p.grad is not None for p in model.parameters() if p.requires_grad)
            if has_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if batch_count > 0:
            total_loss += batch_loss
            total_samples += batch_count

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / max(total_samples, 1)
            running = aggregate_global_metrics(all_metrics)
            print(f"    Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"Loss={avg_loss:.4f}, "
                  f"P@1={running.get('precision@1', 0):.4f}, "
                  f"R@1={running.get('recall@1', 0):.4f}, "
                  f"F1@1={running.get('f1@1', 0):.4f}, "
                  f"Time={elapsed:.1f}s")

    # Final gradient step
    if total_samples > 0 and (batch_idx + 1) % gradient_accumulation_steps != 0:
        has_grads = any(
            p.grad is not None for p in model.parameters() if p.requires_grad)
        if has_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = aggregate_global_metrics(all_metrics)
    return avg_loss, avg_metrics


def evaluate_phase2(model, dataloader, loss_fn, device):
    """Evaluate Phase 2 on validation set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if not batch:
                continue

            for sample in batch:
                try:
                    data = sample["data"].to(device)
                    commit_indices = sample["commit_indices"].to(device)
                    gt_positions = sample["ground_truth_positions"]
                    num_commits = sample["num_commits"]

                    scores, _ = model(data, commit_indices, num_commits)

                    loss = loss_fn(scores, gt_positions)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        total_samples += 1

                    metrics = compute_metrics(scores, gt_positions)
                    for k, v in metrics.items():
                        all_metrics[k].append(v)

                except Exception:
                    continue

            if (batch_idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = aggregate_global_metrics(all_metrics)
    return avg_loss, avg_metrics


def train_phase2_fold(
    fold_idx, train_indices, val_indices,
    phase2_dataset, frozen_encoder, config
):
    """Train Phase 2 (commit ranking with frozen Phase 1 encoder) for one fold."""
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — Fold {fold_idx + 1}: Commit Ranking (Frozen Encoder)")
    print(f"{'─'*60}")

    # Resolve multi-GPU setup
    primary_device, dp_ids = setup_devices(config.get("gpu_ids"))

    num_workers = config.get("num_workers", 0)

    # Create data loaders
    train_subset = Subset(phase2_dataset, train_indices)
    val_subset = Subset(phase2_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config["phase2_batch_size"],
        shuffle=True,
        collate_fn=collate_fn_phase2,
        num_workers=num_workers,
        pin_memory=(primary_device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config["phase2_batch_size"],
        shuffle=False,
        collate_fn=collate_fn_phase2,
        num_workers=num_workers,
        pin_memory=(primary_device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Build unified model with frozen encoder
    model = UnifiedPhase2Model(
        encoder=frozen_encoder,
        hidden_dim=config["hidden_dim"],
        num_heads=config.get("phase2_num_heads", 8),
        num_commit_transformer_layers=config["num_commit_transformer_layers"],
        max_commits=config["max_commits"],
        dropout=config["dropout"],
    ).to(primary_device)

    # DataParallel is NOT compatible with CommitRankingModule: each sample has
    # a variable number of nodes and commits, so tensors cannot be stacked to
    # the same shape across a batch — DataParallel would raise dimension errors
    # and silently swallow every sample in the except branch (loss=0, P@1=0).
    #
    # Instead we use pipeline parallelism: pin the frozen encoder on GPU 1 so
    # it does not consume GPU 0 VRAM, leaving GPU 0 fully available for the
    # trainable CommitRankingModule and its optimizer state.
    # Both GPUs stay busy: GPU 1 runs encoder.forward() (torch.no_grad),
    # GPU 0 runs commit_ranker.forward() + backward.
    if dp_ids and len(dp_ids) >= 2:
        encoder_device = torch.device(f"cuda:{dp_ids[1]}")
        model.encoder = model.encoder.to(encoder_device)
        print(f"  Pipeline parallel: encoder on cuda:{dp_ids[1]}, "
              f"commit_ranker on cuda:{dp_ids[0]}")
    else:
        encoder_device = primary_device
        print(f"  Single device: {primary_device}")

    # Only optimize commit ranking parameters (encoder is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {num_trainable:,} / {num_total:,} total "
          f"({100*num_trainable/num_total:.1f}%)")

    optimizer = AdamW(
        trainable_params,
        lr=config["phase2_lr"],
        weight_decay=config["phase2_weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    loss_fn = LabelSmoothingRankingLoss(
        temperature=config["phase2_temperature"],
        margin=1.0,
        smoothing=config["phase2_label_smoothing"],
    )

    early_stopping = EarlyStopping(
        patience=config["phase2_patience"], mode='max')

    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    fold_history = {
        "train_loss": [], "val_loss": [],
        "train_f1@1": [], "val_f1@1": [],
    }

    for epoch in range(1, config["phase2_epochs"] + 1):
        gc.collect()
        torch.cuda.empty_cache()

        epoch_start = time.time()

        # Train
        train_loss, train_metrics = train_epoch_phase2(
            model, train_loader, optimizer, loss_fn, primary_device, epoch,
            config["log_interval"], config["phase2_gradient_accumulation_steps"])

        gc.collect()
        torch.cuda.empty_cache()

        # Validate
        val_loss, val_metrics = evaluate_phase2(
            model, val_loader, loss_fn, primary_device)

        gc.collect()
        torch.cuda.empty_cache()

        val_f1 = val_metrics.get('f1@1', 0)
        scheduler.step(val_f1)

        epoch_time = time.time() - epoch_start

        if epoch % 5 == 0 or epoch == 1:
            print(f"\n    Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Train P@1={train_metrics.get('precision@1', 0):.4f}, "
                  f"R@1={train_metrics.get('recall@1', 0):.4f}, "
                  f"F1@1={train_metrics.get('f1@1', 0):.4f} | "
                  f"Val P@1={val_metrics.get('precision@1', 0):.4f}, "
                  f"R@1={val_metrics.get('recall@1', 0):.4f}, "
                  f"F1@1={val_f1:.4f}, "
                  f"Time={epoch_time:.1f}s")

        fold_history["train_loss"].append(train_loss)
        fold_history["val_loss"].append(val_loss)
        fold_history["train_f1@1"].append(train_metrics.get('f1@1', 0))
        fold_history["val_f1@1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.commit_ranker.state_dict())
            print(f"    ✓ New best Phase 2 "
                  f"(P@1={val_metrics.get('precision@1', 0):.4f}, "
                  f"R@1={val_metrics.get('recall@1', 0):.4f}, "
                  f"F1@1={val_f1:.4f})")

        if early_stopping(val_f1, epoch):
            print(f"    ⚠️  Phase 2 early stopping at epoch {epoch}")
            break

    # Load best and final eval
    if best_model_state is not None:
        model.commit_ranker.load_state_dict(best_model_state)

    final_val_loss, final_val_metrics = evaluate_phase2(
        model, val_loader, loss_fn, primary_device)

    print(f"\n  Phase 2 Fold {fold_idx + 1} Best (Epoch {best_epoch}):")
    print(f"    P@1={final_val_metrics.get('precision@1', 0):.4f}, "
          f"R@1={final_val_metrics.get('recall@1', 0):.4f}, "
          f"F1@1={final_val_metrics.get('f1@1', 0):.4f}, "
          f"MRR={final_val_metrics.get('mrr', 0):.4f}")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fold": fold_idx + 1,
        "best_epoch": best_epoch,
        "best_val_f1@1": best_val_f1,
        "final_metrics": final_val_metrics,
        "history": fold_history,
        "best_commit_ranker_state": best_model_state,
    }


# ==========================================================================
#  Main: Unified Two-Phase K-Fold Training
# ==========================================================================

def get_num_commits_for_stratification(test_cases, data_path):
    """Get commit count bins for stratified splitting."""
    commit_bins = []
    for test_name in test_cases:
        commits_path = Path(data_path) / test_name / "commits.json"
        if commits_path.exists():
            with open(commits_path) as f:
                commits_data = json.load(f)
            all_commits = set()
            all_commits.add(commits_data.get("fix_commit", ""))
            all_commits.update(commits_data.get("ground_truth_inducers", []))
            all_commits.update(commits_data.get("all_commits_in_history", []))
            all_commits.discard("")
            commit_bins.append(min(len(all_commits), 5))
        else:
            commit_bins.append(2)
    return commit_bins


def main():
    parser = argparse.ArgumentParser(
        description="Unified Two-Phase Training: "
                    "Deletion Line Ranking → Commit Ranking (Frozen Encoder)")

    # Phase 1 overrides
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase1-lr", type=float, default=None,
                        help="Single LR for all params (fallback)")
    parser.add_argument("--phase1-bert-lr", type=float, default=None,
                        help="LR for CodeBERT (e.g. 2e-5)")
    parser.add_argument("--phase1-rest-lr", type=float, default=None,
                        help="LR for graph layers + ranker (e.g. 1e-4)")
    parser.add_argument("--phase1-bert-freeze-bottom-layers", type=int, default=None,
                        help="Freeze bottom N BERT layers + embeddings (0 = train all, 6 = train top 6 only)")
    parser.add_argument("--phase1-batch-size", type=int, default=None)
    parser.add_argument("--phase1-patience", type=int, default=None)

    # Phase 2 overrides
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--phase2-lr", type=float, default=None)
    parser.add_argument("--phase2-batch-size", type=int, default=None)
    parser.add_argument("--phase2-patience", type=int, default=None)

    # Shared
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-gt-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--top-k-lines", type=int, default=None,
                        help="Top-K deletion lines to extract commits from")
    parser.add_argument("--no-stratify", action="store_true")

    # Skip Phase 1 (use existing checkpoints)
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 training and load existing "
                             "checkpoints from save-dir")
    parser.add_argument("--phase1-checkpoint-dir", type=str, default=None,
                        help="Directory with existing Phase 1 checkpoints "
                             "(fold*_phase1_best.pt). If not set, uses save-dir.")

    args = parser.parse_args()

    # Override config
    if args.phase1_epochs is not None: CONFIG["phase1_epochs"] = args.phase1_epochs
    if args.phase1_lr is not None: CONFIG["phase1_lr"] = args.phase1_lr
    if args.phase1_bert_lr is not None: CONFIG["phase1_bert_lr"] = args.phase1_bert_lr
    if args.phase1_rest_lr is not None: CONFIG["phase1_rest_lr"] = args.phase1_rest_lr
    if args.phase1_bert_freeze_bottom_layers is not None:
        CONFIG["phase1_bert_freeze_bottom_layers"] = args.phase1_bert_freeze_bottom_layers
    if args.phase1_batch_size is not None: CONFIG["phase1_batch_size"] = args.phase1_batch_size
    if args.phase1_patience is not None: CONFIG["phase1_patience"] = args.phase1_patience
    if args.phase2_epochs is not None: CONFIG["phase2_epochs"] = args.phase2_epochs
    if args.phase2_lr is not None: CONFIG["phase2_lr"] = args.phase2_lr
    if args.phase2_batch_size is not None: CONFIG["phase2_batch_size"] = args.phase2_batch_size
    if args.phase2_patience is not None: CONFIG["phase2_patience"] = args.phase2_patience
    if args.hidden_dim is not None: CONFIG["hidden_dim"] = args.hidden_dim
    if args.num_gt_layers is not None: CONFIG["num_gt_layers"] = args.num_gt_layers
    if args.dropout is not None: CONFIG["dropout"] = args.dropout
    if args.n_folds is not None: CONFIG["n_folds"] = args.n_folds
    if args.seed is not None: CONFIG["seed"] = args.seed
    if args.save_dir is not None: CONFIG["save_dir"] = args.save_dir
    if args.top_k_lines is not None: CONFIG["phase2_top_k_lines"] = args.top_k_lines
    if args.no_stratify: CONFIG["stratify_by_commits"] = False

    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    phase1_ckpt_dir = Path(args.phase1_checkpoint_dir) if args.phase1_checkpoint_dir else Path(CONFIG["save_dir"])

    # ================================================================
    # Multi-GPU setup + performance flags
    # ================================================================
    primary_device, dp_ids = setup_devices(CONFIG.get("gpu_ids"))
    # Patch the module-level DEVICE used by Phase 1 helpers
    import model_architecture as _ma
    _ma.DEVICE = primary_device

    # Speed flags
    torch.backends.cudnn.benchmark = True   # auto-tune conv kernels
    torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere+
    torch.backends.cudnn.allow_tf32 = True

    n_gpus = len(dp_ids) if dp_ids else 1
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Primary device: {primary_device}  |  DataParallel GPUs: {dp_ids or 'disabled'}")
    print(f"torch.backends.cudnn.benchmark = True")

    # ================================================================
    # Print header
    # ================================================================
    print("=" * 70)
    print("UNIFIED TWO-PHASE TRAINING")
    print("  Phase 1: Deletion Line Ranking (SharedEncoder + DeletionLineRanker)")
    print("  Phase 2: Commit Ranking (Frozen Encoder + CommitRankingModule)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()

    # ================================================================
    # Load test cases
    # ================================================================
    test_cases_path = Path(CONFIG["data_path"]) / CONFIG["test_cases_file"]
    with open(test_cases_path) as f:
        all_test_cases = json.load(f)
    print(f"Total test cases: {len(all_test_cases)}")

    # ================================================================
    # Create K-Fold splits (SAME for both phases — leak-free)
    # ================================================================
    all_indices = np.arange(len(all_test_cases))

    if CONFIG["stratify_by_commits"]:
        commit_bins = get_num_commits_for_stratification(
            all_test_cases, CONFIG["data_path"])
        kfold = StratifiedKFold(
            n_splits=CONFIG["n_folds"],
            shuffle=True,
            random_state=CONFIG["seed"],
        )
        splits = list(kfold.split(all_indices, commit_bins))
    else:
        kfold = KFold(
            n_splits=CONFIG["n_folds"],
            shuffle=True,
            random_state=CONFIG["seed"],
        )
        splits = list(kfold.split(all_indices))

    print(f"\nK-Fold splits (shared by both phases):")
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")

    # ================================================================
    # Load Phase 1 dataset (deletion line graphs)
    # ================================================================
    # ================================================================
    # Initialize CodeBERT embedder (shared by Phase 1 caching & Phase 2)
    # ================================================================
    print("\nInitializing CodeBERT tokenizer ...")
    embedder = CodeBERTEmbedder(tokenizer_only=True)
    graph_builder = TemporalGraphBuilder(embedder)

    print("\n" + "=" * 70)
    print("LOADING PHASE 1 DATASET (Deletion Line Graphs)")
    print("=" * 70)

    phase1_dataset = DeletionLineDataset(
        data_path=CONFIG["data_path"],
        test_cases=all_test_cases,
        embedder=embedder,
        graph_mode=CONFIG["graph_mode"],
        prebuilt_dir=CONFIG["prebuilt_dir"],
        lazy=True,
    )

    # ================================================================
    # Per-fold training loop
    # ================================================================
    all_phase1_results = []
    all_phase2_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{CONFIG['n_folds']}")
        print(f"{'='*70}")
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

        train_test_cases = [all_test_cases[i] for i in train_indices]
        val_test_cases = [all_test_cases[i] for i in val_indices]

        # ==============================================================
        # Phase 1: Train or load deletion line ranker
        # ==============================================================
        phase1_ckpt_path = phase1_ckpt_dir / f"fold{fold_idx}_phase1_best.pt"

        if args.skip_phase1 and phase1_ckpt_path.exists():
            print(f"\n  Loading existing Phase 1 checkpoint: {phase1_ckpt_path}")
            ckpt = torch.load(phase1_ckpt_path, map_location=DEVICE)
            phase1_state = ckpt["model_state_dict"]
            phase1_result = {"model_state": phase1_state, "loaded_from": str(phase1_ckpt_path)}
        else:
            phase1_result = train_phase1_fold(
                fold_idx, train_test_cases, val_test_cases,
                embedder, CONFIG)

            if phase1_result is None:
                print(f"  SKIP: Phase 1 failed for fold {fold_idx + 1}")
                continue

            phase1_state = phase1_result["model_state"]

            # Save Phase 1 checkpoint
            torch.save(
                {"model_state_dict": phase1_state},
                os.path.join(CONFIG["save_dir"], f"fold{fold_idx}_phase1_best.pt"))

        all_phase1_results.append(phase1_result)

        # ==============================================================
        # Extract Phase 1 encoder
        # ==============================================================
        print(f"\n  Extracting frozen encoder from Phase 1...")
        temp_model = DeletionLineRankingModel(
            input_dim=EMB_DIM,
            hidden_dim=CONFIG["hidden_dim"],
            num_gt_layers=CONFIG["num_gt_layers"],
            num_heads=CONFIG["num_heads"],
            num_edge_types=CONFIG["num_edge_types"],
            dropout=CONFIG["dropout"],
            include_bert=CONFIG.get("include_bert", True),
        )
        temp_model.load_state_dict(phase1_state, strict=False)
        frozen_encoder = copy.deepcopy(temp_model.encoder)
        del temp_model
        gc.collect()

        # Move encoder to primary device and freeze
        frozen_encoder = frozen_encoder.to(primary_device)
        frozen_encoder.eval()
        for param in frozen_encoder.parameters():
            param.requires_grad = False

        # ==============================================================
        # Bridge: Extract commits from top-K deletion lines
        # ==============================================================
        print(f"\n  Extracting commits from top-{CONFIG['phase2_top_k_lines']} "
              f"deletion lines...")

        bridge_model = DeletionLineRankingModel(
            input_dim=EMB_DIM,
            hidden_dim=CONFIG["hidden_dim"],
            num_gt_layers=CONFIG["num_gt_layers"],
            num_heads=CONFIG["num_heads"],
            num_edge_types=CONFIG["num_edge_types"],
            dropout=CONFIG["dropout"],
            include_bert=CONFIG.get("include_bert", True),
        ).to(primary_device)
        bridge_model.load_state_dict(phase1_state, strict=False)

        # Extract commits for ALL test cases in this fold
        # (train + val — for training Phase 2 we need train commits too)
        all_fold_test_cases = train_test_cases + val_test_cases
        phase1_commits = extract_phase1_commits(
            bridge_model, phase1_dataset, all_fold_test_cases,
            CONFIG["data_path"], top_k=CONFIG["phase2_top_k_lines"])

        del bridge_model
        gc.collect()
        torch.cuda.empty_cache()

        n_with_commits = len(phase1_commits)
        print(f"  Extracted commits for {n_with_commits}/{len(all_fold_test_cases)} test cases")

        # Check GT coverage
        gt_found = 0
        for v in phase1_commits.values():
            gt_set = set(g[:12] for g in v.get("ground_truth", []))
            sel_set = set(s[:12] for s in v.get("selected_commits", []))
            if gt_set & sel_set:
                gt_found += 1
        if n_with_commits > 0:
            print(f"  GT commit in selected: {gt_found}/{n_with_commits} "
                  f"({100*gt_found/n_with_commits:.1f}%)")

        # ==============================================================
        # Phase 2: Build dataset and train commit ranker
        # ==============================================================
        print(f"\n  Building Phase 2 dataset (unified temporal graphs)...")

        phase2_dataset = TemporalGraphDataset(
            data_path=CONFIG["data_path"],
            test_cases=all_test_cases,  # Full list for index alignment
            graph_builder=graph_builder,
            phase1_commits=phase1_commits,
        )

        # Train Phase 2
        phase2_result = train_phase2_fold(
            fold_idx, train_indices.tolist(), val_indices.tolist(),
            phase2_dataset, frozen_encoder, CONFIG)

        all_phase2_results.append(phase2_result)

        # Save Phase 2 checkpoint
        torch.save(
            {
                "encoder_state_dict": phase1_state,
                "commit_ranker_state_dict": phase2_result["best_commit_ranker_state"],
                "config": CONFIG,
            },
            os.path.join(CONFIG["save_dir"], f"fold{fold_idx}_unified_best.pt"))

        # Cleanup for next fold
        del frozen_encoder, phase2_dataset, phase1_commits
        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # Aggregate CV results
    # ================================================================
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    metrics_to_aggregate = [
        "recall@1", "recall@2", "recall@3", "recall@5",
        "precision@1", "precision@2", "precision@3", "precision@5",
        "f1@1", "f1@2", "f1@3", "f1@5",
        "mrr", "accuracy", "first_rank",
    ]

    # Phase 1 results
    if all_phase1_results and all(
        "metrics" in r for r in all_phase1_results
    ):
        print("\n" + "─" * 60)
        print("PHASE 1 — Deletion Line Ranking (per-fold validation)")
        print("─" * 60)
        # evaluate_top1_metrics returns "F1@1", "precision@1", "recall@1" keys
        p1_f1s = [r["metrics"].get("F1@1", r["metrics"].get("f1@1", 0)) for r in all_phase1_results if "metrics" in r]
        p1_p1s = [r["metrics"].get("precision@1", 0) for r in all_phase1_results if "metrics" in r]
        p1_r1s = [r["metrics"].get("recall@1", 0) for r in all_phase1_results if "metrics" in r]
        print(f"  Mean P@1:  {np.mean(p1_p1s):.4f} ± {np.std(p1_p1s):.4f}")
        print(f"  Mean R@1:  {np.mean(p1_r1s):.4f} ± {np.std(p1_r1s):.4f}")
        print(f"  Mean F1@1: {np.mean(p1_f1s):.4f} ± {np.std(p1_f1s):.4f}")

    # Phase 2 results
    print("\n" + "─" * 60)
    print("PHASE 2 — Commit Ranking (Frozen Encoder, per-fold validation)")
    print("─" * 60)

    aggregated = {m: [] for m in metrics_to_aggregate}
    for fold_result in all_phase2_results:
        for metric in metrics_to_aggregate:
            val = fold_result["final_metrics"].get(metric, 0)
            aggregated[metric].append(val)

    print(f"\n{'Metric':<15} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 60)
    for metric in metrics_to_aggregate:
        values = aggregated[metric]
        if values:
            print(f"{metric:<15} | {np.mean(values):>8.4f} | "
                  f"{np.std(values):>8.4f} | "
                  f"{np.min(values):>8.4f} | "
                  f"{np.max(values):>8.4f}")

    print("\n" + "-" * 60)
    print("Per-Fold Phase 2 Results:")
    for fold_result in all_phase2_results:
        fm = fold_result['final_metrics']
        print(f"  Fold {fold_result['fold']}: "
              f"P@1={fm.get('precision@1', 0):.4f}, "
              f"R@1={fm.get('recall@1', 0):.4f}, "
              f"F1@1={fm.get('f1@1', 0):.4f}, "
              f"MRR={fm.get('mrr', 0):.4f}, "
              f"Best Epoch={fold_result['best_epoch']}")

    # ================================================================
    # Save summary
    # ================================================================
    summary = {
        "config": CONFIG,
        "data_split": {
            "total_samples": len(all_test_cases),
            "n_folds": CONFIG["n_folds"],
            "split_type": "unified two-phase, same splits for both phases",
        },
        "phase1_summary": {
            "mean_F1@1": float(np.mean(p1_f1s)) if 'p1_f1s' in dir() and p1_f1s else None,
        },
        "phase2_cv_metrics": {
            metric: {
                "mean": float(np.mean(aggregated[metric])),
                "std": float(np.std(aggregated[metric])),
                "values": [float(v) for v in aggregated[metric]],
            }
            for metric in metrics_to_aggregate
            if aggregated[metric]
        },
        "per_fold_phase2": [
            {k: v for k, v in r.items()
             if k not in ("best_commit_ranker_state", "history")}
            for r in all_phase2_results
        ],
    }

    summary_path = os.path.join(CONFIG["save_dir"], "unified_kfold_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — UNIFIED TWO-PHASE TRAINING")
    print("=" * 70)
    if aggregated["f1@1"]:
        print(f"\n  Phase 2 (Commit Ranking with Frozen Phase 1 Encoder):")
        print(f"    Mean P@1:        {np.mean(aggregated['precision@1']):.4f} ± {np.std(aggregated['precision@1']):.4f}")
        print(f"    Mean R@1:        {np.mean(aggregated['recall@1']):.4f} ± {np.std(aggregated['recall@1']):.4f}")
        print(f"    Mean F1@1:       {np.mean(aggregated['f1@1']):.4f} ± {np.std(aggregated['f1@1']):.4f}")
        print(f"    Mean MRR:        {np.mean(aggregated['mrr']):.4f} ± {np.std(aggregated['mrr']):.4f}")
        print(f"    Mean First Rank: {np.mean(aggregated['first_rank']):.4f} ± {np.std(aggregated['first_rank']):.4f}")

    print(f"\n✓ Results saved to: {CONFIG['save_dir']}")
    print(f"  - unified_kfold_summary.json (all results)")
    print(f"  - fold*_phase1_best.pt (Phase 1 encoder checkpoints)")
    print(f"  - fold*_unified_best.pt (encoder + commit ranker)")


if __name__ == "__main__":
    main()
