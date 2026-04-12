"""
training/utils.py
──────────────────
Stateless training utilities shared by both phases.

  set_seed                   — fix all random seeds for reproducibility
  setup_device               — single-GPU device configuration
  EarlyStopping              — patience-based stopping criterion
  compute_metrics            — per-sample TP/FP/MRR (used inside training loops)
  aggregate_global_metrics   — fold-level aggregation of raw per-sample counts
  coerce_idx                 — safely extract an int from a Tensor / int index
  clip_and_step              — clip gradients, step, zero grads
  build_phase1_model         — construct a DeletionLineRankingModel from config
  build_phase1_optimizer     — Adam with differential LRs (Phase 1)
  log_pair_distribution      — print pos/neg/tie pair counts
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(gpu_id: int = 0) -> torch.device:
    """
    Return the single training device.

    Parameters
    ----------
    gpu_id : int
        GPU index to use.  Ignored when CUDA is unavailable.

    Returns
    -------
    torch.device
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device("cpu")
    gpu_id = min(gpu_id, torch.cuda.device_count() - 1)
    return torch.device(f"cuda:{gpu_id}")


class EarlyStopping:
    """
    Patience-based early stopping.

    Parameters
    ----------
    patience  : epochs without improvement before stopping
    min_delta : minimum change to count as an improvement
    mode      : 'max' (higher is better) or 'min'
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Return True if training should stop."""
        if self.best_score is None:
            self.best_score, self.best_epoch = score, epoch
            return False
        improved = (score > self.best_score + self.min_delta
                    if self.mode == "max"
                    else score < self.best_score - self.min_delta)
        if improved:
            self.best_score, self.best_epoch = score, epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


def compute_metrics(scores: torch.Tensor, ground_truth_positions: List[int],
                    k_values: List[int] = [1, 2, 3, 5]) -> Dict:
    """
    Compute per-sample TP@k, FP@k, MRR, and first rank.

    Returns a flat dict of raw counts suitable for aggregation by
    ``aggregate_global_metrics``.
    """
    if not ground_truth_positions:
        result: Dict = {"num_gt": 0, "mrr": 0.0, "first_rank": float("inf")}
        for k in k_values:
            result[f"tp@{k}"] = 0
            result[f"fp@{k}"] = 0
        return result

    ranked = torch.argsort(scores, descending=True).tolist()
    gt_set = set(ground_truth_positions)
    result = {"num_gt": 1}

    for k in k_values:
        hit = bool(set(ranked[:k]) & gt_set)
        result[f"tp@{k}"] = 1 if hit else 0
        result[f"fp@{k}"] = 0 if hit else 1

    for rank, idx in enumerate(ranked, 1):
        if idx in gt_set:
            result["mrr"] = 1.0 / rank
            result["first_rank"] = rank
            break
    else:
        result["mrr"] = 0.0
        result["first_rank"] = len(ranked) + 1
    return result


def aggregate_global_metrics(all_metrics: Dict[str, List],
                              k_values: List[int] = [1, 2, 3, 5]) -> Dict:
    """
    Aggregate per-sample raw counts into global precision / recall / F1.

    Parameters
    ----------
    all_metrics : dict mapping metric_name → list of per-sample values
                  (accumulated by appending ``compute_metrics`` results)
    """
    result: Dict = {}
    total_gt = sum(all_metrics.get("num_gt", [0]))

    for k in k_values:
        tp = sum(all_metrics.get(f"tp@{k}", [0]))
        fp = sum(all_metrics.get(f"fp@{k}", [0]))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / total_gt  if total_gt   else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        result[f"precision@{k}"] = precision
        result[f"recall@{k}"]    = recall
        result[f"f1@{k}"]        = f1
        result[f"tp_total@{k}"]  = tp
        result[f"fp_total@{k}"]  = fp

    mrrs = all_metrics.get("mrr", [])
    result["mrr"]        = float(np.mean(mrrs)) if mrrs else 0.0
    frs = all_metrics.get("first_rank", [])
    result["first_rank"] = float(np.mean(frs)) if frs else 0.0
    result["total_gt"]   = total_gt
    return result


# ---------------------------------------------------------------------------
# Generic tensor / gradient helpers
# ---------------------------------------------------------------------------

def coerce_idx(idx) -> int:
    """Safely extract a plain ``int`` from a Tensor, list, or int index."""
    if isinstance(idx, torch.Tensor):
        return idx.item() if idx.numel() == 1 else idx[0].item()
    return int(idx)


def clip_and_step(model: nn.Module, optimizer) -> None:
    """Clip gradients and step only when at least one gradient exists."""
    if any(p.grad is not None for p in model.parameters() if p.requires_grad):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Phase 1 model / optimizer factories
# ---------------------------------------------------------------------------

def build_phase1_model(config: Dict, device: torch.device):
    """Construct a :class:`DeletionLineRankingModel` from *config*."""
    from models.phase1_model import DeletionLineRankingModel
    from data.constants import NUM_EDGE_TYPES

    return DeletionLineRankingModel(
        input_dim=config.get("emb_dim", 768),
        hidden_dim=config["model"]["hidden_dim"],
        num_gt_layers=config["model"]["num_gt_layers"],
        num_heads=config["model"]["num_heads"],
        num_edge_types=NUM_EDGE_TYPES,
        dropout=config["phase1"].get("dropout", config["model"]["dropout"]),
        include_bert=config["model"].get("include_bert", True),
        num_bert_layers_freeze=config["phase1"].get("bert_freeze_bottom_layers", 0),
        bert_chunk=config["model"].get("bert_chunk", 256),
    ).to(device)


def build_phase2_model(config: Dict, device: torch.device):
    """Construct a :class:`CommitRankingModule` from *config*."""
    from models.phase2_model import CommitRankingModule

    # torch.manual_seed(config["defaults"]["seed"])
    return CommitRankingModule(
        input_dim=config["model"]["hidden_dim"],
        hidden_dim=config["phase2"]["hidden_dim"],
        num_heads=config["phase2"].get("num_heads", 4),
        num_commit_transformer_layers=config["phase2"]["num_commit_transformer_layers"],
        dropout=config["model"]["dropout"],
        max_temporal_dist=config["phase2"].get("max_temporal_dist", 50),
    ).to(device)


def build_phase1_optimizer(model, config: Dict) -> torch.optim.Optimizer:
    """
    Build Adam with differential learning rates for Phase 1.

    CodeBERT → small LR to preserve pre-trained knowledge.
    Graph layers + ranker (random init) → larger LR to converge faster.
    Falls back to a single param group when BERT is disabled or no
    differential LRs are configured.
    """
    include_bert = config["model"].get("include_bert", True)
    bert_lr      = config["phase1"].get("bert_lr")
    rest_lr      = config["phase1"].get("rest_lr")
    fallback     = config["phase1"]["lr"]

    if not include_bert or (bert_lr is None and rest_lr is None):
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=fallback
        )

    # ── KEY FIX: use id() to strictly separate the two groups ──
    # Collect BERT param ids first, then exclude them from rest
    bert_param_ids = set()
    bert_params    = []

    if include_bert and hasattr(model.encoder, "bert_model"):
        for p in model.encoder.bert_model.parameters():
            if p.requires_grad and id(p) not in bert_param_ids:
                bert_param_ids.add(id(p))
                bert_params.append(p)

    rest_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and id(p) not in bert_param_ids
    ]
    groups = []
    if bert_params:
        groups.append({"params": bert_params,
                       "lr": bert_lr if bert_lr is not None else fallback})
    if rest_params:
        groups.append({"params": rest_params,
                       "lr": rest_lr if rest_lr is not None else fallback})

    if not groups:
        # absolute fallback — should never happen
        groups = [{"params": [p for p in model.parameters()
                               if p.requires_grad], "lr": fallback}]

    return torch.optim.Adam(groups)


# def log_pair_distribution(pairs) -> None:
    """Print the pos / neg / tie breakdown of a pair list."""
    counts: Dict = defaultdict(int)
    for p in pairs:
        counts[p.prob] += 1
    print(f"  Pair distribution: pos={counts.get(1.0, 0)}, "
          f"neg={counts.get(0.0, 0)}, tie={counts.get(0.5, 0)}")