"""
training/utils.py
Shared training utilities used across both phases.

  set_seed              — fix all random seeds for reproducibility
  setup_device          — single-GPU device configuration
  EarlyStopping         — patience-based stopping criterion
  coerce_idx            — safely extract an int from a Tensor / int index
  clip_and_step         — clip gradients, step, zero grads
  build_phase1_model    — construct a DeletionLineRankingModel from config
  build_phase2_model    — construct a CommitRankingModule from config
  log_pair_distribution — print pos/neg/tie pair counts
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

    return CommitRankingModule(
        input_dim=config["model"]["hidden_dim"],
        hidden_dim=config["phase2"]["hidden_dim"],
        num_heads=config["phase2"].get("num_heads", 4),
        num_commit_transformer_layers=config["phase2"]["num_commit_transformer_layers"],
        dropout=config["model"]["dropout"],
        max_temporal_dist=config["phase2"].get("max_temporal_dist", 50),
    ).to(device)


def log_pair_distribution(pairs) -> None:
    """Print the pos / neg / tie breakdown of a pair list."""
    counts: Dict = defaultdict(int)
    for p in pairs:
        counts[p.prob] += 1
    print(f"  Pair distribution: pos={counts.get(1.0, 0)}, "
          f"neg={counts.get(0.0, 0)}, tie={counts.get(0.5, 0)}")