"""training — loss functions, metrics, utilities, and per-phase trainers."""

from .loss import PairwiseRankingLoss, LabelSmoothingRankingLoss
from .evaluation import (
    load_true_commit_map,
    evaluate_top1_metrics,
    evaluate_topk_metrics,
    evaluate_ranking,
    print_metrics,
    compute_summary_statistics,
)
from .utils import (
    set_seed,
    setup_device,
    EarlyStopping,
    compute_metrics,
    aggregate_global_metrics,
    coerce_idx,
    clip_and_step,
    build_phase1_model,
    build_phase1_optimizer,
    log_pair_distribution,
)
from .phase1_trainer import Phase1Trainer, train_phase1_fold
from .phase2_trainer import train_phase2_fold

__all__ = [
    "PairwiseRankingLoss",
    "LabelSmoothingRankingLoss",
    "load_true_commit_map",
    "evaluate_top1_metrics",
    "evaluate_topk_metrics",
    "evaluate_ranking",
    "print_metrics",
    "compute_summary_statistics",
    "set_seed",
    "setup_device",
    "EarlyStopping",
    "compute_metrics",
    "aggregate_global_metrics",
    "coerce_idx",
    "clip_and_step",
    "build_phase1_model",
    "build_phase1_optimizer",
    "log_pair_distribution",
    "Phase1Trainer",
    "train_phase1_fold",
    "train_phase2_fold",
]