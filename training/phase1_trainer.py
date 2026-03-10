"""
training/phase1_trainer.py
──────────────────────────
Phase 1 trainer: Deletion Line Ranking.

This module contains ONLY training orchestration logic.
Model factories, evaluation, and generic utilities live in
``training.utils`` and ``training.evaluation``.

Public API
----------
Phase1Trainer      — manages one fold of Phase 1 training
train_phase1_fold  — convenience wrapper that creates a trainer and runs it
"""

import copy
import contextlib
import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from data.phase1.dataset import DeletionLineDataset
from data.phase1.pairs import combine_pairs_to_batches
from training.evaluation import evaluate_ranking, load_true_commit_map
from training.loss import PairwiseRankingLoss
from training.utils import (
    EarlyStopping,
    build_phase1_model,
    build_phase1_optimizer,
    coerce_idx,
    log_pair_distribution,
    setup_device,
)


class Phase1Trainer:
    """
    Manages Phase 1 training for one cross-validation fold.

    Args:
        config  : full CONFIG dict (keys used: phase1_*, hidden_dim, …)
        dataset : pre-loaded DeletionLineDataset containing ALL test cases
        device  : torch device for training (if None, inferred from config)

    Usage::

        trainer = Phase1Trainer(config, dataset, device)
        result  = trainer.train(fold_idx, train_cases, val_cases)
        # result → {"model_state", "best_epoch", "best_f1", "metrics"}
    """

    def __init__(self, config: Dict, dataset: DeletionLineDataset,
                 device: torch.device = None) -> None:
        self.config = config
        self.dataset = dataset
        self.device = device 

        # Built lazily in train()
        self.model = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.criterion = PairwiseRankingLoss()

        # History (populated during train())
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_f1: float = 0.0
        self.best_epoch: int = 0

    def train(
        self,
        fold_idx: int,
        train_cases: List[str],
        val_cases: List[str],
    ) -> Optional[Dict]:
        """
        Train Phase 1 for one fold.

        Returns
        -------
        dict with keys:
            model_state : best model state_dict
            best_epoch  : epoch at which best val F1@1 was achieved
            best_f1     : best validation F1@1
            metrics     : full metrics dict from evaluate_top1_metrics
        None if no training pairs could be generated.
        """
        cfg = self.config
        ds  = self.dataset
        print(f"  PHASE 1 — Fold {fold_idx + 1}: Deletion Line Ranking")

        val_cid = load_true_commit_map(val_cases, cfg["data_path"])
        print(f"  Val inducing commits: {sum(len(v) for v in val_cid.values())}")

        # Pairs (generated from the shared dataset, filtered by fold split)
        print("  Generating pairs...")
        max_pairs   = cfg["phase1_max_pairs_per_test"]
        train_pairs = ds.get_pairs_for_cases(train_cases, max_pairs)
        val_pairs   = ds.get_pairs_for_cases(val_cases, max_pairs)
        print(f"  Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs")

        if not train_pairs:
            print(f"  WARNING: No training pairs for fold {fold_idx + 1}")
            return None

        log_pair_distribution(train_pairs)

        # Model + optimisation
        self.model = build_phase1_model(cfg, self.device)
        self.optimizer = build_phase1_optimizer(self.model, cfg)
        for i, g in enumerate(self.optimizer.param_groups):
            print(f"  Optimizer group {i}: lr={g['lr']:.2e}, "
                  f"params={sum(p.numel() for p in g['params']):,}")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3)

        # Training loop
        best_state: Optional[Dict] = None
        self.best_f1    = 0.0
        self.best_epoch = 0
        self.train_losses.clear()
        self.val_losses.clear()

        stopper = EarlyStopping(patience=cfg["phase1_patience"], mode="max")
        bs    = cfg["phase1_batch_size"]
        max_n = cfg.get("max_nodes_per_graph", 9500)

        best_metrics = None
        for epoch in range(1, cfg["phase1_epochs"] + 1):

            tr_loss = self._train_epoch(train_pairs, bs, max_n)

            with self.model.cache_context():
                vl_loss = self._validate_epoch(val_pairs, bs, max_n)

            val_m   = self._evaluate(ds, val_cases)
            f1      = val_m.get("f1@1", 0.0)

            self.train_losses.append(tr_loss)
            self.val_losses.append(vl_loss)
            self.scheduler.step(f1)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch}: train={tr_loss:.4f}, val={vl_loss:.4f}, "
                      f"P@1={val_m.get('precision@1', 0):.4f}, "
                      f"R@1={val_m.get('recall@1', 0):.4f}, F1@1={f1:.4f}")

            if f1 > self.best_f1:
                self.best_f1    = f1
                self.best_epoch = epoch
                best_state   = copy.deepcopy(self.model.state_dict())
                best_metrics = val_m
                print(f"  ✓ New best (F1@1={f1:.4f}, epoch {epoch})")

            if stopper(f1, epoch):
                print(f"  ⚠ Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        final_m = best_metrics if best_metrics is not None else self._evaluate(ds, val_cases)
        print(f"\n  Fold {fold_idx + 1} best (epoch {self.best_epoch}): "
              f"P@1={final_m.get('precision@1', 0):.4f}, "
              f"R@1={final_m.get('recall@1', 0):.4f}, "
              f"F1@1={final_m.get('f1@1', 0.0):.4f}")

        # Extract raw state_dict for return value
        model_state = self.model.state_dict()
        return {
            "model_state": model_state,
            "best_epoch":  self.best_epoch,
            "best_f1":     self.best_f1,
            "metrics":     final_m,
        }

    def _train_epoch(self, pairs, batch_size: int, max_nodes: int) -> float:
        """Train for one epoch. Returns average loss per batch."""
        return _run_pairs_epoch(
            self.model, pairs, self.criterion, self.optimizer,
            batch_size, max_nodes, self.device, training=True,
        )

    def _validate_epoch(self, pairs, batch_size: int, max_nodes: int) -> float:
        """Validate for one epoch. Returns average loss per batch."""
        return _run_pairs_epoch(
            self.model, pairs, self.criterion, None,
            batch_size, max_nodes, self.device, training=False,
        )

    def _evaluate(self, dataset, test_cases: List[str]) -> Dict:
        """Score every deletion line and compute P/R/F1."""
        return evaluate_ranking(
            self.model, dataset, test_cases, self.config["data_path"],
            device=self.device,
        )


def train_phase1_fold(
    fold_idx: int,
    train_cases: List[str],
    val_cases: List[str],
    dataset: DeletionLineDataset,
    config: Dict,
    device: torch.device = None,
) -> Optional[Dict]:
    """
    Convenience wrapper: create a fresh Phase1Trainer and run one fold.

    Parameters
    ----------
    fold_idx    : zero-based fold index (used only for progress printing)
    train_cases : test-case names assigned to the training split
    val_cases   : test-case names assigned to the validation split
    dataset     : pre-loaded DeletionLineDataset containing ALL test cases
    config      : full CONFIG dict
    device      : torch device (inferred from config if None)

    Returns
    -------
    dict with keys ``model_state``, ``best_epoch``, ``best_f1``, ``metrics``,
    or None if no training pairs could be generated for this fold.
    """
    return Phase1Trainer(config, dataset, device).train(
        fold_idx, train_cases, val_cases)


def _run_pairs_epoch(
    model,
    pairs,
    criterion,
    optimizer,
    batch_size: int,
    max_nodes: int,
    device: torch.device,
    training: bool,
) -> float:
    """
    Run one epoch over pairwise data — shared by train and validate paths.

    When training=True:  model.train(), shuffle pairs, backprop, clip+step.
    When training=False: model.eval(), torch.no_grad(), no backprop.

    Uses model to batch-encode all unique graphs per batch,
    reducing encoder calls from ~2*N per batch to ~1 batched call.

    Returns average loss per batch.
    """
    if training:
        model.train()
        random.shuffle(pairs)
    else:
        model.eval()

    batches = combine_pairs_to_batches(pairs, batch_size)

    # ── DEBUG: show epoch level info ──
    phase = "Train" if training else "Val"
    print(f"\n  [{phase}] Starting epoch: {len(pairs)} pairs → {len(batches)} batches of size {batch_size}")

    loss_sum, total_valid, skipped_large = 0.0, 0, 0
    errors: Dict = defaultdict(int)

    _target_cache = {
        0.0: torch.tensor([0.0], dtype=torch.float32, device=device),
        0.5: torch.tensor([0.5], dtype=torch.float32, device=device),
        1.0: torch.tensor([1.0], dtype=torch.float32, device=device),
    }

    ctx = torch.no_grad() if not training else contextlib.nullcontext()
    with ctx:
        for batch in batches:

            cache_ctx = model.cache_context() if training else contextlib.nullcontext()
            with cache_ctx:
                if training:
                    optimizer.zero_grad()

                # ── Step 1: Validate pairs, build pair_specs ─────────────────
                pair_specs: List[Tuple] = []
                pair_probs: List[float] = []

                for pair in batch.pairs:
                    x_g, y_g = pair.x.pyg, pair.y.pyg
                if x_g is None or y_g is None:
                    continue
                if x_g.num_nodes > max_nodes or y_g.num_nodes > max_nodes:
                    skipped_large += 1
                    continue
                x_idx = coerce_idx(pair.x.del_idx)
                y_idx = coerce_idx(pair.y.del_idx)
                if x_idx >= x_g.num_nodes or y_idx >= y_g.num_nodes:
                    continue
                pair_specs.append((x_g, x_idx, y_g, y_idx))
                pair_probs.append(pair.prob)

            if not pair_specs:
                continue

            # ── Step 2: Batch encode + extract deletion embeddings ────────
            # forward now returns stacked tensors ready for ranker

            try:
                emb_x, emb_y, valid_mask = model(
                    pair_specs, device, max_nodes
                )
            except Exception as exc:
                etype = type(exc).__name__
                errors[etype] += 1
                if errors[etype] <= 3:
                    print(f"  [{phase}] forward {etype}: {str(exc)[:120]}")

                if "CUDA" in str(exc) or "out of memory" in str(exc):
                    torch.cuda.empty_cache()
                
                continue

            if emb_x is None or len(valid_mask) == 0:
                continue
            
            # ── Step 3: Single batched ranker call ────────────────────────
            # emb_x: [N, hidden_dim], emb_y: [N, hidden_dim]
            # ranker does ONE matrix multiply for all N pairs
            try:
                probs = model.ranker(emb_x, emb_y)   # [N]
            except Exception as exc:
                errors[type(exc).__name__] += 1
                continue

            # ── Step 4: Build target tensor aligned with valid pairs ──────
            targets = torch.cat([
                _target_cache[pair_probs[i]] for i in valid_mask
            ])   # [N]

            # ── Step 5: Single vectorized loss + single backward ──────────
            batch_loss = criterion(probs, targets)   # scalar

            if training:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                

            loss_sum    += batch_loss.item()
            total_valid += len(valid_mask)

    # ── DEBUG: end of epoch summary ──
    print(f"  [{phase}] Epoch done: valid={total_valid}/{len(pairs)}, "
            f"loss={loss_sum/max(len(batches),1):.4f}")


    if errors or skipped_large:
        phase = "Train" if training else "Val"
        print(f"  [{phase}] valid={total_valid}/{len(pairs)}, "
              f"skipped_large={skipped_large}, errors={dict(errors)}")

    return loss_sum / max(len(batches), 1)
