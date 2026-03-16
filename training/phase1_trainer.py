"""
training/phase1_trainer.py
───────────────────────────
Phase 1 trainer: Deletion Line Ranking.

Testcase-level batching replaces the old pair-level batching:
  - combine_testcases_to_batches groups test cases (not pairs) into batches
  - Each batch encodes all its graphs once in a single forward pass
  - Pair embeddings are extracted by lookup — no graph encoded twice per batch
"""

import contextlib
import copy
import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from data.dataset import DeletionLineDataset
from data.phase1.pairs import combine_testcases_to_batches
from training.evaluation import evaluate_ranking, load_true_commit_map
from training.loss import PairwiseRankingLoss
from training.utils import (
    EarlyStopping,
    build_phase1_model,
    build_phase1_optimizer,
)


class Phase1Trainer:
    """
    Args:
        config  : full CONFIG
        dataset : pre-loaded DeletionLineDataset containing ALL test cases
        device  : torch device
    Usage:
        trainer = Phase1Trainer(config, dataset, device)
        result  = trainer.train(fold_idx, train_cases, val_cases)
    """

    def __init__(
        self,
        config: Dict,
        dataset: DeletionLineDataset,
        device: torch.device = None,
    ) -> None:
        self.config    = config
        self.dataset   = dataset
        self.device    = device
        self.criterion = PairwiseRankingLoss()

        self.model     = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

    def train(
        self,
        fold_idx:    int,
        train_cases: List[str],
        val_cases:   List[str],
    ) -> Optional[Dict]:

        cfg = self.config
        ds  = self.dataset
        print(f"  PHASE 1 — Fold {fold_idx + 1}: Deletion Line Ranking")

        val_cid = load_true_commit_map(val_cases, cfg["data_path"])
        print(f"  Val inducing commits: {sum(len(v) for v in val_cid.values())}")

        # Count pairs for logging only (pairs are now built inside batches)
        max_pairs = cfg["phase1_max_pairs_per_test"]
        n_train_pairs = sum(
            len(ds.mini_graphs.get(n, [])) for n in train_cases
        )
        n_val_pairs = sum(
            len(ds.mini_graphs.get(n, [])) for n in val_cases
        )
        print(f"  Train graphs: {n_train_pairs} | Val graphs: {n_val_pairs}")
        print(f"  Max pairs per test case: {max_pairs}")

        # Model + optimisation
        self.model     = build_phase1_model(cfg, self.device)
        self.optimizer = build_phase1_optimizer(self.model, cfg)

        for i, g in enumerate(self.optimizer.param_groups):
            print(
                f"  Optimizer group {i}: lr={g['lr']:.2e}, "
                f"params={sum(p.numel() for p in g['params']):,}"
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        stopper  = EarlyStopping(patience=cfg["phase1_patience"], mode="max")
        max_gpb  = cfg.get("max_graphs_per_batch", 32)   # VRAM knob
        max_n    = cfg.get("max_nodes_per_graph",  9500)

        best_f1, best_epoch        = 0.0, 0
        best_state:   Optional[Dict] = None
        best_metrics: Optional[Dict] = None
        history = {
            "train_loss": [], "val_loss": [],
            "train_f1@1": [], "val_f1@1": [],
        }

        for epoch in range(1, cfg["phase1_epochs"] + 1):
            tr_loss = self._train_epoch(train_cases, max_pairs, max_gpb, max_n)
            vl_loss = self._validate_epoch(val_cases,  max_pairs, max_gpb, max_n)
            val_m   = self._evaluate(ds, val_cases)

            f1 = val_m.get("f1@1", 0.0)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["val_f1@1"].append(f1)

            self.scheduler.step(f1)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch}: train={tr_loss:.4f}, val={vl_loss:.4f}, "
                    f"P@1={val_m.get('precision@1', 0):.4f}, "
                    f"R@1={val_m.get('recall@1', 0):.4f}, F1@1={f1:.4f}"
                )

            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                best_state   = copy.deepcopy(self.model.state_dict())
                best_metrics = val_m
                print(f"  ✓ New best (F1@1={f1:.4f}, epoch {epoch})")

            if stopper(f1, epoch):
                print(f"  ⚠ Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

        final_m = best_metrics or self._evaluate(ds, val_cases)
        print(
            f"\n  Fold {fold_idx + 1} best (epoch {best_epoch}): "
            f"P@1={final_m.get('precision@1', 0):.4f}, "
            f"R@1={final_m.get('recall@1', 0):.4f}, "
            f"F1@1={final_m.get('f1@1', 0.0):.4f}"
        )

        return {
            "model_state": self.model.state_dict(),
            "best_epoch":  best_epoch,
            "best_f1":     best_f1,
            "metrics":     final_m,
            "history":     history,
        }

    def _train_epoch(
        self,
        cases:             List[str],
        max_pairs:         int,
        max_graphs_per_batch: int,
        max_nodes:         int,
    ) -> float:
        return _run_epoch(
            model=self.model,
            dataset=self.dataset,
            cases=cases,
            criterion=self.criterion,
            optimizer=self.optimizer,
            max_pairs=max_pairs,
            max_graphs_per_batch=max_graphs_per_batch,
            max_nodes=max_nodes,
            device=self.device,
            training=True,
        )

    def _validate_epoch(
        self,
        cases:             List[str],
        max_pairs:         int,
        max_graphs_per_batch: int,
        max_nodes:         int,
    ) -> float:
        return _run_epoch(
            model=self.model,
            dataset=self.dataset,
            cases=cases,
            criterion=self.criterion,
            optimizer=None,
            max_pairs=max_pairs,
            max_graphs_per_batch=max_graphs_per_batch,
            max_nodes=max_nodes,
            device=self.device,
            training=False,
        )

    def _evaluate(self, dataset: DeletionLineDataset, test_cases: List[str]) -> Dict:
        return evaluate_ranking(
            self.model, dataset, test_cases, self.config["data_path"],
            device=self.device,
        )


def train_phase1_fold(
    fold_idx:    int,
    train_cases: List[str],
    val_cases:   List[str],
    dataset:     DeletionLineDataset,
    config:      Dict,
    device:      torch.device = None,
) -> Optional[Dict]:
    return Phase1Trainer(config, dataset, device).train(
        fold_idx, train_cases, val_cases
    )


def _run_epoch(
    model,
    dataset:              DeletionLineDataset,
    cases:                List[str],
    criterion,
    optimizer,
    max_pairs:            int,
    max_graphs_per_batch: int,
    max_nodes:            int,
    device:               torch.device,
    training:             bool,
) -> float:
    """
    Run one epoch over testcase-level batches.

    When training=True:  model.train(), shuffle cases, backprop, clip+step.
    When training=False: model.eval(), torch.no_grad(), no backprop.

    Each batch:
      1. Encodes all its graphs in ONE forward pass
      2. Extracts deletion-line embeddings by lookup (node 0 of each graph)
      3. Runs the ranker over all pairs in a single call
      4. Backpropagates (training only)

    Returns average loss per batch.
    """
    phase = "Train" if training else "Val"

    if training:
        model.train()
        cases = list(cases)
        random.shuffle(cases)
    else:
        model.eval()

    batches = combine_testcases_to_batches(
        dataset=dataset,
        cases=cases,
        max_pairs=max_pairs,
        max_graphs_per_batch=max_graphs_per_batch,
    )

    total_pairs  = sum(len(b) for b in batches)
    total_graphs = sum(len(b.mini_graphs) for b in batches)
    print(
        f"\n  [{phase}] Starting epoch: "
        f"{len(cases)} cases → {len(batches)} batches, "
        f"{total_graphs} graphs, {total_pairs} pairs"
    )

    loss_sum     = 0.0
    total_valid  = 0
    skipped_large = 0
    errors: Dict = defaultdict(int)

    _target_cache = {
        0.0: torch.tensor([0.0], dtype=torch.float32, device=device),
        0.5: torch.tensor([0.5], dtype=torch.float32, device=device),
        1.0: torch.tensor([1.0], dtype=torch.float32, device=device),
    }

    grad_ctx = contextlib.nullcontext() if training else torch.no_grad()

    with grad_ctx:
        for batch in batches:
            if not batch.pairs:
                continue

            if training:
                optimizer.zero_grad()

            # ── Step 1: encode all graphs once + extract pair embeddings ──
            try:
                emb_x, emb_y, valid_mask = model(
                    batch.mini_graphs,
                    batch.pairs,
                    device,
                    max_nodes,
                )
            except Exception as exc:
                etype = type(exc).__name__
                errors[etype] += 1
                if errors[etype] <= 3:
                    print(f"  [{phase}] forward {etype}: {str(exc)[:120]}")
                if "CUDA" in str(exc) or "out of memory" in str(exc):
                    torch.cuda.empty_cache()
                continue

            if emb_x is None or not valid_mask:
                continue

            # ── Step 2: single batched ranker call ────────────────────────
            try:
                probs = model.ranker(emb_x, emb_y)   # [N]
            except Exception as exc:
                errors[type(exc).__name__] += 1
                continue

            # ── Step 3: build targets aligned to valid pairs ──────────────
            targets = torch.cat(
                [_target_cache[batch.pairs[i].prob] for i in valid_mask]
            )

            # ── Step 4: loss + backward ───────────────────────────────────
            batch_loss = criterion(probs, targets)

            if training:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum    += batch_loss.item()
            total_valid += len(valid_mask)

    avg_loss = loss_sum / max(len(batches), 1)
    print(
        f"  [{phase}] Epoch done: valid_pairs={total_valid}/{total_pairs}, "
        f"loss={avg_loss:.4f}"
    )
    if errors or skipped_large:
        print(f"  [{phase}] skipped_large={skipped_large}, errors={dict(errors)}")

    return avg_loss