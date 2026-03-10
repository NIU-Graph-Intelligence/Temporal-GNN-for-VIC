"""
training/phase2_trainer.py
──────────────────────────
Phase 2 training loop: commit ranking on pre-computed node embeddings.

The frozen encoder has already been applied during the embedding
pre-computation step.  This trainer only optimises CommitRankingModule.
"""

import copy
import gc
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from models.phase2_model import CommitRankingModule
from training.loss import LabelSmoothingRankingLoss
from training.utils import (
    EarlyStopping,
    aggregate_global_metrics,
    clip_and_step,
    compute_metrics,
    setup_device,
)
from data.phase2.dataset import collate_phase2


def _train_epoch(model: CommitRankingModule, loader: DataLoader,
                 optimizer, loss_fn: nn.Module, device: torch.device,
                 log_interval: int, grad_accum: int) -> Tuple[float, Dict]:
    """Train for one epoch. Returns (avg_loss, aggregated_metrics)."""
    model.train()
    total_loss, total_samples = 0.0, 0
    all_m: Dict = defaultdict(list)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for b_idx, batch in enumerate(loader):
        if not batch:
            continue

        try:
            scores_all = model(
               batch["node_embeddings"].to(device),
               batch["commit_indices"].to(device),
           )

            # Split [C_total] back into per-sample scores using commit_counts
            scores_list = torch.split(scores_all, batch["commit_counts"])

            b_loss  = torch.tensor(0.0, device=device)
            b_count = 0
            for scores, gt in zip(scores_list,
                                batch["ground_truth_positions"]):
                if not gt:
                    continue
                loss = loss_fn(scores, gt) / grad_accum
                if loss.requires_grad and not torch.isnan(loss):
                    b_loss  = b_loss + loss
                    b_count += 1
                with torch.no_grad():
                    for k, v in compute_metrics(
                            scores, gt).items():
                        all_m[k].append(v)

            if b_count and b_loss.requires_grad:
                b_loss.backward()
                total_loss    += b_loss.item() * grad_accum
                total_samples += b_count

        except Exception as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()

        if (b_idx + 1) % grad_accum == 0:
            clip_and_step(model, optimizer)

        if (b_idx + 1) % log_interval == 0:
            running = aggregate_global_metrics(all_m)
            print(f"  Batch {b_idx+1}/{len(loader)}: "
                  f"loss={total_loss/max(total_samples,1):.4f}, "
                  f"P@1={running.get('precision@1',0):.4f}, "
                  f"F1@1={running.get('f1@1',0):.4f}, "
                  f"t={time.time()-t0:.1f}s")

    # Flush any remaining accumulated gradients at end of epoch
    if total_samples and (len(loader)) % grad_accum != 0:
        clip_and_step(model, optimizer)

    return total_loss / max(total_samples, 1), aggregate_global_metrics(all_m)



def _validate_epoch(model: CommitRankingModule, loader: DataLoader,
                    loss_fn: nn.Module,
                    device: torch.device) -> Tuple[float, Dict]:
    """Validate for one epoch. Returns (avg_loss, aggregated_metrics)."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_m: Dict = defaultdict(list)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if not batch:
                continue
            
            try:
                scores_all = model(
                        batch["node_embeddings"].to(device),
                        batch["commit_indices"].to(device),
                    )
                scores_list = torch.split(scores_all, batch["commit_counts"])

                for scores, gt in zip(scores_list, batch["ground_truth_positions"]):
                    if not gt:
                        continue
                    loss = loss_fn(scores, gt)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        total_samples += 1
                    
                    for k,v in compute_metrics(scores, gt).items():
                        all_m[k].append(v)

            except Exception:
                print(f"  [Val] batch error: ")
                continue

    return total_loss / max(total_samples, 1), aggregate_global_metrics(all_m)


# def train_phase2_fold(fold_idx: int, train_indices: List[int],
#                       val_indices: List[int], phase2_dataset,
#                       config: Dict) -> Dict:
#     """
#     Train Phase 2 (commit ranking) for one fold on pre-computed embeddings.

#     Returns a dict with:
#         fold                     : int
#         best_epoch               : int
#         best_val_f1@1            : float
#         final_metrics            : Dict
#         history                  : Dict
#         best_commit_ranker_state : state dict of CommitRankingModule
#     """
#     print(f"\n{'─'*60}")
#     print(f"  PHASE 2 — Fold {fold_idx + 1}: Commit Ranking")
#     print(f"{'─'*60}")

#     primary = setup_device(config.get("gpu_id", 0))
    
#     loader_kw = dict(collate_fn=collate_phase2, num_workers=0,
#                      pin_memory=False,
#                      persistent_workers= False,
#                      )

#     train_loader = DataLoader(Subset(phase2_dataset, train_indices),
#                               batch_size=config["phase2_batch_size"],
#                               shuffle=True, **loader_kw)
#     val_loader   = DataLoader(Subset(phase2_dataset, val_indices),
#                               batch_size=config["phase2_batch_size"],
#                               shuffle=False, **loader_kw)

#     model = CommitRankingModule(
#         input_dim=config["hidden_dim"],
#         hidden_dim=config["phase2_hidden_dim"],
#         num_heads=config.get("phase2_num_heads", 4),
#         num_commit_transformer_layers=config["num_commit_transformer_layers"],
#         dropout=config["dropout"],
#     ).to(primary)

#     print(f"  Device: {primary}")
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"  Trainable parameters: {n_params:,}")

#     optimizer = AdamW(model.parameters(), lr=config["phase2_lr"],
#                       weight_decay=config["phase2_weight_decay"])
#     scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
#                                   patience=5)
#     loss_fn = LabelSmoothingRankingLoss(
#         temperature=config["phase2_temperature"], margin=1.0,
#         smoothing=config["phase2_label_smoothing"])
#     stopper = EarlyStopping(patience=config["phase2_patience"], mode="max")

#     best_f1, best_epoch, best_state = 0.0, 0, None
#     history = {"train_loss": [], "val_loss": [],
#                "train_f1@1": [], "val_f1@1": []}
#     ga = config["phase2_gradient_accumulation_steps"]

#     for epoch in range(1, config["phase2_epochs"] + 1):
#         t0 = time.time()

#         tr_loss, tr_m = _train_epoch(model, train_loader, optimizer, loss_fn,
#                                      primary, config["log_interval"], ga)

#         vl_loss, vl_m = _validate_epoch(model, val_loader, loss_fn, primary)

#         vl_f1 = vl_m.get("f1@1", 0)
#         scheduler.step(vl_f1)

#         if epoch % 5 == 0 or epoch == 1:
#             print(f"\n  Epoch {epoch}: train={tr_loss:.4f}, val={vl_loss:.4f} | "
#                   f"train P@1={tr_m.get('precision@1',0):.4f}, "
#                   f"F1@1={tr_m.get('f1@1',0):.4f} | "
#                   f"val P@1={vl_m.get('precision@1',0):.4f}, "
#                   f"R@1={vl_m.get('recall@1',0):.4f}, "
#                   f"F1@1={vl_f1:.4f}  [{time.time()-t0:.1f}s]")

#         history["train_loss"].append(tr_loss)
#         history["val_loss"].append(vl_loss)
#         history["train_f1@1"].append(tr_m.get("f1@1", 0))
#         history["val_f1@1"].append(vl_f1)

#         if vl_f1 > best_f1:
#             best_f1, best_epoch = vl_f1, epoch
#             best_state = copy.deepcopy(model.state_dict())
#             print(f"  ✓ New best (F1@1={vl_f1:.4f}, epoch {epoch})")

#         if stopper(vl_f1, epoch):
#             print(f"  ⚠ Early stopping at epoch {epoch}")
#             break

#     if best_state:
#         model.load_state_dict(best_state)

#     _, final_m = _validate_epoch(model, val_loader, loss_fn, primary)
#     print(f"\n  Fold {fold_idx+1} best (epoch {best_epoch}): "
#           f"P@1={final_m.get('precision@1',0):.4f}, "
#           f"R@1={final_m.get('recall@1',0):.4f}, "
#           f"F1@1={final_m.get('f1@1',0):.4f}, "
#           f"MRR={final_m.get('mrr',0):.4f}")

#     del model, optimizer, scheduler
#     gc.collect()
#     torch.cuda.empty_cache()

#     return {"fold": fold_idx + 1, "best_epoch": best_epoch,
#             "best_val_f1@1": best_f1, "final_metrics": final_m,
#             "history": history, "best_commit_ranker_state": best_state}

def train_phase2_fold(fold_idx: int, train_indices: List[int],
                      val_indices: List[int], phase2_dataset,
                      config: Dict) -> Dict:
    """
    Train Phase 2 (commit ranking) for one fold on pre-computed embeddings.

    Returns a dict with:
        fold                     : int
        best_epoch               : int
        best_val_f1@1            : float
        final_metrics            : Dict
        history                  : Dict
        best_commit_ranker_state : state dict of CommitRankingModule
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — Fold {fold_idx + 1}: Commit Ranking")
    print(f"{'─'*60}")

    primary = setup_device(config.get("gpu_id", 0))

    loader_kw = dict(
        collate_fn=collate_phase2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    train_loader = DataLoader(Subset(phase2_dataset, train_indices),
                              batch_size=config["phase2_batch_size"],
                              shuffle=True, **loader_kw)
    val_loader   = DataLoader(Subset(phase2_dataset, val_indices),
                              batch_size=config["phase2_batch_size"],
                              shuffle=False, **loader_kw)

   
    model = CommitRankingModule(
        input_dim=config["hidden_dim"],
        hidden_dim=config["phase2_hidden_dim"],
        num_heads=config.get("phase2_num_heads", 4),
        num_commit_transformer_layers=config["num_commit_transformer_layers"],
        dropout=config["dropout"],
    ).to(primary)

    print(f"  Device: {primary}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=config["phase2_lr"],
                      weight_decay=config["phase2_weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=5)
    loss_fn = LabelSmoothingRankingLoss(
        temperature=config["phase2_temperature"], margin=1.0,
        smoothing=config["phase2_label_smoothing"])
    stopper = EarlyStopping(patience=config["phase2_patience"], mode="max")

    best_f1, best_epoch, best_state = 0.0, 0, None
    history = {"train_loss": [], "val_loss": [],
               "train_f1@1": [], "val_f1@1": []}
    ga = config["phase2_gradient_accumulation_steps"]

    for epoch in range(1, config["phase2_epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_m = _train_epoch(model, train_loader, optimizer, loss_fn,
                                     primary, config["log_interval"], ga)

        vl_loss, vl_m = _validate_epoch(model, val_loader, loss_fn, primary)

        vl_f1 = vl_m.get("f1@1", 0)
        scheduler.step(vl_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"\n  Epoch {epoch}: train={tr_loss:.4f}, val={vl_loss:.4f} | "
                  f"train P@1={tr_m.get('precision@1',0):.4f}, "
                  f"F1@1={tr_m.get('f1@1',0):.4f} | "
                  f"val P@1={vl_m.get('precision@1',0):.4f}, "
                  f"R@1={vl_m.get('recall@1',0):.4f}, "
                  f"F1@1={vl_f1:.4f}  [{time.time()-t0:.1f}s]")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1@1"].append(tr_m.get("f1@1", 0))
        history["val_f1@1"].append(vl_f1)

        if vl_f1 > best_f1:
            best_f1, best_epoch = vl_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best (F1@1={vl_f1:.4f}, epoch {epoch})")

        if stopper(vl_f1, epoch):
            print(f"  ⚠ Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    _, final_m = _validate_epoch(model, val_loader, loss_fn, primary)
    print(f"\n  Fold {fold_idx+1} best (epoch {best_epoch}): "
          f"P@1={final_m.get('precision@1',0):.4f}, "
          f"R@1={final_m.get('recall@1',0):.4f}, "
          f"F1@1={final_m.get('f1@1',0):.4f}, "
          f"MRR={final_m.get('mrr',0):.4f}")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {"fold": fold_idx + 1, "best_epoch": best_epoch,
            "best_val_f1@1": best_f1, "final_metrics": final_m,
            "history": history, "best_commit_ranker_state": best_state}