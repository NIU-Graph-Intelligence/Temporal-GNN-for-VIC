"""
main.py
───────
Entry point for unified two-phase training:
    Phase 1  →  Deletion Line Ranking
    Phase 2  →  Commit Ranking (pre-computed embeddings from frozen encoder)
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from models.shared_encoder import CodeBERTEmbedder
from data.phase1.dataset import DeletionLineDataset
from data.phase2.dataset import (
    Phase2EmbeddingDataset,
    score_and_cache_top_embeddings,
    precompute_phase2_embeddings,
)
from training.utils import set_seed, setup_device, build_phase1_model
from training.phase1_trainer import train_phase1_fold
from training.phase2_trainer import train_phase2_fold
from config import CONFIG



def _parse_args():
    p = argparse.ArgumentParser(
        description="Unified two-phase training: "
                    "Deletion Line Ranking → Commit Ranking")
    # Phase 1
    p.add_argument("--phase1-epochs", type=int)
    p.add_argument("--phase1-lr", type=float, help="Single LR fallback")
    p.add_argument("--phase1-bert-lr", type=float)
    p.add_argument("--phase1-rest-lr", type=float)
    p.add_argument("--phase1-bert-freeze-bottom-layers", type=int)
    p.add_argument("--phase1-batch-size", type=int)
    p.add_argument("--phase1-patience", type=int)
    # Phase 2
    p.add_argument("--phase2-epochs", type=int)
    p.add_argument("--phase2-lr", type=float)
    p.add_argument("--phase2-batch-size", type=int)
    p.add_argument("--phase2-patience", type=int)
    # Shared
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--num-gt-layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--n-folds", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--save-dir", type=str)
    p.add_argument("--no-stratify", action="store_true")
    # Skip Phase 1
    p.add_argument("--skip-phase1", action="store_true",
                   help="Load existing Phase 1 checkpoints instead of training")
    p.add_argument("--phase1-checkpoint-dir", type=str)
    return p.parse_args()


def _apply_cli(args):
    """Overwrite CONFIG in-place with any CLI arguments that were provided."""
    mapping = {
        "phase1_epochs": "phase1_epochs",
        "phase1_lr": "phase1_lr",
        "phase1_bert_lr": "phase1_bert_lr",
        "phase1_rest_lr": "phase1_rest_lr",
        "phase1_bert_freeze_bottom_layers": "phase1_bert_freeze_bottom_layers",
        "phase1_batch_size": "phase1_batch_size",
        "phase1_patience": "phase1_patience",
        "phase2_epochs": "phase2_epochs",
        "phase2_lr": "phase2_lr",
        "phase2_batch_size": "phase2_batch_size",
        "phase2_patience": "phase2_patience",
        "hidden_dim": "hidden_dim",
        "num_gt_layers": "num_gt_layers",
        "dropout": "dropout",
        "n_folds": "n_folds",
        "seed": "seed",
        "save_dir": "save_dir",
    }
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            CONFIG[key] = val


# ── Per-fold pipeline ───────────────────────────────────────────────────────

def _run_fold(fold_idx, train_idx, val_idx, all_cases,
              phase1_dataset, p1_ckpt_dir, skip_p1,
              device, embedder):

    """Execute Phase 1 + Phase 2 for one fold. Returns (p1_result, p2_result)."""
    train_cases = [all_cases[i] for i in train_idx]
    val_cases   = [all_cases[i] for i in val_idx]

    #  Phase 1: train or load 
    p1_ckpt = p1_ckpt_dir / f"fold{fold_idx}_phase1_best.pt"
    if skip_p1 and p1_ckpt.exists():
        print(f"\n  Loading Phase 1 checkpoint: {p1_ckpt}")
        t0 = time.perf_counter()
        checkpoint = torch.load(p1_ckpt, map_location="cpu")
        
        if "encoder_state_dict" in checkpoint:
            state = checkpoint["encoder_state_dict"]
        

        p1_result = {"model_state": state, "loaded_from": str(p1_ckpt)}
        print(f"  [{time.perf_counter()-t0:.2f}s] Phase 1 checkpoint load")
    else:
        t0 = time.perf_counter()
        p1_result = train_phase1_fold(
            fold_idx, train_cases, val_cases, phase1_dataset, CONFIG,
            device=device)
        print(f"  [{time.perf_counter()-t0:.2f}s] Phase 1 training")
        if p1_result is None:
            return None, None
        state = p1_result["model_state"]
        torch.save({"model_state_dict": state},
                   CONFIG["save_dir"] + f"/fold{fold_idx}_phase1_best.pt")

    # Build Phase 1 model for scoring 
    t0 = time.perf_counter()
    p1_model = build_phase1_model(CONFIG, device)
    p1_model.load_state_dict(state, strict=False)

    frozen_enc = p1_model.encoder
    for p in frozen_enc.parameters():
        p.requires_grad = False

    frozen_enc.eval()

    print(f"  [{time.perf_counter()-t0:.2f}s] Build Phase 1 model for scoring")
    #  Score deletion lines + cache encoder output
    all_fold_cases = train_cases + val_cases
    print(f"\n  Scoring deletion lines for {len(all_fold_cases)} test cases...")
    t0 = time.perf_counter()
    scored = score_and_cache_top_embeddings(
        p1_model, phase1_dataset, all_fold_cases, device)
    print(f"  [{time.perf_counter()-t0:.2f}s] Score and cache top embeddings")
    del p1_model; gc.collect(); torch.cuda.empty_cache()

    n_with = len(scored)
    gt_hits = sum(
        1 for mg, _ in scored.values()
        if any(sha[:12] in {g[:12] for g in mg.inducing_commits}
               for sha in mg.tp_to_commit.values())
    )
    print(f"  Top graphs selected: {n_with}/{len(all_fold_cases)}")
    if n_with:
        print(f"  GT in history:       {gt_hits}/{n_with} "
              f"({100*gt_hits/n_with:.1f}%)")

    # ── Phase 2: pre-compute embeddings (encoder runs only on fix commits)
    print(f"\n  Pre-computing Phase 2 embeddings...")
    t0 = time.perf_counter()
    p2_items = precompute_phase2_embeddings(
        scored, CONFIG["data_path"], list(all_cases),
    )
    print(f"  [{time.perf_counter()-t0:.2f}s] Precompute Phase 2 embeddings")
    del frozen_enc, scored; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: train CommitRankingModule 
    p2_ds = Phase2EmbeddingDataset(p2_items)
    t0 = time.perf_counter()
    p2_result = train_phase2_fold(
        fold_idx, list(train_idx), list(val_idx), p2_ds, CONFIG)
    print(f"  [{time.perf_counter()-t0:.2f}s] Phase 2 training")

    torch.save(
        {"encoder_state_dict": state,
         "commit_ranker_state_dict": p2_result["best_commit_ranker_state"],
         "config": CONFIG},
         os.path.join(CONFIG["save_dir"], f"fold{fold_idx}_phase1_best.pt")
    )
    del p2_ds, p2_items; gc.collect(); torch.cuda.empty_cache()
    return p1_result, p2_result


# Results aggregation 

REPORT_METRICS = [
    "recall@1", "recall@2", "recall@3", "recall@5",
    "precision@1", "precision@2", "precision@3", "precision@5",
    "f1@1", "f1@2", "f1@3", "f1@5",
    "mrr", "accuracy", "first_rank",
]


def _print_and_save(p1_results, p2_results, n_total):
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    p1_f1s = [r["metrics"].get("f1@1", 0)
              for r in p1_results if "metrics" in r]
    if p1_f1s:
        p1_p1 = [r["metrics"].get("precision@1", 0) for r in p1_results if "metrics" in r]
        p1_r1 = [r["metrics"].get("recall@1",    0) for r in p1_results if "metrics" in r]
        print("\nPhase 1 (Deletion Line Ranking) ")
        print(f"  P@1:  {np.mean(p1_p1):.4f} ± {np.std(p1_p1):.4f}")
        print(f"  R@1:  {np.mean(p1_r1):.4f} ± {np.std(p1_r1):.4f}")
        print(f"  F1@1: {np.mean(p1_f1s):.4f} ± {np.std(p1_f1s):.4f}")

    agg = {m: [r["final_metrics"].get(m, 0) for r in p2_results]
           for m in REPORT_METRICS}
    print("\nPhase 2 (Commit Ranking) ")
    print(f"\n{'Metric':<15} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 60)
    for m in REPORT_METRICS:
        vs = agg[m]
        if vs:
            print(f"{m:<15} | {np.mean(vs):>8.4f} | {np.std(vs):>8.4f} | "
                  f"{np.min(vs):>8.4f} | {np.max(vs):>8.4f}")

    print("\nPer-fold results")
    for r in p2_results:
        fm = r["final_metrics"]
        print(f"  Fold {r['fold']}: P@1={fm.get('precision@1',0):.4f}, "
              f"R@1={fm.get('recall@1',0):.4f}, F1@1={fm.get('f1@1',0):.4f}, "
              f"MRR={fm.get('mrr',0):.4f}, best_epoch={r['best_epoch']}")

    summary = {
        "config": CONFIG,
        "data_split": {"total_samples": n_total, "n_folds": CONFIG["n_folds"]},
        "phase1_summary": {"mean_f1@1": float(np.mean(p1_f1s)) if p1_f1s else None},
        "phase2_cv_metrics": {
            m: {"mean": float(np.mean(agg[m])), "std": float(np.std(agg[m])),
                "values": [float(v) for v in agg[m]]}
            for m in REPORT_METRICS if agg[m]
        },
        "per_fold_phase2": [
            {k: v for k, v in r.items()
             if k not in ("best_commit_ranker_state", "history")}
            for r in p2_results
        ],
    }
    path = os.path.join(CONFIG["save_dir"], "unified_kfold_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {path}")
    return agg



def main():
    args = _parse_args()
    _apply_cli(args)

    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    p1_ckpt_dir = Path(
        args.phase1_checkpoint_dir
        if args.phase1_checkpoint_dir else CONFIG["save_dir"])

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    primary = setup_device(CONFIG.get("gpu_id", 0))

    print("=" * 70)
    print("UNIFIED TWO-PHASE TRAINING")
    print(f"  device: {primary}")
    print("=" * 70)

    with open(Path(CONFIG["data_path"]) / CONFIG["test_cases_file"]) as f:
        all_cases = json.load(f)
    print(f"Total test cases: {len(all_cases)}")

    # K-Fold splits
    idx = np.arange(len(all_cases))
    splits = list(KFold(CONFIG["n_folds"], shuffle=True,
                        random_state=CONFIG["seed"]).split(idx))

    print(f"\nK-Fold splits ({CONFIG['n_folds']} folds):")
    for i, (tr, va) in enumerate(splits):
        print(f"  Fold {i+1}: train={len(tr)}, val={len(va)}")

    # Shared objects
    print("\nInitialising CodeBERT tokenizer...")
    t0 = time.perf_counter()
    embedder = CodeBERTEmbedder(tokenizer_only=True)
    print(f"  [{time.perf_counter()-t0:.2f}s] CodeBERT tokenizer init")

    print("\nLoading Phase 1 dataset...")
    t0 = time.perf_counter()
    p1_dataset = DeletionLineDataset(
        data_path=CONFIG["data_path"], test_cases=all_cases,
        embedder=embedder, prebuilt_dir=CONFIG["prebuilt_dir"],
    )
    print(f"  [{time.perf_counter()-t0:.2f}s] Phase 1 dataset load")

    # Per-fold loop
    all_p1, all_p2 = [], []
    total_start = time.perf_counter()
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{CONFIG['n_folds']}  "
              f"(train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*70}")
        p1_r, p2_r = _run_fold(
            fold_idx, train_idx, val_idx, all_cases,
            p1_dataset, p1_ckpt_dir, args.skip_phase1,
            device=primary, embedder=embedder)
        if p1_r:
            all_p1.append(p1_r)
        if p2_r:
            all_p2.append(p2_r)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  [Total: {total_elapsed:.1f}s] All folds completed")

    t0 = time.perf_counter()
    agg = _print_and_save(all_p1, all_p2, len(all_cases))
    print(f"  [{time.perf_counter()-t0:.2f}s] Results aggregation + save")
    if agg.get("f1@1"):
        print("\n Final summary ")
        for m in ["precision@1", "recall@1", "f1@1", "mrr"]:
            vs = agg[m]
            print(f"  {m:<12}: {np.mean(vs):.4f} ± {np.std(vs):.4f}")
    print(f"\n✓ All outputs saved to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()
