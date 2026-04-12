import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config_utils import ConfigManager
CONFIG = ConfigManager().raw
from data.dataset import (
    DeletionLineDataset,
    CommitRankingDataset,
    collate_commit_ranking,
)
from models.shared_encoder import CodeBERTEmbedder
from torch.utils.data import DataLoader, Subset
from training.embedding_cache import score_deletion_lines, build_phase2_items
from training.loss import LabelSmoothingRankingLoss
from training.phase1_trainer import train_phase1_fold
from training.phase2_trainer import _run_epoch, train_phase2_fold
from training.utils import build_phase1_model, build_phase2_model, set_seed, setup_device



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified two-phase training: "
                    "Deletion Line Ranking → Commit Ranking"
    )
    # Phase 1
    p.add_argument("--phase1-epochs",type=int)
    p.add_argument("--phase1-lr",type=float)
    p.add_argument("--phase1-bert-lr",type=float)
    p.add_argument("--phase1-rest-lr",type=float)
    p.add_argument("--phase1-bert-freeze-bottom-layers", type=int)
    p.add_argument("--max-graphs-per-batch", type=int)
    p.add_argument("--phase1-patience",    type=int)

    # Phase 2
    p.add_argument("--phase2-epochs",   type=int)
    p.add_argument("--phase2-lr",      type=float)
    p.add_argument("--phase2-batch-size",  type=int)
    p.add_argument("--phase2-patience",    type=int)

    # Shared
    p.add_argument("--hidden-dim",    type=int)
    p.add_argument("--num-gt-layers", type=int)
    p.add_argument("--dropout",       type=float)
    p.add_argument("--seed",          type=int)
    p.add_argument("--save-dir",      type=str)
    p.add_argument("--no-stratify",   action="store_true")

    # Skip Phase 1
    p.add_argument("--skip-phase1", action="store_true",
                   help="Load existing Phase 1 checkpoint instead of training")
    p.add_argument("--phase1-checkpoint-dir", type=str)
    return p.parse_args()


def _apply_cli(args: argparse.Namespace) -> None:
    """Overwrite CONFIG in-place with any CLI arguments that were provided."""
    # Maps CLI arg name → (section, key_in_section)
    CLI_MAP = {
        "phase1_epochs":                     ("phase1", "epochs"),
        "phase1_lr":                         ("phase1", "lr"),
        "phase1_bert_lr":                    ("phase1", "bert_lr"),
        "phase1_rest_lr":                    ("phase1", "rest_lr"),
        "phase1_bert_freeze_bottom_layers":  ("phase1", "bert_freeze_bottom_layers"),
        "phase1_patience":                   ("phase1", "patience"),
        "max_graphs_per_batch":              ("phase1", "max_graphs_per_batch"),
        "phase2_epochs":                     ("phase2", "epochs"),
        "phase2_lr":                         ("phase2", "lr"),
        "phase2_batch_size":                 ("phase2", "batch_size"),
        "phase2_patience":                   ("phase2", "patience"),
        "hidden_dim":                        ("model",  "hidden_dim"),
        "num_gt_layers":                     ("model",  "num_gt_layers"),
        "dropout":                           ("model",  "dropout"),
        "seed":                              ("defaults", "seed"),
        "save_dir":                          ("paths",   "save_dir"),
    }
    for cli_key, (section, yaml_key) in CLI_MAP.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            CONFIG.setdefault(section, {})[yaml_key] = val



def _run_phase1(
    train_cases: List[str],
    val_cases: List[str],
    phase1_dataset: DeletionLineDataset,
    p1_ckpt_path: Path,
    skip_p1: bool,
    device: torch.device,
) -> Optional[Dict]:
    """
    Train Phase 1 or load an existing checkpoint.
    """
    if skip_p1:
        if not p1_ckpt_path.exists():
            raise FileNotFoundError(
                f"--skip-phase1 was set but checkpoint not found: {p1_ckpt_path}"
            )
        print(f"\n  Loading Phase 1 checkpoint: {p1_ckpt_path}")
        state = torch.load(p1_ckpt_path, map_location="cpu")["model_state_dict"]
        return {"model_state": state, "loaded_from": str(p1_ckpt_path)}

    result = train_phase1_fold(
        0, train_cases, val_cases, phase1_dataset, CONFIG, device=device
    )
    if result is None:
        return None

    ckpt_path = Path(CONFIG["paths"]["save_dir"]) / "phase1_best.pt"
    torch.save({"model_state_dict": result["model_state"]}, ckpt_path)
    print(f"  ✓ Phase 1 checkpoint saved to {ckpt_path}")
    return result

def diagnose_phase1_accuracy(scored, cases, label, top_k=1):
    correct = sum(
        1 for name in cases
        if name in scored and any(
            mg.rootcause 
            for _, mg, _ in scored[name][:top_k]
        )
    )
    total = sum(1 for name in cases if name in scored)
    print(f"  Phase 1 top-{top_k} correct ({label}): "
          f"{correct}/{total} = {correct/max(total,1)*100:.1f}%")
        
def _prepare_phase2_data(
    p1_state: Dict,
    phase1_dataset: DeletionLineDataset,
    all_cases: List[str],
    train_cases: List[str],   
    val_cases: List[str],     
    test_cases: List[str],
    device: torch.device,
) -> Tuple[CommitRankingDataset, Dict[str, int]]:

    p1_model = build_phase1_model(CONFIG, device)
    p1_model.load_state_dict(p1_state, strict=False)
    p1_model.encoder.eval()
    for param in p1_model.encoder.parameters():
        param.requires_grad = False

    print(f"\n  Scoring deletion lines for {len(all_cases)} test cases...")
    scored = score_deletion_lines(
        p1_model, phase1_dataset, all_cases, device,
        max_nodes=CONFIG["defaults"]["max_nodes_per_batch"],
        top_k=CONFIG["phase2"]["top_k_lines"],
    )
    print(f"  Top graphs selected: {len(scored)}/{len(all_cases)}")

    # ── Diagnose BEFORE deleting scored ──────────────────────────
    diagnose_phase1_accuracy(scored, train_cases, "train")
    diagnose_phase1_accuracy(scored, val_cases,   "val")
    diagnose_phase1_accuracy(scored, test_cases,  "test")

    del p1_model
    gc.collect()

    print("\n  Building Phase 2 embedding items...")
    p2_items = build_phase2_items(scored, all_cases)

    del scored
    gc.collect()
    torch.cuda.empty_cache()

    case_to_idx = {name: i for i, name in enumerate(all_cases)}
    return CommitRankingDataset(p2_items), case_to_idx



def _run_phase2(
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    p2_ds: CommitRankingDataset,
    case_to_idx: Dict[str, int],
    device: torch.device,
) -> Dict:
    """
    Train Phase 2 and evaluate on the held-out test set.

    Returns the p2_result dict augmented with ``test_metrics``.
    """
    train_idx = [case_to_idx[c] for c in train_cases if c in case_to_idx]
    val_idx   = [case_to_idx[c] for c in val_cases   if c in case_to_idx]
    test_idx  = [case_to_idx[c] for c in test_cases  if c in case_to_idx]

    p2_result = train_phase2_fold(0, train_idx, val_idx, p2_ds, CONFIG)

    print(f"\n  Evaluating on held-out test set ({len(test_idx)} cases)...")
    test_model = build_phase2_model(CONFIG, device)
    if p2_result.get("best_commit_ranker_state"):
        test_model.load_state_dict(p2_result["best_commit_ranker_state"])

    test_loader = DataLoader(
        Subset(p2_ds, test_idx),
        batch_size=CONFIG["phase2"]["batch_size"],
        shuffle=False,
        collate_fn=collate_commit_ranking,
        num_workers=0,
    )
    loss_fn = LabelSmoothingRankingLoss(
        temperature=CONFIG["phase2"]["temperature"],
        smoothing=CONFIG["phase2"]["label_smoothing"],
    )

    _, test_m = _run_epoch(
        test_model, test_loader, None, loss_fn, device, training=False
    )

    print(
        f"\n  Test Set Results\n"
        f"  P@1={test_m.get('precision@1', 0):.4f}  "
        f"R@1={test_m.get('recall@1', 0):.4f}  "
        f"F1@1={test_m.get('f1@1', 0):.4f}  "
        f"MRR={test_m.get('mrr', 0):.4f}"
    )

    p2_result["test_metrics"] = test_m

    del test_model
    gc.collect()
    torch.cuda.empty_cache()

    return p2_result



def _run(
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    phase1_dataset: DeletionLineDataset,
    p1_ckpt_dir: Path,
    skip_p1: bool,
    device: torch.device,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Execute Phase 1 + Phase 2 for the single train/val/test split."""

    p1_result = _run_phase1(
        train_cases, val_cases,
        phase1_dataset, p1_ckpt_dir / "phase1_best.pt",
        skip_p1, device,
    )
    if p1_result is None:
        return None, None

    all_cases = train_cases + val_cases + test_cases
   
    p2_ds, case_to_idx = _prepare_phase2_data(
        p1_result["model_state"], phase1_dataset, all_cases,
        train_cases, val_cases, test_cases,
        device,
    )
    p2_result = _run_phase2(
        train_cases, val_cases, test_cases,
        p2_ds, case_to_idx, device,
    )

    torch.save(
        {"model_state_dict": p2_result["best_commit_ranker_state"]},
        Path(CONFIG["paths"]["save_dir"]) / "phase2_best.pt",
    )

    del p2_ds
    gc.collect()
    torch.cuda.empty_cache()

    return p1_result, p2_result



REPORT_METRICS = ["recall@1", "precision@1", "f1@1"]


def _print_and_save(
    p1_result: Dict,
    p2_result: Dict,
    split_info: Dict,
) -> None:
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if p1_result and "metrics" in p1_result:
        m = p1_result["metrics"]
        print("\nPhase 1 (Deletion Line Ranking)")
        print(f"  P@1:  {m.get('precision@1', 0):.4f}")
        print(f"  R@1:  {m.get('recall@1',    0):.4f}")
        print(f"  F1@1: {m.get('f1@1',        0):.4f}")

    val_m  = p2_result.get("final_metrics", {})
    test_m = p2_result.get("test_metrics",  {})

    print("\nPhase 2 — Validation (used for model selection)")
    print(f"  Best epoch : {p2_result.get('best_epoch', 0)}")
    print(
        f"  P@1={val_m.get('precision@1', 0):.4f}  "
        f"R@1={val_m.get('recall@1', 0):.4f}  "
        f"F1@1={val_m.get('f1@1', 0):.4f}  "
        f"MRR={val_m.get('mrr', 0):.4f}"
    )

    print("\nPhase 2 — Test (held-out, final numbers)")
    print(f"{'Metric':<15} | {'Value':>8}")
    print("-" * 28)
    for name in REPORT_METRICS:
        val = test_m.get(name)
        if val is not None:
            print(f"{name:<15} | {val:>8.4f}")

    summary = {
        "config":              CONFIG,
        "data_split":          split_info,
        "phase1_metrics":      p1_result.get("metrics", {}),
        "phase2_val_metrics":  val_m,
        "phase2_test_metrics": test_m,
        "best_epoch":          p2_result.get("best_epoch", 0),
    }
    path = Path(CONFIG["paths"]["save_dir"]) / "results_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {path}")



def main() -> None:
    args = _parse_args()
    _apply_cli(args)


    set_seed(CONFIG["defaults"]["seed"])
    os.makedirs(CONFIG["paths"]["save_dir"], exist_ok=True)

    p1_ckpt_dir = Path(
        args.phase1_checkpoint_dir
        if args.phase1_checkpoint_dir
        else CONFIG["paths"]["save_dir"]
    )

    device = setup_device(CONFIG["defaults"]["gpu_id"])

    print("=" * 70)
    print("UNIFIED TWO-PHASE TRAINING")
    print(f"  device : {device}")
    print("=" * 70)

    with open(Path(CONFIG["paths"]["data_root"]) / CONFIG["paths"]["test_cases_file"]) as f:
        all_cases: List[str] = json.load(f)
    print(f"Total test cases: {len(all_cases)}")

    # Fixed 70 / 15 / 15 split — seeded for reproducibility
    seed = CONFIG["defaults"]["seed"]
    train_cases, temp_cases = train_test_split(
        all_cases, test_size=0.30, random_state=seed, shuffle=True
    )
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=0.50, random_state=seed, shuffle=True
    )

    n = len(all_cases)
    print(f"\nData split (seed={seed}):")
    for label, subset in [("Train", train_cases), ("Val", val_cases), ("Test", test_cases)]:
        print(f"  {label:<6}: {len(subset)} cases ({100*len(subset)/n:.1f}%)")

    split_info = {
        "total": n, "train": len(train_cases),
        "val": len(val_cases), "test": len(test_cases),
        "seed": seed, "strategy": "random_70_15_15",
    }

    print("\nInitialising CodeBERT tokenizer...")
    embedder = CodeBERTEmbedder(tokenizer_only=True)

    print("\nLoading dataset...")
    p1_dataset = DeletionLineDataset(
        data_path=CONFIG["paths"]["data_root"],
        test_cases=all_cases,
        embedder=embedder,
        prebuilt_dir=CONFIG["paths"]["prebuilt_dir"],
    )

    
    p1_r, p2_r = _run(
        train_cases, val_cases, test_cases,
        p1_dataset, p1_ckpt_dir, args.skip_phase1, device,
    )
    if p1_r and p2_r:
        _print_and_save(p1_r, p2_r, split_info)

    print(f"\n✓ All outputs saved to: {CONFIG['paths']['save_dir']}")


if __name__ == "__main__":
    main()