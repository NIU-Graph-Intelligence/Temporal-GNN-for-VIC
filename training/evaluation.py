"""
Evaluation metrics for both phases.

Metrics are computed EXACTLY as in NeuralSZZ's train.py (eval_top),
ensuring results are directly comparable to the baseline.

Public API
----------
load_true_commit_map      — load ground-truth inducing commits from info.json
evaluate_topk_metrics     — precision@k / recall@k / f1@k (NeuralSZZ)
evaluate_top1_metrics     — backward-compatible wrapper for k=1
evaluate_ranking          — score every deletion line and compute P/R/F1@1
print_metrics             — pretty-print a metrics dict
compute_summary_statistics — average metrics across folds
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch


def load_true_commit_map(
    test_cases: List[str],
    data_path: str,
) -> Dict[str, set]:
    """
    Load the true bug-inducing commit set for each test case from info.json.

    Returns {test_name: set_of_inducing_commit_shas}.
    """
    result = {}
    base = Path(data_path)
    for name in test_cases:
        path = base / name / "info.json"
        if path.exists():
            with open(path) as f:
                result[name] = set(json.load(f).get("induce", []))
        else:
            result[name] = set()
    return result


def evaluate_topk_metrics(
    test_cases_graphs: Dict[str, List],
    true_cid_map: Optional[Dict[str, set]] = None,
    data_path: Optional[str] = None,
    k: int = 1,
) -> Dict[str, float]:

    if true_cid_map is None:
        if data_path is None:
            raise ValueError(
                "Either true_cid_map or data_path must be provided.")
        true_cid_map = load_true_commit_map(
            list(test_cases_graphs.keys()), data_path)

    tp = fp = total_t = 0

    for test_name, ranked in test_cases_graphs.items():
        if not ranked:
            continue

        gt_set   = true_cid_map.get(test_name, set())
        gt_short = {g[:12] for g in gt_set}
        total_t += len(gt_set)
        cid_set  = set()

        for mg in ranked[:k]:
            mg_commits_short = {sha[:12] for sha in mg.tp_to_commit.values()}
            gt_in_this_line  = mg_commits_short & gt_short

            if gt_in_this_line:
                new_gt = gt_in_this_line - cid_set
                if new_gt:
                    tp += len(new_gt)
                    cid_set.update(new_gt)
                continue # GT line — never FP regardless of new_gt being empty
            
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / total_t   if total_t   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        f'precision@{k}': precision,
        f'recall@{k}':    recall,
        f'f1@{k}':        f1,
        f'tp@{k}':        tp,
        f'fp@{k}':        fp,
        'total_inducing_commits': total_t,
    }


def evaluate_top1_metrics(
    test_cases_graphs: Dict[str, List],
    true_cid_map: Optional[Dict[str, set]] = None,
    data_path: Optional[str] = None,
) -> Dict[str, float]:
    """Backward-compatible wrapper: evaluate at k=1."""
    return evaluate_topk_metrics(
        test_cases_graphs, true_cid_map, data_path, k=1)


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict (supports @1 and optionally @2)."""
    tag = f"{prefix} " if prefix else ""
    print(f"{tag}Precision@1: {metrics['precision@1']:.4f} | "
          f"Recall@1: {metrics['recall@1']:.4f} | "
          f"F1@1: {metrics['f1@1']:.4f}")
    if "precision@2" in metrics:
        print(f"{tag}Precision@2: {metrics['precision@2']:.4f} | "
              f"Recall@2: {metrics['recall@2']:.4f} | "
              f"F1@2: {metrics['f1@2']:.4f}")


def compute_summary_statistics(all_metrics: List[Dict]) -> Dict[str, float]:
    """Average metrics across multiple folds."""
    if not all_metrics:
        return {"precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0}

    n = len(all_metrics)
    result = {
        "precision@1": sum(m["precision@1"] for m in all_metrics) / n,
        "recall@1":    sum(m["recall@1"]    for m in all_metrics) / n,
        "f1@1":        sum(m["f1@1"]        for m in all_metrics) / n,
    }
    if "precision@2" in all_metrics[0]:
        result["precision@2"] = sum(m.get("precision@2", 0) for m in all_metrics) / n
        result["recall@2"]    = sum(m.get("recall@2",    0) for m in all_metrics) / n
        result["f1@2"]        = sum(m.get("f1@2",        0) for m in all_metrics) / n
    if "commit_precision@1" in all_metrics[0]:
        result["commit_precision@1"] = sum(m.get("commit_precision@1", 0) for m in all_metrics) / n
        result["commit_recall@1"]    = sum(m.get("commit_recall@1",    0) for m in all_metrics) / n
        result["commit_f1@1"]        = sum(m.get("commit_f1@1",        0) for m in all_metrics) / n
    return result


def evaluate_ranking(
    model, dataset, test_cases: List[str], data_path: str,
    device: torch.device = None,
) -> Dict:
    """
    Score every deletion line in *dataset* and compute P/R/F1@1.

    This is the Phase 1 end-of-epoch evaluation routine.  It does NOT
    mutate the dataset's internal graph lists.
    """
    from training.utils import coerce_idx

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    true_cid    = load_true_commit_map(test_cases, data_path)
    ranked: Dict = {}

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict or not graphs_dict[name]:
                continue
            for mg in graphs_dict[name]:
                try:
                    gd  = mg.pyg.to(device)
                    idx = coerce_idx(mg.del_idx)
                    mg.score = (
                        model.predict(gd, idx).item()
                        if idx < gd.num_nodes else 0.0
                    )
                except Exception:
                    mg.score = 0.0
            ranked[name] = sorted(
                graphs_dict[name], key=lambda g: g.score, reverse=True
            )

    return evaluate_top1_metrics(ranked, true_cid, data_path)



# Phase 2 metrics (per-sample + aggregation)

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
    all_metrics : dict mapping metric_name -> list of per-sample values
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