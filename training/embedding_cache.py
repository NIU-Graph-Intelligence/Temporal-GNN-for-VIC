"""
Scores Phase 1 deletion lines and caches encoder outputs for Phase 2 training.
"""

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Batch

from training.utils import coerce_idx


# Step 1: score deletion lines, cache top-1 encoder output

def score_deletion_lines(
    model,
    dataset,
    test_cases: List[str],
    device: torch.device,
    max_nodes: int = 4096,
) -> Dict[str, Tuple]:

    """
    Score every deletion-line graph and return the top-1 MiniGraph per test
    case together with its full encoder output tensor.
    """
    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    results: Dict[str, Tuple] = {}

    def _score_batch(batch_list: List[Tuple]) -> None:
        """Batch-encode (mg, pyg) pairs; update best_* for the current case."""
        nonlocal best_score, best_mg, best_h
        if not batch_list:
            return
        try:
            batch_data = Batch.from_data_list(
                [gd for _, gd in batch_list]
            ).to(device)
            h_all = model.encoder.encode_pyg(batch_data).cpu()

            for i, (mg, _) in enumerate(batch_list):
                start = batch_data.ptr[i].item()
                end   = batch_data.ptr[i + 1].item()
                h_i   = h_all[start:end]
                idx   = coerce_idx(mg.del_idx)
                if idx < h_i.size(0):
                    s = model.ranker.score(h_i[idx].to(device)).item()
                    if s > best_score:
                        best_score, best_mg, best_h = s, mg, h_i.clone()
        except Exception:
            # Batching failed — fall back to one-by-one encoding
            for mg, gd in batch_list:
                _score_single(mg, gd)

    def _score_single(mg, gd) -> None:
        """Encode one graph; update best_* if its score is higher."""
        nonlocal best_score, best_mg, best_h
        try:
            h   = model.encoder.encode_pyg(gd.to(device)).cpu()
            idx = coerce_idx(mg.del_idx)
            if idx < h.size(0):
                s = model.ranker.score(h[idx].to(device)).item()
                if s > best_score:
                    best_score, best_mg, best_h = s, mg, h
        except Exception:
            pass

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict:
                continue

            best_score  = float("-inf")
            best_mg     = None
            best_h      = None
            batch_list: List[Tuple] = []
            batch_nodes = 0

            for mg in graphs_dict[name]:
                n = mg.pyg.num_nodes

                if n > max_nodes:
                    # Graph too large to batch — flush current batch, then encode alone
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0
                    _score_single(mg, mg.pyg)
                    continue

                if batch_nodes + n > max_nodes and batch_list:
                    # Would overflow batch capacity — flush first
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0

                batch_list.append((mg, mg.pyg))
                batch_nodes += n

            _score_batch(batch_list)

            if best_mg is not None:
                results[name] = (best_mg, best_h)

    return results


#  Step 2: convert cached encoder outputs into Phase 2 training items 

def build_phase2_items(
    scored: Dict[str, Tuple],
    all_cases: List[str],
) -> List[dict]:
    """
    
    For each test case:
      - node_embeddings is taken directly from the cached encoder output produced by score_deletion_lines.
      - commit_indices comes from mg.pyg.temporal_pos
      - ground_truth_position maps each inducing commit SHA to its temporal-position index via the tp_to_commit dict stored on the MiniGraph.
   """

    def _make_invalid(name: str) -> dict:
        return {
            "test_name": name, "valid": False,
            "node_embeddings": None, "commit_indices": None,
            "ground_truth_positions": [],
        }

    items:   List[dict] = []
    n_valid: int        = 0

    for tc_idx, test_name in enumerate(all_cases):
        entry = scored.get(test_name)
        if entry is None:
            items.append(_make_invalid(test_name))
            continue

        mg, cached_h = entry


        commit_to_tp = {sha[:12]: tp for tp, sha in mg.tp_to_commit.items() if tp > 0}

        gt_positions_raw = sorted({
            commit_to_tp[sha[:12]]
            for sha in mg.inducing_commits
            if sha[:12] in commit_to_tp
        })

        node_embeddings = cached_h[1:]
        commit_indices = mg.pyg.temporal_pos.cpu()[1:] - 1 # re-indexed 
        gt_positions = [tp - 1 for tp in gt_positions_raw] 

        items.append({
            "test_name":              test_name,
            "valid":                  True,
            "node_embeddings":        node_embeddings,
            "commit_indices":         commit_indices,
            "ground_truth_positions": gt_positions,
        })
        n_valid += 1

        if (tc_idx + 1) % 50 == 0:
            print(f"    [{tc_idx+1}/{len(all_cases)}] {n_valid} valid embeddings built")

    print(f"  Phase 2 embeddings: {n_valid}/{len(all_cases)} valid")
    _log_embedding_stats([i for i in items if i["valid"]])
    return items



def _log_embedding_stats(valid_items: List[dict]) -> None:
    """Print tensor memory statistics for a list of valid Phase 2 items."""
    if not valid_items:
        return

    total_bytes = sum(
        item["node_embeddings"].nelement() * item["node_embeddings"].element_size()
        + item["commit_indices"].nelement() * item["commit_indices"].element_size()
        for item in valid_items
    )

    node_counts   = [item["node_embeddings"].size(0) for item in valid_items]
    commit_counts = [int(item["commit_indices"].max().item()) + 1
                     for item in valid_items]
    hidden_dim    = valid_items[0]["node_embeddings"].size(1)
    n             = len(valid_items)
    total_mb      = total_bytes / (1024 ** 2)

    print(f"\n  ── Embedding Cache Stats ─────────────────────────────────")
    print(f"  Valid items  : {n}")
    print(f"  Hidden dim   : {hidden_dim}")
    print(f"  Node counts  : min={min(node_counts)}, max={max(node_counts)}, "
          f"mean={sum(node_counts)/n:.1f}")
    print(f"  Commit counts: min={min(commit_counts)}, max={max(commit_counts)}, "
          f"mean={sum(commit_counts)/n:.1f}")
    print(f"  Total memory : {total_mb:.1f} MB ({total_bytes / (1024**3):.2f} GB)")
    print(f"  Avg per item : {total_mb / n:.2f} MB")