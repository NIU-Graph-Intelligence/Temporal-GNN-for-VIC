"""
Scores Phase 1 deletion lines and caches encoder outputs for Phase 2 training.
"""

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Batch

from training.utils import coerce_idx
from data.constants import EdgeType

# Step 1: score deletion lines, cache top-1 encoder output
def score_deletion_lines(
    model,
    dataset,
    test_cases: List[str],
    device: torch.device,
    max_nodes: int = 4096,
    top_k: int = 1,
) -> Dict[str, Tuple]:

    """
    Score every deletion-line graph and return the top-k MiniGraph per test
    case together with its full encoder output tensor.
    """
    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    results: Dict[str, Tuple] = {}

    def _score_batch(batch_list: List[Tuple]) -> None:
        nonlocal all_scored        
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
                    all_scored.append((s, mg, h_i.clone()))
        except Exception:
            # Batching failed — fall back to one-by-one encoding
            for mg, gd in batch_list:
                _score_single(mg, gd)

    def _score_single(mg, gd) -> None:
        nonlocal all_scored
        try:
            h   = model.encoder.encode_pyg(gd.to(device)).cpu()
            idx = coerce_idx(mg.del_idx)
            if idx < h.size(0):
                s = model.ranker.score(h[idx].to(device)).item()
                all_scored.append((s, mg, h))
        except Exception:
            pass

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict:
                continue

            all_scored: List[Tuple] = []
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

            if all_scored:
                all_scored.sort(key=lambda x: x[0], reverse=True)
                results[name] = all_scored[:top_k]
    return results



#  Step 2: convert cached encoder outputs (top-1 deletion line embeddings) into Phase 2 training items 

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
            "test_name": name, 
            "valid": False,
            "node_embeddings": None, 
            "commit_indices": None,
            "is_temporal_node": None,
            "ground_truth_positions": [],
            "is_correct_deletion_line": False,
            "p1_score": 0.0,
        }

    items:   List[dict] = []
    n_valid: int        = 0

    for tc_idx, test_name in enumerate(all_cases):
        entry = scored.get(test_name)
        if entry is None:
            items.append(_make_invalid(test_name))
            continue

        all_node_embeddings = []
        all_commit_indices = []
        all_is_temporal_nodes = []
        all_gt_positions = []
        commit_offset = 0
        any_valid = False
        best_p1_score = float("-inf")

        for p1_score, mg, cached_h in entry:
            best_p1_score = max(best_p1_score, p1_score)

            # GT Position for this deletion line
            commit_to_tp = {
                sha[:12]: tp
                for tp, sha in mg.tp_to_commit.items()
                if tp > 0
            }

            gt_positions_raw = sorted({
                commit_to_tp[sha[:12]]
                for sha in mg.inducing_commits
                if sha[:12] in commit_to_tp
            })
            gt_positions = [tp-1 for tp in gt_positions_raw]

            # Node Embeddings and commit indices
            node_embeddings = cached_h[1:]
            commit_indices = mg.pyg.temporal_pos.cpu()[1:] - 1

            if node_embeddings.size(0) == 0 or commit_indices.numel() == 0:
                continue

            # Temporal Mask
            edge_index = mg.pyg.edge_index
            edge_type = mg.pyg.edge_type
            temporal_fwd_type = EdgeType.TEMPORAL_FWD
            temporal_mask_full = torch.zeros(
                mg.pyg.num_nodes,
                dtype=torch.bool
            )

            temporal_dst = edge_index[1][edge_type == temporal_fwd_type]
            temporal_mask_full[temporal_dst] = True
            is_temporal_node = temporal_mask_full[1:]

            # number of commits in this deletion line
            n_commits = int(commit_indices.max().item()) + 1

            # Validate ground truth positions are in bounds
            gt_positions = [g for g in gt_positions if g < n_commits]

            # Adjust commit indices by offset to make them sequential across all deletion lines
            adjusted_commit_indices = commit_indices + commit_offset

            # Adjust ground truth positions by offset
            adjusted_gt_positions = [g + commit_offset for g in gt_positions]

            # Accumulate
            all_node_embeddings.append(node_embeddings)
            all_commit_indices.append(adjusted_commit_indices)
            all_is_temporal_nodes.append(is_temporal_node)
            all_gt_positions.extend(adjusted_gt_positions)

            # Update offset for next deletion line
            commit_offset += n_commits
            any_valid = True

        if not any_valid or len(all_node_embeddings) == 0:
            items.append(_make_invalid(test_name))
            continue

        # concatenate all deletion line data
        pooled_node_embeddings = torch.cat(all_node_embeddings, dim=0)
        pooled_commit_indices = torch.cat(all_commit_indices, dim=0)
        pooled_is_temporal_nodes = torch.cat(all_is_temporal_nodes, dim=0)

        # Final validation
        total_commits = commit_offset
        all_gt_positions_valid = [g for g in all_gt_positions if g < total_commits]
        is_correct = len(all_gt_positions_valid) > 0

        items.append({
            "test_name":              test_name,
            "valid":                  True,
            "node_embeddings":        pooled_node_embeddings,
            "commit_indices":         pooled_commit_indices,
            "is_temporal_node":       pooled_is_temporal_nodes,
            "ground_truth_positions": all_gt_positions_valid,
            "is_correct_deletion_line": is_correct,
            "p1_score":               best_p1_score,
        })
            
        n_valid += 1

        if (tc_idx + 1) % 50 == 0:
            print(f"    [{tc_idx+1}/{len(all_cases)}] "
                  f"{n_valid} valid items built")

    print(f"  Phase 2 items: {n_valid}/{len(all_cases)} valid")

    # Stats
    n_correct = sum(1 for i in items if i["valid"] and i["is_correct_deletion_line"])
    n_wrong   = sum(1 for i in items if i["valid"] and not i["is_correct_deletion_line"])
    print(f"  Correct deletion lines: {n_correct} | Wrong deletion lines: {n_wrong}")

    _log_embedding_stats([i for i in items if i["valid"]])
    return items


           

#  Step 2: convert cached encoder outputs (top-1 and top-3 deletion line embeddings) into Phase 2 training items 
# def build_phase2_items(
#     scored: Dict[str, List[Tuple]],
#     all_cases: List[str],
# ) -> List[dict]:
    """
    For each test case, find the first deletion line in the scored list
    that contains the inducing commit and use that for Phase 2.

    Works for both top_k=1 and top_k=3 — always produces exactly one
    item per test case (or invalid if no deletion line has inducing commit).
    """

    def _make_invalid(name: str) -> dict:
        return {
            "test_name":              name,
            "valid":                  False,
            "node_embeddings":        None,
            "commit_indices":         None,
            "is_temporal_node":       None,
            "ground_truth_positions": [],
        }

    items:   List[dict] = []
    n_valid: int        = 0

    for tc_idx, test_name in enumerate(all_cases):
        entry = scored.get(test_name)
        if entry is None:
            items.append(_make_invalid(test_name))
            continue

        # Find first deletion line that has the inducing commit
        selected = None
        for p1_score, mg, cached_h in entry:
            commit_to_tp = {
                sha[:12]: tp
                for tp, sha in mg.tp_to_commit.items() if tp > 0
            }
            gt_positions_raw = sorted({
                commit_to_tp[sha[:12]]
                for sha in mg.inducing_commits
                if sha[:12] in commit_to_tp
            })
            if gt_positions_raw:
                selected = (mg, cached_h, gt_positions_raw)
                break

        if selected is None:
            items.append(_make_invalid(test_name))
            continue

        mg, cached_h, gt_positions_raw = selected

        # Node embeddings and commit indices
        node_embeddings = cached_h[1:]
        commit_indices  = mg.pyg.temporal_pos.cpu()[1:] - 1

        if node_embeddings.size(0) == 0 or commit_indices.numel() == 0:
            items.append(_make_invalid(test_name))
            continue

        # Temporal mask
        edge_index        = mg.pyg.edge_index
        edge_type         = mg.pyg.edge_type
        temporal_fwd_type = EdgeType.TEMPORAL_FWD
        temporal_mask_full = torch.zeros(mg.pyg.num_nodes, dtype=torch.bool)
        temporal_dst = edge_index[1][edge_type == temporal_fwd_type]
        temporal_mask_full[temporal_dst] = True
        is_temporal_node = temporal_mask_full[1:]

        # Validate gt positions in bounds
        gt_positions = [tp - 1 for tp in gt_positions_raw]
        n_commits    = int(commit_indices.max().item()) + 1
        gt_positions = [g for g in gt_positions if g < n_commits]

        if not gt_positions:
            items.append(_make_invalid(test_name))
            continue

        items.append({
            "test_name":              test_name,
            "valid":                  True,
            "node_embeddings":        node_embeddings,
            "commit_indices":         commit_indices,
            "is_temporal_node":       is_temporal_node,
            "ground_truth_positions": gt_positions,
        })
        n_valid += 1

        if (tc_idx + 1) % 50 == 0:
            print(f"    [{tc_idx+1}/{len(all_cases)}] "
                  f"{n_valid} valid embeddings built")

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
    commit_counts = [int(item["commit_indices"].max().item()) + 1 if item["commit_indices"].numel() > 0 else 0
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