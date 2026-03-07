"""
Phase 2 data layer: everything needed to go from Phase 1 results to
ready-to-train pre-computed embeddings.

  build_fix_commit_pyg             build fix commit's PyG graph for encoding
  score_and_cache_top_embeddings   score deletion lines, cache encoder output
  precompute_phase2_embeddings     combine cached + fix-commit embeddings
  Phase2EmbeddingDataset           serves pre-computed items to DataLoader
  collate_phase2                   DataLoader collate function

Flow (called from main.py → _run_fold):
  1. score_and_cache_top_embeddings   — Phase 1 model scores every MiniGraph,
                                        caches encoder output for top-1
  2. precompute_phase2_embeddings     — reuses cached history embeddings,
                                        encodes fix commit (only new encoder call)
  3. Phase2EmbeddingDataset           — wraps the items for DataLoader
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from config import CONFIG, EDGE_TYPES
from data.phase1.processing import _find_commit_dir


# ── Fix commit graph construction ───────────────────────────────────────────

def _build_section_edges(
    nodes: List[Dict], node_offset: int,
) -> List[Tuple[int, int, int]]:
    """
    Build CFG/DFG/LINEMAP edges for one commit section using
    nodeIndex → global mapping (handles non-sequential nodeIndex values).
    """
    local_to_global = {
        node["nodeIndex"]: node_offset + i
        for i, node in enumerate(nodes)
    }
    edges: List[Tuple[int, int, int]] = []

    for node in nodes:
        src = local_to_global[node["nodeIndex"]]
        for t in node.get("cfgs", []):
            if isinstance(t, int) and t in local_to_global:
                dst = local_to_global[t]
                edges.append((src, dst, EDGE_TYPES["CFG_FWD"]))
                edges.append((dst, src, EDGE_TYPES["CFG_BWD"]))
        for t in node.get("dfgs", []):
            if isinstance(t, int) and t in local_to_global:
                dst = local_to_global[t]
                edges.append((src, dst, EDGE_TYPES["DFG_FWD"]))
                edges.append((dst, src, EDGE_TYPES["DFG_BWD"]))

    linemap_groups: Dict[int, List[int]] = defaultdict(list)
    for node in nodes:
        lm = node.get("lineMapIndex", -1)
        if lm is not None and lm >= 0:
            linemap_groups[lm].append(local_to_global[node["nodeIndex"]])
    for group in linemap_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append((group[i], group[j], EDGE_TYPES["LINEMAP"]))
                edges.append((group[j], group[i], EDGE_TYPES["LINEMAP"]))

    return edges


def build_fix_commit_pyg(
    test_dir: Path,
    fix_commit: str,
    embedder,
) -> Optional[Data]:
    """
    Load the fix commit's ``graph.json`` and return a PyG Data suitable
    for encoding through the SharedEncoder.

    Returns None if the graph cannot be loaded.
    """
    fix_dir = _find_commit_dir(test_dir, fix_commit)
    if fix_dir is None:
        return None
    fix_graph_path = fix_dir / "graph.json"
    if not fix_graph_path.exists():
        return None
    try:
        with open(fix_graph_path) as f:
            fix_nodes = json.load(f)
    except Exception:
        return None
    if not fix_nodes:
        return None

    codes = [n.get("code", "") for n in fix_nodes]
    edges = _build_section_edges(fix_nodes, 0)
    temporal_pos = torch.zeros(len(fix_nodes), dtype=torch.long)

    if edges:
        src   = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst   = torch.tensor([e[1] for e in edges], dtype=torch.long)
        etype = torch.tensor([e[2] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        etype      = torch.empty((0,),   dtype=torch.long)

    if getattr(embedder, "tokenizer_only", False):
        toks = embedder.tokenize_texts(codes)
        return Data(
            token_ids=toks["token_ids"],
            attention_mask=toks["attention_mask"],
            edge_index=edge_index,
            edge_type=etype,
            temporal_pos=temporal_pos,
            num_nodes=len(fix_nodes),
        )
    else:
        X = embedder.encode_texts(codes)
        if X.size(0) == 0:
            return None
        return Data(
            x=X, edge_index=edge_index, edge_type=etype,
            temporal_pos=temporal_pos, num_nodes=X.size(0),
        )


# ── Scoring + encoder output caching ───────────────────────────────────────

def score_and_cache_top_embeddings(
    model,
    dataset,
    test_cases: List[str],
    device: torch.device,
) -> Dict[str, Tuple]:
    """
    Score every deletion line and return the top-1 MiniGraph per test case
    along with its encoder output (cached on CPU).

    Uses batched encoding via PyG Batch.from_data_list to reduce encoder
    forward passes. Batches are capped by max_nodes_per_batch to avoid OOM.

    Returns
    -------
    {test_name: (MiniGraph, encoder_h [N, hidden_dim] on CPU)}
    """
    from training.utils import coerce_idx

    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    results: Dict[str, Tuple] = {}
    max_nodes = CONFIG.get("max_nodes_per_batch", 4096)

    def _score_batch(batch_list: List[Tuple]) -> None:
        """Encode a batch of (mg, gd), score each, update best for this case."""
        nonlocal best_score, best_mg, best_h
        if not batch_list:
            return
        graphs = [gd for _, gd in batch_list]
        try:
            batch_data = Batch.from_data_list(graphs)
            batch_data = batch_data.to(device)
            h_all = model.encoder.encode_pyg(batch_data).cpu()
        except Exception:
            for mg, gd in batch_list:
                try:
                    h = model.encoder.encode_pyg(gd.to(device)).cpu()
                    idx = coerce_idx(mg.del_idx)
                    if idx < h.size(0):
                        score = model.ranker.score(h[idx].to(device)).item()
                        if score > best_score:
                            best_score, best_mg, best_h = score, mg, h
                except Exception:
                    pass
            return
        ptr = 0
        for mg, gd in batch_list:
            n = gd.num_nodes
            h_i = h_all[ptr : ptr + n]
            ptr += n
            idx = coerce_idx(mg.del_idx)
            if idx < n:
                score = model.ranker.score(h_i[idx].to(device)).item()
                if score > best_score:
                    best_score, best_mg, best_h = score, mg, h_i.clone()

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict:
                continue

            best_score = float("-inf")
            best_mg    = None
            best_h     = None
            batch_list: List[Tuple] = []
            batch_nodes = 0

            for mg in graphs_dict[name]:
                gd = mg.pyg
                n = gd.num_nodes

                if n > max_nodes:
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0
                    try:
                        h = model.encoder.encode_pyg(gd.to(device)).cpu()
                        idx = coerce_idx(mg.del_idx)
                        if idx < h.size(0):
                            score = model.ranker.score(h[idx].to(device)).item()
                            if score > best_score:
                                best_score, best_mg, best_h = score, mg, h
                    except Exception:
                        pass
                    continue

                if batch_nodes + n > max_nodes and batch_list:
                    _score_batch(batch_list)
                    batch_list, batch_nodes = [], 0

                batch_list.append((mg, gd))
                batch_nodes += n

            _score_batch(batch_list)
            if best_mg is not None:
                results[name] = (best_mg, best_h)

    return results


# ── Embedding pre-computation ───────────────────────────────────────────────

def precompute_phase2_embeddings(
    scored: Dict[str, Tuple],
    frozen_encoder,
    embedder,
    data_path: str,
    all_cases: List[str],
    device: torch.device,
) -> List[Dict]:
    """
    Build Phase 2 training items from cached Phase 1 encoder outputs.

    For each test case:
      1. History embeddings — reused directly from the cached encoder output
         (no encoder run; these are the representations that Phase 1 produced
         in context of the full temporal graph).
      2. Fix commit embeddings — the only new encoder call: loads the fix
         commit's ``graph.json``, builds a PyG, and runs the frozen encoder.
      3. Combines [fix_h | history_h] with commit_indices derived from
         temporal_pos.

    Returns a list aligned with ``all_cases`` (one item per test case).
    """
    frozen_encoder.eval()
    items: List[Dict] = []
    n_valid = 0

    for tc_idx, test_name in enumerate(all_cases):
        invalid = {
            "test_name": test_name, "valid": False,
            "node_embeddings": None, "commit_indices": None,
            "num_commits": 0, "ground_truth_positions": [],
        }

        entry = scored.get(test_name)
        if entry is None:
            items.append(invalid)
            continue

        mg, cached_h = entry
        history_h = cached_h[1:]      # skip deletion node (index 0)

        test_dir = Path(data_path) / test_name

        try:
            with open(test_dir / "commits.json") as f:
                fix_commit = json.load(f)["fix_commit"]
        except Exception:
            items.append(invalid)
            continue

        fix_pyg = build_fix_commit_pyg(test_dir, fix_commit, embedder)
        if fix_pyg is None:
            items.append(invalid)
            continue

        try:
            with torch.no_grad():
                fix_h = frozen_encoder.encode_pyg(fix_pyg.to(device)).cpu()
        except Exception:
            items.append(invalid)
            continue

        F_nodes = fix_h.size(0)
        combined_h  = torch.cat([fix_h, history_h], dim=0)
        tp_hist = mg.pyg.temporal_pos[1:]
        tp_dev = tp_hist.device
        combined_tp = torch.cat([
            torch.zeros(F_nodes, dtype=torch.long, device=tp_dev),
            tp_hist,
        ])
        num_commits = int(combined_tp.max().item()) + 1 if len(combined_tp) > 0 else 0

        tp_to_commit = {0: fix_commit[:12]}
        tp_to_commit.update(mg.tp_to_commit)
        commit_to_tp = {sha[:12]: tp for tp, sha in tp_to_commit.items()}

        gt_positions = sorted(set(
            commit_to_tp[sha[:12]]
            for sha in mg.inducing_commits
            if sha[:12] in commit_to_tp
        ))

        items.append({
            "test_name": test_name,
            "valid": True,
            "node_embeddings": combined_h,
            "commit_indices": combined_tp,
            "num_commits": num_commits,
            "ground_truth_positions": gt_positions,
        })
        n_valid += 1

        if (tc_idx + 1) % 50 == 0:
            print(f"    [{tc_idx+1}/{len(all_cases)}] "
                  f"{n_valid} valid embeddings computed")

    print(f"  Phase 2 embeddings: {n_valid}/{len(all_cases)} valid")
    return items


# ── Dataset + collate ───────────────────────────────────────────────────────

class Phase2EmbeddingDataset(Dataset):
    """
    One item = pre-computed node embeddings + commit metadata for one
    test case.

    Items are produced by ``precompute_phase2_embeddings`` and stored
    as plain dicts with keys:

        test_name              : str
        valid                  : bool
        node_embeddings        : Tensor [N, hidden_dim]
        commit_indices         : Tensor [N]
        num_commits            : int
        ground_truth_positions : List[int]
    """

    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_phase2(batch: List[Dict]) -> Optional[List[Dict]]:
    """
    DataLoader collate function.

    Filters out samples where pre-computation failed (valid=False).
    Returns the valid samples as a plain list, or None if the entire
    batch was invalid.
    """
    valid = [item for item in batch
             if item.get("valid", False)
             and item.get("node_embeddings") is not None]
    return valid if valid else None
