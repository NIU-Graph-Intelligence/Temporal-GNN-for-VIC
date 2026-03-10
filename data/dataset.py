"""
data/phase1/dataset.py
──────────────────────
DeletionLineDataset — loads pre-built Phase 1 graphs for training and evaluation.

The graph structure is assembled offline by ``scripts/build_temporal_graphs.py``
(which calls ``data.phase1.processing.build_full_graph_structure``).
This file only handles loading those pre-built files and embedding them.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from data.phase1.minigraph import MiniGraph
from data.phase1.pairs import DeletionLinePair, build_pairs
from data.phase1.processing import _build_pyg, _build_tp_to_commit


class DeletionLineDataset:
    """
    Loads pre-built deletion-line graphs for all test cases.

    Workflow
    --------
    1. Run ``scripts/build_temporal_graphs.py`` once to pre-compute and save
       ``del_*.json`` files under ``prebuilt_dir/full_graph/<test_name>/``.
    2. On the first training run, ``DeletionLineDataset`` reads each JSON,
       embeds nodes with CodeBERT, and writes a ``.pt`` cache alongside it.
    3. All subsequent runs load directly from the ``.pt`` caches — fast RAM-only.

    Parameters
    ----------
    data_path    : path to trainData/ directory (used only for info.json lookups)
    test_cases   : list of test-case folder names
    embedder     : CodeBERTEmbedder (tokenizer_only=True for fine-tuning)
    prebuilt_dir : path to the temporal_graph/ directory produced by the
                   pre-computation script (required)
    """

    def __init__(
        self,
        data_path: str,
        test_cases: List[str],
        embedder,
        prebuilt_dir: str,
    ) -> None:
        self.data_path    = Path(data_path)
        self.test_cases   = test_cases
        self.embedder     = embedder
        self.prebuilt_dir = Path(prebuilt_dir)
        self.mini_graphs: Dict[str, List[MiniGraph]] = {}

        self._load_from_prebuilt()
        self._populate_inducing_commits()

    # Loader 
    def _load_from_prebuilt(self) -> None:
        """
        Load all graphs from pre-built JSON files, using .pt caches.

        Cache versioning: if the embedder uses tokenize_texts (v2 format) but
        only v1 caches exist (with pre-computed ``x`` tensors), old caches are
        cleared automatically and regenerated.
        """
        mode_dir = self.prebuilt_dir / "full_graph"

        if not mode_dir.exists():
            raise FileNotFoundError(
                f"Pre-built graph directory not found: {mode_dir}\n"
                f"Run:  python scripts/build_temporal_graphs.py"
            )

        # Invalidate v1 caches when switching to tokenizer-only (v2) format
        if getattr(self.embedder, "tokenizer_only", False):
            sentinel = mode_dir / ".cache_format_v2"
            if not sentinel.exists():
                import glob as _glob
                old_pts = _glob.glob(str(mode_dir / "**" / "*.pt"), recursive=True)
                if old_pts:
                    print(f"  Clearing {len(old_pts)} stale embedding caches "
                          f"(switching to tokenized format)...")
                    for p in old_pts:
                        try:
                            Path(p).unlink()
                        except OSError:
                            pass
                try:
                    sentinel.parent.mkdir(parents=True, exist_ok=True)
                    sentinel.touch()
                except OSError:
                    pass

        total, skipped = 0, 0

        for tc_idx, test_name in enumerate(self.test_cases):
            test_dir = mode_dir / test_name
            if not test_dir.exists():
                skipped += 1
                continue

            graphs: List[MiniGraph] = []
            for json_path in sorted(test_dir.glob("del_*.json")):
                mg = self._load_one(json_path, test_name)
                if mg is not None:
                    graphs.append(mg)

            if graphs:
                self.mini_graphs[test_name] = graphs
                total += len(graphs)

            if (tc_idx + 1) % 50 == 0:
                print(f"  [dataset] {tc_idx + 1}/{len(self.test_cases)} "
                      f"test cases — {total} graphs loaded")

        print(f"  [dataset] {len(self.mini_graphs)} test cases, "
              f"{total} graphs, {skipped} skipped")

    def _load_one(self, json_path: Path, test_name: str) -> Optional[MiniGraph]:
        """
        Load one deletion-line graph, using the .pt cache when available.
        Falls back to JSON -> embed -> write cache on first run.

        The cache stores history_chains and deletion-node metadata so that
        Phase 2 can build temporal graphs directly from Phase 1 results.
        """
        cache_path = json_path.with_suffix(".pt")

        if cache_path.exists():
            try:
                cached          = torch.load(cache_path, map_location="cpu")
                mg              = MiniGraph([], cached["pyg"], test_name,
                                            cached.get("del_idx", 0))
                mg.rootcause    = cached.get("rootcause", False)
                mg.tp_to_commit = cached.get("tp_to_commit", {})
                mg.history_chains = cached.get("history_chains")
                mg.del_line_beg   = cached.get("del_line_beg")
                mg.del_code       = cached.get("del_code")

                if mg.history_chains is not None:
                    return mg

                # Old cache format — backfill from JSON and re-save
                with open(json_path) as f:
                    data = json.load(f)
                nodes = data.get("nodes", [])
                if nodes:
                    mg.history_chains = nodes[0].get("history_chains", [])
                    mg.del_line_beg   = nodes[0].get("lineBeg", 0)
                    mg.del_code       = nodes[0].get("code", "")
                    cached["history_chains"] = mg.history_chains
                    cached["del_line_beg"]   = mg.del_line_beg
                    cached["del_code"]       = mg.del_code
                    try:
                        torch.save(cached, cache_path)
                    except Exception:
                        pass
                else:
                    mg.history_chains = []
                return mg
            except Exception:
                pass  # corrupt cache — fall through and rebuild

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            return None

        nodes              = data.get("nodes", [])
        edges              = data.get("edges", [])
        temporal_positions = data.get("temporal_positions", [])
        if not nodes:
            return None

        pyg = _build_pyg(nodes, edges, temporal_positions, self.embedder)
        if pyg is None:
            return None

        rootcause      = data.get("rootcause", False)
        del_idx_val    = data.get("del_idx", 0)
        tp_to_commit   = _build_tp_to_commit(nodes, temporal_positions)
        history_chains = nodes[0].get("history_chains", [])
        del_line_beg   = nodes[0].get("lineBeg", 0)
        del_code       = nodes[0].get("code", "")

        try:
            torch.save(
                {"pyg": pyg, "rootcause": rootcause,
                 "del_idx": del_idx_val, "tp_to_commit": tp_to_commit,
                 "history_chains": history_chains,
                 "del_line_beg": del_line_beg, "del_code": del_code},
                cache_path,
            )
        except Exception:
            pass

        mg                = MiniGraph(nodes, pyg, test_name, del_idx_val)
        mg.rootcause      = rootcause
        mg.tp_to_commit   = tp_to_commit
        mg.history_chains = history_chains
        mg.del_line_beg   = del_line_beg
        mg.del_code       = del_code
        return mg

    def get_mini_graphs_dict(self) -> Dict[str, List[MiniGraph]]:
        return self.mini_graphs

    def get_test_cases(self) -> List[str]:
        return list(self.mini_graphs.keys())

    def get_all_pairs(self, max_pairs_per_test: int = 50) -> List[DeletionLinePair]:
        """Generate pairwise training examples across all test cases."""
        all_pairs: List[DeletionLinePair] = []
        for graphs in self.mini_graphs.values():
            all_pairs.extend(build_pairs(graphs, max_pairs_per_test))
        return all_pairs

    def get_pairs_for_cases(
        self, test_cases: List[str], max_pairs_per_test: int = 50,
    ) -> List[DeletionLinePair]:
        """Generate pairwise training examples for a subset of test cases."""
        all_pairs: List[DeletionLinePair] = []
        for name in test_cases:
            if name in self.mini_graphs:
                all_pairs.extend(build_pairs(self.mini_graphs[name],
                                             max_pairs_per_test))
        return all_pairs

    def _populate_inducing_commits(self) -> None:
        """
        Read info.json for each test case and attach the inducing commit
        set (full SHA + 12-char prefix) to every MiniGraph in that case.
        """
        for test_name, graphs in self.mini_graphs.items():
            info_path = self.data_path / test_name / "info.json"
            inducing: set = set()
            if info_path.exists():
                try:
                    with open(info_path) as f:
                        for sha in json.load(f).get("induce", []):
                            inducing.add(sha)
                            inducing.add(sha[:12])
                except Exception:
                    pass
            for mg in graphs:
                mg.inducing_commits = inducing


# Phase 2 — pre-computed embedding dataset

class CommitRankingDataset(Dataset):
    
    """
    Wraps pre-computed node embeddings and commit metadata for Phase 2 training.
    Items are produced by training.embedding_cache.build_phase2_items and stored as plain dicts.
    """

    def __init__(self, items: List[Dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_commit_ranking(batch: List[Dict]) -> Optional[Dict]:
    """
    DataLoader collate function for CommitRankingDataset.
    """
    valid = [
        item for item in batch
        if item.get("valid", False) and item.get("node_embeddings") is not None
    ]
    if not valid:
        return None

    offset          = 0
    all_embeddings: List[torch.Tensor] = []
    all_indices:    List[torch.Tensor] = []
    commit_counts:  List[int]          = []
    gt_positions:   List[List[int]]    = []

    for item in valid:
        ci = item["commit_indices"]
        n_commits = int(ci.max().item()) + 1
        all_embeddings.append(item["node_embeddings"])
        all_indices.append(ci + offset)
        commit_counts.append(n_commits)
        gt_positions.append(item["ground_truth_positions"])
        offset += n_commits

    return {
        "node_embeddings":        torch.cat(all_embeddings, dim=0),
        "commit_indices":         torch.cat(all_indices,    dim=0),
        "commit_counts":          commit_counts,
        "ground_truth_positions": gt_positions,
    }