"""
Pre-compute full_graph temporal subgraphs for all deletion lines and save as JSON.

This builds the graph structure only (nodes, edges, temporal positions).

Usage (from project root):
    python scripts/build_temporal_graphs.py

Output structure:
    temporal_graph/full_graph/<test_name>/del_<node_idx>.json
"""

import json
import argparse
import time
from pathlib import Path

from data.phase1.processing import build_full_graph_structure, _find_commit_dir


def build_graphs(data_path: Path, output_dir: Path, test_cases: list):
    """Build and save full_graph temporal subgraphs for all deletion lines."""
    mode_dir = output_dir / "full_graph"

    total_saved = 0
    skipped = 0
    start_time = time.time()

    for tc_idx, test_name in enumerate(test_cases):
        test_dir = data_path / test_name

        # ── Load info.json ──
        info_path = test_dir / "info.json"
        if not info_path.exists():
            skipped += 1
            continue

        try:
            with open(info_path) as f:
                info = json.load(f)
            fix_commit = info["fix"]
        except Exception:
            skipped += 1
            continue

        # ── Find fixing commit directory ──
        fix_commit_dir = _find_commit_dir(test_dir, fix_commit)
        if not fix_commit_dir:
            skipped += 1
            continue

        # ── Load graph_vszz_full_history.json ──
        graph_path = fix_commit_dir / "graph_vszz_full_history.json"
        if not graph_path.exists():
            skipped += 1
            continue

        try:
            with open(graph_path) as f:
                all_nodes = json.load(f)
        except Exception:
            skipped += 1
            continue

        if not all_nodes:
            skipped += 1
            continue

        # ── Create output directory for this test case ──
        test_output_dir = mode_dir / test_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # ── Build subgraph for each deletion node ──
        for node_idx, node in enumerate(all_nodes):
            if not node.get("isDel", False):
                continue

            structure = build_full_graph_structure(
                all_nodes, node_idx, test_name, data_path
            )
            if structure is None:
                continue

            output = {
                "test_name": test_name,
                "graph_mode": "full_graph",
                "node_idx_in_graph": node_idx,
                "rootcause": node.get("rootcause", False),
                "del_idx": 0,
                "nodes": structure["nodes"],
                "edges": [[s, d, t] for s, d, t in structure["edges"]],
                "temporal_positions": structure["temporal_positions"],
            }

            save_path = test_output_dir / f"del_{node_idx}.json"
            with open(save_path, "w") as f:
                json.dump(output, f)

            total_saved += 1

        # Progress
        if (tc_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (tc_idx + 1) / elapsed
            print(f"  {tc_idx + 1}/{len(test_cases)} test cases | "
                  f"{total_saved} graphs saved | {rate:.1f} cases/sec")

    elapsed = time.time() - start_time
    print(f"\nDone: {total_saved} graphs saved, "
          f"{skipped} test cases skipped ({elapsed:.1f}s)")
    print(f"Output: {mode_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute full_graph temporal subgraphs and save as JSON")

    parser.add_argument(
        "--data_path", type=str,
        default="/mnt/data/NeuralSZZ/replication/replication/trainData",
        help="Path to trainData/ directory")
    parser.add_argument(
        "--output_dir", type=str,
        default="/mnt/data/NeuralSZZ/replication/replication/temporal_graph",
        help="Output directory for pre-built graphs")
    parser.add_argument(
        "--test_cases_file", type=str,
        default="/mnt/data/NeuralSZZ/replication/replication/trainData/"
                "successfultestcase661foundGT.json",
        help="JSON file listing test case names")

    args = parser.parse_args()

    with open(args.test_cases_file) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    total = build_graphs(
        Path(args.data_path),
        Path(args.output_dir),
        test_cases,
    )

    print(f"\nALL DONE: {total} graphs saved")


if __name__ == "__main__":
    main()
