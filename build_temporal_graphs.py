"""
This builds the graph structure only (nodes, edges, temporal positions).

Usage (from project root):
    python build_temporal_graphs.py

Output structure:
    temporal_graph/full_graph/<test_name>/del_<node_idx>.json
"""

import json
import argparse
import logging
import time
from pathlib import Path

from config_utils import ConfigManager
from data.phase1.processing import build_full_graph_structure, find_commit_dir

CONFIG = ConfigManager().raw

logger = logging.getLogger(__name__)


def build_graphs(data_path: Path, output_dir: Path, test_cases: list) -> int:
    """Build and save full_graph temporal subgraphs for all deletion lines."""
    mode_dir = output_dir / "full_graph"

    total_saved = 0
    skipped = 0
    skipped_names: list = []
    start_time = time.time()

    for tc_idx, test_name in enumerate(test_cases):
        test_dir = data_path / test_name

        # ── Load info.json ──
        info_path = test_dir / "info.json"
        if not info_path.exists():
            skipped += 1
            skipped_names.append((test_name, "info.json not found"))
            continue

        try:
            with open(info_path) as f:
                info = json.load(f)
            fix_commit = info["fix"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped += 1
            skipped_names.append((test_name, f"info.json error: {exc}"))
            continue

        # Find fixing commit directory
        fix_commit_dir = find_commit_dir(test_dir, fix_commit)
        if not fix_commit_dir:
            skipped += 1
            skipped_names.append((test_name, f"commit dir not found: {fix_commit[:12]}"))
            continue

        # Load graph_vszz_full_history_trailByremovingStep3.json
        graph_path = fix_commit_dir / "graph_vszz_full_history_trailByremovingStep3.json"
        if not graph_path.exists():
            skipped += 1
            skipped_names.append((test_name, "vszz graph file not found"))
            continue

        try:
            with open(graph_path) as f:
                all_nodes = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            skipped += 1
            skipped_names.append((test_name, f"graph read error: {exc}"))
            continue

        # Create output directory for this test case
        test_output_dir = mode_dir / test_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Build subgraph for each deletion node
        for node_idx, node in enumerate(all_nodes):
            if not node.get("isDel", False):
                continue

            structure = build_full_graph_structure(
                all_nodes, node_idx, test_name, data_path
            )
            if structure is None:
                logger.warning("No graph structure for %s node %d", test_name, node_idx)
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
            try:
                with open(save_path, "w") as f:
                    json.dump(output, f)
                total_saved += 1
            except OSError as exc:
                logger.error("Failed to save %s: %s", save_path, exc)

        # Progress
        if (tc_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (tc_idx + 1) / elapsed
            print(f"  {tc_idx + 1}/{len(test_cases)} test cases | "
                  f"{total_saved} graphs saved | {rate:.1f} cases/sec")

    elapsed = time.time() - start_time
    print(f"\nDone: {total_saved} graphs saved, "
          f"{skipped} test cases skipped ({elapsed:.1f}s)")
    if skipped_names:
        print(f"\nSkipped test cases:")
        for name, reason in skipped_names:
            print(f"  {name}: {reason}")
    print(f"Output: {mode_dir}")
    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute full_graph temporal subgraphs and save as JSON")

    parser.add_argument(
        "--data_path", type=str,
        default=CONFIG["paths"]["data_root"],
        help="Path to trainData/ directory")
    parser.add_argument(
        "--output_dir", type=str,
        default=CONFIG["paths"]["prebuilt_dir"],
        help="Output directory for pre-built graphs")
    parser.add_argument(
        "--test_cases_file", type=str,
        default=str(Path(CONFIG["paths"]["data_root"]) / CONFIG["paths"]["test_cases_file"]),
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
