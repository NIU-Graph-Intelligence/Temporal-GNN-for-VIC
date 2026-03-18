"""
data/phase1/processing.py
─────────────────────────
Low-level graph processing helpers for Phase 1 full-graph construction.

  _find_commit_dir        — locate a commit's subdirectory by SHA prefix
  _find_history_node      — match a V-SZZ history entry to a graph.json node
  _make_synthetic_node    — create a placeholder node for empty graph.json files
  _build_cfg_dfg_edges    — build intra-section CFG/DFG/LINEMAP edges
  _build_pyg              — convert raw nodes+edges into a PyG Data object
  _build_tp_to_commit     — extract temporal_pos → commit SHA mapping
  build_full_graph_structure — assemble raw {nodes, edges, temporal_positions}
                               from V-SZZ data (used by scripts/build_temporal_graphs.py)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data

from config import EDGE_TYPES


# Filesystem helpers 

def _find_commit_dir(test_dir: Path, commit_sha: str) -> Optional[Path]:
    """Return the subdirectory whose name starts with the first 12 chars of SHA."""
    prefix = commit_sha[:12]
    for d in test_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


#  Node matching 

def _find_history_node(
    hist_graph_nodes: List[Dict], hist_entry: Dict
) -> Optional[int]:
    """
    Find the node index in graph.json matching a V-SZZ history entry.

    Priority:
      1. Line-number range  ∩  code prefix match
      2. Line-number range only
      3. Code prefix match only
    """
    target_line = hist_entry.get("line_num")
    target_code = hist_entry.get("code", "").strip()

    if target_line is not None and target_code:
        for i, n in enumerate(hist_graph_nodes):
            lb, le = n.get("lineBeg", -1), n.get("lineEnd", -1)
            if lb <= target_line <= le:
                nc = n.get("code", "")
                if target_code[:20] in nc or nc[:20] in target_code:
                    return i

    if target_line is not None:
        for i, n in enumerate(hist_graph_nodes):
            if n.get("lineBeg", -1) <= target_line <= n.get("lineEnd", -1):
                return i

    if target_code:
        for i, n in enumerate(hist_graph_nodes):
            nc = n.get("code", "")
            if target_code[:20] in nc or nc[:20] in target_code:
                return i

    return None


def _make_synthetic_node(hist_entry: Dict) -> Dict:
    """
    Minimal graph.json-style node from a V-SZZ history entry.

    Used when a history commit's graph.json is empty so the temporal
    chain stays connected.
    """
    line_num = hist_entry.get("line_num", 0)
    return {
        "nodeIndex": 0,
        "lineBeg":   line_num,
        "lineEnd":   line_num,
        "code":      hist_entry.get("code", ""),
        "cfgs":      [],
        "dfgs":      [],
        "rootcause": False,
    }


#  Edge construction

def _build_cfg_dfg_edges(
    subgraph_nodes: List[Dict], section_start: int, section_end: int
) -> List[tuple]:
    """
    Build CFG, DFG, and LINEMAP edges for one contiguous section of the graph.

    graph.json node indices are local (0-based within each section), so the
    global index = section_start + local_idx.
    """
    edges = []
    for i in range(section_start, section_end):
        node = subgraph_nodes[i]

        for cfg_t in node.get("cfgs", []):
            target = section_start + cfg_t
            if section_start <= target < section_end:
                edges.append((i, target, EDGE_TYPES["CFG_FWD"]))
                edges.append((target, i, EDGE_TYPES["CFG_BWD"]))

        for dfg_t in node.get("dfgs", []):
            target = section_start + dfg_t
            if section_start <= target < section_end:
                edges.append((i, target, EDGE_TYPES["DFG_FWD"]))
                edges.append((target, i, EDGE_TYPES["DFG_BWD"]))

        lmi = node.get("lineMapIndex", -1)
        if lmi != -1:
            target = section_start + lmi
            if section_start <= target < section_end:
                edges.append((i, target, EDGE_TYPES["LINEMAP"]))
                edges.append((target, i, EDGE_TYPES["LINEMAP"]))

    return edges


# PyG conversion 

def _build_pyg(
    nodes: List[Dict],
    edges: List[tuple],
    temporal_positions: List[int],
    embedder,
) -> Optional[Data]:
    """
    Convert graph nodes + edges into a PyG ``Data`` object.

    Tokenize mode (embedder.tokenizer_only=True):
        Stores ``token_ids`` + ``attention_mask``; CodeBERT runs inside
        the model at training time.

    Embed mode (embedder.tokenizer_only=False):
        Calls ``embedder.encode_texts()`` to pre-compute ``x`` embeddings.
    """
    node_texts = [n.get("code", "") for n in nodes]
    if not node_texts:
        return None

    temporal_pos = torch.tensor(temporal_positions, dtype=torch.long)

    if edges:
        src        = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst        = torch.tensor([e[1] for e in edges], dtype=torch.long)
        etype      = torch.tensor([e[2] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        etype      = torch.empty((0,),   dtype=torch.long)

    if getattr(embedder, "tokenizer_only", False):
        toks = embedder.tokenize_texts(node_texts)
        return Data(
            token_ids      = toks["token_ids"],
            attention_mask = toks["attention_mask"],
            edge_index     = edge_index,
            edge_type      = etype,
            num_nodes      = toks["token_ids"].size(0),
            temporal_pos   = temporal_pos,
        )
    else:
        X = embedder.encode_texts(node_texts)
        if X.size(0) == 0:
            return None
        return Data(
            x            = X,
            edge_index   = edge_index,
            edge_type    = etype,
            num_nodes    = X.size(0),
            temporal_pos = temporal_pos,
        )


#  Metadata extraction

def _build_tp_to_commit(
    nodes: List[Dict], temporal_positions: List[int]
) -> Dict[int, str]:
    """
    Build {temporal_pos → 12-char commit SHA} from the deletion node's
    history_chains.
    """
    if not nodes:
        return {}
    actual_tps = set(temporal_positions)
    tp_map: Dict[int, str] = {}
    for chain in nodes[0].get("history_chains", []):
        for i, entry in enumerate(chain.get("history", [])):
            tp = i + 1
            if tp not in tp_map and tp in actual_tps:
                tp_map[tp] = entry.get("commit", "")
    return tp_map



def build_full_graph_structure(
    all_nodes: List[Dict],
    del_node_idx: int,
    test_name: str,
    data_path: Path,
) -> Optional[Dict]:
    """
    Build the raw full-graph structure for one deletion line.

    Called by ``scripts/build_temporal_graphs.py`` to produce the
    ``del_*.json`` files that ``DeletionLineDataset`` loads at training time.

    Layout
    ------
    Section 0     — the single deletion-line node (from the fixing commit)
    Sections 1..N — complete graph.json from each historical commit,
                    one section per history step per chain

    Edges
    -----
    CFG/DFG/LINEMAP  — intra-section (within each history graph)
    TEMPORAL_FWD/BWD — per chain: deletion node -> C1 match -> C2 match ...

    Returns {nodes, edges, temporal_positions} or None.
    """
    del_node = all_nodes[del_node_idx]
    test_dir = data_path / test_name

    subgraph_nodes:     List[Dict]      = [del_node]
    temporal_positions: List[int]       = [0]
    section_starts:     List[int]       = [0]
    temporal_chains:    List[List[int]] = []

    for chain in del_node.get("history_chains", []):
        chain_globals: List[int] = []

        for hist_idx, hist_entry in enumerate(chain.get("history", [])):
            commit_sha = hist_entry.get("commit", "")
            temp_pos   = hist_idx + 1

            commit_dir = _find_commit_dir(test_dir, commit_sha)

            graph_path = commit_dir / "graph.json"
            
            try:
                with open(graph_path) as f:
                    hist_nodes = json.load(f)
            except Exception as e:
                print(f"  [processing] Error reading {graph_path}: {e}")
                continue

            sec_start = len(subgraph_nodes)
            section_starts.append(sec_start)

            if not hist_nodes:
                # Empty graph.json — insert a synthetic node to preserve chain
                subgraph_nodes.append(_make_synthetic_node(hist_entry))
                temporal_positions.append(temp_pos)
                chain_globals.append(sec_start)
                continue

            matched_idx = _find_history_node(hist_nodes, hist_entry) or 0
            chain_globals.append(sec_start + matched_idx)
            for node in hist_nodes:
                subgraph_nodes.append(node)
                temporal_positions.append(temp_pos)

        if chain_globals:
            temporal_chains.append(chain_globals)

    edges: List[tuple] = []
    num_nodes = len(subgraph_nodes)

    for chain_globals in temporal_chains:
        prev = 0
        for g_idx in chain_globals:
            edges.append((prev, g_idx, EDGE_TYPES["TEMPORAL_FWD"]))
            edges.append((g_idx, prev, EDGE_TYPES["TEMPORAL_BWD"]))
            prev = g_idx

    for s_idx in range(1, len(section_starts)):
        s_start = section_starts[s_idx]
        s_end   = (section_starts[s_idx + 1]
                   if s_idx + 1 < len(section_starts) else num_nodes)
        edges.extend(_build_cfg_dfg_edges(subgraph_nodes, s_start, s_end))

    return {
        "nodes":              subgraph_nodes,
        "edges":              edges,
        "temporal_positions": temporal_positions,
    }

