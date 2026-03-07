# NeuralSZZ

Identify bug-inducing commits using a two-phase neural approach: first rank deletion lines in the fix commit to find root-cause lines, then rank candidate commits using a frozen encoder and a commit-level ranking head.

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Graph Construction](#graph-construction)
- [Models](#models)
- [Training](#training)
- [Data](#data)

## Motivation

Traditional SZZ-based bug-inducing commit identification relies on heuristics and line-level tracing. This project explores a neural two-phase pipeline:

1. **Phase 1 (Deletion Line Ranking)**: Given a bug-fix commit, rank deletion lines to identify the most likely root-cause lines using pairwise learning and a graph encoder.
2. **Phase 2 (Commit Ranking)**: Given the top-ranked deletion line and its temporal history, rank candidate bug-inducing commits using pre-computed node embeddings and a commit-level transformer head.

The encoder is trained only in Phase 1; Phase 2 reuses its frozen representations and trains a lightweight ranking module, avoiding redundant encoder passes and reducing training cost.

## Key Features

- **Two-phase pipeline**: Deletion line ranking → commit ranking
- **Shared graph encoder**: CodeBERT + Graph Transformer with temporal positional encoding
- **Pre-computed embeddings**: Phase 2 uses cached encoder output; encoder runs once per test case, not per epoch
- **Batched MiniGraph encoding**: PyG `Batch.from_data_list` for efficient scoring
- **K-fold cross-validation**: Stratified or plain K-fold over test cases
- **Pairwise (Phase 1) and listwise (Phase 2) losses**: BCE pairwise + label-smoothing listwise CE
- **Timing instrumentation**: Per-step timing in `main.py` for profiling

## Repository Layout

```
config.py                 Global hyperparameters, paths, edge types
main.py                   Entry point: K-fold loop, Phase 1 + Phase 2 orchestration
build_temporal_graphs.py  Pre-compute full_graph temporal subgraphs (Step 1)

data/
  phase1/
    dataset.py            DeletionLineDataset — load pre-built graphs, .pt cache
    minigraph.py          MiniGraph — one deletion line + PyG graph context
    pairs.py              DeletionLinePair, Batch, build_pairs, combine_pairs_to_batches
    processing.py         build_full_graph_structure, _build_pyg, _build_tp_to_commit
  phase2/
    dataset.py            build_fix_commit_pyg, score_and_cache_top_embeddings,
                          precompute_phase2_embeddings, Phase2EmbeddingDataset, collate_phase2

models/
  shared_encoder.py        CodeBERTEmbedder, GraphTransformerLayer, SharedEncoder
  phase1_model.py         DeletionLineRanker, DeletionLineRankingModel
  phase2_model.py         CommitRankingModule

training/
  phase1_trainer.py       Phase1Trainer, train_phase1_fold
  phase2_trainer.py       train_phase2_fold (CommitRankingModule only)
  loss.py                 PairwiseRankingLoss, LabelSmoothingRankingLoss
  evaluation.py           load_true_commit_map, evaluate_topk_metrics, evaluate_ranking
  utils.py                set_seed, setup_device, EarlyStopping, build_phase1_model, etc.

scripts/
  run_dummy_smoke_test.py Minimal run (12 test cases, 2 folds, 2 epochs) with timing
  build_temporal_graphs.py (if moved) Pre-compute temporal graphs

trainData/                Raw test cases: <test_name>/info.json, commits.json, <sha>/graph.json
temporal_graph/           Pre-built graphs: full_graph/<test_name>/del_*.json
checkpoints_unified_two_phase/  Saved models, unified_kfold_summary.json
```

## Installation

```bash
conda create -n neuralszz python=3.7
conda activate neuralszz
pip install torch torchvision torchaudio  # adjust for your CUDA version
pip install torch-geometric transformers
pip install scikit-learn
```

Ensure `trainData/` and `temporal_graph/` paths in `config.py` point to your data directories.

## Quick Start

```bash
# 1. Pre-compute temporal graphs (Step 1 — run once)
python build_temporal_graphs.py

# 2. Full training (K-fold, default config)
python main.py

# 3. Dummy smoke test (12 test cases, 2 folds, 2 epochs)
python scripts/run_dummy_smoke_test.py

# 4. Override config via CLI
python main.py --n-folds 5 --phase1-epochs 10 --phase2-epochs 20 --save-dir ./my_checkpoints
```

## Graph Construction

*Details of the graph construction pipeline (Step 1) will be added here. This includes:*

- *Building full temporal graphs from V-SZZ history*
- *Node and edge types (CFG, DFG, LINEMAP, TEMPORAL)*
- *Output structure: `temporal_graph/full_graph/<test_name>/del_*.json`*
- *Integration with `build_temporal_graphs.py` and `data.phase1.processing`*

## Models

### Shared Encoder (`models/shared_encoder.py`)

- **CodeBERTEmbedder**: Tokenizer wrapper around `microsoft/unixcoder-base-nine`; `tokenizer_only=True` for fine-tuning (tokens passed to SharedEncoder).
- **GraphTransformerLayer**: Multi-head attention over graph edges with per-edge-type bias, followed by FFN (GELU) and residual connections.
- **SharedEncoder**: CodeBERT → linear projection → sinusoidal temporal PE → stacked GraphTransformer layers. Output: `[N, hidden_dim]` node embeddings.

### Phase 1: Deletion Line Ranking (`models/phase1_model.py`)

- **DeletionLineRanker**: RankNet-style MLP (hidden_dim → 32 → 16 → 8 → 1). Pairwise forward: `sigmoid(score(emb1) - score(emb2))`.
- **DeletionLineRankingModel**: SharedEncoder + DeletionLineRanker. Extracts the deletion-line node embedding and scores it. Trained with pairwise BCE.

### Phase 2: Commit Ranking (`models/phase2_model.py`)

- **CommitRankingModule**: Operates on pre-computed node embeddings (no encoder). Multi-head attention pooling (nodes → one vector per commit) → TransformerEncoder over commits → ranking head (Linear → GELU → Dropout → Linear) → scores `[C]`. Trained with label-smoothing listwise CE + pairwise margin.

### Edge Types (`config.py`)

- CFG_FWD, CFG_BWD, DFG_FWD, DFG_BWD, LINEMAP, TEMPORAL_FWD, TEMPORAL_BWD (7 types).

## Training

### Phase 1

- **Data**: Pairwise examples from `DeletionLineDataset` via `build_pairs` (rootcause vs non-rootcause, ties).
- **Loss**: `PairwiseRankingLoss` (BCE).
- **Optimizer**: Adam with differential LRs (CodeBERT vs graph layers + ranker).
- **Flow**: `train_phase1_fold` → `Phase1Trainer.train` → `_run_pairs_epoch` (accumulate loss, single backward per batch).

### Phase 2

- **Data**: Pre-computed embeddings from `score_and_cache_top_embeddings` + `precompute_phase2_embeddings`. `Phase2EmbeddingDataset` serves `node_embeddings`, `commit_indices`, `ground_truth_positions`.
- **Loss**: `LabelSmoothingRankingLoss` (listwise CE + pairwise margin).
- **Optimizer**: AdamW.
- **Flow**: `train_phase2_fold` → `CommitRankingModule` only; encoder is frozen and not used in the training loop.

### Metrics

- Phase 1: Precision@1, Recall@1, F1@1.
- Phase 2: Precision@k, Recall@k, F1@k (k=1,2,3,5), MRR.

### Timing

`main.py` prints elapsed time for: tokenizer init, dataset load, Phase 1 training, build model for scoring, score and cache, precompute Phase 2 embeddings, Phase 2 training, results aggregation.

## Data

### Directory Structure

```
trainData/
  <test_name>/
    info.json           fix commit, inducing commits (ground truth)
    commits.json        fix_commit, ground_truth, vszz_introducer_commits
    <sha12...>/         per-commit directories
      graph.json        CFG/DFG nodes and edges
      graph_vszz_full_history.json  (used by build_temporal_graphs)

temporal_graph/
  full_graph/
    <test_name>/
      del_0.json, del_1.json, ...   pre-built temporal subgraphs per deletion line
      del_*.pt                       cached tokenized PyG (created on first run)
```

### Phase 1 Data

- **DeletionLineDataset**: Loads `del_*.json` from `temporal_graph/full_graph/<test_name>/`. First run: tokenize with CodeBERT, save `.pt`. Subsequent runs: load `.pt` directly.
- **MiniGraph**: Holds `pyg` (PyG Data), `tp_to_commit`, `inducing_commits`, `history_chains`, `del_line_beg`, `del_code`.
- **Pairs**: `build_pairs` generates `DeletionLinePair` (x, y, prob). `combine_pairs_to_batches` splits into batches for training.

### Phase 2 Data

- **score_and_cache_top_embeddings**: Scores every MiniGraph per test case, caches encoder output for the top-1. Uses batched encoding (`Batch.from_data_list`) with `max_nodes_per_batch` to avoid OOM.
- **precompute_phase2_embeddings**: Reuses cached history embeddings; encodes only the fix commit’s `graph.json`. Produces `node_embeddings`, `commit_indices`, `ground_truth_positions` per test case.
- **Phase2EmbeddingDataset**: Wraps pre-computed items for the DataLoader. `collate_phase2` filters invalid samples.

### Config (`config.py`)

- Paths: `data_path`, `test_cases_file`, `prebuilt_dir`, `save_dir`
- Phase 1: `phase1_epochs`, `phase1_batch_size`, `phase1_max_pairs_per_test`, `phase1_bert_freeze_bottom_layers`, etc.
- Phase 2: `phase2_epochs`, `phase2_batch_size`, `phase2_patience`, `phase2_temperature`, `phase2_label_smoothing`, etc.
- Model: `hidden_dim`, `num_gt_layers`, `num_heads`, `max_nodes_per_graph`, `max_nodes_per_batch`, `num_workers`
