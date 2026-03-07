"""
Single source of truth for all hyperparameters, data paths, and architecture settings used across both training phases.
"""

# ── Graph edge types (shared by data/ and models/) ─────────────────────────────
EDGE_TYPES = {
    "CFG_FWD":      0,
    "CFG_BWD":      1,
    "DFG_FWD":      2,
    "DFG_BWD":      3,
    "LINEMAP":      4,
    "TEMPORAL_FWD": 5,
    "TEMPORAL_BWD": 6,
}
NUM_EDGE_TYPES = len(EDGE_TYPES)   # 7

CONFIG = {
    "data_path": "/mnt/data/NeuralSZZ/replication/replication/trainData",
    "test_cases_file": "successfultestcase661foundGT.json",
    "prebuilt_dir": "/mnt/data/NeuralSZZ/replication/replication/temporal_graph",
    "graph_mode": "full_graph",
    "save_dir": (
        "/mnt/data/NeuralSZZ/replication/replication/"
        "checkpoints_unified_two_phase"
    ),

    "phase1_epochs": 50,
    "phase1_lr": 5e-6,                    
    "phase1_bert_lr": 2e-5,                 # CodeBERT only
    "phase1_rest_lr": 1e-4,                 # graph layers + ranker
    "phase1_bert_freeze_bottom_layers": 12,
    "phase1_batch_size": 8,
    "phase1_patience": 10,
    "phase1_max_pairs_per_test": 100,

    "phase2_epochs": 100,
    "phase2_lr": 5e-5,
    "phase2_weight_decay": 0.05,
    "phase2_batch_size": 4,
    "phase2_patience": 15,
    "phase2_gradient_accumulation_steps": 2,
    "phase2_temperature": 1.0,
    "phase2_label_smoothing": 0.1,
    "phase2_top_k_lines": 1,

    "hidden_dim": 1536,
    "num_gt_layers": 1,
    "num_heads": 2,          
    "num_edge_types": NUM_EDGE_TYPES,
    "dropout": 0.1,
    "include_bert": True,
    "max_nodes_per_graph": 9500,    
    "bert_chunk": 256,            

  
    "phase2_num_heads": 8,
    "num_commit_transformer_layers": 2,
    "max_commits": 100,

    "n_folds": 10,
    "seed": 42,

    # GPU
    "gpu_id": 0,

    # DataLoader
    "num_workers": 4,

    # Phase 2 scoring (batch encoding)
    "max_nodes_per_batch": 4096,

    # Logging
    "log_interval": 20,
}