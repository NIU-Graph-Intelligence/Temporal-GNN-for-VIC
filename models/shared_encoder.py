import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from transformers import AutoTokenizer, AutoModel

from config import NUM_EDGE_TYPES

MODEL_NAME = "microsoft/unixcoder-base-nine"
MAX_LEN = 64
EMB_DIM = 768


# Module-level sinusoidal PE (shared by SharedEncoder and CommitRankingModule)

def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """
    Return a fixed sinusoidal positional encoding table [max_len, d_model].

    Shared by SharedEncoder (temporal node positions) and
    CommitRankingModule (commit-order positions) 
    """
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# CodeBERT embedder 

class CodeBERTEmbedder(nn.Module):
    """
    Tokeniser wrapper around CodeBERT.

    Converts raw code strings into token_ids + attention_mask (CPU tensors)
    for consumption by SharedEncoder when include_bert=True.
    Instantiate with tokenizer_only=True (the default use-case) to avoid
    loading model weights here — CodeBERT lives inside SharedEncoder instead.
    """

    def __init__(self, model_name: str = MODEL_NAME, max_len: int = MAX_LEN,
                 tokenizer_only: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_only = tokenizer_only
        self.max_len = max_len

    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenise code strings to CPU tensors (no gradient).

        Returns {"token_ids": [N, max_len], "attention_mask": [N, max_len]}.
        """
        if not texts:
            return {
                "token_ids": torch.zeros(0, self.max_len, dtype=torch.long),
                "attention_mask": torch.zeros(0, self.max_len, dtype=torch.long),
            }
        toks = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {"token_ids": toks["input_ids"],
                "attention_mask": toks["attention_mask"]}


# Graph attention layer

class GraphTransformerLayer(nn.Module):
    """
    Single graph-attention layer with edge-type-specific bias.

    Multi-head attention over graph edges, with per-edge-type learned
    bias terms added to attention logits before softmax.  Followed by
    a two-layer FFN with GELU activation and residual connections.
    """

    def __init__(self, d_model: int, num_heads: int = 8,
                 num_edge_types: int = NUM_EDGE_TYPES,
                 dropout: float = 0.1, use_edge_features: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_edge_features = use_edge_features
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_edge_features:
            self.edge_bias = nn.Embedding(num_edge_types, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, d_model] node features
            edge_index : [2, E] edge indices
            edge_type  : [E]    edge type ids
        Returns:
            x          : [N, d_model] updated node features
        """
        N = x.size(0)
        if edge_index.size(1) == 0:
            return self.norm2(x + self.dropout(self.ffn(x)))

        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        src, dst = edge_index
        attn = (Q[dst] * K[src]).sum(dim=-1) * self.scale   # [E, H]
        if self.use_edge_features:
            attn = attn + self.edge_bias(edge_type)
        attn = pyg_softmax(attn, dst, num_nodes=N)
        attn = self.dropout(attn)

        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim),
            attn.unsqueeze(-1) * V[src],
        )
        out = self.out_proj(out.view(N, self.d_model))

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# Shared encoder 

class SharedEncoder(nn.Module):
    """
    Full encoder shared by Phase 1 (deletion line ranking) and Phase 2
    (commit ranking, where it is frozen).

    Pipeline
    --------
    token_ids [N, seq_len]
        ↓  CodeBERT (fine-tuned, chunked + gradient-checkpointed)
    CLS embeddings [N, 768]
        ↓  input_proj  (Linear → LayerNorm → GELU → Dropout)
    [N, hidden_dim]
        ↓  + fixed sinusoidal temporal PE
        ↓  GraphTransformerLayer × num_gt_layers
    node representations [N, hidden_dim]

    When include_bert=False the encoder expects pre-computed 768-dim
    embeddings in ``x`` (backward-compatible / frozen-CodeBERT mode).
    """

    def __init__(self, input_dim: int = EMB_DIM, hidden_dim: int = 1536,
                 num_gt_layers: int = 4, num_heads: int = 8,
                 num_edge_types: int = NUM_EDGE_TYPES, dropout: float = 0.2,
                 max_temporal_positions: int = 20, use_checkpoint: bool = True,
                 include_bert: bool = True, num_bert_layers_freeze: int = 0,
                 bert_chunk: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_bert = include_bert
        self.bert_chunk = bert_chunk
        self.use_checkpoint = use_checkpoint

        if include_bert:
            self.bert_model = AutoModel.from_pretrained(MODEL_NAME)
            if num_bert_layers_freeze > 0:
                for p in self.bert_model.embeddings.parameters():
                    p.requires_grad = False
                for i in range(min(num_bert_layers_freeze,
                                   len(self.bert_model.encoder.layer))):
                    for p in self.bert_model.encoder.layer[i].parameters():
                        p.requires_grad = False
            # self._bert_anchor = self._find_bert_anchor()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.register_buffer("temporal_pos_embedding",
                             sinusoidal_pe(max_temporal_positions, hidden_dim))

        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, num_edge_types, dropout)
            for _ in range(num_gt_layers)
        ])

    # def _find_bert_anchor(self):
    #     """Find the first trainable non-pooler BERT param (for checkpointing)."""
    #     if not self.include_bert:
    #         return None
    #     return next(
    #         (p for name, p in self.bert_model.named_parameters()
    #          if p.requires_grad and "pooler" not in name),
    #         None,
    #     )

    def _run_bert(self, token_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Run CodeBERT in chunks with optional gradient checkpointing.

        Checkpointing discards intermediate activations and recomputes
        them during backward — essential for graphs with many nodes.
        Called only when include_bert=True.
        """
        dev = next(self.bert_model.parameters()).device


        bert_anchor = next(
                    (p for name, p in self.bert_model.named_parameters()
                    if p.requires_grad and "pooler" not in name),
                    None,
                    )
        # bert_anchor = self._bert_anchor
        can_checkpoint = self.use_checkpoint and bert_anchor is not None
        # chunk = min(self.bert_chunk, 64) if can_checkpoint else self.bert_chunk

        def _cls(anchor, ids, mask):
            return self.bert_model(input_ids=ids,
                                   attention_mask=mask).last_hidden_state[:, 0, :]

        pieces = []
        for i in range(0, token_ids.size(0), self.bert_chunk):
            ids_c = token_ids[i:i + self.bert_chunk].to(dev)
            mask_c = attention_mask[i:i + self.bert_chunk].to(dev)
            if self.training and can_checkpoint:
                emb = torch.utils.checkpoint.checkpoint(
                    _cls, bert_anchor, ids_c, mask_c)
            else:
                emb = _cls(bert_anchor, ids_c, mask_c)
            pieces.append(emb)
        return torch.cat(pieces, dim=0)

    def forward(self, x: torch.Tensor = None, edge_index: torch.Tensor = None,
                edge_type: torch.Tensor = None,
                temporal_pos: torch.Tensor = None,
                token_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x              : [N, 768]      pre-computed embeddings (include_bert=False)
            token_ids      : [N, seq_len]  raw token ids          (include_bert=True)
            attention_mask : [N, seq_len]
            edge_index     : [2, E]
            edge_type      : [E]
            temporal_pos   : [N]  temporal position index per node
        Returns:
            h : [N, hidden_dim]
        """
        if self.include_bert and token_ids is not None:
            x = self._run_bert(token_ids, attention_mask)

        h = self.input_proj(x)
        if temporal_pos is not None:
            clamped = temporal_pos.clamp(0, self.temporal_pos_embedding.size(0) - 1)
            h = h + self.temporal_pos_embedding[clamped]
        for layer in self.gt_layers:
            h = layer(h, edge_index, edge_type)
        return h

    def encode_pyg(self, pyg_data) -> torch.Tensor:
        """
        Unpack a PyG Data object and run the encoder — single entry-point
        used by both Phase 1 and Phase 2 models so unpacking logic is
        not duplicated across model files.

        Args:
            pyg_data : PyG Data with edge_index, edge_type, temporal_pos,
                       and either (token_ids + attention_mask) or x.
        Returns:
            h : [N, hidden_dim]
        """
        dev = next(self.parameters()).device
        kw = {
            "edge_index": pyg_data.edge_index.to(dev),
            "edge_type":  pyg_data.edge_type.to(dev),
            "temporal_pos": (pyg_data.temporal_pos.to(dev)
                             if hasattr(pyg_data, "temporal_pos")
                             and pyg_data.temporal_pos is not None else None),
        }
        if self.include_bert and hasattr(pyg_data, "token_ids"):
            kw["token_ids"]      = pyg_data.token_ids.to(dev)
            kw["attention_mask"] = pyg_data.attention_mask.to(dev)
        else:
            kw["x"] = pyg_data.x.to(dev)
        return self.forward(**kw)