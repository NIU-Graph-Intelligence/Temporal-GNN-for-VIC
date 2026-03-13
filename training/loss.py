"""
training/loss.py
────────────────
Loss functions for both training phases.

  PairwiseRankingLoss        — BCE pairwise loss (Phase 1).
  LabelSmoothingRankingLoss  — Listwise CE + pairwise margin (Phase 2).
"""

import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss using BCELoss.

    pred_probs  : P(x > y) for each pair (model output)
    target_probs:
        1.0 — x is rootcause,     y is not
        0.0 — x is not rootcause, y is
        0.5 — both have the same label
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred_probs: torch.Tensor,
                target_probs: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred_probs, target_probs)


# class LabelSmoothingRankingLoss(nn.Module):
#     """
#     ListNet-style ranking loss with label smoothing.

#     Soft targets replace hard one-hot targets:
#       - Ground-truth positions receive   (1 − ε) / num_gt
#       - All other positions receive      ε / max(C − num_gt, 1)

#     Parameters
#     ----------
#     temperature : float — softmax temperature (higher → softer distribution)
#     smoothing   : float — label smoothing coefficient ε ∈ [0, 1)
#     """

#     def __init__(self, temperature: float = 1.0,
#                  smoothing: float = 0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.smoothing = smoothing

#     def forward(self, scores: torch.Tensor,
#                 ground_truth_positions) -> torch.Tensor:
#         """
#         Args:
#             scores                 : [C] one scalar score per commit
#             ground_truth_positions : list[int] indices of inducing commits
#         """

#         C     = scores.size(0)
#         n_gt  = len(ground_truth_positions)

#         # Temperature scaling
#         scores = scores / self.temperature

#         # Label smoothing
#         smooth_val          = self.smoothing / max(C - n_gt, 1)
#         targets             = torch.full((C,), smooth_val, device=scores.device)
#         targets[ground_truth_positions] = 1.0 - self.smoothing + smooth_val

#         # ListNet ranking loss
#         log_probs = F.log_softmax(scores, dim=0)
#         return -(targets * log_probs).sum()
        


# ---------- Label Smoothing Loss ----------
class LabelSmoothingRankingLoss(nn.Module):
    """
    Listwise ranking loss with label smoothing for regularization.
    
    Instead of hard targets (1 for ground truth, 0 for others),
    uses soft targets: (1 - smoothing) for GT, smoothing/(n-1) for others.
    """
    def __init__(self, temperature: float = 1.0, margin: float = 1.0, smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.smoothing = smoothing
    
    def forward(self, scores: torch.Tensor, ground_truth_positions: List[int]) -> torch.Tensor:
        """
        Compute loss with label smoothing.
        
        Args:
            scores: [num_commits] predicted scores
            ground_truth_positions: List of ground truth inducer indices
        
        Returns:
            Scalar loss
        """
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        num_commits = len(scores)
        num_gt = len(ground_truth_positions)
        
        if num_gt == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # Create soft targets with label smoothing
        # GT positions get (1 - smoothing), others get smoothing / (num_commits - num_gt)
        soft_targets = torch.full_like(scores, self.smoothing / max(num_commits - num_gt, 1))
        for gt_pos in ground_truth_positions:
            if 0 <= gt_pos < num_commits:
                soft_targets[gt_pos] = (1.0 - self.smoothing) / num_gt
        
        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum()
        
        # Compute cross-entropy with soft targets
        log_probs = F.log_softmax(scores / self.temperature, dim=0)
        ce_loss = -torch.sum(soft_targets * log_probs)
        
        # Add margin-based ranking loss for hard negatives
        margin_loss = torch.tensor(0.0, device=scores.device)
        for gt_pos in ground_truth_positions:
            if 0 <= gt_pos < num_commits:
                gt_score = scores[gt_pos]
                # Push GT score above all non-GT scores by margin
                for i in range(num_commits):
                    if i not in ground_truth_positions:
                        margin_loss += F.relu(self.margin - (gt_score - scores[i]))
        
        if num_gt > 0 and num_commits > num_gt:
            margin_loss = margin_loss / (num_gt * (num_commits - num_gt))
        
        return ce_loss + 0.5 * margin_loss
