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
        

        # Create GT Mask
        gt_mask = torch.zeros(num_commits, dtype=torch.bool, device=scores.device)
        gt_mask[ground_truth_positions] = True
        neg_mask = ~gt_mask

        # ---------- Soft Targets ----------
        # GT positions get (1 - smoothing), others get smoothing / (num_commits - num_gt)
        soft_targets = torch.full_like(scores, self.smoothing / max(num_commits - num_gt, 1))
        
        soft_targets[gt_mask] = (1.0 - self.smoothing) / num_gt
    
        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum()
        
        # Compute cross-entropy with soft targets
        log_probs = F.log_softmax(scores / self.temperature, dim=0)
        ce_loss = -torch.sum(soft_targets * log_probs)
        
        
        # Margin Loss
        gt_scores = scores[gt_mask]
        neg_scores = scores[neg_mask]

        if gt_scores.numel() > 0 and neg_scores.numel() > 0:
            diffs = gt_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            # Margin loss
            margin_loss = F.relu(self.margin - diffs).mean()
        else:
            margin_loss = torch.tensor(0.0, device=scores.device)
        
        return ce_loss + 0.5 * margin_loss
