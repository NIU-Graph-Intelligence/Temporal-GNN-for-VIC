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

# Phase 1
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




# Phase 1 without Pairs
class FocalListWiseRankingLoss(nn.Module):
    """
    Uses softmax over list
    Applies focal weighting to both positives and negatives
    Supports multiple GT items
    """

    def __init__(self, focal_gamma: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.eps = eps

    def forward(self, scores: torch.Tensor, gt_indices: List[int]) -> torch.Tensor:

        N = scores.size(0)

        if len(gt_indices) == 0:
            raise ValueError("At least one GT index required.")

        # Create GT mask
        gt_mask = torch.zeros(N, dtype=torch.float32, device = scores.device)
        gt_mask[gt_indices] = 1.0 

        # Softmax probabilities
        probs = F.softmax(scores, dim=0)

        # Clamp for numerical stability
        probs = probs.clamp(min=self.eps, max=1.0 - self.eps)

        # Positive (GT) focal loss
        pos_loss = -((1.0 - probs) ** self.focal_gamma) * torch.log(probs)

        # Negative (non-GT) focal loss
        neg_loss = -(probs ** self.focal_gamma) * torch.log(1.0 - probs)

        # Combine
        loss = gt_mask * pos_loss + (1.0 - gt_mask) * neg_loss

        # Normalize for stability
        return loss.mean()
        

# Phase 2 Focal + Label Smoothing + Margin
class LabelSmoothingRankingLoss(nn.Module):
    """
    Listwise ranking loss combining:
      - Focal loss: down-weights easy cases, focuses on hard ones
      - Label smoothing: regularization to prevent overconfidence
      - Margin loss: pushes GT scores above non-GT scores
    """
    def __init__(
        self,
        temperature: float = 1.0,
        margin:      float = 1.0,
        smoothing:   float = 0.1,
        focal_gamma: float = 2.0,    
        focal_alpha: float = 1.0, 
    ):
        super().__init__()
        self.temperature = temperature
        self.margin      = margin
        self.smoothing   = smoothing
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(
        self,
        scores: torch.Tensor,
        ground_truth_positions: List[int],
    ) -> torch.Tensor:


        num_commits = len(scores)
        num_gt      = len(ground_truth_positions)


        # Create GT mask
        gt_mask  = torch.zeros(num_commits, dtype=torch.bool, device=scores.device)
        gt_mask[ground_truth_positions] = True
        neg_mask = ~gt_mask

        # Soft targets with label smoothing
        soft_targets = torch.full_like(
            scores, self.smoothing / max(num_commits - num_gt, 1))
        soft_targets[gt_mask] = (1.0 - self.smoothing) / num_gt

        # Normalize to sum to 1
        soft_targets = soft_targets / soft_targets.sum()

        # Focal loss
        # probs[i] = softmax probability assigned to commit i
        probs     = F.softmax(scores / self.temperature, dim=0)
        log_probs = torch.log(probs + 1e-8)

        # p_gt = probability the model assigns to the GT commit(s)
        # averaged over all GT positions
        p_gt = probs[gt_mask].mean()

        # Focal modulating factor: down-weights easy (high p_gt) examples
        focal_weight = self.focal_alpha * (1.0 - p_gt) ** self.focal_gamma

        # Focal cross-entropy with soft targets
        ce_loss = -torch.sum(soft_targets * log_probs)
        focal_loss = focal_weight * ce_loss

        # Margin loss
        gt_scores  = scores[gt_mask]
        neg_scores = scores[neg_mask]

        if gt_scores.numel() > 0 and neg_scores.numel() > 0:
            diffs       = gt_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
            margin_loss = F.relu(self.margin - diffs).mean()
        else:
            margin_loss = torch.tensor(0.0, device=scores.device)

        return focal_loss + 0.5 * margin_loss



