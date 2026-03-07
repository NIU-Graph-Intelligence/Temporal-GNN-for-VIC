"""
training/loss.py
────────────────
Loss functions for both training phases.

  PairwiseRankingLoss        — BCE pairwise loss (Phase 1).
  LabelSmoothingRankingLoss  — Listwise CE + pairwise margin (Phase 2).
"""

import torch
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


class LabelSmoothingRankingLoss(nn.Module):
    """
    Listwise ranking loss with label smoothing and a pairwise margin term.

    Soft targets replace hard one-hot targets:
      - Ground-truth positions receive   (1 − ε) / num_gt
      - All other positions receive      ε / max(C − num_gt, 1)
      - Targets are L1-normalised to sum to 1

    The cross-entropy term encourages correct ranking of the full list.
    The margin term additionally penalises cases where a non-GT commit
    scores higher than a GT commit by more than ``margin``.

    Parameters
    ----------
    temperature : float — softmax temperature (higher → softer distribution)
    margin      : float — minimum required score gap for the margin term
    smoothing   : float — label smoothing coefficient ε ∈ [0, 1)
    """

    def __init__(self, temperature: float = 1.0, margin: float = 1.0,
                 smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.smoothing = smoothing

    def forward(self, scores: torch.Tensor,
                ground_truth_positions) -> torch.Tensor:
        """
        Args:
            scores                 : [C] one scalar score per commit
            ground_truth_positions : list[int] indices of inducing commits
        """
        if len(scores) < 2:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        C = len(scores)
        num_gt = len(ground_truth_positions)
        if num_gt == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        # Soft targets 
        gt_t = torch.tensor(
            [p for p in ground_truth_positions if 0 <= p < C],
            device=scores.device, dtype=torch.long,
        )
        soft = torch.full_like(scores, self.smoothing / max(C - num_gt, 1))
        if gt_t.numel() > 0:
            soft[gt_t] = (1.0 - self.smoothing) / num_gt
        soft = soft / soft.sum()

        # Cross-entropy 
        ce = -torch.sum(soft * F.log_softmax(scores / self.temperature, dim=0))

        # Pairwise margin (vectorised) 
        non_gt_mask = torch.ones(C, dtype=torch.bool, device=scores.device)
        non_gt_mask[gt_t] = False

        if gt_t.numel() > 0 and non_gt_mask.any():
            gt_scores = scores[gt_t]                     # [G]
            non_gt_scores = scores[non_gt_mask]           # [C-G]
            # Broadcasting: [G, 1] - [1, C-G] → [G, C-G]
            margin_loss = F.relu(
                self.margin - (gt_scores.unsqueeze(1) - non_gt_scores.unsqueeze(0))
            ).mean()
        else:
            margin_loss = torch.tensor(0.0, device=scores.device)

        return ce + 0.5 * margin_loss


