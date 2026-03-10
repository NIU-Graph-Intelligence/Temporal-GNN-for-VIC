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

    def __init__(self, temperature: float = 1.0,
                 smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, scores: torch.Tensor,
                ground_truth_positions) -> torch.Tensor:
        """
        Args:
            scores                 : [C] one scalar score per commit
            ground_truth_positions : list[int] indices of inducing commits
        """

        C     = scores.size(0)
        n_gt  = len(ground_truth_positions)

        # Temperature scaling
        scores = scores / self.temperature

        # Label smoothing
        smooth_val          = self.smoothing / max(C - n_gt, 1)
        targets             = torch.full((C,), smooth_val, device=scores.device)
        targets[gt_positions] = 1.0 - self.smoothing + smooth_val

        # ListNet ranking loss
        log_probs = F.log_softmax(scores, dim=0)
        return -(targets * log_probs).sum()
        


