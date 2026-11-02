"""Numerically stable binary segmentation losses with optional pixel validity."""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySegmentationLoss(nn.Module):
    """Combine masked BCE-with-logits and soft Dice losses."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight == 0:
            raise ValueError("Segmentation loss weights must be non-negative and not both zero.")
        if smooth <= 0:
            raise ValueError("Dice smoothing must be positive.")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    @staticmethod
    def _with_channel(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor.unsqueeze(1)
        if tensor.ndim == 4 and tensor.shape[1] == 1:
            return tensor
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W].")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self._with_channel(logits, "Logits")
        target = self._with_channel(target, "Target").to(device=logits.device)
        if logits.shape != target.shape:
            raise ValueError(f"Logit shape {tuple(logits.shape)} does not match target shape {tuple(target.shape)}.")

        target_values = target.to(dtype=logits.dtype)
        valid = torch.isfinite(target_values)
        if validity_mask is not None:
            valid = valid & self._with_channel(validity_mask, "Validity mask").to(
                device=logits.device, dtype=torch.bool
            )
        if valid.shape != logits.shape:
            raise ValueError("Validity mask dimensions do not match logits.")
        if not valid.any():
            raise ValueError("Segmentation batch contains no valid target pixels.")
        invalid_labels = valid & (target_values.ne(0) & target_values.ne(1))
        if invalid_labels.any():
            values = torch.unique(target_values[invalid_labels]).detach().cpu().tolist()
            raise ValueError(
                f"Valid binary target pixels must be 0 or 1; found {values}. "
                "Mark nodata pixels invalid before calculating loss."
            )
        if not torch.isfinite(logits[valid]).all():
            raise ValueError("Segmentation logits contain non-finite values on valid pixels.")

        valid_float = valid.to(dtype=logits.dtype)
        safe_target = torch.where(valid, target_values, torch.zeros_like(target_values))
        bce_pixels = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
        bce_loss = (bce_pixels * valid_float).sum() / valid_float.sum()

        probabilities = torch.sigmoid(logits) * valid_float
        masked_target = safe_target * valid_float
        reduce_dims = tuple(range(1, logits.ndim))
        intersection = (probabilities * masked_target).sum(dim=reduce_dims)
        denominator = probabilities.sum(dim=reduce_dims) + masked_target.sum(dim=reduce_dims)
        valid_samples = valid_float.sum(dim=reduce_dims) > 0
        dice_per_sample = 1.0 - (
            (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        )
        dice_loss = dice_per_sample[valid_samples].mean()
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return {"total": total_loss, "bce": bce_loss, "dice": dice_loss}
