"""Lightweight shared spatial decoder with hazard-specific binary logit heads."""
import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HazardSegmentationHead(nn.Module):
    """Decode fused spatial features into independent binary hazard masks."""

    hazards = ("flood", "landslide", "wildfire")

    def __init__(
        self,
        embed_dim: int,
        decoder_channels: int = 64,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or decoder_channels <= 0:
            raise ValueError("Segmentation channel dimensions must be positive.")
        if num_classes != 1:
            raise ValueError("Hazard segmentation currently supports one binary logit per head.")

        norm_groups = min(8, decoder_channels)
        while decoder_channels % norm_groups:
            norm_groups -= 1

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, decoder_channels),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            hazard: nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
            for hazard in self.hazards
        })

    def forward(
        self,
        spatial_features: torch.Tensor,
        output_size: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        if spatial_features.ndim != 4:
            raise ValueError("Spatial features must have shape [batch, channels, height, width].")
        if len(output_size) != 2 or min(int(value) for value in output_size) <= 0:
            raise ValueError("Segmentation output size must contain two positive dimensions.")

        size: Tuple[int, int] = (int(output_size[0]), int(output_size[1]))
        features = F.interpolate(
            spatial_features,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder(features)
        return {hazard: head(decoded) for hazard, head in self.heads.items()}
