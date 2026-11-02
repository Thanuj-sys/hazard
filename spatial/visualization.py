"""Lightweight Pillow visualization for an image and predicted footprint."""
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image


def _image_array(image: Any, bands: Sequence[int]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    values = np.asarray(image)
    if values.ndim == 2:
        values = np.repeat(values[..., np.newaxis], 3, axis=-1)
    elif values.ndim == 3 and values.shape[-1] in (1, 3, 4):
        if values.shape[-1] == 1:
            values = np.repeat(values, 3, axis=-1)
        else:
            values = values[..., list(bands)]
    elif values.ndim == 3 and values.shape[0] <= 16:
        if max(bands) >= values.shape[0]:
            raise ValueError("Requested visualization band is not present in the image.")
        values = values[list(bands)].transpose(1, 2, 0)
    elif values.ndim == 3 and values.shape[-1] >= 3:
        values = values[..., list(bands)]
    else:
        raise ValueError("Image must be 2D or a 3D band-first/band-last array.")

    channels = []
    for channel in range(3):
        plane = values[..., channel].astype(np.float32)
        finite = np.isfinite(plane)
        if not finite.any():
            channels.append(np.zeros(plane.shape, dtype=np.uint8))
            continue
        low, high = np.percentile(plane[finite], (2, 98))
        if high <= low:
            high = low + 1.0
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        scaled[~finite] = 0.0
        channels.append((scaled * 255.0).astype(np.uint8))
    return np.stack(channels, axis=-1)


def save_footprint_overlay(
    image: Any,
    mask: Any,
    output_path: Any,
    *,
    bands: Sequence[int] = (2, 1, 0),
    validity_mask: Optional[Any] = None,
    alpha: float = 0.45,
) -> str:
    """Save an RGB preview with the predicted binary footprint highlighted in red."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    prediction = np.asarray(mask)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("Footprint mask must be a 2D raster.")
    valid = np.isfinite(prediction)
    if validity_mask is not None:
        if isinstance(validity_mask, torch.Tensor):
            validity_mask = validity_mask.detach().cpu().numpy()
        validity_mask = np.asarray(validity_mask, dtype=bool)
        if validity_mask.shape != prediction.shape:
            raise ValueError("Overlay validity-mask dimensions do not match the footprint.")
        valid &= validity_mask
    footprint = (prediction == 1) & valid
    rgb = _image_array(image, bands)
    if rgb.shape[:2] != prediction.shape:
        raise ValueError("Image and footprint dimensions must match for visualization.")
    overlay = rgb.astype(np.float32)
    red = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    overlay[footprint] = (1.0 - alpha) * overlay[footprint] + alpha * red
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)
    return str(path)
