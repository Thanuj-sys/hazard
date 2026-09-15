import glob
import json
import os
from typing import Dict, Any, Optional

import torch

from models.multihazard_vlm import MultiHazardVLM


def resolve_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    return checkpoints[-1] if checkpoints else None


def load_model_with_fallback() -> MultiHazardVLM:
    config_path = os.path.join("configs", "model_config.yaml")
    model = MultiHazardVLM(config_path=config_path if os.path.exists(config_path) else None)

    checkpoint_path = resolve_latest_checkpoint()
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        try:
            model.load_state_dict(checkpoint_state, strict=True)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as exc:
            print(f"Checkpoint mismatch: {exc}")
            print("Using current initialized model weights instead.")
            current_state = model.state_dict()
            filtered = {}
            for key, value in checkpoint_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    filtered[key] = value
            if filtered:
                model.load_state_dict(filtered, strict=False)
    else:
        print("No checkpoint found. Using fresh initialized model weights.")

    model.eval()
    return model


def build_demo_sample() -> Dict[str, Any]:
    sample_images = torch.randn(1, 5, 10, 64, 64)
    sample_quality = torch.ones(1, 5, dtype=torch.bool)
    sample_h3 = {
        4: torch.randint(0, 1000, (1, 10)),
        6: torch.randint(0, 1000, (1, 20)),
        8: torch.randint(0, 1000, (1, 30)),
    }
    sample_geo = {
        'elevation': torch.randn(1, 1),
        'slope': torch.randn(1, 1),
        'ndvi': torch.rand(1, 1),
        'aspect': torch.randn(1, 1),
        'lithology_class': torch.randint(0, 10, (1,)),
    }
    aoi_bounds = torch.tensor([[10.0, 70.0, 15.0, 75.0]])

    return {
        'images': sample_images,
        'quality_flags': sample_quality,
        'h3_indices': sample_h3,
        'geo_attributes': sample_geo,
        'aoi_bounds': aoi_bounds,
    }


def run_demo() -> Dict[str, Any]:
    model = load_model_with_fallback()
    sample = build_demo_sample()

    with torch.no_grad():
        outputs = model(
            images=sample['images'],
            quality_flags=sample['quality_flags'],
            h3_indices=sample['h3_indices'],
            geo_attributes=sample['geo_attributes'],
            aoi_bounds=sample['aoi_bounds'],
        )

    class_names = ['Flood', 'Wildfire', 'Landslide']
    hazard_idx = int(torch.argmax(outputs['hazard_logits'][0]).item())
    hazard_name = class_names[min(hazard_idx, len(class_names) - 1)]
    probabilities = torch.softmax(outputs['hazard_logits'][0], dim=-1).tolist()
    coords = outputs['coords_geo'][0].tolist()
    radius = float(torch.clamp(outputs['radius_km'][0, 0], min=0.5, max=150.0).item())

    payload = {
        'status': 'success',
        'hazard_type': hazard_name,
        'hazard_probabilities': {
            'flood': round(float(probabilities[0]), 5),
            'wildfire': round(float(probabilities[1]), 5),
            'landslide': round(float(probabilities[2]), 5),
        },
        'epicenter': {
            'lat': round(float(coords[0]), 5),
            'lon': round(float(coords[1]), 5),
        },
        'impact_radius_km': round(radius, 3),
        'report': (
            f"{hazard_name.upper()} ALERT: The model predicts a {hazard_name.lower()} hazard in the target AOI. "
            "This assessment is generated from the fused geospatial and visual representation of the scene."
        )
    }
    return payload


if __name__ == '__main__':
    result = run_demo()
    print(json.dumps(result, indent=2))
