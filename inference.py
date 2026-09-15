import torch
import argparse
import os
from typing import Dict, Any

from models.multihazard_vlm import MultiHazardVLM


def parse_args():
    parser = argparse.ArgumentParser(description="MultiHazard-VLM Inference")
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weights')
    return parser.parse_args()


def run_inference(model: MultiHazardVLM, sample_data: Dict[str, Any], aoi_bounds: torch.Tensor) -> Dict[str, Any]:
    """Run end-to-end inference using the model's actual API contract."""
    device = next(model.parameters()).device

    images = sample_data['images'].to(device)
    if images.dim() == 4:
        images = images.unsqueeze(0)

    quality_flags = sample_data['quality_flags'].to(device)
    if quality_flags.dim() == 1:
        quality_flags = quality_flags.unsqueeze(0)

    h3_indices = {
        4: torch.tensor([[123456]], device=device, dtype=torch.long),
        6: torch.tensor([[234567]], device=device, dtype=torch.long),
        8: torch.tensor([[345678]], device=device, dtype=torch.long),
    }

    geo_attributes = {
        'elevation': sample_data['geo_attributes']['elevation'].to(device).view(1, 1),
        'slope': sample_data['geo_attributes']['slope'].to(device).view(1, 1),
        'ndvi': sample_data['geo_attributes']['ndvi'].to(device).view(1, 1),
        'aspect': sample_data['geo_attributes']['aspect'].to(device).view(1, 1),
        'lithology_class': sample_data['geo_attributes']['lithology_class'].to(device).view(1),
    }

    report_dict = model.generate_report(
        images=images,
        quality_flags=quality_flags,
        h3_indices=h3_indices,
        geo_attributes=geo_attributes,
        aoi_bounds=aoi_bounds,
    )

    hazard_classes = ["No Hazard", "Flood", "Wildfire", "Landslide"]
    pred_class_idx = int(report_dict['hazard_class'][0].item())
    hazard_type = hazard_classes[min(pred_class_idx, len(hazard_classes) - 1)]

    raw_coords = report_dict['epicenter_coords_geo'][0].detach().cpu().tolist() if report_dict['epicenter_coords_geo'] is not None else None
    raw_radius = float(report_dict['impact_radius_km'][0].item())
    clamped_radius = max(0.5, min(raw_radius, 150.0))

    api_response = {
        "status": "success",
        "data": {
            "hazard_type": hazard_type,
            "epicenter_lat": float(raw_coords[0]) if raw_coords is not None else None,
            "epicenter_lon": float(raw_coords[1]) if raw_coords is not None else None,
            "impact_radius_km": clamped_radius,
            "h3_index": "8a2a1072b59ffff",
            "analysis_report": "Generated report tokens available in model output."
        }
    }

    return api_response


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiHazardVLM(config_path=args.config if os.path.exists(args.config) else None)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    sample_data = {
        'images': torch.randn(5, 10, 16, 16),
        'quality_flags': torch.ones(5, dtype=torch.bool),
        'geo_attributes': {
            'elevation': torch.tensor([100.0]),
            'slope': torch.tensor([15.0]),
            'ndvi': torch.tensor([0.45]),
            'aspect': torch.tensor([90.0]),
            'lithology_class': torch.tensor([3]),
        },
    }

    aoi_bounds = torch.tensor([[10.0, 20.0, 15.0, 25.0]], device=device)

    result = run_inference(model, sample_data, aoi_bounds)

    print("\n--- MultiHazard-VLM Inference Report ---")
    print(f"Hazard Type : {result['data']['hazard_type']}")
    print(f"Epicenter   : ({result['data']['epicenter_lat']:.4f}, {result['data']['epicenter_lon']:.4f})")
    print(f"Radius (km) : {result['data']['impact_radius_km']:.2f}")
    print(f"H3 Index    : {result['data']['h3_index']}")
    print(f"Summary     : {result['data']['analysis_report']}")
    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
