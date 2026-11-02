"""Evaluate reviewed independent test cases; --local-smoke runs generated diagnostics separately."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from evaluation.metrics import MultiHazardEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



HAZARDS = ("flood", "wildfire", "landslide")
HAZARD_LABELS = {hazard: index for index, hazard in enumerate(HAZARDS)}


def _classification_metrics(
    predictions: list[int],
    targets: list[int],
    evaluator: MultiHazardEvaluator,
) -> dict[str, float]:
    import torch
    return evaluator.evaluate(torch.tensor(predictions), torch.tensor(targets))


def _segmentation_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, float]:
    """Calculate pixel metrics only when trained segmentation outputs exist."""
    import torch
    total_intersection = 0.0
    total_union = 0.0
    total_predicted = 0.0
    total_target = 0.0
    total_correct = 0.0
    total_valid = 0.0

    for index, hazard in enumerate(batch["hazards"]):
        logits = outputs[hazard][index, 0]
        target = batch["target_masks"][index, 0] > 0.5
        valid = batch["validity_masks"][index, 0].bool()
        predicted = torch.sigmoid(logits) >= 0.5
        predicted = predicted[valid]
        target = target[valid]
        intersection = torch.logical_and(predicted, target).sum().item()
        union = torch.logical_or(predicted, target).sum().item()
        total_intersection += intersection
        total_union += union
        total_predicted += predicted.sum().item()
        total_target += target.sum().item()
        total_correct += torch.eq(predicted, target).sum().item()
        total_valid += valid.sum().item()

    precision = total_intersection / total_predicted if total_predicted else 0.0
    recall = total_intersection / total_target if total_target else 0.0
    dice = (
        2 * total_intersection / (total_predicted + total_target)
        if total_predicted + total_target
        else 0.0
    )
    return {
        "pixel_accuracy": total_correct / total_valid if total_valid else 0.0,
        "iou": total_intersection / total_union if total_union else 0.0,
        "dice": dice,
        "segmentation_precision": precision,
        "segmentation_recall": recall,
    }


def run_evaluation(checkpoint_path: Path, output_path: Path, *, allow_generated_samples=False) -> dict[str, Any]:
    if not allow_generated_samples:
        raise ValueError("Local sample files have no independently verified provenance. Use the benchmark dataset or explicitly enable --local-smoke.")
    import torch
    from api.app import adapt_checkpoint_state
    from data.datasets import CASLandslideDataset, HLSBurnScarDataset, S1S2WaterDataset
    from evaluation.metrics import MultiHazardEvaluator
    from models.multihazard_vlm import MultiHazardVLM
    from train import prepare_reconstruction_batch
    DATASET_TYPES = {"flood": S1S2WaterDataset, "wildfire": HLSBurnScarDataset, "landslide": CASLandslideDataset}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHazardVLM(config_path=str(PROJECT_ROOT / "configs" / "model_config.yaml")).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(adapt_checkpoint_state(checkpoint_state, model), strict=True)
    model.eval()

    samples = []
    for hazard, dataset_type in DATASET_TYPES.items():
        dataset = dataset_type()
        if not len(dataset):
            raise RuntimeError(f"No labeled {hazard} samples were found.")
        sample = dataset[0]
        if sample.get("target_mask") is None:
            raise RuntimeError(f"The {hazard} sample has no target mask.")
        samples.append(sample)

    batch = prepare_reconstruction_batch(samples, model, device)
    with torch.inference_mode():
        outputs = model(**batch["model_inputs"])

    predictions = torch.argmax(outputs["hazard_logits"], dim=-1).cpu().tolist()
    targets = [HAZARD_LABELS[hazard] for hazard in batch["hazards"]]
    evaluator = MultiHazardEvaluator()
    overall = _classification_metrics(predictions, targets, evaluator)

    results_by_hazard: dict[str, dict[str, float]] = {"overall": overall}
    sample_counts: dict[str, int] = {"overall": len(targets)}
    per_sample = []
    for index, hazard in enumerate(batch["hazards"]):
        hazard_metrics = _classification_metrics([predictions[index]], [targets[index]], evaluator)
        results_by_hazard[hazard] = hazard_metrics
        sample_counts[hazard] = sample_counts.get(hazard, 0) + 1
        per_sample.append({
            "hazard": hazard,
            "target_class": hazard,
            "predicted_class": HAZARDS[predictions[index]],
            "confidence": float(outputs["hazard_probs"][index, predictions[index]].cpu()),
            "hazard_probabilities": {name: float(outputs["hazard_probs"][index, j].cpu()) for j, name in enumerate(HAZARDS)},
            "event_id": batch["event_ids"][index],
            "ground_truth_source": "unverified generated sample; not research ground truth",
        })

    segmentation_keys = checkpoint_state.keys() if isinstance(checkpoint_state, dict) else ()
    has_trained_segmentation_head = any(
        key.startswith("segmentation_head.") for key in segmentation_keys
    )
    segmentation_status = "evaluated" if has_trained_segmentation_head else "not_evaluated"
    if has_trained_segmentation_head and outputs.get("segmentation"):
        results_by_hazard["overall"].update(
            _segmentation_metrics(outputs["segmentation"], batch)
        )

    report = {
        "results_by_hazard": results_by_hazard,
        "sample_counts": sample_counts,
        "metadata": {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "evaluation_kind": "generated_sample_smoke_test",
            "benchmark_status": "not_a_historical_event_benchmark",
            "ground_truth_status": "UNVERIFIED generated sample files; all scores are software smoke diagnostics only",
            "checkpoint": str(checkpoint_path.resolve()),
            "device": str(device),
            "datasets": {
                hazard: DATASET_TYPES[hazard].__name__ for hazard in HAZARDS
            },
            "segmentation_status": segmentation_status,
            "segmentation_note": (
                "Skipped because the selected checkpoint has no trained segmentation_head weights."
                if not has_trained_segmentation_head
                else "Segmentation metrics were calculated from the trained segmentation head."
            ),
            "geospatial_regression_status": "not_evaluated",
            "geospatial_regression_note": (
                "No verified event coordinates or radii are present in the local event manifest; "
                "coordinate and radius scores are intentionally omitted."
            ),
            "per_sample": per_sample,
        },
    }
    from evaluation.diagnostics import report_diagnostics
    report["detailed_evaluation"] = report_diagnostics(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-smoke", action="store_true", help="Explicitly run generated local samples as software diagnostics only")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "model_epoch_9.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.local_smoke:
        output = args.output or PROJECT_ROOT / "evaluation" / "smoke_metrics_report.json"
        if output.resolve() == (PROJECT_ROOT / "evaluation" / "metrics_report.json").resolve():
            parser.error("Smoke diagnostics cannot overwrite the benchmark report")
        report = run_evaluation(args.checkpoint, output, allow_generated_samples=True)
    else:
        from evaluation.benchmark import dump_json, evaluate_dataset
        output = args.output or PROJECT_ROOT / "evaluation" / "metrics_report.json"
        predictions = args.predictions or args.dataset / "predictions" / "predictions.jsonl"
        report = evaluate_dataset(args.dataset, predictions)
        dump_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
