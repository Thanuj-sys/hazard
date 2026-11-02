"""Check readiness or reconstruct one historical event without live lookback."""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reconstruction import check_event_readiness, reconstruct_event


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Historical disaster event reconstruction")
    parser.add_argument("--event", required=True, help="Event ID from data/events/event_manifest.csv")
    parser.add_argument("--checkpoint", help="Segmentation model checkpoint; required only when inputs are ready")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Model configuration")
    parser.add_argument("--manifest", help="Override the event manifest path")
    parser.add_argument("--data-root", help="Root for manifest-referenced local input files")
    parser.add_argument("--results-root", help="Output root; defaults to results/events")
    parser.add_argument("--check-only", action="store_true", help="Check inputs and ground truth without model loading")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        readiness = check_event_readiness(
            args.event,
            manifest_path=args.manifest,
            data_root=args.data_root,
        )
        print(json.dumps(readiness, indent=2))
        if readiness["status"] == "not_ready":
            print("EVENT NOT READY")
            return 2
        return 0

    result = reconstruct_event(
        args.event,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        manifest_path=args.manifest,
        data_root=args.data_root,
        results_root=args.results_root,
    )
    print(json.dumps(result["summary"], indent=2))
    if result["readiness"]["status"] == "not_ready":
        print("EVENT NOT READY")
        for missing in result["readiness"]["missing_inputs"]:
            print(f"- {missing}")
        return 2
    print("Reconstruction artifacts:")
    for name, path in result["artifacts"].items():
        print(f"- {name}: {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
