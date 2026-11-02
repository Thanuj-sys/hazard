"""Evaluate independent test cases against actual saved model predictions."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmark import dump_json, evaluate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/metrics_report.json')
    args = parser.parse_args(argv)
    predictions = args.predictions or args.dataset / 'predictions/predictions.jsonl'
    report = evaluate_dataset(args.dataset, predictions)
    dump_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
