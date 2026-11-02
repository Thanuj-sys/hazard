"""Compare a fixed-checkpoint transfer test with additional, non-baseline source cases."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.diagnostics import classification_diagnostics
LABELS = ["flood", "wildfire", "landslide"]


def diagnostic(rows):
    result = classification_diagnostics(rows, LABELS)
    for metric in ("precision", "recall", "f1"):
        result["summary"]["macro_" + metric] = (
            sum((row[metric] or 0.) for row in result["per_class"]) / len(LABELS) if rows else None
        )
    return result


def compare(baseline, expanded, baseline_manifest, expanded_manifest):
    if baseline["metadata"]["checkpoint_sha256"] != expanded["metadata"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed: this comparison must hold the model fixed")
    for key in ("input_size", "frames"):
        if baseline["metadata"][key] != expanded["metadata"][key]:
            raise ValueError(f"Input protocol changed: {key}")
    old_ids = {row["case_id"] for row in baseline_manifest["records"]}
    def signature(row):
        return tuple(sorted(asset["sha256"] for key, asset in row["assets"].items() if key != "mask"))
    old_images = {signature(row) for row in baseline_manifest["records"]}
    records = {row["case_id"]: row for row in expanded_manifest["records"]}
    all_rows = expanded["metadata"]["per_sample"]
    ids = [row["case_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Expanded evaluation contains duplicate case IDs")
    new_rows = [row for row in all_rows if row["case_id"] not in old_ids
                and signature(records[row["case_id"]]) not in old_images]
    old_rows = baseline["metadata"]["per_sample"]
    old_by_id = {row["case_id"]: row for row in old_rows}
    repeated = [row for row in all_rows if row["case_id"] in old_by_id]
    changed = [row["case_id"] for row in repeated
               if row["predicted_hazard"] != old_by_id[row["case_id"]]["predicted_hazard"]]
    flood_groups = {}
    for row in all_rows:
        if row["true_hazard"] == "flood":
            group = row["case_id"].removeprefix("flood_").rsplit("_", 1)[0]
            flood_groups.setdefault(group, []).append(row)
    return {"checkpoint_sha256": baseline["metadata"]["checkpoint_sha256"],
            "model_changed": False, "baseline": diagnostic(old_rows),
            "expanded": diagnostic(all_rows), "new_cases_only": diagnostic(new_rows),
            "repeated_baseline_evaluated_cases": len(repeated),
            "changed_argmax_on_repeated_cases": changed,
            "flood_source_prefix_diagnostics": {key: diagnostic(rows) for key, rows in flood_groups.items()},
            "new_case_definition": "Not in the baseline acquisition manifest by case ID or exact image combination SHA256; empty baseline masks also count as previously acquired.",
            "limitations": ["Positive-only, source-confounded transfer diagnostics; no operational validation claim.",
                "Newly added cases have not been used to tune or retrain this checkpoint in this run; historical training lineage remains unknown.",
                "More evaluated patches do not remove source confounding, temporal/context ablation, or event clustering.",
                "Source-prefix slices are descriptive; they do not establish event-independent training/test separation.",
                "Wilson intervals assume independent patches and can be optimistic under geographic/event clustering."]}


def percent(value):
    return "Not measured" if value is None else f"{value * 100:.2f}%"


def markdown(report):
    lines = ["# Expanded fixed-checkpoint transfer evaluation", "",
             "The model was not retrained. Changes in aggregate metrics reflect a different evaluated sample, not a model improvement.", "",
             "| Cohort | Eligible patches | Accuracy | Macro recall | Macro F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in [("baseline", "Original baseline"), ("expanded", "Expanded total"), ("new_cases_only", "New source cases only")]:
        result = report[key]; summary = result["summary"]
        lines.append(f"| {label} | {result['sample_count']} | {percent(summary['accuracy'])} | {percent(summary['macro_recall'])} | {percent(summary['macro_f1'])} |")
    for key, label in [("expanded", "Expanded total"), ("new_cases_only", "New cases only")]:
        result = report[key]
        lines += ["", f"## {label}: class-wise results", "",
                  "| Class | Cases | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in result["per_class"]:
            precision = "Undefined (no predictions)" if row["precision"] is None else percent(row["precision"])
            lines.append(f"| {row['class']} | {row['support']} | {precision} | {percent(row['recall'])} | {percent(row['f1'])} |")
        lines += ["", "Confusion matrix: rows true labels; columns flood / wildfire / landslide.", "", "```text"]
        lines += [f"{label:10s} " + " ".join(f"{count:4d}" for count in row) for label, row in zip(LABELS, result["confusion_matrix"])]
        lines += ["```"]
    lines += ["", "## Interpretation limits", ""] + ["- " + text for text in report["limitations"]]
    lines += ["", "Figures: `figures/overall/index.html`. Raw scored cases and exclusions: `metrics_report.json`. New-case-only diagnostics: `comparison.json`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "evaluation/real_transfer/metrics_report.json")
    parser.add_argument("--expanded", type=Path, default=ROOT / "evaluation/real_transfer_expanded/metrics_report.json")
    parser.add_argument("--baseline-manifest", type=Path, default=ROOT / "dataset/real_holdout/manifest.json")
    parser.add_argument("--expanded-manifest", type=Path, default=ROOT / "dataset/real_holdout_expanded/manifest.json")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = compare(read(args.baseline), read(args.expanded), read(args.baseline_manifest), read(args.expanded_manifest))
    output = args.expanded.parent
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(markdown(result), encoding="utf-8")
    for key in ("baseline", "expanded", "new_cases_only"):
        print(key, result[key]["sample_count"], result[key]["summary"])


if __name__ == "__main__":
    main()
