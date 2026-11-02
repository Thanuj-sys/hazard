"""Checks for honest expanded-validation cohorts; fixtures are not benchmark data."""
import pytest
from scripts.compare_transfer_evaluations import compare


def record(case_id, signature):
    return {"case_id": case_id, "assets": {"image": {"sha256": signature}, "mask": {"sha256": "mask"}}}


def prediction(case_id):
    return {"case_id": case_id, "true_hazard": "flood", "predicted_hazard": "wildfire",
            "hazard_probabilities": {"flood": .2, "wildfire": .6, "landslide": .2}}


def report(rows, checkpoint="fixed"):
    return {"metadata": {"checkpoint_sha256": checkpoint, "frames": 4, "input_size": 64, "per_sample": rows}}


def test_new_cases_exclude_old_ids_and_identical_images():
    old = record("old", "same-image")
    duplicate = record("renamed", "same-image")
    new = record("new", "new-image")
    result = compare(report([prediction("old")]), report([prediction("old"), prediction("renamed"), prediction("new")]),
                     {"records": [old]}, {"records": [old, duplicate, new]})
    assert result["expanded"]["sample_count"] == 3
    assert result["new_cases_only"]["sample_count"] == 1
    assert result["new_cases_only"]["per_case"][0]["case_id"] == "new"
    assert result["repeated_baseline_evaluated_cases"] == 1
    assert result["model_changed"] is False


def test_comparison_rejects_changed_checkpoint_or_protocol():
    with pytest.raises(ValueError, match="Checkpoint changed"):
        compare(report([]), report([], "different"), {"records": []}, {"records": []})
    changed = report([]); changed["metadata"]["frames"] = 1
    with pytest.raises(ValueError, match="Input protocol changed"):
        compare(report([]), changed, {"records": []}, {"records": []})


def test_comparison_rejects_duplicate_scored_cases():
    row = prediction("case")
    with pytest.raises(ValueError, match="duplicate case IDs"):
        compare(report([]), report([row, row]), {"records": []}, {"records": [record("case", "sha")]})
