"""Contract/integrity tests. All fixture pixels are synthetic software tests only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from evaluation.benchmark import (METRICS, digest, dump_json, evaluate_dataset,
    new_case, new_prediction, raster_arrays, score_case, validate_dataset, write_jsonl)
from scripts.build_evaluation_dataset import init, save


@pytest.fixture
def benchmark(tmp_path):
    init(tmp_path)
    case = new_case('fixture_flood', 'flood')
    case.update(split='test', event_id='fixture_event', region_id='fixture_region',
                annotation_status='reviewed', training_exposure='not_used',
                review={'annotator_id': 'fixture_author', 'reviewer_id': 'fixture_reviewer',
                        'protocol': 'software test only', 'reviewed_at': '2026-01-01T00:00:00Z'})
    source = {'source_id': 'fixture', 'url': 'https://example.test/software-fixture'}
    dump_json(tmp_path / 'metadata/sources.json', {'sources': [source]})
    transform = from_origin(75, 15, .01, .01)
    profile = dict(driver='GTiff', width=2, height=2, count=1, dtype='uint8',
                   crs='EPSG:4326', transform=transform)
    target = np.array([[1, 0], [0, 1]], dtype='uint8')
    for field, name, data in [('image_path', 'image', target + 2),
                               ('mask_path', 'mask', target),
                               ('validity_mask_path', 'valid', np.ones((2, 2), dtype='uint8'))]:
        path = tmp_path / 'masks' / f'{name}.tif'
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)
        case[field] = path.relative_to(tmp_path).as_posix()
        case['assets'][field] = {'sha256': digest(path)}
    case.update(true_hazard='flood', hazard_present=True, true_latitude=14.99,
                true_longitude=75.01, true_impact_radius_km=2.,
                verified_claims=['flood_verified'],
                geo_attributes={'slope': {'value': 0., 'absolute_tolerance': .5,
                    'unit': 'degrees', 'source_id': 'fixture', 'reference': 'software fixture'}},
                required_interventions=['evacuate', 'rescue'],
                standard_checklist=['evacuate', 'rescue'],
                priority_order=['evacuate', 'rescue'], unsafe_actions=['enter_floodwater'],
                mask_metadata={'crs': 'EPSG:4326', 'width': 2, 'height': 2,
                    'transform': list(transform)[:6], 'resolution': [.01, .01], 'nodata': None})
    for field, value in case.items():
        if field in case['missing_reasons'] and value is not None:
            case['missing_reasons'].pop(field)
            case['evidence'][field] = {'source_id': 'fixture', 'reference': 'software fixture',
                                       'method': 'published_label', 'status': 'verified'}
    case['evidence']['priority_order'].update(method='expert_annotation', annotator_id='a',
        reviewer_id='b', protocol='software fixture', reviewed_at='2026-01-01T00:00:00Z')
    save(tmp_path, [case])
    pred = new_prediction(case['case_id'])
    pred.update(prediction_status='completed', predicted_hazard='flood',
                hazard_probabilities={'flood': 1., 'wildfire': 0., 'landslide': 0.},
                predicted_latitude=14.99, predicted_longitude=75.01, predicted_radius_km=2.,
                predicted_claims=['flood_verified'], predicted_geo_claims={'slope': .4},
                predicted_interventions=['evacuate', 'rescue'],
                predicted_action_priorities=['evacuate', 'rescue'], predicted_mask_path=case['mask_path'],
                model_version='software-fixture-only', inference_timestamp='2026-01-01T00:00:00Z')
    write_jsonl(tmp_path / 'predictions/predictions.jsonl', [pred])
    return tmp_path, case, pred, {'fixture': source}


def test_all_17_metrics_and_fixed_class_macro(benchmark):
    root, _, _, _ = benchmark
    report = evaluate_dataset(root, root / 'predictions/predictions.jsonl')
    scores = report['results_by_hazard']['overall']
    assert all(scores[m] is not None for m in METRICS)
    assert scores['accuracy'] == 1
    assert scores['macro_f1'] == .25  # Four fixed classes, only one represented.
    assert scores['haversine_distance_km'] == 0
    assert scores['radius_absolute_error_km'] == 0
    assert scores['dice_score'] == 1
    assert scores['average_reasoning_score'] == 1
    assert not report['metadata']['validation']['research_ready']


def test_missing_predictions_remain_null(benchmark):
    root, case, _, _ = benchmark
    write_jsonl(root / 'predictions/predictions.jsonl', [new_prediction(case['case_id'])])
    report = evaluate_dataset(root, root / 'predictions/predictions.jsonl')
    assert all(report['results_by_hazard']['overall'][m] is None for m in METRICS)
    assert report['metric_coverage']['overall']['accuracy']['eligible_count'] == 0


def test_unknown_ground_truth_never_becomes_zero(benchmark):
    root, case, pred, sources = benchmark
    case['true_latitude'] = None
    case['true_impact_radius_km'] = 0
    scores = score_case(root, case, pred, sources)
    assert 'haversine_distance_km' not in scores
    assert 'radius_relative_error' not in scores
    assert scores['radius_absolute_error_km'] == 2


def test_unverified_evidence_cannot_score(benchmark):
    root, case, pred, sources = benchmark
    case['evidence']['verified_claims']['status'] = 'unverified'
    scores = score_case(root, case, pred, sources)
    assert 'direct_assessment_precision' not in scores
    assert 'average_reasoning_score' not in scores
    save(root, [case])
    assert any('no verified field-level evidence' in x for x in validate_dataset(root)['errors'])


@pytest.mark.parametrize('field', ['event_id', 'region_id'])
def test_event_and_region_leakage_rejected(benchmark, field):
    root, case, _, _ = benchmark
    other = copy.deepcopy(case)
    other.update(case_id='fixture_other', split='train')
    other[field] = case[field]
    save(root, [case, other])
    assert any(f'{field} leaks' in e for e in validate_dataset(root)['errors'])


def test_file_tamper_rejected(benchmark):
    root, case, _, _ = benchmark
    with rasterio.open(root / case['image_path'], 'r+') as dst:
        dst.write(np.zeros((2, 2), dtype='uint8'), 1)
    assert any('Checksum' in e for e in validate_dataset(root)['errors'])


def test_shifted_prediction_grid_rejected(benchmark):
    root, case, pred, _ = benchmark
    path = root / 'masks/shifted.tif'
    with rasterio.open(root / case['mask_path']) as src:
        profile = src.profile
        profile['transform'] = from_origin(76, 15, .01, .01)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(src.read())
    pred['predicted_mask_path'] = 'masks/shifted.tif'
    with pytest.raises(ValueError, match='align exactly'):
        raster_arrays(root, case, pred)


def test_invalid_pixels_excluded_and_all_background_defined(benchmark):
    root, case, pred, sources = benchmark
    with rasterio.open(root / case['mask_path'], 'r+') as dst:
        dst.write(np.array([[255, 0], [0, 0]], dtype='uint8'), 1)
    with rasterio.open(root / case['validity_mask_path'], 'r+') as dst:
        dst.write(np.array([[0, 1], [1, 1]], dtype='uint8'), 1)
    scores = score_case(root, case, pred, sources)
    assert scores['intersection_over_union'] == scores['dice_score'] == 1
    with rasterio.open(root / case['validity_mask_path'], 'r+') as dst:
        dst.write(np.ones((2, 2), dtype='uint8'), 1)
    with pytest.raises(ValueError, match='binary'):
        raster_arrays(root, case, pred)


def test_unknown_geo_claim_and_unsafe_action_penalized(benchmark):
    root, case, pred, sources = benchmark
    pred['predicted_geo_claims']['invented'] = 10
    pred['predicted_interventions'].append('enter_floodwater')
    scores = score_case(root, case, pred, sources)
    assert scores['factual_correctness'] == .5
    assert scores['recovery_necessity'] == pytest.approx(2/3)


def test_missing_and_reversed_action_priorities_penalized(benchmark):
    root, case, pred, sources = benchmark
    for order in (['evacuate'], ['rescue', 'evacuate']):
        pred['predicted_action_priorities'] = order
        assert score_case(root, case, pred, sources)['action_priority_precision'] == 0


def test_path_traversal_rejected(benchmark):
    root, case, _, _ = benchmark
    case['image_path'] = '../outside.tif'
    save(root, [case])
    assert any('escapes dataset root' in e for e in validate_dataset(root)['errors'])


def test_duplicate_predictions_rejected(benchmark):
    root, _, pred, _ = benchmark
    write_jsonl(root / 'predictions/predictions.jsonl', [pred, pred])
    with pytest.raises(ValueError, match='duplicate prediction'):
        evaluate_dataset(root, root / 'predictions/predictions.jsonl')


def test_unreviewed_and_exposed_test_cases_rejected(benchmark):
    root, case, _, _ = benchmark
    case.update(training_exposure='unknown', annotation_status='pending')
    save(root, [case])
    errors = validate_dataset(root)['errors']
    assert any('independence' in e for e in errors)
    assert any('reviewed annotation' in e for e in errors)


def test_candidates_do_not_enter_splits_or_metrics(tmp_path):
    init(tmp_path)
    case = new_case('candidate_only', 'landslide')
    save(tmp_path, [case])
    report = validate_dataset(tmp_path)
    assert report['valid'] and not report['research_ready']
    result = evaluate_dataset(tmp_path, tmp_path / 'predictions/predictions.jsonl')
    assert result['sample_counts'] == {}
    assert result['results_by_hazard']['overall']['accuracy'] is None

@pytest.mark.parametrize('field', ['evidence', 'assets', 'missing_reasons', 'review'])
def test_malformed_annotation_objects_return_errors(benchmark, field):
    root, case, _, _ = benchmark
    case[field] = None
    save(root, [case])
    report = validate_dataset(root)
    assert not report['valid']
    assert any(field in message for message in report['errors'])


def test_invalid_date_type_returns_validation_error(benchmark):
    root, case, _, _ = benchmark
    case['event_start'] = 20260101
    save(root, [case])
    assert any('event_start' in message for message in validate_dataset(root)['errors'])


def test_null_prediction_reasons_raise_useful_error(benchmark):
    root, case, _, _ = benchmark
    pred = new_prediction(case['case_id'])
    pred['missing_reasons'] = None
    write_jsonl(root / 'predictions/predictions.jsonl', [pred])
    with pytest.raises(ValueError, match='missing_reasons'):
        evaluate_dataset(root, root / 'predictions/predictions.jsonl')


def test_probability_dictionary_rejects_unknown_classes(benchmark):
    root, _, pred, _ = benchmark
    pred['hazard_probabilities'] = {'flood': .5, 'wildfire': 0., 'landslide': 0., 'typo': .5}
    write_jsonl(root / 'predictions/predictions.jsonl', [pred])
    with pytest.raises(ValueError, match='probabilities'):
        evaluate_dataset(root, root / 'predictions/predictions.jsonl')


def test_smoke_metric_helpers_work_when_called_directly():
    import importlib.util
    import torch
    script = Path(__file__).resolve().parents[1] / 'scripts/run_evaluation.py'
    spec = importlib.util.spec_from_file_location('isolated_evaluation_script', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module._segmentation_metrics(
        {'flood': torch.tensor([[[[2., -2.]]]])},
        {'hazards': ['flood'], 'target_masks': torch.tensor([[[[1., 0.]]]]),
         'validity_masks': torch.ones((1, 1, 1, 2), dtype=torch.bool)})
    assert result['iou'] == 1


def test_acquisition_failure_exits_nonzero_without_losing_completed_cases(tmp_path, monkeypatch):
    import requests
    import scripts.build_evaluation_dataset as builder
    monkeypatch.setattr(builder, 'download', lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError('offline')))
    def partial(*args, **kwargs):
        yield new_case('completed_before_failure', 'wildfire')
        raise ValueError('download cap reached')
    monkeypatch.setattr(builder, 'acquire_fire', partial)
    code = builder.main(['acquire', '--root', str(tmp_path), '--hazards', 'wildfire'])
    assert code == 2
    assert validate_dataset(tmp_path)['case_count'] == 1
