"""Diagnostics tests use software fixtures only. No real benchmark claims."""
import pytest
from evaluation.diagnostics import classification_diagnostics, report_diagnostics, metric_distributions


def test_confusion_counts_and_errors():
    records = [{'true_hazard':t,'predicted_hazard':p} for t,p in [('flood','flood'),('flood','wildfire'),('wildfire','wildfire')]]
    result = classification_diagnostics(records,['flood','wildfire','landslide'])
    assert result['confusion_matrix'] == [[1,1,0],[0,1,0],[0,0,0]]
    assert result['per_class'][0]['recall'] == .5
    assert result['per_class'][1]['precision'] == .5
    assert result['per_class'][2]['recall'] is None
    assert result['confusion_matrix_row_normalized'][2] == [None,None,None]
    assert result['summary']['accuracy'] == pytest.approx(2/3)
    assert result['summary']['error_count'] == 1


def test_empty_diagnostics_remain_unavailable():
    result = classification_diagnostics([],['flood','wildfire','landslide','none'])
    assert result['status'] == 'not_evaluated'
    assert result['summary']['accuracy'] is None
    assert result['summary']['cohens_kappa'] is None
    assert result['summary']['multiclass_mcc'] is None
    assert result['summary']['accuracy_ci95'] is None


def test_three_perfect_cases_still_have_wide_uncertainty():
    result = classification_diagnostics([{'true_hazard':x,'predicted_hazard':x} for x in ['flood','wildfire','landslide']],['flood','wildfire','landslide'])
    interval = result['summary']['accuracy_ci95']
    assert .43 < interval['lower'] < .44
    assert interval['upper'] == pytest.approx(1)
    assert result['warnings']


def test_probability_metrics_use_full_vectors():
    rows = [{'true_hazard': 'flood', 'predicted_hazard':'flood','hazard_probabilities':{'flood':.8,'wildfire':.2}},
            {'true_hazard': 'wildfire','predicted_hazard':'wildfire','hazard_probabilities':{'flood':.1,'wildfire':.9}}]
    result = classification_diagnostics(rows,['flood','wildfire'])['probability_diagnostics']
    assert result['eligible_count'] == 2
    assert result['curves']['flood']['roc_auc'] == 1
    assert result['curves']['wildfire']['average_precision'] == 1
    assert result['multiclass_brier_score'] == pytest.approx(.05)
    assert result['expected_calibration_error'] == pytest.approx(.15)
    assert sum(b['count'] for b in result['calibration_bins']) == 2


def test_confidence_only_does_not_invent_probability_vector():
    report = {'metadata': {'evaluation_kind':'generated_sample_smoke_test','per_sample':[{'hazard':'flood','target_class':'flood','predicted_class':'flood','confidence':.9}]}}
    result = report_diagnostics(report)
    assert result['provenance'] == 'generated_sample_smoke_test'
    overall = result['classification']['overall']
    assert overall['sample_count'] == 1
    assert overall['probability_diagnostics']['status'] == 'not_evaluated'


def test_metric_distribution_quantiles():
    result = metric_distributions({'overall':{'haversine_distance_km':[0,2,4]}})['overall']['haversine_distance_km']
    assert result['mean'] == result['median'] == 2
    assert result['p25'] == 1
    assert result['p95'] == pytest.approx(3.8)


def test_aggregate_macro_scores_do_not_get_fake_per_case_variance():
    result = metric_distributions({'overall': {'macro_f1': [.5,.5,.5]}})['overall']['macro_f1']
    assert result['mean'] == .5
    assert 'std_population' not in result
    assert 'not_applicable' in result['distribution_status']


def test_malformed_legacy_metadata_does_not_crash():
    assert report_diagnostics({'metadata': []})['classification']['overall']['sample_count'] == 0
    assert report_diagnostics({'metadata': {'per_sample': None}})['classification']['overall']['sample_count'] == 0
