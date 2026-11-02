"""Detailed diagnostics from saved predictions; never generates predictions or truth."""
import math


def classification_diagnostics(records, labels):
    labels = list(labels)
    records = [r for r in records if r.get('true_hazard') in labels and r.get('predicted_hazard') in labels]
    n = len(records)
    matrix = [[0 for _ in labels] for _ in labels]
    for row in records:
        matrix[labels.index(row['true_hazard'])][labels.index(row['predicted_hazard'])] += 1
    supports = [sum(row) for row in matrix]
    predicted_counts = [sum(row[j] for row in matrix) for j in range(len(labels))]
    per_class = []
    divide = lambda a, b: a / b if b else None
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp, fn = predicted_counts[i] - tp, supports[i] - tp
        tn = n - tp - fp - fn
        per_class.append({'class': label, 'support': supports[i], 'predicted_count': predicted_counts[i],
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': divide(tp, tp + fp), 'recall': divide(tp, tp + fn),
            'f1': divide(2 * tp, 2 * tp + fp + fn), 'specificity': divide(tn, tn + fp),
            'negative_predictive_value': divide(tn, tn + fn)})
    correct = sum(matrix[i][i] for i in range(len(labels)))
    accuracy = divide(correct, n)
    expected = divide(sum(a * b for a, b in zip(supports, predicted_counts)), n*n)
    mcc_den = math.sqrt(max(0, (n*n-sum(v*v for v in supports)) * (n*n-sum(v*v for v in predicted_counts))))
    supported = [row for row in per_class if row['support']]
    ci = None
    if n:
        z = 1.959963984540054
        denominator = 1 + z*z/n
        center = (accuracy + z*z/(2*n)) / denominator
        half = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / denominator
        ci = {'lower': max(0., center-half), 'upper': min(1., center+half), 'level': .95,
              'method': 'Wilson score interval; assumes independent cases, not clustered event sampling'}
    summary = {'accuracy': accuracy,
        'balanced_accuracy': sum(row['recall'] for row in supported)/len(supported) if supported else None,
        'weighted_f1': sum((row['f1'] or 0)*row['support'] for row in per_class)/n if n else None,
        'micro_f1': accuracy,
        'cohens_kappa': (accuracy-expected)/(1-expected) if n and expected < 1 else None,
        'multiclass_mcc': (correct*n-sum(a*b for a,b in zip(supports,predicted_counts)))/mcc_den if mcc_den else None,
        'error_count': n-correct, 'correct_count': correct, 'accuracy_ci95': ci}
    result = {'status': 'available' if n else 'not_evaluated', 'sample_count': n,
        'labels': labels, 'confusion_matrix': matrix,
        'confusion_matrix_row_normalized': [[v/supports[i] if supports[i] else None for v in row] for i,row in enumerate(matrix)],
        'orientation': 'Rows = true class; columns = predicted class', 'per_class': per_class,
        'summary': summary, 'per_case': records,
        'warnings': ['Small samples do not establish generalization.'] if 0 < n < 100 else []}
    result['probability_diagnostics'] = probability_diagnostics(records, labels)
    return result


def probability_diagnostics(records, labels):
    eligible = []
    for row in records:
        probs = row.get('hazard_probabilities')
        if (not isinstance(probs, dict) or set(probs) != set(labels)
                or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())
                or abs(sum(probs.values())-1) > 1e-5):
            continue
        eligible.append(row)
    unavailable = {'status': 'not_evaluated', 'eligible_count': 0, 'total_classified': len(records),
                   'reason': 'Full probabilities for every evaluation class are required; top confidence alone is insufficient.'}
    if not eligible:
        return unavailable
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    except ImportError:
        return {**unavailable, 'reason': 'Install scikit-learn and numpy to calculate probability diagnostics.'}
    truth = np.array([labels.index(r['true_hazard']) for r in eligible])
    scores = np.array([[r['hazard_probabilities'][label] for label in labels] for r in eligible])
    onehot = np.eye(len(labels))[truth]
    log_loss = float(-np.log(np.clip(scores[np.arange(len(truth)),truth], 1e-15, 1)).mean())
    brier = float(((scores-onehot)**2).sum(axis=1).mean())
    # Calibration uses the probability argmax, not a possibly thresholded emitted class.
    confidence = scores.max(axis=1)
    correct = scores.argmax(axis=1) == truth
    bins, ece = [], 0.
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        count = int(mask.sum())
        mean_conf = float(confidence[mask].mean()) if count else None
        mean_acc = float(correct[mask].mean()) if count else None
        bins.append({'lower': i/10, 'upper': (i+1)/10, 'count': count, 'confidence': mean_conf, 'accuracy': mean_acc})
        if count:
            ece += count/len(truth)*abs(mean_conf-mean_acc)
    curves = {}
    for i, label in enumerate(labels):
        binary = truth == i
        if not binary.any() or binary.all():
            curves[label] = {'status': 'not_evaluated', 'reason': 'Both positive and negative references are required.'}
            continue
        fpr,tpr,_ = roc_curve(binary, scores[:,i])
        precision,recall,_ = precision_recall_curve(binary, scores[:,i])
        curves[label] = {'status': 'available', 'roc_auc': float(roc_auc_score(binary,scores[:,i])),
            'average_precision': float(average_precision_score(binary,scores[:,i])),
            'positive_prevalence': float(binary.mean()),
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'precision_recall': {'precision': precision.tolist(), 'recall': recall.tolist()}}
    return {'status': 'available', 'eligible_count': len(eligible), 'total_classified': len(records),
            'log_loss': log_loss, 'multiclass_brier_score': brier, 'expected_calibration_error': float(ece),
            'calibration_bins': bins, 'curves': curves,
            'definitions': {'brier': 'Mean sum of squared errors across all classes; range [0,2]',
                            'ece': '10 equal-width confidence bins; weighted absolute confidence-accuracy gap'}}


def metric_distributions(values):
    result = {}
    for scope, metrics in values.items():
        result[scope] = {}
        for name, numbers in metrics.items():
            if not numbers:
                continue
            if name in ('macro_precision', 'macro_recall', 'macro_f1'):
                result[scope][name] = {'count': len(numbers), 'mean': numbers[0],
                    'distribution_status': 'not_applicable: aggregate class metric, not per-case observations'}
                continue
            ordered = sorted(numbers)
            def percentile(q):
                pos = (len(ordered)-1)*q
                lo,hi = math.floor(pos),math.ceil(pos)
                return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
            mean = sum(numbers)/len(numbers)
            result[scope][name] = {'count': len(numbers), 'mean': mean,
                'std_population': math.sqrt(sum((v-mean)**2 for v in numbers)/len(numbers)),
                'minimum': ordered[0], 'p25': percentile(.25), 'median': percentile(.5),
                'p75': percentile(.75), 'p95': percentile(.95), 'maximum': ordered[-1]}
    return result


def report_diagnostics(report):
    """Enrich older saved reports using actual per-sample class records only."""
    existing = report.get('detailed_evaluation')
    if isinstance(existing, dict):
        return existing
    metadata = report.get('metadata')
    metadata = metadata if isinstance(metadata, dict) else {}
    rows = []
    saved_rows = metadata.get('per_sample')
    saved_rows = saved_rows if isinstance(saved_rows, list) else []
    for row in saved_rows:
        if isinstance(row, dict):
            rows.append({'case_id': row.get('case_id') or row.get('hazard'),
                'true_hazard': row.get('target_class'), 'predicted_hazard': row.get('predicted_class'),
                'hazard_probabilities': row.get('hazard_probabilities')})
    smoke = metadata.get('evaluation_kind') == 'generated_sample_smoke_test'
    labels = ['flood','wildfire','landslide'] + ([] if smoke else ['none'])
    by_scope = {'overall': classification_diagnostics(rows, labels)}
    for hazard in ('flood', 'wildfire', 'landslide'):
        by_scope[hazard] = classification_diagnostics([r for r in rows if r['true_hazard'] == hazard], labels)
    return {'classification': by_scope,
            'metric_distributions': {}, 'provenance': 'generated_sample_smoke_test' if smoke else 'saved_report',
            'notice': 'GENERATED-DATA DIAGNOSTICS ONLY. Not independent benchmark evidence.' if smoke else 'Only saved prediction/reference records are used.'}
