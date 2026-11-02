"""Evidence-first benchmark contract, integrity checks, and held-out evaluation.

Independent of torch, the API, and training loaders. Never fills missing truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HAZARDS = ('flood', 'wildfire', 'landslide')
SPLITS = ('train', 'validation', 'test')
METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
           'haversine_distance_km', 'radius_absolute_error_km', 'radius_relative_error',
           'direct_assessment_precision', 'direct_discovery_recall', 'factual_correctness',
           'recovery_necessity', 'strategic_completeness', 'action_priority_precision',
           'average_reasoning_score', 'pixel_accuracy', 'intersection_over_union', 'dice_score')
TRUTH_FIELDS = ('true_hazard', 'hazard_present', 'severity', 'true_latitude',
                'true_longitude', 'true_impact_radius_km', 'affected_area_km2',
                'event_start', 'event_end', 'event_polygon', 'location_uncertainty_km',
                'verified_claims', 'geo_attributes', 'required_interventions',
                'standard_checklist', 'priority_order', 'unsafe_actions',
                'affected_population', 'affected_infrastructure')
OBSERVATION_FIELDS = ('image_path', 'mask_path', 'validity_mask_path',
                      'acquisition_datetime', 'latitude', 'longitude', 'bbox',
                      'source_satellite', 'source_organization', 'spatial_resolution_m',
                      'crs', 'quality', 'administrative_region', 'state', 'district',
                      'country', 'mask_metadata', 'label_confidence')
PREDICTION_FIELDS = ('predicted_hazard', 'hazard_probabilities', 'predicted_latitude',
                     'predicted_longitude', 'predicted_radius_km', 'predicted_claims',
                     'predicted_geo_claims', 'predicted_interventions',
                     'predicted_action_priorities', 'predicted_mask_path',
                     'model_version', 'inference_timestamp')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if not isinstance(value, dict):
                    raise ValueError(f'{path}:{line_number}: expected object')
                records.append(value)
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records), encoding='utf-8')


def new_case(case_id, hazard):
    if hazard not in HAZARDS:
        raise ValueError(f'Unsupported hazard: {hazard}')
    case = dict.fromkeys(OBSERVATION_FIELDS + TRUTH_FIELDS)
    case.update(schema_version=1, case_id=case_id, hazard_type=hazard, split=None,
                event_id=None, region_id=None, annotation_status='pending',
                training_exposure='unknown', evidence={}, assets={},
                missing_reasons={field: 'Not supplied or independently verified.'
                                 for field in OBSERVATION_FIELDS + TRUTH_FIELDS})
    return case


def new_prediction(case_id):
    record = dict.fromkeys(PREDICTION_FIELDS)
    record.update(case_id=case_id, prediction_status='not_run',
                  missing_reasons={k: 'Inference has not been run on this case.' for k in PREDICTION_FIELDS})
    return record


def local_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError('Expected a relative asset path')
    root = Path(root).resolve()
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f'Asset escapes dataset root: {value}')
    return path


def has_evidence(case, field, sources):
    evidence_map = case.get('evidence')
    if not isinstance(evidence_map, dict):
        return False
    evidence = evidence_map.get(field)
    if not isinstance(evidence, dict):
        return False
    source_id = evidence.get('source_id')
    if not isinstance(source_id, str):
        return False
    source = sources.get(source_id, {})
    url = source.get('url') if isinstance(source, dict) else None
    if not isinstance(url, str) or not url.startswith(('https://', 'http://')):
        return False
    if not evidence.get('reference') or evidence.get('status') != 'verified':
        return False
    method = evidence.get('method')
    if field == 'priority_order' and method != 'expert_annotation':
        return False
    if method == 'documented_derivation' and not evidence.get('reference'):
        return False
    if method == 'expert_annotation':
        return bool(evidence.get('annotator_id') and evidence.get('reviewer_id')
                    and evidence['annotator_id'] != evidence['reviewer_id']
                    and evidence.get('protocol') and evidence.get('reviewed_at'))
    return method in ('source_metadata', 'published_label', 'documented_derivation')


def finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def raster_arrays(root, case, prediction=None):
    import numpy as np
    import rasterio
    with rasterio.open(local_path(root, case['image_path'])) as image, \
         rasterio.open(local_path(root, case['mask_path'])) as mask, \
         rasterio.open(local_path(root, case['validity_mask_path'])) as validity:
        rasters = [image, mask, validity]
        predicted = rasterio.open(local_path(root, prediction['predicted_mask_path'])) if prediction else None
        if predicted:
            rasters.append(predicted)
        try:
            if not image.crs or image.transform.is_identity:
                raise ValueError('Image has no usable georeferencing')
            for raster in rasters[1:]:
                if (raster.crs != image.crs or raster.shape != image.shape
                        or not raster.transform.almost_equals(image.transform)):
                    raise ValueError('Image, target, validity, and prediction grids must align exactly')
                if raster.count != 1:
                    raise ValueError('Masks must have one band')
            target = mask.read(1)
            valid_values = validity.read(1)
            if not np.isin(valid_values, (0, 1)).all():
                raise ValueError('Validity mask must contain only 0 and 1')
            valid = valid_values == 1
            if not valid.any():
                raise ValueError('No valid evaluation pixels')
            if np.any(valid & ((mask.read_masks(1) == 0) | (image.dataset_mask() == 0))):
                raise ValueError('Validity mask includes source no-data pixels')
            if not np.isin(target[valid], (0, 1)).all():
                raise ValueError('Valid target labels must be binary 0/1')
            metadata = case.get('mask_metadata') or {}
            expected = {'crs': mask.crs.to_string(), 'width': mask.width, 'height': mask.height,
                        'transform': list(mask.transform)[:6], 'resolution': list(mask.res),
                        'nodata': mask.nodata}
            for key, value in expected.items():
                if key not in metadata or metadata[key] != value:
                    raise ValueError(f'Mask metadata mismatch: {key}')
            if predicted:
                pred = predicted.read(1)
                if np.any(valid & (predicted.read_masks(1) == 0)):
                    raise ValueError('Prediction omits valid evaluation pixels')
                if not np.isin(pred[valid], (0, 1)).all():
                    raise ValueError('Predicted mask must contain binary class IDs, not probabilities')
                return target[valid].astype(bool), pred[valid].astype(bool)
            return target[valid].astype(bool), None
        finally:
            if predicted:
                predicted.close()


def validate_dataset(root):
    root = Path(root)
    cases = read_jsonl(root / 'annotations/cases.jsonl')
    source_data = json.loads((root / 'metadata/sources.json').read_text(encoding='utf-8'))
    sources = {s['source_id']: s for s in source_data['sources']}
    errors, warnings = [], []
    ids, groups, hashes = set(), {}, {}
    footprints = []
    counts = {s: {h: 0 for h in HAZARDS} for s in SPLITS}
    for case in cases:
        cid = case.get('case_id')
        def error(message):
            errors.append(f'{cid}: {message}')
        if not isinstance(cid, str) or not cid or cid in ids:
            error('Missing or duplicate case_id')
            continue
        ids.add(cid)
        malformed = False
        for name in ('evidence', 'assets', 'missing_reasons'):
            if not isinstance(case.get(name), dict):
                error(f'{name} must be a JSON object')
                malformed = True
        if not isinstance(case.get('review', {}), dict):
            error('review must be a JSON object')
            malformed = True
        if malformed:
            continue
        for field in OBSERVATION_FIELDS + TRUTH_FIELDS:
            if field not in case:
                error(f'Missing required field {field}')
            elif case[field] is None:
                if not case.get('missing_reasons', {}).get(field):
                    error(f'Null {field} needs a missing reason')
            elif not has_evidence(case, field, sources):
                error(f'{field} has no verified field-level evidence')
        hazard, split = case.get('hazard_type'), case.get('split')
        if hazard not in HAZARDS:
            error('Unknown hazard_type')
        if split not in SPLITS and split is not None:
            error('Unknown split')
        if case.get('true_hazard') not in (*HAZARDS, 'none', None):
            error('Unknown true_hazard')
        if case.get('hazard_present') is not None and type(case['hazard_present']) is not bool:
            error('hazard_present must be boolean or null')
        if case.get('hazard_present') is False and case.get('true_hazard') not in ('none', None):
            error('A non-event must have true_hazard=none')
        if case.get('hazard_present') is True and case.get('true_hazard') not in HAZARDS:
            error('Positive case must identify true_hazard')
        if case.get('severity') not in (None, 'low', 'medium', 'high', 'critical'):
            error('Unknown severity')
        for field in ('latitude', 'longitude', 'true_latitude', 'true_longitude',
                      'true_impact_radius_km', 'affected_area_km2', 'location_uncertainty_km'):
            value = case.get(field)
            if value is not None:
                limit = 90 if 'latitude' in field else 180 if 'longitude' in field else None
                if not finite_number(value) or (limit and abs(value) > limit) or (not limit and value < 0):
                    error(f'Invalid numeric value for {field}')
        for field in ('verified_claims', 'required_interventions', 'standard_checklist', 'priority_order', 'unsafe_actions'):
            value = case.get(field)
            if value is not None and (not isinstance(value, list) or any(not isinstance(v, str) for v in value)
                                      or len(value) != len(set(value))):
                error(f'{field} must contain unique canonical string IDs')
        if case.get('geo_attributes') is not None:
            if not isinstance(case['geo_attributes'], dict):
                error('geo_attributes must be an object')
            else:
                for name, attr in case['geo_attributes'].items():
                    if (not isinstance(attr, dict) or not finite_number(attr.get('value'))
                            or not finite_number(attr.get('absolute_tolerance')) or attr['absolute_tolerance'] < 0
                            or not attr.get('unit') or not attr.get('source_id') in sources or not attr.get('reference')):
                        error(f'Invalid geo attribute {name}; requires value, absolute_tolerance, unit, source_id, reference')
        for field in ('event_start', 'event_end', 'acquisition_datetime'):
            if case.get(field) is not None:
                try:
                    parsed = datetime.fromisoformat(case[field].replace('Z', '+00:00'))
                    if field == 'acquisition_datetime' and parsed.tzinfo is None:
                        raise ValueError('Timezone missing')
                except (ValueError, TypeError, AttributeError):
                    error(f'Invalid ISO date/time: {field}')
        if case.get('event_start') and case.get('event_end'):
            try:
                start = datetime.fromisoformat(case['event_start'].replace('Z', '+00:00')).date()
                end = datetime.fromisoformat(case['event_end'].replace('Z', '+00:00')).date()
                if start > end:
                    error('Event end precedes start')
            except (ValueError, TypeError, AttributeError):
                pass  # The malformed field has already been reported above.
        for field in ('image_path', 'mask_path', 'validity_mask_path'):
            if not case.get(field):
                continue
            try:
                path = local_path(root, case[field])
                if not path.is_file():
                    error(f'Missing asset {field}')
                    continue
                sha = digest(path)
                asset = case['assets'].get(field, {})
                if not isinstance(asset, dict):
                    error(f'Asset metadata must be an object: {field}')
                    continue
                if asset.get('sha256') != sha:
                    error(f'Checksum missing or incorrect: {field}')
                if field == 'image_path' and split:
                    if sha in hashes and hashes[sha] != split:
                        error('Duplicate image bytes cross split boundaries')
                    hashes[sha] = split
                if path.name.startswith('sample_'):
                    error('Repository generated sample files are prohibited')
            except ValueError as exc:
                error(str(exc))
        if split:
            review = case.get('review') or {}
            if (not review.get('annotator_id') or not review.get('reviewer_id')
                    or review.get('annotator_id') == review.get('reviewer_id')
                    or not review.get('protocol') or not review.get('reviewed_at')):
                error('Split membership requires independent annotation review and protocol')
            if case.get('bbox') is not None:
                bbox = case['bbox']
                if (not isinstance(bbox, list) or len(bbox) != 4
                        or not all(finite_number(v) for v in bbox)
                        or not -180 <= bbox[0] < bbox[2] <= 180
                        or not -90 <= bbox[1] < bbox[3] <= 90):
                    error('bbox must be ordered WGS84 [west,south,east,north]')
                else:
                    for other_id, other_split, other in footprints:
                        if split != other_split and max(bbox[0], other[0]) < min(bbox[2], other[2]) and max(bbox[1], other[1]) < min(bbox[3], other[3]):
                            error(f'Image footprint overlaps another split: {other_id}')
                    footprints.append((cid, split, bbox))
            if case.get('annotation_status') != 'reviewed':
                error('Split membership requires reviewed annotation')
            for key in ('event_id', 'region_id'):
                group = case.get(key)
                if not group:
                    error(f'Split membership requires {key}')
                elif (key, group) in groups and groups[key, group] != split:
                    error(f'{key} leaks across splits: {group}')
                else:
                    groups[key, group] = split
            if hazard in HAZARDS:
                counts[split][hazard] += 1
            if split == 'test' and case.get('training_exposure') != 'not_used':
                error('Test independence has not been attested')
            if all(case.get(f) for f in ('image_path', 'mask_path', 'validity_mask_path')):
                try:
                    raster_arrays(root, case)
                except (ValueError, OSError) as exc:
                    error(str(exc))
            else:
                error('Split member requires image, binary target, and validity assets')
        else:
            warnings.append(f'{cid}: candidate excluded from all benchmark splits')
    for split in SPLITS:
        path = root / 'splits' / f'{split}.txt'
        actual = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
        expected = [c['case_id'] for c in cases if c.get('split') == split]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            errors.append(f'{split}.txt disagrees with cases.jsonl')
    blockers = [f'{h}: {counts["test"][h]}/100 recommended test cases' for h in HAZARDS if counts['test'][h] < 100]
    test = [c for c in cases if c.get('split') == 'test']
    for h in HAZARDS:
        subset = [c for c in test if c['hazard_type'] == h]
        if not any(c.get('hazard_present') is False for c in subset):
            blockers.append(f'{h}: no verified negative/non-event test cases')
        if len({c.get('region_id') for c in subset}) < 2:
            blockers.append(f'{h}: fewer than two test regions')
        if len({c.get('event_id') for c in subset}) < 2:
            blockers.append(f'{h}: fewer than two test events')
        if len({c.get('severity') for c in subset if c.get('severity')}) < 2:
            blockers.append(f'{h}: fewer than two verified severity levels')
    for split in ('train', 'validation'):
        if not any(counts[split].values()):
            blockers.append(f'{split}: no reviewed cases')
    required_complete = tuple(f for f in TRUTH_FIELDS + OBSERVATION_FIELDS
                              if f not in ('event_polygon', 'affected_population', 'affected_infrastructure'))
    incomplete = [c['case_id'] for c in test if any(c.get(f) is None for f in required_complete)]
    if incomplete:
        blockers.append(f'{len(incomplete)} test cases have unavailable requested annotations')
    return {'schema_version': 1, 'generated_at': utc_now(), 'valid': not errors,
            'research_ready': bool(test) and not errors and not blockers,
            'case_count': len(cases), 'split_counts': counts, 'errors': errors,
            'warnings': warnings, 'readiness_blockers': blockers}


def _priority_score(predicted, reference):
    # Correct ordered expert pairs / all expert pairs plus unknown predicted pairs.
    if len(reference) < 2:
        return None
    pred_rank = {item: i for i, item in enumerate(predicted)}
    correct = sum(a in pred_rank and b in pred_rank and pred_rank[a] < pred_rank[b]
                  for i, a in enumerate(reference) for b in reference[i + 1:])
    unknown = sum(a not in reference or b not in reference
                  for i, a in enumerate(predicted) for b in predicted[i + 1:])
    return correct / (len(reference) * (len(reference) - 1) / 2 + unknown)


def score_case(root, case, pred, sources):
    values = {}
    def truth(*fields):
        return all(case.get(f) is not None and has_evidence(case, f, sources) for f in fields)
    if truth('true_latitude', 'true_longitude') and all(pred.get(f) is not None for f in ('predicted_latitude', 'predicted_longitude')):
        lat1, lat2 = map(math.radians, (pred['predicted_latitude'], case['true_latitude']))
        dlat, dlon = lat2 - lat1, math.radians(case['true_longitude'] - pred['predicted_longitude'])
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        values['haversine_distance_km'] = 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, a))))
    if truth('true_impact_radius_km') and pred.get('predicted_radius_km') is not None:
        radius = case['true_impact_radius_km']
        error = abs(pred['predicted_radius_km'] - radius)
        values['radius_absolute_error_km'] = error
        if radius > 0:
            values['radius_relative_error'] = error / radius
    if truth('verified_claims') and pred.get('predicted_claims') is not None:
        p, t = set(pred['predicted_claims']), set(case['verified_claims'])
        values['direct_assessment_precision'] = len(p & t) / len(p) if p else 0.0
        if t:
            values['direct_discovery_recall'] = len(p & t) / len(t)
    if truth('geo_attributes') and pred.get('predicted_geo_claims') is not None:
        claims, attrs = pred['predicted_geo_claims'], case['geo_attributes']
        if claims:
            correct = sum(k in attrs and abs(v - attrs[k]['value']) <= attrs[k]['absolute_tolerance']
                          for k, v in claims.items())
            values['factual_correctness'] = correct / len(claims)
        else:
            values['factual_correctness'] = 0.0
    for field, metric, precision in (
        ('required_interventions', 'recovery_necessity', True),
        ('standard_checklist', 'strategic_completeness', False)):
        if truth(field, 'unsafe_actions') and pred.get('predicted_interventions') is not None:
            p, t = set(pred['predicted_interventions']), set(case[field])
            correct = len((p & t) - set(case['unsafe_actions']))
            denominator = len(p) if precision else len(t)
            if denominator:
                values[metric] = correct / denominator
            elif precision:
                values[metric] = 0.0
    if truth('priority_order') and pred.get('predicted_action_priorities') is not None:
        value = _priority_score(pred['predicted_action_priorities'], case['priority_order'])
        if value is not None:
            values['action_priority_precision'] = value
    reasoning = METRICS[7:13]
    if all(k in values for k in reasoning):
        values['average_reasoning_score'] = sum(values[k] for k in reasoning) / 6
    if truth('mask_path', 'validity_mask_path') and pred.get('predicted_mask_path'):
        import numpy as np
        target, prediction = raster_arrays(root, case, pred)
        intersection = int(np.logical_and(target, prediction).sum())
        union = int(np.logical_or(target, prediction).sum())
        denominator = int(target.sum() + prediction.sum())
        values['pixel_accuracy'] = float((target == prediction).mean())
        values['intersection_over_union'] = intersection / union if union else 1.0
        values['dice_score'] = 2 * intersection / denominator if denominator else 1.0
    return values


def validate_predictions(predictions, case_ids):
    seen = set()
    for pred in predictions:
        cid = pred.get('case_id')
        if not isinstance(cid, str) or cid not in case_ids or cid in seen:
            raise ValueError(f'Unknown or duplicate prediction case_id: {cid}')
        seen.add(cid)
        if not isinstance(pred.get('missing_reasons'), dict):
            raise ValueError(f'{cid}: missing_reasons must be a JSON object')
        for field in PREDICTION_FIELDS:
            if field not in pred:
                raise ValueError(f'{cid}: prediction missing field {field}')
            if pred[field] is None and not pred.get('missing_reasons', {}).get(field):
                raise ValueError(f'{cid}: null prediction {field} requires reason')
        if pred.get('prediction_status') == 'not_run':
            if any(pred.get(k) is not None for k in PREDICTION_FIELDS):
                raise ValueError(f'{cid}: not_run prediction contains invented output')
            continue
        if pred.get('prediction_status') != 'completed' or not pred.get('model_version'):
            raise ValueError(f'{cid}: completed inference requires model_version')
        try:
            timestamp = datetime.fromisoformat(pred['inference_timestamp'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError('timezone missing')
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f'{cid}: inference_timestamp requires an ISO timestamp with timezone')
        if pred.get('predicted_hazard') not in (*HAZARDS, 'none', None):
            raise ValueError(f'{cid}: invalid predicted hazard')
        probs = pred.get('hazard_probabilities')
        if probs is not None and (not isinstance(probs, dict) or set(probs) not in (set(HAZARDS), {*HAZARDS, 'none'})
                or any(not finite_number(v) or not 0 <= v <= 1 for v in probs.values())
                or abs(sum(probs.values()) - 1) > 1e-5):
            raise ValueError(f'{cid}: invalid hazard probabilities')
        for field in ('predicted_latitude', 'predicted_longitude', 'predicted_radius_km'):
            v = pred.get(field)
            if v is not None and (not finite_number(v) or (field.endswith('radius_km') and v < 0)
                    or (field.endswith('latitude') and abs(v) > 90)
                    or (field.endswith('longitude') and abs(v) > 180)):
                raise ValueError(f'{cid}: invalid {field}')
        for field in ('predicted_claims', 'predicted_interventions', 'predicted_action_priorities'):
            value = pred.get(field)
            if value is not None and (not isinstance(value, list) or any(not isinstance(v, str) for v in value)
                                      or len(value) != len(set(value))):
                raise ValueError(f'{cid}: {field} requires unique canonical IDs')
        if pred.get('predicted_geo_claims') is not None:
            claims = pred['predicted_geo_claims']
            if not isinstance(claims, dict) or any(not finite_number(v) for v in claims.values()):
                raise ValueError(f'{cid}: invalid numeric geo claims')


def evaluate_dataset(root, predictions_path):
    root = Path(root)
    validation = validate_dataset(root)
    if validation['errors']:
        raise ValueError('Dataset integrity failed: ' + '; '.join(validation['errors'][:10]))
    cases = read_jsonl(root / 'annotations/cases.jsonl')
    predictions = read_jsonl(predictions_path)
    validate_predictions(predictions, {c['case_id'] for c in cases})
    indexed = {p['case_id']: p for p in predictions}
    sources = {s['source_id']: s for s in json.loads((root / 'metadata/sources.json').read_text(encoding='utf-8'))['sources']}
    samples, values, classification = Counter(), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    skipped = []
    diagnostic_rows = defaultdict(list)
    for case in cases:
        if case.get('split') != 'test':
            continue
        scopes = ('overall', case['hazard_type'])
        for scope in scopes:
            samples[scope] += 1
        pred = indexed.get(case['case_id'])
        if not pred or pred.get('prediction_status') != 'completed':
            skipped.append({'case_id': case['case_id'], 'reason': 'No completed prediction'})
            continue
        scores = score_case(root, case, pred, sources)
        for scope in scopes:
            for metric, value in scores.items():
                values[scope][metric].append(value)
            if (case.get('true_hazard') is not None and has_evidence(case, 'true_hazard', sources)
                    and pred.get('predicted_hazard') is not None):
                classification[scope].append((case['true_hazard'], pred['predicted_hazard']))
                diagnostic_rows[scope].append({'case_id': case['case_id'], 'true_hazard': case['true_hazard'],
                    'predicted_hazard': pred['predicted_hazard'], 'hazard_probabilities': pred.get('hazard_probabilities')})
    for scope, pairs in classification.items():
        classes = (*HAZARDS, 'none')
        ps, rs, fs = [], [], []
        for cls in classes:
            tp = sum(t == cls and p == cls for t, p in pairs)
            fp = sum(t != cls and p == cls for t, p in pairs)
            fn = sum(t == cls and p != cls for t, p in pairs)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            ps.append(p); rs.append(r); fs.append(2*p*r/(p+r) if p+r else 0.0)
        values[scope]['accuracy'] = [float(t == p) for t, p in pairs]
        for key, scores in (('macro_precision', ps), ('macro_recall', rs), ('macro_f1', fs)):
            values[scope][key] = [sum(scores)/4] * len(pairs)
    results, coverage = {}, {}
    for scope in ('overall', *HAZARDS):
        results[scope], coverage[scope] = {}, {}
        for metric in METRICS:
            numbers = values[scope][metric]
            results[scope][metric] = sum(numbers)/len(numbers) if numbers else None
            coverage[scope][metric] = {'eligible_count': len(numbers), 'test_count': samples[scope],
                'reason': None if numbers else 'No test cases with verified required annotations and completed predictions.'}
    from evaluation.diagnostics import classification_diagnostics, metric_distributions
    details = {'classification': {scope: classification_diagnostics(diagnostic_rows[scope], (*HAZARDS, 'none'))
               for scope in ('overall', *HAZARDS)}, 'metric_distributions': metric_distributions(values),
               'provenance': 'independent_evidence_benchmark',
               'notice': 'Scores use eligible reviewed references; inspect coverage and dataset readiness.'}
    aliases = dict(zip(METRICS, ('accuracy', 'precision', 'recall', 'f1',
        'haversine_dist_km', 'radius_abs_err_km', 'radius_rel_err', 'dap', 'ddr',
        'factual_correctness', 'recovery_necessity', 'strategic_completeness',
        'action_priority_precision', 'average_score', 'pixel_accuracy', 'iou', 'dice')))
    for scope in results:
        results[scope].update({alias: results[scope][key] for key, alias in aliases.items()})
    return {'results_by_hazard': results, 'sample_counts': dict(samples), 'metric_coverage': coverage,
            'detailed_evaluation': details,
            'metadata': {'evaluation_kind': 'independent_evidence_benchmark', 'generated_at': utc_now(),
                         'benchmark_status': 'research_ready' if validation['research_ready'] else 'incomplete',
                         'ground_truth_status': 'Only field-level verified references are scored',
                         'dataset_sha256': digest(root / 'annotations/cases.jsonl'),
                         'predictions_sha256': digest(predictions_path),
                         'aggregation': 'Case-macro mean; classification macro uses flood/wildfire/landslide/none',
                         'skipped_predictions': skipped, 'validation': validation}}
