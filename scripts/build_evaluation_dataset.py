"""Build and audit an evidence-first multi-hazard dataset.

Examples:
  python scripts/build_evaluation_dataset.py init
  python scripts/build_evaluation_dataset.py acquire --per-hazard 2
  python scripts/build_evaluation_dataset.py validate
  python scripts/build_evaluation_dataset.py import --cases reviewed_cases.jsonl
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmark import (HAZARDS, SPLITS, digest, dump_json, local_path,
    new_case, new_prediction, read_jsonl, utc_now, validate_dataset, write_jsonl)

SOURCES = [
    {'source_id': 'sen1floods11', 'organization': 'Cloud to Street / dataset authors',
     'url': 'https://github.com/cloudtostreet/Sen1Floods11',
     'data_url': 'https://storage.googleapis.com/sen1floods11/v1.1/',
     'license': 'See upstream license; retain downloaded source documentation',
     'label_method': 'Published hand-labeled water masks',
     'independence': 'Human labels; exclude Otsu and weak model labels',
     'limitations': ['Water includes permanent water; cannot establish flood presence or inundation footprint.',
                    'Not accepted as flood truth without independent event-water verification.']},
    {'source_id': 'hls_burn_scars', 'organization': 'NASA IMPACT / IBM-NASA dataset authors',
     'url': 'https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars',
     'license': 'CC-BY-4.0', 'doi': '10.57967/hf/0956',
     'label_method': 'MTBS burn scar polygons; manual image and scar screening',
     'independence': 'MTBS reference labels; no project model output used',
     'limitations': ['Burn scars describe past burning, not active fire or response urgency.',
                    'Upstream random train/validation split is not an independent event holdout.',
                    'Severity, event radius, expert response labels require further evidence.']},
    {'source_id': 'landslide4sense', 'organization': 'IARAI / Landslide4Sense authors; IBM-NASA mirror',
     'url': 'https://github.com/iarai/Landslide4Sense-2022',
     'data_url': 'https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense',
     'license': 'Not declared in mirror card; confirm upstream data redistribution terms',
     'label_method': 'Published pixel-level landslide inventory labels',
     'independence': 'Published annotations; no project model output used',
     'limitations': ['HDF5 patches do not supply reliable per-patch CRS/transform in the card.',
                    'No guessed georeferencing is assigned; patches stay outside geospatial benchmark.',
                    'Event and region identities require upstream recovery before split allocation.']},
    {'source_id': 'nrsc_wayanad', 'organization': 'NRSC / ISRO / NDEM',
     'url': 'https://ndem.nrsc.gov.in/documents/Disaster_Document/2024/klwayanadlandslide_31072024.pdf',
     'license': 'Confirm official product terms before redistribution',
     'label_method': 'Official event documentation; not a pixel label',
     'limitations': ['Narrative/map evidence alone cannot supply aligned raster masks.']},
]


def init(root):
    root.mkdir(parents=True, exist_ok=True)
    for directory in [*(f'images/{h}' for h in HAZARDS), 'masks', 'validity',
                      'annotations', 'metadata/evidence', 'splits', 'predictions', 'raw']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    source_path = root / 'metadata/sources.json'
    if not source_path.exists():
        dump_json(source_path, {'schema_version': 1, 'checked_at': utc_now(), 'sources': SOURCES})
    for file in ['annotations/cases.jsonl', 'predictions/predictions.jsonl', *(f'splits/{s}.txt' for s in SPLITS)]:
        path = root / file
        if not path.exists():
            path.write_text('', encoding='utf-8')
    if not (root / 'annotations/case_template.json').exists():
        dump_json(root / 'annotations/case_template.json', new_case('REPLACE_WITH_UNIQUE_ID', 'flood'))
        dump_json(root / 'predictions/prediction_template.json', new_prediction('REPLACE_WITH_UNIQUE_ID'))


def get(session, url, **kwargs):
    response = session.get(url, timeout=(20, 90), **kwargs)
    response.raise_for_status()
    return response


def download(session, url, path, expected_sha=None, expected_md5=None, max_bytes=128*1024*1024):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected_sha and digest(path) == expected_sha:
        return
    partial = path.with_suffix(path.suffix + '.partial')
    total = 0
    try:
        with get(session, url, stream=True) as response, partial.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError('Source file exceeds download size limit')
                stream.write(chunk)
        if expected_sha and digest(partial) != expected_sha:
            raise ValueError('Upstream SHA-256 mismatch')
        if expected_md5 and base64.b64encode(hashlib.md5(partial.read_bytes()).digest()).decode() != expected_md5:
            raise ValueError('Upstream MD5 mismatch')
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def evidence(source, reference, method='source_metadata'):
    return {'source_id': source, 'reference': reference, 'method': method, 'status': 'verified'}


def assign(case, field, value, source, reference, method='source_metadata'):
    case[field] = value
    case['evidence'][field] = evidence(source, reference, method)
    case['missing_reasons'].pop(field, None)


def asset(root, case, field, path, source, reference):
    assign(case, field, path.relative_to(root).as_posix(), source, reference, 'published_label' if field != 'image_path' else 'source_metadata')
    case['assets'][field] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                             'source_id': source, 'reference': reference}


def raster_metadata(root, case, source, reference):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(root / case['image_path']) as raster:
        if not raster.crs or raster.transform.is_identity:
            return
        bbox = list(transform_bounds(raster.crs, 'EPSG:4326', *raster.bounds))
        assign(case, 'bbox', bbox, source, reference, 'documented_derivation')
        assign(case, 'crs', raster.crs.to_string(), source, reference)
        # The observation position is the image center, never an event epicenter.
        assign(case, 'latitude', (bbox[1]+bbox[3])/2, source, 'WGS84 bounding-box midpoint from image grid', 'documented_derivation')
        assign(case, 'longitude', (bbox[0]+bbox[2])/2, source, 'WGS84 bounding-box midpoint from image grid', 'documented_derivation')
        if raster.crs.is_projected and raster.crs.linear_units == 'metre':
            assign(case, 'spatial_resolution_m', list(raster.res), source, reference)


def binary_mask(root, case, source, raw_mask, reference):
    import numpy as np
    import rasterio
    with rasterio.open(raw_mask) as mask, rasterio.open(root / case['image_path']) as image:
        if not image.crs or mask.crs != image.crs or mask.shape != image.shape or not mask.transform.almost_equals(image.transform):
            raise ValueError('Published image and mask do not have aligned georeferencing')
        values = mask.read(1)
        valid = (mask.read_masks(1) > 0) & np.isin(values, (0, 1)) & (image.dataset_mask() > 0)
        if not valid.any():
            raise ValueError('Published image has no valid labeled pixels')
        profile = mask.profile.copy()
        profile.update(dtype='uint8', count=1, nodata=255, compress='deflate')
        target = root / 'masks' / f'{case["case_id"]}.tif'
        validity = root / 'validity' / f'{case["case_id"]}.tif'
        with rasterio.open(target, 'w', **profile) as dst:
            dst.write(np.where(valid, values, 255).astype('uint8'), 1)
        profile['nodata'] = None
        with rasterio.open(validity, 'w', **profile) as dst:
            dst.write(valid.astype('uint8'), 1)
        asset(root, case, 'mask_path', target, source, reference)
        asset(root, case, 'validity_mask_path', validity, source, reference)
        with rasterio.open(target) as raster:
            meta = {'crs': raster.crs.to_string(), 'width': raster.width, 'height': raster.height,
                    'transform': list(raster.transform)[:6], 'resolution': list(raster.res),
                    'nodata': raster.nodata, 'classes': {'0': 'background', '1': 'burn_scar'},
                    'validity': '1=evaluate, 0=ignore; image and target source no-data excluded'}
        assign(case, 'mask_metadata', meta, source, 'Binary conversion of published -1/0/1 labels; -1 mapped to 255 nodata', 'documented_derivation')
        present = bool((values[valid] == 1).any())
        assign(case, 'hazard_present', present, source, reference, 'published_label')
        assign(case, 'true_hazard', 'wildfire' if present else 'none', source, reference, 'published_label')


def acquire_flood(session, root, count):
    base = 'https://storage.googleapis.com/sen1floods11/'
    response = get(session, 'https://storage.googleapis.com/storage/v1/b/sen1floods11/o',
                   params={'prefix': 'v1.1/data/flood_events/HandLabeled/LabelHand/', 'maxResults': 1000}).json()
    dump_json(root / 'metadata/evidence/sen1floods11_objects.json', response)
    candidates = sorted(response.get('items', []), key=lambda x: ('India' not in x['name'], x['name']))[:count]
    result = []
    for obj in candidates:
        stem = Path(obj['name']).name.replace('_LabelHand.tif', '')
        case = new_case('sen1floods11_' + stem.lower(), 'flood')
        case['upstream_event_group'] = stem.split('_')[0]
        case['exclusion_reasons'] = ['Water/no-water labels include permanent water; flood-specific truth not established.',
                                     'Independent event/region holdout and per-case narrative annotations not reviewed.']
        image_name = obj['name'].replace('/LabelHand/', '/S1Hand/').replace('_LabelHand.tif', '_S1Hand.tif')
        image = root / 'images/flood' / f'{case["case_id"]}.tif'
        raw = root / 'raw/sen1floods11' / Path(obj['name']).name
        download(session, base + image_name, image)
        download(session, base + obj['name'] + '?generation=' + obj['generation'], raw, expected_md5=obj['md5Hash'])
        asset(root, case, 'image_path', image, 'sen1floods11', base + image_name)
        case['raw_label'] = {'path': raw.relative_to(root).as_posix(), 'sha256': digest(raw),
                             'reference': base + obj['name'], 'semantics': 'water, not independently verified flood footprint'}
        raster_metadata(root, case, 'sen1floods11', base + image_name)
        assign(case, 'source_satellite', 'Sentinel-1', 'sen1floods11', SOURCES[0]['url'])
        assign(case, 'source_organization', 'ESA / Copernicus; Cloud to Street dataset', 'sen1floods11', SOURCES[0]['url'])
        result.append(case)
        print(f'Staged flood-source candidate {case["case_id"]}', flush=True)
    return result


def acquire_landslide(session, root, count):
    repo = 'ibm-nasa-geospatial/Landslide4sense'
    info = get(session, f'https://huggingface.co/api/datasets/{repo}').json()
    revision = info['sha']
    objects = get(session, f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/images/test',
                  params={'limit': min(count, 1000)}).json()
    dump_json(root / 'metadata/evidence/landslide4sense_revision.json', {'revision': revision, 'objects': objects})
    result = []
    for obj in objects[:count]:
        if obj['type'] != 'file':
            continue
        number = Path(obj['path']).stem.removeprefix('image_')
        case = new_case(f'landslide4sense_test_{number}', 'landslide')
        case['upstream_split'] = 'test'
        case['exclusion_reasons'] = ['HDF5 has no verified patch georeferencing/event identity; no CRS or coordinates invented.',
                                     'Upstream test membership alone cannot certify independence from project model training.']
        for kind, name in [('images', f'image_{number}.h5'), ('annotations', f'mask_{number}.h5')]:
            remote = f'{kind}/test/{name}'
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}'
            path = root / ('images/landslide' if kind == 'images' else 'raw/landslide4sense') / f'{case["case_id"]}_{name}'
            expected = obj.get('lfs', {}).get('oid') if kind == 'images' else None
            download(session, url, path, expected_sha=expected)
            if kind == 'images':
                asset(root, case, 'image_path', path, 'landslide4sense', url)
            else:
                case['raw_label'] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path),
                                     'reference': url, 'semantics': 'published landslide binary HDF5; georeferencing unavailable'}
        assign(case, 'source_satellite', 'Sentinel-2 plus ALOS PALSAR ancillary layers', 'landslide4sense', SOURCES[2]['data_url'])
        assign(case, 'source_organization', 'ESA / IARAI dataset authors; IBM-NASA mirror', 'landslide4sense', SOURCES[2]['data_url'])
        result.append(case)
        print(f'Staged landslide-source candidate {case["case_id"]}', flush=True)
    return result


class LimitedReader:
    def __init__(self, raw, limit):
        self.raw, self.limit, self.used = raw, limit, 0
    def read(self, size=-1):
        block = self.raw.read(size)
        self.used += len(block)
        if self.used > self.limit:
            raise ValueError('Archive streaming budget reached; retained source assets remain candidates')
        return block


def acquire_fire(session, root, count, archive_mb):
    repo = 'ibm-nasa-geospatial/hls_burn_scars'
    revision_path = root / 'metadata/evidence/hls_revision.json'
    existing = json.loads(revision_path.read_text(encoding='utf-8')) if revision_path.exists() else {}
    retained = 0
    for image in sorted((root / 'raw/hls_burn_scars').glob('*_merged.tif')):
        key = image.name.removesuffix('_merged.tif')
        mask = image.with_name(key + '.mask.tif')
        if not mask.exists() or not existing.get('url'):
            continue
        case = new_case('hls_' + key.lower(), 'wildfire')
        target = root / 'images/wildfire' / f'{case["case_id"]}.tif'
        shutil.copyfile(image, target)
        ref = existing['url'] + '#basename=' + image.name
        asset(root, case, 'image_path', target, 'hls_burn_scars', ref)
        raster_metadata(root, case, 'hls_burn_scars', ref)
        binary_mask(root, case, 'hls_burn_scars', mask, existing['url'] + '#basename=' + mask.name)
        assign(case, 'source_satellite', 'Harmonized Landsat / Sentinel-2 (specific platform not recovered)', 'hls_burn_scars', SOURCES[1]['url'])
        assign(case, 'source_organization', 'NASA HLS / NASA IMPACT; MTBS reference labels', 'hls_burn_scars', SOURCES[1]['url'])
        case['exclusion_reasons'] = ['Event/region grouping and training exposure require review.',
                                     'Past burn scar only; event severity, response, and exact timing not established.']
        yield case
        retained += 1
        if retained >= count:
            return
    revision = get(session, f'https://huggingface.co/api/datasets/{repo}').json()['sha']
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/hls_burn_scars.tar.gz'
    dump_json(root / 'metadata/evidence/hls_revision.json', {'revision': revision, 'url': url,
              'note': 'Bounded streaming extraction; archive checksum cannot be verified unless entire archive is read.'})
    found, pairs = {}, {}
    result = []
    with get(session, url, stream=True) as response:
        with tarfile.open(fileobj=LimitedReader(response.raw, archive_mb*1024*1024), mode='r|gz') as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or member.size > 64*1024*1024:
                    continue
                if name.endswith('_merged.tif'):
                    key, kind = name.removesuffix('_merged.tif'), 'image'
                elif name.endswith('.mask.tif'):
                    key, kind = name.removesuffix('.mask.tif'), 'mask'
                else:
                    continue
                # Never extract archive paths; retain only basename under controlled root.
                path = root / 'raw/hls_burn_scars' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                found.setdefault(key, {})[kind] = (path, member.name)
                if len(found[key]) == 2 and key not in pairs:
                    pairs[key] = True
                    case = new_case('hls_' + key.lower(), 'wildfire')
                    image, member_name = found[key]['image']
                    target = root / 'images/wildfire' / f'{case["case_id"]}.tif'
                    shutil.copyfile(image, target)
                    ref = url + '#member=' + member_name
                    asset(root, case, 'image_path', target, 'hls_burn_scars', ref)
                    raster_metadata(root, case, 'hls_burn_scars', ref)
                    binary_mask(root, case, 'hls_burn_scars', found[key]['mask'][0], url + '#member=' + found[key]['mask'][1])
                    assign(case, 'source_satellite', 'Harmonized Landsat / Sentinel-2 (specific platform not recovered)', 'hls_burn_scars', SOURCES[1]['url'])
                    assign(case, 'source_organization', 'NASA HLS / NASA IMPACT; MTBS reference labels', 'hls_burn_scars', SOURCES[1]['url'])
                    case['exclusion_reasons'] = ['Event/region grouping and training exposure need review; upstream random split rejected.',
                                                 'Burn scar is not active fire; acquisition timestamp, severity, radius, and expert response remain unverified.']
                    yield case
                    print(f'Staged wildfire-source candidate {case["case_id"]}', flush=True)
                    if len(pairs) >= count:
                        return


def save(root, cases):
    write_jsonl(root / 'annotations/cases.jsonl', cases)
    old = {p['case_id']: p for p in read_jsonl(root / 'predictions/predictions.jsonl')}
    write_jsonl(root / 'predictions/predictions.jsonl', [old.get(c['case_id'], new_prediction(c['case_id'])) for c in cases])
    for split in SPLITS:
        (root / 'splits' / f'{split}.txt').write_text(''.join(c['case_id']+'\n' for c in cases if c.get('split') == split), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'acquire', 'validate', 'import'])
    parser.add_argument('--root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--per-hazard', type=int, default=2)
    parser.add_argument('--archive-budget-mb', type=int, default=128)
    parser.add_argument('--hazards', nargs='+', choices=HAZARDS, default=list(HAZARDS))
    parser.add_argument('--cases', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    init(root)
    failures = []
    if args.command == 'acquire':
        if not 1 <= args.per_hazard <= 1000 or not 1 <= args.archive_budget_mb <= 4096:
            parser.error('per-hazard must be 1..1000; archive-budget-mb must be 1..4096')
        import requests
        session = requests.Session()
        session.headers['User-Agent'] = 'MultiHazard-research-dataset/1.0'
        cases = {c['case_id']: c for c in read_jsonl(root / 'annotations/cases.jsonl')}
        snapshots = []
        for name, url in [
            ('sen1floods11_README.md', 'https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/README.md'),
            ('hls_burn_scars_README.md', 'https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/raw/main/README.md'),
            ('landslide4sense_README.md', 'https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/raw/main/README.md')]:
            try:
                path = root / 'metadata/evidence' / name
                download(session, url, path, max_bytes=2*1024*1024)
                snapshots.append({'url': url, 'path': path.relative_to(root).as_posix(),
                                  'sha256': digest(path), 'retrieved_at': utc_now()})
            except (requests.RequestException, ValueError, OSError) as exc:
                failures.append({'source_document': url, 'error': str(exc)})
        dump_json(root / 'metadata/evidence/source_snapshots.json', snapshots)
        for hazard, function in [('flood', acquire_flood), ('landslide', acquire_landslide), ('wildfire', acquire_fire)]:
            if hazard not in args.hazards:
                continue
            try:
                kwargs = {'archive_mb': args.archive_budget_mb} if hazard == 'wildfire' else {}
                for case in function(session, root, args.per_hazard, **kwargs):
                    if case['case_id'] not in cases:
                        cases[case['case_id']] = case
                    save(root, list(cases.values()))
                save(root, list(cases.values()))
            except (requests.RequestException, ValueError, OSError, tarfile.TarError) as exc:
                failures.append({'hazard': hazard, 'error': str(exc)})
                print(f'{hazard}: acquisition incomplete: {exc}', flush=True)
        dump_json(root / 'metadata/acquisition_report.json', {'generated_at': utc_now(), 'failures': failures,
                  'case_count': len(cases), 'status': 'candidates_only; expert review and evidence still required'})
    elif args.command == 'import':
        if not args.cases:
            parser.error('--cases is required for import')
        incoming = read_jsonl(args.cases)
        current = {c['case_id']: c for c in read_jsonl(root / 'annotations/cases.jsonl')}
        if len({c['case_id'] for c in incoming}) != len(incoming):
            raise ValueError('Duplicate incoming case IDs')
        for case in incoming:
            current[case['case_id']] = case
        # Validate in an isolated metadata staging directory before replacing the registry.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='hazard_review_') as temp:
            stage = Path(temp)
            init(stage)
            shutil.copyfile(root / 'metadata/sources.json', stage / 'metadata/sources.json')
            for directory in ('images', 'masks', 'validity', 'raw'):
                shutil.copytree(root / directory, stage / directory, dirs_exist_ok=True)
            save(stage, list(current.values()))
            report = validate_dataset(stage)
            if report['errors']:
                raise ValueError('Import rejected: ' + '; '.join(report['errors'][:20]))
        save(root, list(current.values()))
    report = validate_dataset(root)
    dump_json(root / 'metadata/validation_report.json', report)
    print(json.dumps(report, indent=2))
    return 2 if report['errors'] or failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
