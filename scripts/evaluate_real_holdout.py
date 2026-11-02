"""Acquire public labelled holdout subsets and test an existing classifier.
This is a source-domain transfer diagnostic, not a reviewed event benchmark.
No training, checkpoint selection, labels from predictions, or threshold tuning.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAZARDS = ['flood', 'wildfire', 'landslide']
SOURCES = {
 'flood': 'https://github.com/cloudtostreet/Sen1Floods11',
 'wildfire': 'https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars',
 'landslide': 'https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense',
}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def get_json(url):
    r = requests.get(url, timeout=(20,90)); r.raise_for_status(); return r.json()

def fetch(url, path, expected=None, max_bytes=64*2**20):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected is None or sha(path)==expected): return path
    partial = path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            total=0; last=0; started=time.monotonic()
            with requests.get(url,stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                with partial.open('wb') as out:
                    for chunk in r.iter_content(2**20):
                        total += len(chunk)
                        if total>max_bytes: raise ValueError('Download exceeds declared size bound')
                        out.write(chunk)
                        if total-last>128*2**20:
                            print(f'Download {path.name}: {total/2**20:.0f} MiB ({time.monotonic()-started:.0f}s)',flush=True); last=total
            if expected and sha(partial)!=expected: raise ValueError('Upstream SHA256 mismatch')
            partial.replace(path); return path
        except (requests.RequestException,ValueError):
            if attempt==2: raise
            time.sleep(2)

def record(root, hazard, ident, split, files, urls, revision=None):
    return dict(case_id=f'{hazard}_{ident}', hazard=hazard, upstream_split=split,
      source=SOURCES[hazard], revision=revision,
      files={k:str(p.relative_to(root)) for k,p in files.items()},
      assets={k:dict(sha256=sha(p),bytes=p.stat().st_size,url=urls[k]) for k,p in files.items()})

def acquire_flood(root, count, seed):
    base='https://storage.googleapis.com/sen1floods11/'
    split_url=base+'v1.1/splits/flood_handlabeled/flood_test_data.csv'
    split=fetch(split_url,root/'flood'/'official_test.csv')
    rows=list(csv.reader(split.read_text().splitlines())); random.Random(seed).shuffle(rows)
    def one(row):
        name=next(x.strip() for x in row if '_S1Hand.tif' in x)
        ident=Path(name).name.replace('_S1Hand.tif','')
        files={}; urls={}
        for key,layer in [('s1','S1Hand'),('s2','S2Hand'),('mask','LabelHand')]:
            filename=f'{ident}_{layer}.tif'; url=base+f'v1.1/data/flood_events/HandLabeled/{layer}/{filename}'
            files[key]=fetch(url,root/'flood'/filename); urls[key]=url
        return record(root,'flood',ident,'official_test',files,urls)
    selected=rows[:count]
    with ThreadPoolExecutor(max_workers=3) as pool: result=list(pool.map(one,selected))
    print(f'Acquired {len(result)} flood-event-source test patches',flush=True); return result

def acquire_landslide(root,count,seed):
    repo='ibm-nasa-geospatial/Landslide4sense'; revision=get_json(f'https://huggingface.co/api/datasets/{repo}')['sha']
    numbers=list(range(1,801)); random.Random(seed).shuffle(numbers)
    def one(number):
        files={}; urls={}
        for key,folder,prefix in [('image','images','image'),('mask','annotations','mask')]:
            name=f'{prefix}_{number}.h5'; url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{folder}/test/{name}'
            files[key]=fetch(url,root/'landslide'/name); urls[key]=url
        return record(root,'landslide',str(number),'official_test',files,urls,revision)
    with ThreadPoolExecutor(max_workers=3) as pool: result=list(pool.map(one,numbers[:count]))
    print(f'Acquired {len(result)} Landslide4Sense test pairs',flush=True); return result

def acquire_fire(root,count,seed):
    import pyarrow.parquet as pq
    repo='ibm-nasa-geospatial/hls_burn_scars'
    revision=get_json(f'https://huggingface.co/api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet')['sha']
    objects=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/hls_burn_scars/test?limit=10')
    # The upstream dataset script aliases validation as "test". Do not claim a distinct test split.
    obj=next(x for x in objects if x['path'].endswith('/0000.parquet'))
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{obj["path"]}'
    destination=root/'wildfire'/'official_validation_0000.parquet'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or sha(destination)!=obj['lfs']['oid']:
        size=obj['size']; block=16*2**20
        ranges=[(start,min(start+block-1,size-1)) for start in range(0,size,block)]
        def part(pair):
            start,end=pair; path=destination.with_suffix(f'.part_{start}')
            if path.exists() and path.stat().st_size==end-start+1:return path
            for attempt in range(3):
                try:
                    with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,120),stream=True) as r:
                        r.raise_for_status()
                        if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                            raise ValueError('Server did not honor exact byte range')
                        total=0
                        with path.open('wb') as out:
                            for data in r.iter_content(2**20):
                                total+=len(data)
                                if total>end-start+1:raise ValueError('Range exceeded')
                                out.write(data)
                    if total!=end-start+1:raise ValueError('Incomplete range')
                    print(f'HLS validation range {start//block+1}/{len(ranges)} saved',flush=True)
                    return path
                except (requests.RequestException,ValueError):
                    if attempt==2:raise
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(part,ranges))
        with destination.open('wb') as out:
            for path in parts:
                with path.open('rb') as inp:
                    while data:=inp.read(2**20):out.write(data)
        if sha(destination)!=obj['lfs']['oid']:raise ValueError('Converted HLS shard SHA256 mismatch')
        for path in parts:path.unlink()
    table=pq.read_table(destination)
    indices=list(range(table.num_rows));random.Random(seed).shuffle(indices)
    result=[]
    for index in indices[:count]:
        row=table.slice(index,1).to_pylist()[0]; files={};urls={}
        for key,column in [('image','image'),('mask','annotation')]:
            cell=row[column];name=Path(cell['path']).name
            path=root/'wildfire'/name
            if not cell['bytes']:raise ValueError('Converted dataset omitted original raster bytes')
            path.write_bytes(cell['bytes']);files[key]=path;urls[key]=url+f'#row={index}&column={column}'
        rec=record(root,'wildfire',Path(files['image']).stem,'official_validation',files,urls,revision)
        rec['selection_population']='fixed-seed sample from first 100 validation rows in converted shard 0000; not full validation population'
        rec['container_sha256']=obj['lfs']['oid'];result.append(rec)
    save(root/'wildfire'/'source_conversion.json',dict(revision=revision,source_url=url,sha256=obj['lfs']['oid'],rows=table.num_rows,
         split_note='Upstream conversion test is an alias of official validation, not an independent test split.',selection_rows=indices[:count]))
    print(f'Acquired {len(result)} HLS validation pairs from verified official conversion',flush=True)
    return result

def acquire(args):
    root=args.data_root.resolve(); root.mkdir(parents=True,exist_ok=True)
    results=[]; errors={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={h:pool.submit(fn,root,args.per_hazard,args.seed) for h,fn in [('flood',acquire_flood),('wildfire',acquire_fire),('landslide',acquire_landslide)]}
        for h,future in futures.items():
            try: results.extend(future.result())
            except Exception as e: errors[h]=str(e); print(f'Acquisition failed for {h}: {e}',flush=True)
    save(root/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),seed=args.seed,requested_per_hazard=args.per_hazard,
      selection='Fixed-seed sample of upstream split; no model predictions used in selection',records=results,errors=errors,sources=SOURCES))
    if errors: raise RuntimeError(f'Acquisition incomplete: {errors}')

def load_sample(root,row):
    import numpy as np, rasterio, h5py
    def raster(key):
        with rasterio.open(root/row['files'][key]) as ds: return ds.read(masked=True).astype('float32').filled(np.nan)
    hazard=row['hazard']
    if hazard=='landslide':
        with h5py.File(root/row['files']['image'],'r') as f: raw=f['img'][:].astype('float32')
        with h5py.File(root/row['files']['mask'],'r') as f: mask=f['mask'][:]
        # Undo the authors' published per-band mean division, then convert DN to reflectance.
        raw=np.moveaxis(raw,-1,0) if raw.shape[-1]==14 else raw
        image=np.zeros((10,*raw.shape[1:]),dtype='float32'); image[:3]=raw[[3,2,1]]*np.array([445.17289745,663.41636217,824.63171476],dtype='float32')[:,None,None]/10000.
        valid=np.isfinite(raw[[3,2,1]]).all(axis=0)
        description='L4S RGB B4/B3/B2: undo published mean division (445.17289745,663.41636217,824.63171476), then divide recovered DN by 10000; other slots zero. Source: authors issue 4 comment 1195457420. Transfer adapter, not known historical training transform.'
    elif hazard=='flood':
        s1=raster('s1'); s2=raster('s2'); mask=raster('mask')[0]
        image=np.zeros((10,*s1.shape[1:]),dtype='float32')
        image[:2]=s1[:2]/100.; image[2:8]=s2[[1,2,3,7,11,12]]/10000.
        valid=np.isfinite(s1).all(axis=0)&np.isfinite(s2).all(axis=0)
        description='VV,VH/100 in slots 0:2; S2 B2,B3,B4,B8,B11,B12/10000 in slots 2:8; remaining slots zero.'
    else:
        raw=raster('image'); mask=raster('mask')[0]
        image=np.zeros((10,*raw.shape[1:]),dtype='float32'); image[:6]=raw[:6]
        valid=np.isfinite(raw).all(axis=0)
        description='Published HLS already-reflectance six bands in slots 0:6 without double scaling; remaining slots zero.'
    labels_valid=np.isin(mask,[0,1]) & valid; positive=int(((mask==1)&labels_valid).sum()); valid_count=int(labels_valid.sum())
    if not positive: raise ValueError('No positive annotated pixels: not eligible for positive-only three-class classification')
    image[:,~valid]=0; image=np.nan_to_num(image,nan=0.,posinf=0.,neginf=0.)
    return image,dict(positive_pixels=positive,valid_label_pixels=valid_count,positive_fraction=positive/valid_count,input_adapter=description,
      input_min=float(image.min()),input_max=float(image.max()))

def evaluate(args):
    import numpy as np, torch, torch.nn.functional as F, yaml
    from models.multihazard_vlm import MultiHazardVLM
    from evaluation.diagnostics import classification_diagnostics
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(4)
    root=args.data_root.resolve(); output=args.output.resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['errors']: raise ValueError('Cannot report complete evaluation with failed acquisitions')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state=checkpoint.get('model_state_dict',checkpoint)
    cfg=yaml.safe_load((ROOT/'configs/model_config.yaml').read_text()); cfg['model']['segmentation_enabled']=False
    if cfg['model'].get('hazard_classes') != HAZARDS:
        raise ValueError('Configured class order does not match evaluation labels')
    cfg_path=output.parent/'inference_config.yaml'; cfg_path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    model=MultiHazardVLM(str(cfg_path))
    # Only architecture adaptation: same interpolation as API for deployment grid.
    key='backbone.patch_embed.spatial_pos_embed'; old=state[key]; target=model.state_dict()[key]
    adaptation=None
    if old.shape!=target.shape:
        old_side=int(old.shape[2]**.5); new_side=int(target.shape[2]**.5)
        x=old.squeeze(0).squeeze(0).T.reshape(1,old.shape[-1],old_side,old_side)
        x=F.interpolate(x,size=(new_side,new_side),mode='bicubic',align_corners=False)
        state=dict(state); state[key]=x.reshape(old.shape[-1],-1).T.reshape_as(target)
        adaptation=dict(from_tokens=old.shape[2],to_tokens=target.shape[2],method='bicubic, align_corners=False; matches API deployment adaptation')
    model.load_state_dict(state,strict=True); model.eval()
    known_hashes={sha(p):str(p) for p in (ROOT/'datasets').rglob('*.tif')}
    rows=[]; excluded=[]; counts={h:0 for h in HAZARDS}; started=time.monotonic()
    seen_ids=set(); seen_images=set()
    for row in manifest['records']:
        try:
            image_signature=tuple(sorted(asset['sha256'] for key,asset in row['assets'].items() if key!='mask'))
            if row['case_id'] in seen_ids or image_signature in seen_images:
                raise ValueError('Duplicate case or exact image combination inside evaluation manifest')
            seen_ids.add(row['case_id']); seen_images.add(image_signature)
            for key,asset in row['assets'].items():
                path=root/row['files'][key]
                if sha(path)!=asset['sha256']: raise ValueError('Asset changed since acquisition')
                if key!='mask' and asset['sha256'] in known_hashes: raise ValueError('Exact duplicate of known local training asset')
            image,info=load_sample(root,row)
        except ValueError as exc:
            excluded.append(dict(case_id=row['case_id'],reason=str(exc))); continue
        x=F.interpolate(torch.from_numpy(image).unsqueeze(0),size=(64,64),mode='bilinear',align_corners=False)
        images=x.unsqueeze(1).repeat(1,4,1,1,1)
        # Empty H3 sequences and zero terrain are an explicit missing-context ablation.
        h3={res:torch.empty((1,0),dtype=torch.long) for res in (4,6,8)}
        geo={name:torch.zeros((1,1)) for name in ('elevation','slope','ndvi','aspect')}; geo['lithology_class']=torch.zeros(1,dtype=torch.long)
        with torch.inference_mode():
            visual,_=model.backbone(images,torch.ones((1,4),dtype=torch.bool))
            geographic,geo_mask=model.prompt_encoder(h3,geo)
            _,fused=model.fusion(visual,geographic,geo_mask)
            logits,probs=model.classification_head(fused)
        p=probs[0].tolist()
        if not np.isfinite(p).all(): raise ValueError('Nonfinite prediction')
        predicted=HAZARDS[int(probs[0].argmax())]; hazard=row['hazard']; counts[hazard]+=1
        rows.append(dict(case_id=row['case_id'],true_hazard=hazard,predicted_hazard=predicted,
          hazard_probabilities=dict(zip(HAZARDS,p)),confidence=max(p),upstream_split=row['upstream_split'],source=row['source'],label_details=info))
        print(f"{len(rows)} / {len(manifest['records'])}: {hazard} -> {predicted} ({max(p):.3f})",flush=True)
        save(output.parent/'predictions.partial.json',rows)
    if any(v==0 for v in counts.values()): raise ValueError(f'Missing eligible hazard class: {counts}')
    detail=classification_diagnostics(rows,HAZARDS)
    for key in ('precision', 'recall', 'f1'):
        detail['summary']['macro_' + key] = sum((r[key] or 0.) for r in detail['per_class']) / len(HAZARDS)
    notices=[
      'Real public imagery, independently published pixel annotations; classification evaluated as source-domain task recognition.',
      'Flood means flood-event water-source image, not verified new inundation; wildfire means burn scar, not active fire.',
      'Only positive annotated patches; no non-event detection evaluated. Mask-empty patches excluded explicitly.',
      'Each hazard uses a different dataset/sensor adapter: source and modality confounding limit interpretation.',
      'No training manifest exists in the selected bare state_dict checkpoint; complete historical training exclusion cannot be proven.',
      'Exact file SHA256 duplicates checked against known local datasets/ training assets; no event/region independence claim.',
      f"HLS official validation subset is evaluation only: {sum(row['hazard']=='wildfire' for row in manifest['records'])} fixed-seed source pairs from first 100-row converted shard; not event-held-out.",
      'L4S RGB reflectance adapter differs from historical 8-bit RGB training assumptions. HLS reflectance is not divided by 10000 again.',
      'Missing-context ablation: empty H3 sequences, zero-filled geographic attributes; 4 repeated single-date frames, 64px deployment grid.',
      'Segmentation, geographic regression, vulnerability and language quality not evaluated; no validated trained outputs/references for these tasks.'
    ]
    result=dict(metadata=dict(evaluation_kind='real_public_data_transfer_test',created_at=datetime.now(timezone.utc).isoformat(),
      checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),seed=args.seed,device='cpu',
      input_size=64,frames=4,position_embedding_adaptation=adaptation,classification_only=True,
      acquisition_manifest_sha256=sha(root/'manifest.json'),requested_per_hazard=manifest['requested_per_hazard'],
      acquired_pairs_by_hazard={h:sum(row['hazard']==h for row in manifest['records']) for h in HAZARDS},
      total_inference_seconds=time.monotonic()-started,training_overlap_status='exact local training-file duplicates excluded; historical training lineage unavailable',
      per_sample=[dict(r,target_class=r['true_hazard']) for r in rows],limitations=notices),
      sample_counts=dict(overall=len(rows),**counts),excluded=excluded,
      detailed_evaluation=dict(classification=dict(overall=detail),metric_distributions={},provenance='real_public_data_transfer_test',notice='REAL-DATA TRANSFER TEST - limited sample, incomplete training lineage; not an operational hazard benchmark.'))
    save(output,result)
    with (output.parent/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(output),counts=counts,overall=detail['summary']),indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['acquire','evaluate','all'],default='all')
    p.add_argument('--data-root',type=Path,default=ROOT/'dataset/real_holdout')
    p.add_argument('--output',type=Path,default=ROOT/'evaluation/real_transfer/metrics_report.json')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/model_epoch_9.pt')
    p.add_argument('--per-hazard',type=int,default=30); p.add_argument('--seed',type=int,default=20261027)
    a=p.parse_args()
    if not 1<=a.per_hazard<=100: p.error('--per-hazard must be between 1 and 100')
    if a.phase in ('all','acquire'): acquire(a)
    if a.phase in ('all','evaluate'): evaluate(a)
if __name__=='__main__': main()
