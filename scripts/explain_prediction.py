"""Export hazard attribution figures. Does not train or compute benchmark accuracy."""
from __future__ import annotations
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def move_tensors(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key,item in value.items()}
    return value


def adapt_position_embedding(state, model):
    """Resize the checkpoint's square spatial positional grid; change no other weights."""
    import torch.nn.functional as F
    adapted = dict(state)
    key = 'backbone.patch_embed.spatial_pos_embed'
    old, current = adapted.get(key), model.state_dict().get(key)
    if old is None or current is None or old.shape == current.shape:
        return adapted
    if old.ndim != 4 or current.ndim != 4 or old.shape[:2] != current.shape[:2] or old.shape[-1] != current.shape[-1]:
        raise ValueError('Incompatible positional embedding checkpoint')
    before, after = int(old.shape[2]**.5), int(current.shape[2]**.5)
    if before*before != old.shape[2] or after*after != current.shape[2]:
        raise ValueError('Positional embeddings require square spatial grids')
    grid = old[0,0].transpose(0,1).reshape(1,old.shape[-1],before,before)
    resized = F.interpolate(grid,size=(after,after),mode='bicubic',align_corners=False)
    adapted[key] = resized.reshape(current.shape[-1],after*after).transpose(0,1).reshape_as(current)
    return adapted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-batch', type=Path, help='Tensor-only torch.save model input dictionary for ONE case')
    source.add_argument('--local-smoke', choices=['flood','wildfire','landslide'], help='Explicit generated-data demonstration')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/model_epoch_9.pt')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/model_config.yaml')
    parser.add_argument('--target', choices=['predicted','flood','wildfire','landslide'],default='predicted')
    parser.add_argument('--method', choices=['gradient_times_input','integrated_gradients'],default='gradient_times_input')
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--display-channels',nargs=3,type=int,default=[0,1,2],metavar=('R','G','B'))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'evaluation/explanations')
    args = parser.parse_args(argv)
    if not 2 <= args.steps <= 128:
        parser.error('--steps must be 2..128')
    import torch
    import numpy as np
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error('Install plotting dependencies with: .venv\\Scripts\\python.exe -m pip install -r requirements-evaluation.txt')
    from models.multihazard_vlm import MultiHazardVLM
    from evaluation.explainability import explain_image, HAZARDS
    from evaluation.benchmark import digest, utc_now
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    model = MultiHazardVLM(str(args.config)).to(device)
    # Tensor-only loading avoids arbitrary checkpoint deserialization. Incompatible
    # older pickled checkpoints must be converted by their trusted owner separately.
    checkpoint = torch.load(args.checkpoint,map_location=device,weights_only=True)
    state = checkpoint.get('model_state_dict',checkpoint) if isinstance(checkpoint,dict) else checkpoint
    model.load_state_dict(adapt_position_embedding(state,model),strict=True)
    source_metadata = {}
    if args.local_smoke:
        from data.datasets import S1S2WaterDataset,HLSBurnScarDataset,CASLandslideDataset
        from train import prepare_reconstruction_batch
        kind,path = {'flood':(S1S2WaterDataset,'s1s2_water'), 'wildfire':(HLSBurnScarDataset,'hls_burn_scars'), 'landslide':(CASLandslideDataset,'cas_landslide')}[args.local_smoke]
        dataset = kind(data_dir=str(ROOT/'datasets'/path/'train'))
        if not len(dataset):
            parser.error('Local demonstration sample is missing')
        prepared = prepare_reconstruction_batch([dataset[0]],model,device)
        inputs = prepared['model_inputs']
        source_metadata = {'source_kind':'generated_sample_smoke_test', 'hazard_branch':args.local_smoke,
            'missing_context_features':prepared.get('missing_geospatial_features',[]),
            'notice':'GENERATED INPUT / UNVERIFIED CHECKPOINT — EXPLAINABILITY DEMO ONLY'}
    else:
        inputs = torch.load(args.input_batch,map_location=device,weights_only=True)
        if not isinstance(inputs,dict):
            parser.error('--input-batch must contain a model input dictionary')
        source_metadata = {'source_kind':'user_supplied_preprocessed_inputs', 'input_file':str(args.input_batch.resolve()),
            'input_sha256':digest(args.input_batch),
            'notice':'MODEL ATTRIBUTION — INPUT PROVENANCE AND MODEL VALIDITY REQUIRE INDEPENDENT REVIEW'}
    inputs = move_tensors(inputs,device)
    if not isinstance(inputs.get('images'),torch.Tensor):
        parser.error('Input dictionary requires an images tensor')
    if inputs['images'].ndim != 5 or inputs['images'].shape[0] != 1:
        parser.error('images must have shape [1,T,C,H,W] for exactly one case')
    channels = args.display_channels
    if any(index < 0 or index >= inputs['images'].shape[2] for index in channels):
        parser.error('--display-channels indexes must exist in the image tensor')
    target = None if args.target=='predicted' else HAZARDS.index(args.target)
    result = explain_image(model,inputs,target=target,method=args.method,steps=args.steps)
    meta = {**result['metadata'],**source_metadata,'generated_at':utc_now(),
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint),
        'config_sha256':digest(args.config),'device':str(device),'display_channels':channels,
        'display_note':'Percentile-stretched model-input channels; false color unless explicitly mapped to RGB'}
    output = args.output_dir/(args.local_smoke or 'prepared_case')
    output.mkdir(parents=True,exist_ok=True)
    valid_frame = int(torch.nonzero(inputs['quality_flags'][0],as_tuple=False)[0,0])
    display = inputs['images'][0,valid_frame,channels].detach().cpu().float().numpy().transpose(1,2,0)
    for channel in range(3):
        low,high = np.percentile(display[:,:,channel],[2,98])
        display[:,:,channel] = np.clip((display[:,:,channel]-low)/(high-low),0,1) if high>low else 0
    heatmap = result['heatmap'].numpy()
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(display); axes[0].set_title(f'Model input · display channels {channels}')
    image = axes[1].imshow(heatmap,cmap='inferno',vmin=0,vmax=1)
    axes[1].set_title('Absolute image attribution · normalized')
    axes[2].imshow(display); axes[2].imshow(heatmap,cmap='inferno',vmin=0,vmax=1,alpha=.6*(heatmap>0))
    axes[2].set_title('Attribution overlay · NOT a hazard mask')
    for ax in axes: ax.axis('off')
    fig.colorbar(image,ax=axes[1],fraction=.045,label='Relative attribution [0,1]')
    fig.suptitle(meta['notice'],fontsize=11,color='#985a23')
    fig.text(.5,.025,f"Target: {meta['target_hazard']} | Method: {args.method} | Input: {meta['input_shape']} | Zero attribution: {meta['zero_attribution']}",ha='center',fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.12,wspace=.2)
    for extension in ('png','svg'):
        fig.savefig(output/f'attribution_comparison.{extension}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(9,4))
    shares = result['channel_shares'].numpy()
    ax.bar([f'C{i}' for i in range(len(shares))],shares,color='#287a55')
    ax.set(xlabel='Model input channel (verify sensor/band mapping)',ylabel='Fraction of absolute attribution',title='Relative channel attribution mass')
    fig.suptitle(meta['notice'],fontsize=9,color='#985a23')
    fig.tight_layout()
    fig.savefig(output/'channel_attribution.png',dpi=240)
    plt.close(fig)
    np.savez_compressed(output/'attributions.npz',signed=result['attribution'].numpy(),
                        heatmap=heatmap,raw_heatmap=result['raw_heatmap'].numpy(),channel_shares=shares)
    (output/'explanation.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
    explanation = (f"The model predicts {meta['predicted_hazard']}. The attribution targets {meta['target_hazard']}. "
        'Brighter pixels have greater absolute attribution to that class logit under the chosen method. '
        'This is a model-sensitivity explanation, not a ground-truth footprint or proof of correctness. '
        'Geographic context was held fixed. No LLM or event-specific response advice was generated.')
    (output/'explanation.txt').write_text(meta['notice']+'\n\n'+explanation,encoding='utf-8')
    page = '<!doctype html><html><head><meta charset="utf-8"><title>Hazard attribution</title><style>body{font:16px system-ui;margin:30px;background:#f5f7f5}img{max-width:100%;background:white}p{max-width:1000px;line-height:1.7}h1{color:#925820;font-size:23px}</style></head><body><h1>'+html.escape(meta['notice'])+'</h1><p>'+html.escape(explanation)+'</p><img src="attribution_comparison.png" alt="Image, attribution and overlay"><img src="channel_attribution.png" alt="Relative channel attribution"><p>Full provenance and limitations: explanation.json. Signed numerical arrays: attributions.npz.</p></body></html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    print(f'Attribution outputs saved to {output.resolve()}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
