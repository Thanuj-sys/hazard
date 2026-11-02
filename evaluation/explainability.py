"""Image attribution for the existing hazard classifier (not segmentation truth)."""
from __future__ import annotations
import torch

HAZARDS = ('flood', 'wildfire', 'landslide')


def classification_logits(model, inputs):
    """Use the same classification path as forward, skipping unrelated decoder heads."""
    visual, _ = model.backbone(inputs['images'], inputs['quality_flags'])
    geographic, mask = model.prompt_encoder(inputs['h3_indices'], inputs['geo_attributes'])
    _, pooled = model.fusion(visual, geographic, mask)
    logits, _ = model.classification_head(pooled)
    return logits


def _image_gradient(score, image):
    if not score.requires_grad:
        return torch.zeros_like(image)
    gradient, = torch.autograd.grad(score, image, allow_unused=True)
    return torch.zeros_like(image) if gradient is None else gradient


def explain_image(model, inputs, target=None, method='gradient_times_input', steps=16):
    """Explain one case with fixed geographic context and a zero-image baseline.

    Returns signed attribution at model-input resolution. No labels are used.
    Integrated gradients uses trapezoidal integration and reports completeness
    residual relative to the image-only baseline (all other inputs fixed).
    """
    if method not in ('gradient_times_input', 'integrated_gradients'):
        raise ValueError('Unknown attribution method')
    if not isinstance(steps, int) or not 2 <= steps <= 128:
        raise ValueError('steps must be an integer in [2,128]')
    required = {'images', 'quality_flags', 'h3_indices', 'geo_attributes'}
    if not required.issubset(inputs):
        raise ValueError(f'Missing model input keys: {sorted(required - set(inputs))}')
    images = inputs['images'].detach().float()
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError('Explain one case at a time: images must be [1,T,C,H,W]')
    if not torch.isfinite(images).all():
        raise ValueError('Image input contains non-finite values')
    quality = inputs['quality_flags']
    if quality.dtype != torch.bool or quality.shape != images.shape[:2] or not quality.any():
        raise ValueError('quality_flags must be boolean [1,T] with a valid frame')
    fixed = {key: inputs[key] for key in required}
    fixed['images'] = images
    original_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            actual_logits = classification_logits(model, fixed)
            if actual_logits.shape != (1, len(HAZARDS)) or not torch.isfinite(actual_logits).all():
                raise ValueError('Classifier must produce three finite hazard logits')
            probabilities = actual_logits.softmax(dim=-1)
            selected = int(actual_logits.argmax(-1).item()) if target is None else target
            if isinstance(selected, bool) or not isinstance(selected, int) or selected not in range(len(HAZARDS)):
                raise ValueError('target must identify flood=0, wildfire=1, or landslide=2')
            baseline_inputs = {**fixed, 'images': torch.zeros_like(images)}
            baseline_logit = float(classification_logits(model, baseline_inputs)[0, selected])
        with torch.enable_grad():
            if method == 'gradient_times_input':
                differentiated = images.clone().requires_grad_(True)
                score = classification_logits(model, {**fixed, 'images': differentiated})[0, selected]
                gradient = _image_gradient(score, differentiated)
                attribution = differentiated.detach() * gradient.detach()
            else:
                accumulated = torch.zeros_like(images)
                for step in range(steps + 1):
                    interpolated = (images * (step / steps)).detach().requires_grad_(True)
                    score = classification_logits(model, {**fixed, 'images': interpolated})[0, selected]
                    gradient = _image_gradient(score, interpolated)
                    accumulated += gradient.detach() * (.5 if step in (0, steps) else 1.)
                attribution = images * accumulated / steps
        attribution *= quality[:, :, None, None, None]
        if not torch.isfinite(attribution).all():
            raise ValueError('Attribution contains non-finite values')
        heatmap = attribution.abs().sum(dim=2).sum(dim=1)[0] / quality.sum()
        magnitude = float(heatmap.max())
        scale = float(torch.quantile(heatmap.flatten(), .99))
        if scale <= 0:
            scale = magnitude
        normalized = (heatmap / scale).clamp(0, 1) if scale > 0 else torch.zeros_like(heatmap)
        channel_mass = attribution.abs().sum(dim=(0, 1, 3, 4))
        total = channel_mass.sum()
        shares = channel_mass / total if float(total) > 0 else torch.zeros_like(channel_mass)
        actual_logit = float(actual_logits[0, selected])
        return {'attribution': attribution.detach().cpu(), 'heatmap': normalized.detach().cpu(),
            'raw_heatmap': heatmap.detach().cpu(), 'channel_shares': shares.detach().cpu(),
            'metadata': {'method': method, 'target_hazard': HAZARDS[selected],
                'predicted_hazard': HAZARDS[int(actual_logits.argmax(-1))],
                'hazard_probabilities': {name: float(probabilities[0,i]) for i,name in enumerate(HAZARDS)},
                'target_logit': actual_logit, 'zero_image_baseline_logit': baseline_logit,
                'signed_attribution_sum': float(attribution.sum()),
                'completeness_residual': actual_logit-baseline_logit-float(attribution.sum()) if method == 'integrated_gradients' else None,
                'integration_steps': steps if method == 'integrated_gradients' else None,
                'normalization': 'Absolute channel-summed attribution, valid-frame mean, divided by spatial p99 and clipped [0,1]',
                'normalization_scale': scale, 'zero_attribution': magnitude == 0,
                'input_shape': list(images.shape), 'baseline': 'All image channels zero; geographic inputs and frame flags fixed',
                'limitations': ['Attribution explains model sensitivity, not correctness, causal damage, or a hazard footprint.',
                               'Zero baseline may be outside the training distribution.',
                               'Visualization is at resized model-input resolution, not native satellite resolution.',
                               'Channel shares are relative attribution mass, not causal feature importance.']}}
    finally:
        for module, mode in original_modes:
            module.training = mode
