# Image explainability

This feature explains the existing three-class hazard classifier. It does not
train a model, establish prediction correctness, create hazard masks, or generate
LLM text. No API key is required. Figures are created only when you run the command.

## Quick demonstration using existing generated samples

Run these yourself from `D:\sem7\hazard`:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.venv\Scripts\python.exe scripts/explain_prediction.py --local-smoke flood
```

Replace `flood` with `wildfire` or `landslide` for another branch. Output:

```text
evaluation/explanations/flood/
  index.html
  attribution_comparison.png
  attribution_comparison.svg
  channel_attribution.png
  attributions.npz
  explanation.json
  explanation.txt
```

Open `index.html` for screenshots. The comparison contains the displayed model
input, absolute attribution heatmap, and overlay. The chart shows relative
attribution mass by input channel. `attributions.npz` retains signed attribution
and unclipped heatmaps; `explanation.json` records class probabilities, target,
method, input shape, baseline, normalization, checkpoint/config hashes, context
limitations and zero-attribution status.

The demonstration uses the old generated sample TIFFs and the selected checkpoint.
Every figure is explicitly labeled as a demonstration. It cannot support a claim
of real-world hazard accuracy. The attribution image is not a predicted footprint.

## Methods

The default `gradient_times_input` requires one gradient pass plus inference for
the original image and zero-image reference. It attributes a selected class LOGIT,
not its probability. Signed contribution is image value times logit gradient.
The quick method is local sensitivity and is not assumed to satisfy completeness.

Integrated gradients is optional and more expensive:

```powershell
.venv\Scripts\python.exe scripts/explain_prediction.py --local-smoke flood --method integrated_gradients --steps 16
```

This integrates gradients on a straight line from zero-image input using a
trapezoidal rule. It requires `steps + 1` gradient passes. The JSON includes
`completeness_residual = original_logit - baseline_logit - signed_attribution_sum`.
A large residual indicates inadequate numerical integration or other limitations;
it is not proof of model correctness. The zero baseline can be outside the model's
training distribution. Runtime depends on the device, image size and frame count;
no runtime guarantee is made for CPU inference.

Use `--target wildfire` to explain a class other than the model's prediction.
Use `--device cuda` only with a working CUDA environment. Existing files in the
selected output directory are replaced by a later run.

The heatmap sums absolute attribution across channels and averages valid frames.
It is normalized by spatial p99 and clipped to [0,1]. Signed arrays preserve
negative contributions. Zero attribution produces an explicitly zero heatmap,
never a manufactured hotspot. The preview shows the first valid frame while the
heatmap aggregates all valid frames; inspect the NPZ for frame-specific results.

Geographic context is held fixed. This feature does not claim to explain the
causal effects of elevation, slope, NDVI or lithology. Missing context values from
the legacy preprocessing path are recorded; zero fills are not verified values.

## Real prepared inputs

To explain a real case, provide the exact preprocessed dictionary used by your
model, saved using `torch.save(model_inputs, "prepared_case.pt")`. Required keys:

- `images`: float tensor `[1, T, C, H, W]`, normalized exactly as during inference.
- `quality_flags`: boolean tensor `[1, T]`; True means a valid frame.
- `h3_indices`: dictionary of resolution to integer token tensors.
- `geo_attributes`: dictionary of the model's geographic input tensors.

Input must already match the selected configuration and checkpoint. No label,
reference footprint, or ground-truth response is passed to the classifier. Do not
save the entire training batch; save its `model_inputs` dictionary. The attribution
path uses backbone, prompt encoder, fusion and classification head, skipping the
language and segmentation decoders. It does not need trained segmentation weights.

```powershell
.venv\Scripts\python.exe scripts/explain_prediction.py --input-batch prepared_case.pt --checkpoint checkpoints/model_epoch_9.pt
```

This writes to `evaluation/explanations/prepared_case/`. The tool does not fetch or
invent real imagery or preprocessed inputs for you. Checkpoint and input files
use tensor-only loading. Unknown older pickled checkpoint objects are rejected;
convert them separately only if you trust their source.

Display channels default to `[0, 1, 2]` with per-channel 2nd–98th percentile stretch.
This is a false-color preview unless those channels genuinely map to RGB. Use
`--display-channels 4 3 2` only if that is the correct mapping for YOUR input.
Channel order differs between sources. The preview is at model resolution; no
native-resolution boundary, geographic coordinates or affected area is inferred.

## Regression tests

Tests use small analytic software models and check known gradients, integrated
attribution completeness for a linear model, signed contributions, masked frames,
zero maps, mode restoration, invalid inputs, and absence of parameter gradient
accumulation. Run them yourself:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_explainability.py -q
```

Implementation was prepared without running inference or tests, per the user's
instruction. Generated images and benchmark-quality validation are not claimed.
