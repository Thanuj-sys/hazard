# Event Reconstruction Validation Policy

- Ground truth must be independent from model input.
- Never provide the final event footprint to the model as an input when that footprint is the prediction target.
- Do not train and test by randomly splitting pixels from the same event.
- The final benchmark must use event-level separation. Keep all samples from a held-out event out of training and model selection.
- Equivalent radius is derived from area: `r_eq = sqrt(A / pi)`.
- Area error is: `abs(predicted_area - actual_area) / actual_area * 100`.
- Missing ground truth means the event cannot be presented as a quantitative validation result. It may be shown only as a clearly labeled demonstration if inference inputs are available.
- Never fabricate metrics. Report a metric only when its inputs and definition are available; otherwise mark it unavailable.

## Separation and provenance

Keep input imagery/ancillary layers and evaluation ground-truth artifacts in distinct records and locations. Record source, acquisition/observation date, CRS, resolution, processing steps, and provenance for each. Do not use reports or reference-only summaries as pixel masks or measured areas. State whether any label was used in training, validation, threshold selection, or final test.
