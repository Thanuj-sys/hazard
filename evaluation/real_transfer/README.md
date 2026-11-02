# Completed real-data classifier transfer test

Checkpoint: `checkpoints/model_epoch_9.pt` (unchanged). Run date: 27 October 2026.

Acquired 90 published image/annotation pairs with fixed seed 20261027. Evaluated 75 positive-labelled patches: 25 flood-source, 30 burn-scar, 20 landslide. Fifteen mask-empty patches were excluded, explicitly listed in `metrics_report.json`. They were not relabelled as disasters.

| Measure | Result |
| --- | ---: |
| Accuracy | 66.67% (50/75 correct) |
| Macro precision | 51.52% |
| Macro recall | 66.67% |
| Macro F1 | 56.86% |
| Weighted F1 | 54.90% |
| Log loss | 0.993309 |
| Multiclass Brier score | 0.557037 |
| Ten-bin ECE | 0.302812 |

| Class | Cases | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flood-source | 25 | Undefined (no flood predictions) | 0% | 0% | 0.884000 |
| Burn-scar / wildfire | 30 | 54.55% | 100% | 70.59% | 0.989630 |
| Landslide | 20 | 100% | 100% | 100% | 1.000000 |

Confusion matrix: rows true labels, columns predicted flood/wildfire/landslide:

```text
flood       0 25  0
wildfire    0 30  0
landslide   0  0 20
```

All flood-source cases were classified as wildfire. AUC measures ranking as the threshold changes, while this confusion matrix uses three-class argmax. A positive flood AUC does not mean the classifier successfully selected flood at its deployed decision rule.

Open `figures/overall/index.html` for all metrics and downloadable PNG/SVG plots, including three separate ROC charts and a side-by-side panel. Raw predictions: `predictions.jsonl`. The complete protocol is `../REAL_TRANSFER_PROTOCOL.md`, with source URLs, adapters and limitations. Acquired files and source hashes are in `../../dataset/real_holdout/manifest.json`.

This is a real-public-data, positive-only source-domain transfer diagnostic, not a certified unseen or operational benchmark. A bare checkpoint provides no historical training manifest; known local exact duplicates were checked, but complete training exclusion cannot be proven. Source/modalities are confounded with class, missing geographic context is ablated, frames are repeated, and the source adapters differ from historical training assumptions. Flood labels represent water-source patches, and burn scars are not active-fire labels. No non-event, trained segmentation, localization, vulnerability or language quality evaluation is claimed. The 95% Wilson interval [55.42%,76.29%] assumes independent cases and does not correct event clustering or these limitations.

Validation: three focused source-adapter tests passed; the saved confusion matrix, accuracy and macro metrics independently matched scikit-learn. The ROC triptych and confusion matrix were visually inspected. Original generated smoke results and the strict reviewed benchmark remain separate and unchanged.
