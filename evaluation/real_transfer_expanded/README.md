# Expanded fixed-checkpoint transfer evaluation

The model was not retrained. Changes in aggregate metrics reflect a different evaluated sample, not a model improvement.

| Cohort | Eligible patches | Accuracy | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Original baseline | 75 | 66.67% | 66.67% | 56.86% |
| Expanded total | 248 | 66.94% | 66.67% | 56.97% |
| New source cases only | 173 | 67.05% | 66.67% | 57.02% |

## Expanded total: class-wise results

| Class | Cases | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| flood | 82 | Undefined (no predictions) | 0.00% | 0.00% |
| wildfire | 100 | 54.95% | 100.00% | 70.92% |
| landslide | 66 | 100.00% | 100.00% | 100.00% |

Confusion matrix: rows true labels; columns flood / wildfire / landslide.

```text
flood         0   82    0
wildfire      0  100    0
landslide     0    0   66
```

## New cases only: class-wise results

| Class | Cases | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| flood | 57 | Undefined (no predictions) | 0.00% | 0.00% |
| wildfire | 70 | 55.12% | 100.00% | 71.07% |
| landslide | 46 | 100.00% | 100.00% | 100.00% |

Confusion matrix: rows true labels; columns flood / wildfire / landslide.

```text
flood         0   57    0
wildfire      0   70    0
landslide     0    0   46
```

## Interpretation limits

- Positive-only, source-confounded transfer diagnostics; no operational validation claim.
- Newly added cases have not been used to tune or retrain this checkpoint in this run; historical training lineage remains unknown.
- More evaluated patches do not remove source confounding, temporal/context ablation, or event clustering.
- Source-prefix slices are descriptive; they do not establish event-independent training/test separation.
- Wilson intervals assume independent patches and can be optimistic under geographic/event clustering.

Figures: `figures/overall/index.html`. Raw scored cases and exclusions: `metrics_report.json`. New-case-only diagnostics: `comparison.json`.

Acquisition and reproducibility details: [PROTOCOL.md](PROTOCOL.md). Improvement priorities: [MODEL_IMPROVEMENT_PLAN.md](../MODEL_IMPROVEMENT_PLAN.md).
