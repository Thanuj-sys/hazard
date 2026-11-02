# Improving MultiHazard-VLM after the transfer diagnostic

The existing checkpoint is frozen while the expanded evaluation runs. Do not tune thresholds, preprocessors or weights using the expanded scored cases and then advertise that same set as an untouched test.

## 1. Resolve the input contract before retraining

- Archive a manifest of the exact training images, labels, events, source splits, SHA256s and transforms with every checkpoint.
- Define named channel slots and physical units, not only a 10-channel tensor length. Current flood, HLS and RGB adapters occupy different slots; the live API additionally applies per-band standardization while the legacy loaders use fixed divisors.
- Check reflectance versus DN, SAR scaling, RGB ordering, nodata and missing-modality treatment. Add numerical adapter checks for each source and use the same versioned contract in training, validation and inference.
- Run preprocessing ablations on a development split only. A higher score on a repeatedly inspected transfer set is exploratory evidence, not unbiased final performance.

## 2. Build development data that cannot be solved by dataset identity

- Obtain hazard and non-event examples with comparable sensor coverage and preprocessing. Separate permanent water from confirmed new inundation, and active fire from burn-scar labels when those are the actual target tasks.
- Keep complete events/regions and nearby patches together. Create training, development/calibration and locked final-test manifests before fitting anything.
- Source-level or event-level holdout is preferable to random patch splitting. Verify historical overlap; if it cannot be proven for the old checkpoint, retain that limitation explicitly.
- Do not label an empty annotation mask as a disaster or automatically as a true non-event. Negative examples need an independently justified target and task definition.

## 3. Retrain and calibrate on development data

- Start with a classifier baseline under the shared contract and compare against the Mamba model under identical splits.
- Audit training-class/event counts before using balanced sampling or class-weighted cross-entropy. Balance across events and sources as well as classes.
- Add genuine temporal sequences and measured geographic context if available; repeated frames and zero terrain inputs do not validate those branches.
- Introduce a verified non-event/rejection target if the application needs event detection. A three-class positive-only classifier cannot measure non-event false-alarm performance.
- Select checkpoints using macro F1 and per-class recall; explicitly track flood-to-wildfire errors. Fit calibration on the calibration split only, then freeze it.

## 4. Final validation and release decision

- Use the frozen model once on the locked final test. Report support, confusion matrix, per-class precision/recall/F1, macro F1, calibration, and event-cluster-aware uncertainty where event IDs are available.
- Report negative-case false alarms only when independently labelled negatives and a defined rejection rule exist.
- Test segmentation, location, radius/area, exposure and generated language separately against independent task-specific references. Classification accuracy does not validate these outputs.
- State source and geographic coverage. An expanded public-data source diagnostic is useful evidence but does not by itself establish operational readiness.

## Expanded evaluation in this workspace

The first expansion requests 100 source pairs per hazard using seed 20261027 and the unchanged checkpoint. Sen1Floods11 has only 90 rows in its official hand-labelled test CSV, so the acquisition total is capped by that source. HLS sampling remains limited to the cached 100-row validation shard. Mask-empty and exact-known-training duplicate exclusions reduce the scored total. Results are written to `evaluation/real_transfer_expanded/`, while the original `evaluation/real_transfer/` remains unchanged. The comparison separates genuinely added source cases from the previously acquired baseline cases using case IDs and image SHA256 combinations.
