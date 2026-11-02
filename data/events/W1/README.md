# W1 - Rajouri/J&K Fire Event Cluster (Benchmark Year: 2026)

This is an event-data checklist, not a claim that individual fires or their data/labels have been verified.

## Required input data
- Pre/post-event Sentinel-2 imagery; active-fire observations where available.
- Rainfall deficit/weather, DEM, derived slope, and land-cover/fuel data where available.
- Exact event dates, AOIs, products, and coverage: `not_verified`; identify and document each event in the cluster separately.

## Required ground truth
- Independent georeferenced burned-area perimeter or mask for each selected event.
- Burn-severity labels only if independently sourced. Burned-area reconstruction is not future fire-spread forecasting.
- Ground-truth source, area, and severity: `not_verified`; runout is `not_applicable`.

## Expected CRS and raster resolution
- Preserve each source CRS in metadata. Reproject analysis layers and labels to one event-local projected CRS in metres (appropriate UTM zone to be selected after AOI verification).
- Provisional common analysis grid: 10 m only where source resolution and alignment support it; confirm per event and record resampling. Do not imply this is the native resolution of every layer.

## Missing-data status
- No event-specific inputs or independent ground-truth files are staged or verified in this directory. External availability: `not_verified`.

## Source and provenance requirements
- Record exact provider/product, source URL, license/terms, acquisition/observation dates, time window, AOI, CRS, resolution, bands, processing, and source artifact. Keep inputs separate from ground truth.
