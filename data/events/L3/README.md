# L3 - Kedarnath Disaster (Benchmark Year: 2013)

This is an event-data checklist, not a claim that this event's data or labels have been verified.

## Required input data
- Pre/post-event Sentinel-2 imagery and Sentinel-1 change/deformation observations where available.
- Event-period and antecedent rainfall, DEM, derived slope/aspect, geology/soil, and land-cover data where available.
- Exact dates, AOI, products, and coverage: `not_verified`.

## Required ground truth
- Independent georeferenced landslide event/source-area mask or polygon.
- Independent runout and severity labels only if a suitable source exists; do not infer them from event summaries.
- Ground-truth source, area, runout, and severity: `not_verified`.

## Expected CRS and raster resolution
- Preserve each source CRS in metadata. Reproject analysis layers and labels to one event-local projected CRS in metres (appropriate UTM zone to be selected after AOI verification).
- Provisional common analysis grid: 10 m only where source resolution and alignment support it; confirm per event and record resampling. Do not imply this is the native resolution of every layer.

## Missing-data status
- No event-specific inputs or independent ground-truth files are staged or verified in this directory. External availability: `not_verified`.

## Source and provenance requirements
- Record exact provider/product, source URL, license/terms, acquisition/observation dates, time window, AOI, CRS, resolution, bands/features, processing, and source artifact. Keep inputs separate from ground truth.
