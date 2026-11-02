# F3 - Kerala Flood (Benchmark Year: 2018)

This is an event-data checklist, not a claim that this event's data or labels have been verified.

## Required input data
- Pre-event, during-event, and post-event Sentinel-1 SAR acquisitions; Sentinel-2 imagery where available.
- Event-period rainfall, a DEM, derived slope, and land-cover data where available.
- Exact dates, AOI, products, and coverage: `not_verified`.

## Required ground truth
- Independent georeferenced flood/inundation mask or polygon for the event.
- Independently measured area and mapping date/method. Do not use this footprint as model input.
- Ground-truth source, area, and severity: `not_verified`; runout is `not_applicable`.

## Expected CRS and raster resolution
- Preserve each source CRS in metadata. Reproject analysis layers and labels to one event-local projected CRS in metres (appropriate UTM zone to be selected after AOI verification).
- Provisional common analysis grid: 10 m only where source resolution and alignment support it; confirm per event and record resampling. Do not imply this is the native resolution of every layer.

## Missing-data status
- No event-specific inputs or independent ground-truth files are staged or verified in this directory. External availability: `not_verified`.

## Source and provenance requirements
- Record exact provider/product, source URL, license/terms, acquisition/download dates, time window, AOI, CRS, resolution, bands, preprocessing, and source artifact for every layer. Keep inputs separate from ground truth.
