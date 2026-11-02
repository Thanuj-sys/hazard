# L1 - Wayanad Debris-Flow Disaster (Benchmark Year: 2024)

This is an event-data checklist, not a claim that this event's data or labels have been verified.

## Required input data
- Pre-event remote sensing: Cartosat-3, acquired 2023-05-22.
- Post-event remote sensing: RISAT SAR, acquired 2024-07-31.
- Event-period and antecedent rainfall, DEM, derived slope/aspect, geology/soil, and land-cover data where available.
- Exact local machine-readable raster files, AOI, and coverage remain `not_verified` for this repository.

## Required ground truth
- Official NRSC Chooralmala impact map products may be used as a reference source only.
- Independent georeferenced landslide event/source-area mask or polygon is still required for quantitative readiness, but no machine-readable boundary has been verified in this repository.
- Ground-truth status: `map_reference_only` (not IoU-ready or validation-ready).

## Expected CRS and raster resolution
- Preserve each source CRS in metadata. Reproject analysis layers and labels to one event-local projected CRS in metres (appropriate UTM zone to be selected after AOI verification).
- Provisional common analysis grid: 10 m only where source resolution and alignment support it; confirm per event and record resampling. Do not imply this is the native resolution of every layer.

## Verified official sources
- NRSC/NDEM: https://ndem.nrsc.gov.in/documents/Disaster_Document/2024/klwayanadlandslide_31072024.pdf
- NRSC/NDEM: https://ndem.nrsc.gov.in/documents/Disaster_Document/2024/KL/kllandslide50dsc05082024/kllandslide50dsc05082024_vap1.pdf
- Bhuvan/NRSC: https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/doc/Charter_1029_VAP_3_31july2024.pdf
- Bhuvan/NRSC: https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/doc/Charter_1029_VAP_7_6Aug2024.pdf
- Kerala ILDM: https://ildm.kerala.gov.in/wp-content/uploads/2025/04/Report-on-wayanad-landslide-1.pdf

## Missing-data status
- Official disaster documentation confirms the event occurrence and affected villages, but no verified local pre/post Sentinel-2 raster files, DEM, slope, or independent machine-readable ground-truth mask are staged in this directory.
- Overall L1 status: `not_ready`.
- The event remains fail-closed until a verified georeferenced landslide mask and local rasters are acquired and validated.

## Source and provenance requirements
- Record exact provider/product, source URL, license/terms, acquisition/observation dates, time window, AOI, CRS, resolution, bands/features, processing, and source artifact. Keep inputs separate from ground truth.
