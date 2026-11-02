# Historical Event Acquisition Status

This inventory reflects the repository state only. It does not claim that the event happened or that external products exist. For every benchmark entry, event dates, locations, AOIs, source provenance, local rasters, and spatial ground truth remain unverified unless explicitly recorded in metadata.

Allowed values: verified, available, missing, unverified, not_applicable.
Canonical readiness states: not_ready, data_ready, validation_ready, demo_only.

| event_id | hazard | event_name | date_status | location_status | aoi_status | satellite_status | ancillary_status | ground_truth_status | provenance_status | local_files_status | overall_status | blocking_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | flood | Upper Assam flood | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Exact event dates, AOI, satellite acquisitions, DEM/slope provenance, and independent inundation mask remain unverified. |
| F2 | flood | Assam flood | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Exact event dates, AOI, satellite acquisitions, DEM/slope provenance, and independent inundation mask remain unverified. |
| F3 | flood | Kerala flood | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Exact event dates, AOI, satellite acquisitions, DEM/slope provenance, and independent inundation mask remain unverified. |
| L1 | landslide | Wayanad debris-flow disaster | verified narrative only (2024-07-30) | verified at district level: Wayanad, Kerala | unverified | official map PDFs only; no local verified georeferenced raster inputs | missing | missing | partial: official NRSC/NDEM and Kerala ILDM disaster documentation found; pre-event source documented as Cartosat-3 (2023-05-22), post-event source documented as RISAT SAR (2024-07-31), but no machine-readable boundary or local raster stack was downloaded or validated | missing | not_ready | The event occurrence is confirmed by official disaster documentation, but no local Cartosat-3/RISAT georeferenced rasters, DEM/slope product, or machine-readable independent landslide mask were staged or validated. |
| L2 | landslide | Chamoli disaster | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Exact dates, AOI, pre/post imagery, DEM/slope/geology provenance, and independent landslide mask remain unverified. |
| L3 | landslide | Kedarnath disaster | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Exact dates, AOI, pre/post imagery, DEM/slope/geology provenance, and independent landslide mask remain unverified. |
| W1 | wildfire | Rajouri/J&K fire event cluster | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Event-specific dates, AOI, pre/post fire imagery, burn-area mask, and provenance remain unverified. |
| W2 | wildfire | Bandipur wildfire | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Event-specific dates, AOI, pre/post fire imagery, burn-area mask, and provenance remain unverified. |
| W3 | wildfire | Uttarakhand wildfire | unverified | unverified | unverified | missing | missing | missing | unverified | missing | not_ready | Event-specific dates, AOI, pre/post fire imagery, burn-area mask, and provenance remain unverified. |

## Verification note

The manifest still marks dates, state, district, AOI, source, inputs, and ground truth as not_verified, and the per-event directories contain checklists only. No event is data_ready, validation_ready, or demo_only on this repository evidence. The code fails closed for these events and does not fabricate input rasters, acquisition dates, or independent labels.
