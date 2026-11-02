# Event Data Sources and Verification

This inventory separates model **input data**, independent **ground truth**, and **reference-only information**. It records requirements and candidate source types, not confirmed acquisitions. No event-specific source, date, spatial extent, label, or measurement has been verified for this benchmark in the repository. Do not treat the generic training samples or current/recent Planetary Computer retrieval as historical event evidence.

## Source categories

- **Input data:** imagery and ancillary layers that could have been available for the event. Select acquisition windows without using the final target footprint to guide inputs.
- **Ground truth:** independently produced, georeferenced event footprint or mask reserved for target generation/evaluation as appropriate. Its source, interpretation method, date, CRS, resolution, and uncertainty must be recorded.
- **Reference-only information:** reports, news, situation updates, and non-spatial summaries can help establish chronology or guide source discovery. They are not pixel-level labels and cannot support quantitative footprint validation by themselves.

Sentinel-1 SAR and Sentinel-2 optical imagery are required candidate input types. Microsoft Planetary Computer is referenced by the current project for recent imagery retrieval, but historical coverage and exact event acquisitions remain **not_verified** for every event below. Candidate rainfall sources (for example, IMD gridded rainfall or GPM IMERG) and elevation sources (for example, Copernicus DEM or SRTM) are not selected or verified for these events. These names are discovery candidates only, not citations or claims of event coverage. No URLs are supplied until an exact source record has been verified.

## Event inventory

| Event | Input data required | Rainfall input | DEM / slope input | Independent ground truth required | Source verified? | Quantitative validation? |
|---|---|---|---|---|---|---|
| F1 - Upper Assam flood (2026) | Pre/during/post Sentinel-1; Sentinel-2 where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified | Independent event inundation mask/polygon; source and area not_verified | No | No; ground truth not_verified |
| F2 - Assam flood (2022) | Pre/during/post Sentinel-1; Sentinel-2 where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified | Independent event inundation mask/polygon; source and area not_verified | No | No; ground truth not_verified |
| F3 - Kerala flood (2018) | Pre/during/post Sentinel-1; Sentinel-2 where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified | Independent event inundation mask/polygon; source and area not_verified | No | No; ground truth not_verified |
| L1 - Wayanad debris-flow disaster (2024) | Pre/post Sentinel-2; Sentinel-1 change/deformation where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified; geology/soil source not_verified | Independent event source-area mask; runout and severity labels only if sourced; all not_verified | No | No; ground truth not_verified |
| L2 - Chamoli disaster (2021) | Pre/post Sentinel-2; Sentinel-1 change/deformation where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified; geology/soil source not_verified | Independent event source-area mask; runout and severity labels only if sourced; all not_verified | No | No; ground truth not_verified |
| L3 - Kedarnath disaster (2013) | Pre/post Sentinel-2; Sentinel-1 change/deformation where available; exact acquisitions not_verified | Candidate source and event coverage not_verified | Candidate DEM and derived slope not_verified; geology/soil source not_verified | Independent event source-area mask; runout and severity labels only if sourced; all not_verified | No | No; ground truth not_verified |
| W1 - Rajouri/J&K fire event cluster (2026) | Pre/post Sentinel-2; active-fire observations where available; exact events/acquisitions not_verified | Rainfall deficit/weather source and coverage not_verified | Candidate DEM and derived slope not_verified; land-cover/fuel data not_verified | Independent burned-area perimeter/mask; severity labels only if sourced; all not_verified | No | No; ground truth not_verified |
| W2 - Bandipur wildfire (2019) | Pre/post Sentinel-2; active-fire observations where available; exact acquisitions not_verified | Rainfall deficit/weather source and coverage not_verified | Candidate DEM and derived slope not_verified; land-cover/fuel data not_verified | Independent burned-area perimeter/mask; severity labels only if sourced; all not_verified | No | No; ground truth not_verified |
| W3 - Uttarakhand wildfire (2024) | Pre/post Sentinel-2; active-fire observations where available; exact acquisitions not_verified | Rainfall deficit/weather source and coverage not_verified | Candidate DEM and derived slope not_verified; land-cover/fuel data not_verified | Independent burned-area perimeter/mask; severity labels only if sourced; all not_verified | No | No; ground truth not_verified |

## Verification required before event use

For every selected source, record the exact provider/product and URL, license/terms, access or download date, event-time coverage, spatial coverage, CRS, resolution, bands/features, preprocessing, and provenance. For every ground-truth product, also record who produced/interpreted it, the observation date, mapping method, class definitions, uncertainty, and whether it is independent of the inputs. Update the event manifest only after checking the source artifact; otherwise retain `not_verified` or `not_available`.

An event with no independently sourced spatial ground truth may be used for an explicitly labeled input/inference demonstration only. It must not be reported as a quantitative validation event.
