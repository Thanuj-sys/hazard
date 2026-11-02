# MultiHazard-VLM Project Detail Report

## 1. Project Overview

This project is an end-to-end multi-hazard Earth observation and decision-support system for India. It is designed to ingest geospatial inputs, reason over hazard context, and produce a hazard classification plus a location-aware impact estimate. The system is organized around a deep learning model that fuses satellite imagery with H3 spatial prompts and geospatial metadata such as elevation, slope, NDVI, aspect, and lithology.

The current implementation is a prototype system for:
- Flood detection
- Wildfire detection
- Landslide assessment
- A point-based hazard prediction workflow for map interactions
- A dashboard for user-facing hazard visualization
- A FastAPI backend with health, prediction, and monitoring endpoints

The project is not a fully validated operational disaster monitoring system yet. The model outputs are research/prototype estimates, and the historical-event reconstruction layer is intentionally fail-closed until proper local geospatial inputs and independent ground truth are available.

---

## 2. Project Purpose

The project’s core objective is to estimate:
- primary hazard class for a selected location
- severity or impact radius in kilometers
- approximate circular affected area
- textual hazard reporting grounded in contextual geospatial signals

The overall design combines:
- satellite imagery-derived visual features
- geospatial attributes
- location-aware grid encoding
- cross-attention fusion
- decoupled prediction heads for hazard detection, coordinates, and radius estimation

---

## 3. High-Level Architecture

The design follows a modular multi-stage pipeline:

1. Data acquisition from EO sources
2. Preprocessing and normalization
3. Geospatial prompt encoding with H3 indices
4. Spatiotemporal visual encoding
5. Cross-modality fusion
6. Multi-head prediction
7. Dashboard presentation and REST API exposure

### Architectural Layers

#### 3.1 Visual backbone
The model uses a Spatiotemporal Mamba-style visual backbone to encode temporal image stacks and patch tokens from satellite imagery.

#### 3.2 Geospatial prompt encoder
The geospatial encoder converts location metadata into hierarchical H3 grid prompts and geospatial attributes. This allows the model to reason about where the AOI lies geographically and relate that to hazard priors.

#### 3.3 Cross-attention fusion layer
The visual embedding and geospatial prompt representation are fused into a shared latent feature representation.

#### 3.4 Decoupled prediction heads
The fused representation branches into separate heads:
- hazard classification head
- coordinate regressor
- radius estimator
- language/semantic reasoning head
- optional segmentation head

This separation is intended to prevent the model from conflating classification, localization, and report generation into a single loosely coupled output.

---

## 4. Main Project Modules

### 4.1 API layer
Location: api/app.py

This is the backend entry point. It exposes:
- health checks
- hazard prediction endpoints
- map interaction endpoints
- monitoring endpoints
- metadata and statistics endpoints

Key responsibilities:
- validate AOI and coordinate inputs
- load model weights
- run point inference
- build prepared model inputs from current location and surrounding data
- compute a circular impact estimate
- return JSON prediction payloads

Important notes:
- The API supports India-only geofencing.
- It can run in a real-data mode using Planetary Computer or in a synthetic demo mode.
- It attempts to degrade gracefully when PostgreSQL or Redis are unavailable.

### 4.2 Dashboard layer
Location: web/src/App.jsx

The active dashboard is a React single-page application built with Vite and React Leaflet. It:
- renders a navigable India map with OpenStreetMap tiles and bundled state/district boundaries
- highlights district geometry and displays district/state names on hover
- lets users click the map or enter coordinates to choose a point
- calls the existing FastAPI endpoints through a same-origin Vite/Nginx proxy
- displays hazard scores, estimated radius/area, scene source, current weather, and advisory text
- exports a downloadable text assessment

The former Streamlit implementation remains under dashboard/ as legacy source; it is no longer used by the local or Docker startup paths.

### 4.3 Model layer
Location: models/

Core model logic is implemented in:
- models/multihazard_vlm.py
- models/backbone/
- models/heads/
- models/fusion.py
- models/prompt_encoder.py
- models/segmentation_head.py

This layer handles the actual deep learning inference and decomposition of the outputs.

### 4.4 Data layer
Location: data/

This layer includes:
- database management
- cache logic
- metrics instrumentation
- event and metadata processing
- geospatial data handling
- historical event reconstruction readiness checks

### 4.5 Evaluation layer
Location: evaluation/

This is focused on comparisons, metrics, and reconstruction readiness logic. It checks event provenance, file presence, and data quality gates. The repository contains explicit event-based reconstruction checks to prevent unverified events from being reported as valid benchmarks.

---

## 5. Core Model Design

### 5.1 Model summary
The project uses a multi-hazard VLM architecture combining visual encodings with spatial prompt embeddings. The network is designed to estimate:
- hazard class probability
- geofenced coordinate estimate
- impact radius estimate in kilometers
- language description grounded in context
- optional segmentation mask

### 5.2 Model components

#### Hazard classification head
This head predicts one of the hazard classes.

Expected classes in the current config include:
- flood
- wildfire
- landslide

#### Coordinate regressor
This head outputs a normalized and georeferenced coordinate estimate. It accepts the fused global representation and converts it into geographic coordinates using AOI context.

#### Radius estimator
The radius estimation module is a small MLP head that predicts the approximate impact radius in km.

Important design detail:
- it is trained as a regression output
- it is passed through ReLU to enforce non-negativity
- it is clamped to a bounded range for realistic outputs
- hazard-specific clamping may be applied depending on the hazard type

#### Language reasoner
This head produces a natural-language hazard report using fused contextual embeddings and geospatial attributes, rather than emitting numeric coordinates directly.

#### Segmentation head
When enabled, the model also predicts a segmentation mask or spatial footprint from the fused feature map.

---

## 6. How the Radius Is Estimated

This is a critical part of the project and is directly relevant to report writing.

### 6.1 Model-level radius output
The radius estimator is defined in models/heads/radius_estimator.py. It performs the following:

1. Takes fused global embedding as input
2. Sends it through a small MLP stack
3. Applies ReLU so values cannot be negative
4. Optionally clamps by hazard type
5. Returns the predicted impact radius in kilometers

The relevant function behavior is:
- raw radius = MLP(fused_global)
- radius_km = ReLU(raw_radius)
- hazard-specific lower and upper bounds are applied when hazard type is known
- fallback clamping is applied otherwise

This means the core model does not output a GIS footprint directly; it outputs a scalar impact radius estimate.

### 6.2 API-level adjustment
In api/app.py, the radius is further adjusted based on location and hazard type. The function compute_location_based_radius adjusts the base model radius using:
- latitude-based variation
- longitude-based variation
- hazard-specific multiplier

The logic is designed to produce more realistic regional variation in impact estimate rather than a constant radius everywhere.

### 6.3 Circular affected area formula
The affected area is not measured from an actual polygon footprint. It is computed as a circular approximation:

Area = π × r²

where r is the radius in kilometers.

This gives an estimated circular impact zone in square kilometers.

### 6.4 Interpretation
This means the reported impact radius and affected area are model-estimated approximations from current point context, not measured historical footprint extents from archived hazard events.

---

## 7. Data Flow During Inference

### 7.1 Prediction request cycle
When a user clicks a point on the dashboard:
1. The dashboard sends a POST request to the API with latitude and longitude.
2. The API validates that the point lies within India and within the mapped boundary.
3. The API either:
   - runs a Planetary Computer-based live inference, or
   - falls back to a synthetic mode if configured
4. The model creates a set of inputs from the point and available geospatial context.
5. The model produces hazard probabilities, geofenced coordinates, a radius estimate, and a report text.
6. The API computes the circular affected area as πr².
7. The dashboard displays: hazard, radius, area, and advisory text.

### 7.2 Example output structure
The backend returns a payload with fields such as:
- status
- inference_mode
- data_source
- hazard_type
- hazard_probabilities
- predicted_coordinates
- radius_km
- affected_area_km2
- language_report

---

## 8. Real-Time and External Data Sources

The project is designed to interface with public EO and weather sources such as:
- Microsoft Planetary Computer
- Sentinel-1 SAR imagery
- Sentinel-2 optical imagery
- Open-Meteo weather API

The live inference pipeline checks for available recent satellite scenes covering a selected point and then builds the input tensor from those scene bands.

### 8.1 Data freshness logic
The app records:
- acquisition date
- source data source description
- age of most recent source relative to the current time

This helps distinguish recent satellite imagery from stale or synthetic fallback data.

---

## 9. Runtime Environment and Setup

The project runs in a Python environment with dependencies such as:
- PyTorch
- FastAPI
- React 19, Vite, React Leaflet, and Leaflet
- rasterio
- shapely
- h3
- numpy
- requests
- pydantic
- PostgreSQL client support
- Redis support
- Prometheus metrics support

### Typical local startup commands
The repository includes startup scripts and quick-start instructions.

Typical flow:
1. create a virtual environment
2. install dependencies from requirements files
3. run the API with uvicorn
4. from web/, run npm install and npm run dev for the React dashboard
5. access the dashboard on port 8503 and API on port 8000

---

## 10. Persistence, Monitoring, and Operations

### 10.1 Database
The database layer stores prediction results and statistics, including:
- hazard type
- impact radius
- affected area
- timestamps
- coordinates

### 10.2 Cache
Redis is used to cache prediction results and statistics to reduce repeated processing.

### 10.3 Monitoring
Prometheus metrics are integrated into the project for monitoring:
- total predictions
- prediction timing
- throughput signals
- operational health monitoring

---

## 11. Historical Event Reconstruction and Data Readiness Policy

This project contains a strict fail-closed event-data management layer for historical reconstruction.

### 11.1 Event readiness hierarchy
The reconstruction logic distinguishes between:
- not_ready
- data_ready
- validation_ready
- demo_only

### 11.2 Principle
A historical event is not considered ready just because there is documentation or a PDF describing it. It must have:
- local machine-readable input files
- valid geospatial metadata
- provenance and acquisition dates
- independent ground truth with sufficient spatial support

### 11.3 Why this matters
The repository explicitly avoids marking an event as validated unless:
- the event’s geospatial inputs exist locally
- there is a georeferenced reference mask or independent verification product
- the raster/context metadata passes validation checks

This is important because it protects the project from false benchmarking claims.

---

## 12. Verified Wayanad L1 Example

The project includes a Wayanad landslide event example under the event metadata layer. The source documentation indicates:
- historical event date: 2024-07-30
- pre-event remote-sensing source: Cartosat-3
- post-event remote-sensing source: RISAT SAR
- ground truth status: map_reference_only or unavailable as a machine-readable validation-ready product

The system intentionally does not claim the event is ready for reconstruction or validation until the actual local raster stack and independent ground truth exist.

This is a good example of the project’s fail-closed logic: the event is recognized and documented, but not treated as a validated benchmark.

---

## 13. Current Project Status and Limitations

### Current status
The project is a working prototype for:
- India-local hazard estimation
- point inference interaction
- interactive dashboard use
- model-based radius estimation
- system orchestration

### Limitations
1. It is not a validated disaster footprint mapping system yet.
2. Radius is an estimate based on a learned scalar output, not a measured boundary.
3. Affected area is a circular approximation rather than a geodesic polygon footprint.
4. Real-time inference depends on data availability and network access.
5. Historical reconstruction demands verified local files and independent ground truth before being treated as ready.
6. The project has a research/development character and should not be mistaken for an operational early-warning platform without separate validation.

---

## 14. Report-Writing Summary

This project is best described as a research prototype for AI-assisted geospatial hazard intelligence. It combines remote sensing, geospatial metadata, location-aware inference, and a multi-head model to estimate hazard class and circular impact zone at a map-click location. The project’s design is modular and extensible, but the current functional outputs should be interpreted as approximate estimates rather than validated operational measurements.

The most important conceptual distinction is this:
- the model predicts a circular impact radius and area estimate
- it does not recover a confirmed historical hazard footprint from previous incidents
- the project’s event reconstruction pipeline enforces strict verification before claims of benchmark readiness

---

## 15. Conclusion

The MultiHazard-VLM project is a multi-hazard GIS AI system that integrates satellite imagery, geospatial prompts, and hazard-specific learning heads to classify likely hazard types and estimate impact radius. It provides an interactive dashboard and backend API for location-based inference, while also implementing a strict fail-closed historical event data policy that avoids claiming quantitative readiness without verifiable geospatial evidence.

This makes it suitable for:
- project demonstration
- research exploration
- prototype hazard intelligence workflows
- educational and technical evaluation

It is not yet suitable for operational event validation without independent, local, machine-readable geospatial truth data.
