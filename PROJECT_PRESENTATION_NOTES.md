# MultiHazard-VLM Project Presentation Notes

## 1. Project Title
MultiHazard-VLM: Vision-Language Framework for Multi-Hazard Mapping on Earth Observation Data

## 2. Team / Institution
- Team: D15
- Institution: Amrita School of AI, Coimbatore

## 3. Problem Statement
Natural hazards such as floods, wildfires, and landslides often cause severe damage to life, infrastructure, and agriculture. Traditional monitoring systems are slow, fragmented, and often rely on delayed or limited data sources.

There is a strong need for an intelligent and scalable system that can:
- analyze Earth observation data,
- detect hazard-prone regions,
- provide quick predictions,
- support decision-making for disaster management.

## 4. Objective of the Project
The goal of this project is to build a geospatial hazard prediction system that combines:
- satellite imagery,
- geospatial features,
- deep learning models,
- and a user-friendly dashboard for decision support.

The system aims to identify the most likely hazard at a location and generate a spatial risk assessment.

## 5. Project Overview
This project uses a multi-modal deep learning framework that fuses:
- Sentinel-1 and Sentinel-2 satellite data,
- topographic and environmental context,
- geospatial regions and prompts,
- and hazard-specific prediction heads.

The framework predicts the likely hazard class and provides a risk footprint, including coordinates and impact radius.

## 6. Key Features
- Multi-hazard prediction for flood, wildfire, and landslide
- Integration with real satellite data from Microsoft Planetary Computer
- Point-based prediction API for geospatial locations
- Interactive dashboard with clickable map interface
- Persistent database storage for predictions
- Redis-based caching for faster repeated requests
- Real-time monitoring support with Prometheus/Grafana
- Containerized deployment using Docker

## 7. System Architecture
The system follows a modular pipeline:

1. Data Collection
   - Sentinel-1 SAR imagery
   - Sentinel-2 optical imagery
   - geospatial context layers such as DEM, slope, NDVI, and lithology

2. Preprocessing
   - geospatial alignment,
   - cloud masking,
   - coordinate normalization,
   - dataset structure preparation.

3. Model Pipeline
   - shared spatiotemporal backbone,
   - geospatial prompt encoder,
   - cross-attention fusion layer,
   - hazard-specific output heads.

4. Prediction Layer
   - hazard type classification,
   - impact radius estimation,
   - predicted coordinates,
   - language-based risk explanation.

5. Deployment Layer
   - FastAPI backend
   - Streamlit dashboard
   - PostgreSQL + Redis
   - Docker-based orchestration

## 8. Methodology
The system uses a vision-language style architecture that combines imagery understanding and geospatial reasoning.

The model is designed to:
- extract visual features from satellite products,
- align them with location-specific geospatial context,
- fuse multi-source information,
- predict hazard probabilities at a selected geographic point,
- generate a human-readable summary for disaster response planning.

## 9. Data Sources Used
- Sentinel-1: radar data for surface and structural patterns
- Sentinel-2: multispectral optical data for land cover and vegetation conditions
- Planetary Computer: cloud-accessible public geospatial data
- India administrative boundary data for geofencing and map overlays

## 10. Model Design
The architecture includes:
- Spatiotemporal Mamba visual backbone
- Geospatial prompt encoder
- Cross-attention fusion layer
- Separate heads for:
  - language reasoning,
  - coordinate regression,
  - radius estimation

This modular design helps the model maintain better spatial and semantic understanding for each hazard type.

## 11. Backend and Deployment
The project includes:
- FastAPI REST API for inference and monitoring
- PostgreSQL database for persistent prediction storage
- Redis cache for faster repeated predictions
- Prometheus and Grafana for operational monitoring
- Docker Compose for environment setup

## 12. Dashboard Features
The dashboard enables:
- interactive India map,
- click-to-predict functionality,
- hazard output visualization,
- predicted risk footprint radius,
- district and state context,
- live weather and environmental context,
- operational status display.

## 13. Results and Achievements
The project achieved the following outcomes:
- real hazard prediction pipeline working end-to-end,
- integration of live satellite data and geospatial processing,
- model-ready API endpoints,
- operational dashboard,
- persistent storage and monitoring stack,
- container deployment ready.

## 14. Example Prediction Use Case
A user clicks a location on the map. The system:
1. validates the coordinates,
2. fetches the relevant satellite context,
3. runs the model,
4. identifies the hazard,
5. returns probabilities and risk area,
6. displays a response message and affected radius.

This makes the platform useful for emergency response and local risk assessment.

## 15. Challenges Faced
During development, we addressed several important issues:
- geospatial CRS compatibility problems,
- real-data image processing constraints,
- timestamp normalization issues,
- API timeout handling,
- connection and CORS handling between dashboard and API,
- ensuring the system works with real satellite imagery rather than just synthetic demo values.

## 16. Significance of the Work
This project demonstrates how AI, remote sensing, and geospatial analytics can be combined to produce a practical hazard monitoring system. It bridges the gap between research models and deployable decision-support tools.

The system is especially relevant for:
- disaster preparedness,
- early warning support,
- local government planning,
- incident investigation,
- geospatial risk monitoring.

## 17. Future Scope
The project can be extended in future phases by:
- improving model accuracy with larger disaster datasets,
- adding more hazard classes,
- integrating real-time alert systems,
- deploying on cloud infrastructure,
- adding APIs for mobile apps and public dashboards,
- improving geospatial automation and scaling.

## 18. Conclusion
MultiHazard-VLM is a complete geospatial disaster intelligence project that combines satellite vision, deep learning, geospatial processing, and user-friendly deployment. It serves as a strong foundation for real-world multi-hazard mapping and early risk assessment.

## 19. One-Slide Summary
- We built an AI-based multi-hazard mapping system using Earth observation data.
- It predicts flood, wildfire, and landslide risk using satellite imagery and geospatial context.
- It includes a real-time dashboard, API, database, cache, and monitoring stack.
- It is useful for proactive disaster risk understanding and operational planning.

## 20. Presentation Tips
Use these talking points during the presentation:
- “This project turns geospatial satellite data into operational hazard intelligence.”
- “The model fuses multiple data sources to improve risk prediction.”
- “The dashboard makes the system accessible for rapid decision-making.”
- “We focused on a deployable architecture, not just a research prototype.”

## 21. Sample Speaker Notes
- The core problem is the lack of fast, accessible multi-hazard assessment tools.
- We designed a pipeline that combines remote sensing and AI.
- The dashboard helps users interact with the model in a practical way.
- Real world deployment requires stability, monitoring, and fast data access.
- This project demonstrates how AI can support disaster response and preparedness.

---

## 22. Quick Demo Script for Presentation
1. Open the dashboard and show the India map.
2. Click a location within India.
3. Show hazard prediction and confidence values.
4. Highlight the impact radius and affected area.
5. Mention the data source used for the prediction.
6. Explain that the backend is connected to live model inference and monitoring services.
7. Conclude with the system’s value for public safety and disaster planning.
