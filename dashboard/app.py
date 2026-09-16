"""
Streamlit-based interactive clickable map dashboard for MultiHazard-VLM.
Click anywhere on the India map to get hazard predictions, impact radius, and affected area.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import h3
import pandas as pd
import json
import math
import os

# Configuration
API_BASE_URL = os.getenv("MULTIHAZARD_API_URL", "http://localhost:8001/api/v1")
INDIA_BOUNDS = [[6.5, 68.0], [35.5, 97.5]]

st.set_page_config(page_title="MultiHazard-VLM Dashboard", layout="wide")

st.title("🗺️ MultiHazard-VLM Interactive Prediction Dashboard")
st.markdown("**Click anywhere on the India map to predict hazards, impact radius, and affected area for that location.**")

# Initialize session state for storing clicked coordinates
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = 20.5937  # India center
if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = 78.9629
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
default_hazard = st.sidebar.selectbox("Filter by Hazard Type", ["All", "Flood", "Wildfire", "Landslide"])

# Function to fetch prediction for clicked coordinates
def fetch_prediction(lat: float, lon: float):
    if not (INDIA_BOUNDS[0][0] <= lat <= INDIA_BOUNDS[1][0] and INDIA_BOUNDS[0][1] <= lon <= INDIA_BOUNDS[1][1]):
        st.warning("Please select a location within India.")
        return None

    try:
        # Call API with clicked coordinates (direct point prediction)
        response = requests.post(
            f"{API_BASE_URL}/predict-point",
            json={
                "latitude": lat,
                "longitude": lon,
                "country_boundary": None
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ API Error: {str(e)}")
        return None

# Create Folium Map - Centered on India
m = folium.Map(
    location=[20.5937, 78.9629],  # India center
    zoom_start=5,
    tiles="OpenStreetMap",
    click=True,
    min_zoom=5,
    max_zoom=10,
    world_copy_jump=False,
    max_bounds=True,
    max_bounds_viscosity=1.0,
)
m.fit_bounds(INDIA_BOUNDS, padding=(12, 12))

# Display stored prediction on map if available
if st.session_state.prediction_data:
    pred = st.session_state.prediction_data
    if pred.get('status') == 'success':
        lat = pred['predicted_coordinates'][0]
        lon = pred['predicted_coordinates'][1]
        radius_km = pred['radius_km']
        hazard_type = pred['hazard_type']
        
        # Determine color based on hazard type
        color_map = {'Flood': 'blue', 'Wildfire': 'orange', 'Landslide': 'red'}
        color = color_map.get(hazard_type, 'gray')
        
        # Plot epicenter
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Predicted Epicenter</b><br>{hazard_type}<br>Lat: {lat:.4f}, Lon: {lon:.4f}",
            icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
        ).add_to(m)
        
        # Plot impact radius circle
        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,  # Convert km to meters
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            weight=2,
            popup=f"Impact Radius: {radius_km:.1f} km",
            tooltip=f"Impact Radius: {radius_km:.1f} km"
        ).add_to(m)

# Render map and get click data
map_data = st_folium(m, width=900, height=600)

# Handle map click events
if map_data and map_data.get('last_clicked'):
    clicked_coords = map_data['last_clicked']
    clicked_lat = clicked_coords['lat']
    clicked_lon = clicked_coords['lng']
    
    # Store in session state
    st.session_state.clicked_lat = clicked_lat
    st.session_state.clicked_lon = clicked_lon
    
    # Fetch prediction
    with st.spinner(f"🔄 Predicting hazards for ({clicked_lat:.4f}, {clicked_lon:.4f})..."):
        pred_data = fetch_prediction(clicked_lat, clicked_lon)
        if pred_data and pred_data.get('status') == 'success':
            st.session_state.prediction_data = pred_data
            st.rerun()  # Rerun to update map with prediction

# Display Hazard Report
if st.session_state.prediction_data:
    pred = st.session_state.prediction_data
    if pred.get('status') == 'success':
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Location Details")
            lat = pred['predicted_coordinates'][0]
            lon = pred['predicted_coordinates'][1]
            st.metric("Latitude", f"{lat:.4f}")
            st.metric("Longitude", f"{lon:.4f}")
        
        with col2:
            st.subheader("⚠️ Hazard Type & Probabilities")
            hazard_type = pred['hazard_type']
            probs = pred['hazard_probabilities']
            selected_probability = probs.get(default_hazard.lower(), 0.0) if default_hazard != "All" else probs.get(hazard_type.lower(), 0.0)
            display_hazard = default_hazard if default_hazard != "All" else hazard_type
            st.metric("Selected Hazard" if default_hazard != "All" else "Primary Hazard", display_hazard, delta=f"{selected_probability*100:.1f}%")

            if default_hazard == "All":
                for haz_name, prob_val in probs.items():
                    st.metric(f"{haz_name.capitalize()}", f"{prob_val*100:.1f}%")
            else:
                st.metric(default_hazard, f"{selected_probability*100:.1f}%")
        
        with col3:
            st.subheader("📏 Impact Assessment")
            radius_km = pred['radius_km']
            # Calculate affected area (circle area = π * r²)
            affected_area_km2 = math.pi * (radius_km ** 2)
            
            st.metric("Impact Radius", f"{radius_km:.1f} km")
            st.metric("Affected Area", f"{affected_area_km2:.1f} km²", 
                     help=f"Circular impact zone area: π × {radius_km:.1f}²")

        if pred.get('inference_mode') == 'demo_synthetic':
            st.warning("Demo mode: this result uses synthetic model inputs and must not be used for operational hazard decisions.")
        elif pred.get('data_source'):
            st.caption(f"Data source: {pred['data_source']}")
        
        st.markdown("---")
        st.subheader("📢 Official Advisory")
        st.info(pred['language_report'])
        
        # Add download button for report
        report_text = f"""
        MULTIHAZARD-VLM PREDICTION REPORT
        ===================================
        
        Location: {lat:.4f}, {lon:.4f}
        Primary Hazard: {hazard_type}
        
        Hazard Probabilities:
        - Flood: {probs.get('flood', 0)*100:.1f}%
        - Wildfire: {probs.get('wildfire', 0)*100:.1f}%
        - Landslide: {probs.get('landslide', 0)*100:.1f}%
        
        Impact Assessment:
        - Impact Radius: {radius_km:.1f} km
        - Affected Area: {affected_area_km2:.1f} km²
        
        Advisory:
        {pred['language_report']}
        """
        
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name=f"hazard_report_{lat:.4f}_{lon:.4f}.txt",
            mime="text/plain"
        )
else:
    st.info("👆 **Click on the map above to see predictions!**")
