"""
Database configuration and connection for PostgreSQL + PostGIS.
Manages prediction storage and query operations.
"""

import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import json
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)
LOCAL_PREDICTIONS_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "prediction_history.json"
LOCAL_PREDICTIONS_LOCK = Lock()

# Database URL - configurable via environment variable
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:hazard_password@localhost:5432/hazard_db"
)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={"connect_timeout": 3},
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============= DATABASE MODELS =============

class Prediction(Base):
    """Store all hazard predictions made by the model."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Location information
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    
    # Prediction results
    hazard_type = Column(String(50), nullable=False, index=True)  # Flood, Wildfire, Landslide
    
    # Probabilities (stored as JSON)
    hazard_probabilities = Column(JSON, nullable=False)
    
    # Impact assessment
    predicted_lat = Column(Float, nullable=False)
    predicted_lon = Column(Float, nullable=False)
    impact_radius_km = Column(Float, nullable=False)
    affected_area_km2 = Column(Float, nullable=False)
    
    # Advisory text
    language_report = Column(String(1000), nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    model_version = Column(String(50), default="v1.0", nullable=False)
    
    def __repr__(self):
        return f"<Prediction(id={self.id}, hazard_type={self.hazard_type}, lat={self.latitude}, lon={self.longitude})>"


class HazardHistory(Base):
    """Track historical hazard events at locations."""
    __tablename__ = "hazard_history"

    id = Column(Integer, primary_key=True, index=True)
    
    # Location (H3 cell or lat/lon)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    h3_index = Column(String(20), nullable=True, index=True)
    
    # Hazard info
    hazard_type = Column(String(50), nullable=False, index=True)
    severity = Column(Float, nullable=False)  # 0-1 scale
    
    # Timeline
    event_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<HazardHistory(hazard_type={self.hazard_type}, lat={self.latitude}, lon={self.longitude}, date={self.event_date})>"


class ModelMetrics(Base):
    """Track model performance and predictions over time."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    
    # Prediction count metrics
    total_predictions = Column(Integer, default=0)
    flood_predictions = Column(Integer, default=0)
    wildfire_predictions = Column(Integer, default=0)
    landslide_predictions = Column(Integer, default=0)
    
    # Average metrics
    avg_confidence = Column(Float, default=0.0)
    avg_radius_km = Column(Float, default=0.0)
    
    # Time period
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<ModelMetrics(total={self.total_predictions}, recorded_at={self.recorded_at})>"


# ============= DATABASE FUNCTIONS =============

def init_db():
    """Initialize database by creating all tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise


def get_db():
    """Get database session for dependency injection in FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    latitude: float,
    longitude: float,
    hazard_type: str,
    hazard_probabilities: dict,
    predicted_lat: float,
    predicted_lon: float,
    impact_radius_km: float,
    affected_area_km2: float,
    language_report: str,
    model_version: str = "v1.0"
) -> Optional[Prediction]:
    """Save a prediction to the database."""
    db = SessionLocal()
    try:
        prediction = Prediction(
            latitude=latitude,
            longitude=longitude,
            hazard_type=hazard_type,
            hazard_probabilities=hazard_probabilities,
            predicted_lat=predicted_lat,
            predicted_lon=predicted_lon,
            impact_radius_km=impact_radius_km,
            affected_area_km2=affected_area_km2,
            language_report=language_report,
            model_version=model_version,
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        _save_local_prediction({
            "id": prediction.id,
            "latitude": latitude,
            "longitude": longitude,
            "hazard_type": hazard_type,
            "hazard_probabilities": hazard_probabilities,
            "predicted_lat": predicted_lat,
            "predicted_lon": predicted_lon,
            "impact_radius_km": impact_radius_km,
            "affected_area_km2": affected_area_km2,
            "language_report": language_report,
            "model_version": model_version,
            "created_at": prediction.created_at.isoformat() + "Z",
            "storage": "database",
        })
        logger.info(f"✅ Prediction saved: ID={prediction.id}, Hazard={hazard_type}")
        return prediction
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error saving prediction: {e}")
        _save_local_prediction({
            "latitude": latitude,
            "longitude": longitude,
            "hazard_type": hazard_type,
            "hazard_probabilities": hazard_probabilities,
            "predicted_lat": predicted_lat,
            "predicted_lon": predicted_lon,
            "impact_radius_km": impact_radius_km,
            "affected_area_km2": affected_area_km2,
            "language_report": language_report,
            "model_version": model_version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "storage": "local_json_fallback",
        })
        return None
    finally:
        db.close()


def get_predictions_by_location(latitude: float, longitude: float, radius_degrees: float = 0.5):
    """Query predictions near a specific location."""
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).filter(
            Prediction.latitude.between(latitude - radius_degrees, latitude + radius_degrees),
            Prediction.longitude.between(longitude - radius_degrees, longitude + radius_degrees),
        ).order_by(Prediction.created_at.desc()).limit(100).all()
        return predictions
    finally:
        db.close()


def get_predictions_by_hazard(hazard_type: str, limit: int = 100):
    """Get recent predictions of a specific hazard type."""
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).filter(
            Prediction.hazard_type == hazard_type
        ).order_by(Prediction.created_at.desc()).limit(limit).all()
        return predictions
    finally:
        db.close()


def get_prediction_stats() -> dict:
    """Get statistics about all predictions."""
    db = SessionLocal()
    try:
        total = db.query(func.count(Prediction.id)).scalar()
        flood_count = db.query(func.count(Prediction.id)).filter(Prediction.hazard_type == "Flood").scalar()
        wildfire_count = db.query(func.count(Prediction.id)).filter(Prediction.hazard_type == "Wildfire").scalar()
        landslide_count = db.query(func.count(Prediction.id)).filter(Prediction.hazard_type == "Landslide").scalar()
        
        avg_radius = db.query(func.avg(Prediction.impact_radius_km)).scalar() or 0.0
        avg_area = db.query(func.avg(Prediction.affected_area_km2)).scalar() or 0.0
        
        return {
            "total_predictions": total or 0,
            "flood_predictions": flood_count or 0,
            "wildfire_predictions": wildfire_count or 0,
            "landslide_predictions": landslide_count or 0,
            "avg_radius_km": float(avg_radius),
            "avg_affected_area_km2": float(avg_area),
        }
    finally:
        db.close()


def _read_local_predictions() -> list[dict]:
    if not LOCAL_PREDICTIONS_PATH.is_file():
        return []
    try:
        payload = json.loads(LOCAL_PREDICTIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _save_local_prediction(prediction: dict) -> None:
    LOCAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_PREDICTIONS_LOCK:
        predictions = _read_local_predictions()
        predictions.append(prediction)
        LOCAL_PREDICTIONS_PATH.write_text(
            json.dumps(predictions[-500:], indent=2) + "\n",
            encoding="utf-8",
        )


def get_local_prediction_history(limit: int = 100) -> list[dict]:
    """Return recent predictions when PostgreSQL is unavailable."""
    return list(reversed(_read_local_predictions()[-max(1, min(limit, 500)):]))
