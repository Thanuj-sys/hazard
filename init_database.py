#!/usr/bin/env python
"""
Database initialization script for MultiHazard-VLM.
Sets up PostgreSQL + PostGIS and creates all necessary tables.

Usage:
    python init_database.py
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.database import DATABASE_URL, init_db, Base, engine
from data.cache import clear_all_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_postgres_extensions():
    """Create PostgreSQL extensions like PostGIS."""
    try:
        with engine.connect() as conn:
            # Create PostGIS extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology;"))
            conn.commit()
            logger.info("✅ PostGIS extensions created successfully!")
    except Exception as e:
        logger.warning(f"⚠️ Could not create PostGIS extensions (may already exist): {e}")


def setup_database():
    """Setup database tables and schema."""
    try:
        logger.info("Initializing database...")
        logger.info(f"Using database: {DATABASE_URL}")
        
        # Create PostGIS extensions
        create_postgres_extensions()
        
        # Create all tables
        init_db()
        
        logger.info("✅ Database setup completed successfully!")
        logger.info("\nDatabase tables created:")
        logger.info("  - predictions: Store all hazard predictions")
        logger.info("  - hazard_history: Historical hazard events")
        logger.info("  - model_metrics: Model performance metrics")
        
        return True
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False


def clear_cache():
    """Clear Redis cache."""
    try:
        logger.info("Clearing Redis cache...")
        count = clear_all_cache()
        logger.info(f"✅ Cleared {count} keys from cache")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear cache: {e}")


def main():
    """Main initialization function."""
    logger.info("=" * 60)
    logger.info("MultiHazard-VLM Database Initialization")
    logger.info("=" * 60)
    
    # Check database connection
    try:
        logger.info("\nTesting database connection...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful!")
    except Exception as e:
        logger.error(f"❌ Cannot connect to database: {e}")
        logger.error(f"\nMake sure PostgreSQL is running:")
        logger.error(f"  Database URL: {DATABASE_URL}")
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Clear cache
    try:
        clear_cache()
    except:
        logger.warning("⚠️ Redis not available (optional)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Initialization Complete!")
    logger.info("=" * 60)
    logger.info("\nYou can now start the API server:")
    logger.info("  python -m uvicorn api.app:app --reload")
    logger.info("\nOr with Gunicorn:")
    logger.info("  gunicorn -w 4 -b 0.0.0.0:8000 api.app:app")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
