"""
Redis caching layer for fast prediction lookups and session management.
Provides 100x faster access to frequently queried data.
"""

import redis
import json
import os
from typing import Optional, Dict, Any
import logging
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "hazard_redis_password")

# Cache TTL settings (in seconds)
PREDICTION_CACHE_TTL = 3600  # 1 hour
STATS_CACHE_TTL = 300  # 5 minutes
LOCATION_CACHE_TTL = 7200  # 2 hours

# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        protocol=2,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True,
    )
    # Test connection
    redis_client.ping()
    logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"⚠️ Redis connection failed: {e}. Caching disabled.")
    redis_client = None


# ============= CACHE KEYS =============

def get_prediction_cache_key(latitude: float, longitude: float) -> str:
    """Generate cache key for location-based predictions."""
    lat_round = round(latitude, 3)
    lon_round = round(longitude, 3)
    return f"prediction:{lat_round}:{lon_round}"


def get_stats_cache_key() -> str:
    """Cache key for global statistics."""
    return "stats:global"


def get_location_history_key(latitude: float, longitude: float) -> str:
    """Cache key for location history."""
    lat_round = round(latitude, 3)
    lon_round = round(longitude, 3)
    return f"history:{lat_round}:{lon_round}"


# ============= CACHE OPERATIONS =============

def cache_get(key: str) -> Optional[Dict[str, Any]]:
    """Retrieve value from cache."""
    if redis_client is None:
        return None
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Cache get error for key {key}: {e}")
    return None


def cache_set(key: str, value: Dict[str, Any], ttl: int = PREDICTION_CACHE_TTL) -> bool:
    """Store value in cache with TTL."""
    if redis_client is None:
        return False
    try:
        redis_client.setex(key, ttl, json.dumps(value))
        return True
    except Exception as e:
        logger.warning(f"Cache set error for key {key}: {e}")
        return False


def cache_delete(key: str) -> bool:
    """Delete value from cache."""
    if redis_client is None:
        return False
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Cache delete error for key {key}: {e}")
        return False


def cache_clear_pattern(pattern: str) -> int:
    """Delete all keys matching a pattern."""
    if redis_client is None:
        return 0
    try:
        keys = redis_client.keys(pattern)
        if keys:
            return redis_client.delete(*keys)
        return 0
    except Exception as e:
        logger.warning(f"Cache clear error for pattern {pattern}: {e}")
        return 0


# ============= CACHING DECORATOR =============

def cached(ttl: int = PREDICTION_CACHE_TTL):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_value = cache_get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value
            
            # Call original function
            result = func(*args, **kwargs)
            
            # Store in cache
            if isinstance(result, (dict, list)):
                cache_set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# ============= PREDICTION CACHE OPERATIONS =============

def cache_prediction(
    latitude: float,
    longitude: float,
    prediction_data: Dict[str, Any]
) -> bool:
    """Cache a prediction result."""
    key = get_prediction_cache_key(latitude, longitude)
    return cache_set(key, prediction_data, PREDICTION_CACHE_TTL)


def get_cached_prediction(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """Get cached prediction for a location."""
    key = get_prediction_cache_key(latitude, longitude)
    return cache_get(key)


def invalidate_prediction_cache(latitude: float, longitude: float) -> bool:
    """Invalidate prediction cache for a location."""
    key = get_prediction_cache_key(latitude, longitude)
    return cache_delete(key)


# ============= STATS CACHE OPERATIONS =============

def cache_stats(stats_data: Dict[str, Any]) -> bool:
    """Cache global statistics."""
    key = get_stats_cache_key()
    return cache_set(key, stats_data, STATS_CACHE_TTL)


def get_cached_stats() -> Optional[Dict[str, Any]]:
    """Get cached statistics."""
    key = get_stats_cache_key()
    return cache_get(key)


def invalidate_stats_cache() -> bool:
    """Clear stats cache."""
    key = get_stats_cache_key()
    return cache_delete(key)


# ============= HEALTH CHECK =============

def check_redis_health() -> Dict[str, Any]:
    """Check Redis connection health."""
    if redis_client is None:
        return {"status": "disconnected", "message": "Redis not configured"}
    
    try:
        info = redis_client.info()
        return {
            "status": "connected",
            "used_memory_mb": info.get("used_memory_mb", 0),
            "connected_clients": info.get("connected_clients", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def clear_all_cache() -> int:
    """Clear entire Redis cache (use with caution!)."""
    if redis_client is None:
        return 0
    try:
        return redis_client.flushdb()
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return 0
