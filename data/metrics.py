"""
Prometheus metrics for monitoring MultiHazard-VLM system performance.
Tracks predictions, latency, errors, and resource usage.
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CollectorRegistry
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Create custom registry
REGISTRY = CollectorRegistry()

# ============= COUNTERS =============
# Count total requests and predictions by type

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["hazard_type"],
    registry=REGISTRY
)

api_requests_total = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
    registry=REGISTRY
)

errors_total = Counter(
    "errors_total",
    "Total errors encountered",
    ["error_type"],
    registry=REGISTRY
)

database_writes_total = Counter(
    "database_writes_total",
    "Total database write operations",
    ["table"],
    registry=REGISTRY
)

cache_hits_total = Counter(
    "cache_hits_total",
    "Total cache hits",
    registry=REGISTRY
)

cache_misses_total = Counter(
    "cache_misses_total",
    "Total cache misses",
    registry=REGISTRY
)

# ============= HISTOGRAMS =============
# Track latency and duration distributions

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Time taken to make a prediction (seconds)",
    ["hazard_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Time taken for database queries (seconds)",
    ["query_type"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
    registry=REGISTRY
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "Time taken for API requests (seconds)",
    ["endpoint", "method"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
    registry=REGISTRY
)

# ============= GAUGES =============
# Track current values (not cumulative)

active_requests = Gauge(
    "active_requests",
    "Number of active API requests",
    registry=REGISTRY
)

database_connection_pool_size = Gauge(
    "database_connection_pool_size",
    "Current database connection pool size",
    registry=REGISTRY
)

cache_size_bytes = Gauge(
    "cache_size_bytes",
    "Current cache size in bytes",
    registry=REGISTRY
)

model_inference_queue_length = Gauge(
    "model_inference_queue_length",
    "Number of predictions waiting in queue",
    registry=REGISTRY
)

avg_prediction_confidence = Gauge(
    "avg_prediction_confidence",
    "Average confidence of predictions",
    registry=REGISTRY
)

# ============= INFO =============
# Metadata about the application

app_info = Info(
    "multihazard_vlm",
    "MultiHazard-VLM application information",
    registry=REGISTRY
)


# ============= DECORATORS & TRACKING FUNCTIONS =============

def track_prediction_time(hazard_type: str):
    """Decorator to track prediction latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                predictions_total.labels(hazard_type=hazard_type).inc()
                return result
            finally:
                elapsed = time.time() - start_time
                prediction_latency_seconds.labels(hazard_type=hazard_type).observe(elapsed)
        return wrapper
    return decorator


def track_request_time(endpoint: str, method: str = "GET"):
    """Decorator to track API request latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            status_code = 200
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                errors_total.labels(error_type=type(e).__name__).inc()
                raise
            finally:
                elapsed = time.time() - start_time
                active_requests.dec()
                api_request_duration_seconds.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(elapsed)
                api_requests_total.labels(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code
                ).inc()
        return wrapper
    return decorator


def track_db_query(query_type: str):
    """Decorator to track database query performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                database_writes_total.labels(table=query_type).inc()
                return result
            except Exception as e:
                errors_total.labels(error_type=f"db_{type(e).__name__}").inc()
                raise
            finally:
                elapsed = time.time() - start_time
                database_query_duration_seconds.labels(query_type=query_type).observe(elapsed)
        return wrapper
    return decorator


def track_cache_hit():
    """Record a cache hit."""
    cache_hits_total.inc()


def track_cache_miss():
    """Record a cache miss."""
    cache_misses_total.inc()


def get_metrics() -> bytes:
    """Generate Prometheus metrics in text format."""
    return generate_latest(REGISTRY)


def set_app_info(version: str = "1.0.0", model_version: str = "v1.0"):
    """Set application information."""
    app_info.info({
        "version": version,
        "model_version": model_version,
        "service": "multihazard-vlm",
    })


# Initialize app info
set_app_info()
