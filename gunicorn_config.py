# Gunicorn configuration file for production deployment
# Usage: gunicorn -c gunicorn_config.py api.app:app

import os
import multiprocessing

# Server socket
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "logs/gunicorn_access.log")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "logs/gunicorn_error.log")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "multihazard-vlm"

# Server mechanics
daemon = False
pidfile = "gunicorn.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# Deployment
preload_app = False
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "127.0.0.1")
secure_scheme_headers = {
    "X-FORWARDED_PROTOCOL": "ssl",
    "X-FORWARDED_PROTO": "https",
    "X-FORWARDED_SSL": "on",
}

# Application
raw_env = [
    "PROMETHEUS_MULTIPROC_DIR=/tmp",
]

# SSL (optional - set these if using HTTPS)
# keyfile = "/path/to/keyfile.key"
# certfile = "/path/to/certfile.crt"
# ca_certs = "/path/to/ca_bundle.crt"
# cert_reqs = 2
# ssl_version = "TLSv1_2"

# Setup environment
os.makedirs("logs", exist_ok=True)
