import os

# Bind to PORT from environment or default
port = os.getenv('PORT', '10000')
bind = f"0.0.0.0:{port}"

# Worker configuration for memory efficiency
workers = 1  # Keep at 1 for free tier to avoid OOM
threads = int(os.getenv('GUNICORN_THREADS', '2'))
timeout = int(os.getenv('GUNICORN_TIMEOUT', '120'))

# Logging
accesslog = '-'
errorlog = '-'
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')

# Preload app for faster startup
preload_app = True