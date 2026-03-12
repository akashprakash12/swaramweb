import os

# Render injects PORT at runtime. Falling back to 10000 helps local tests.
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Keep a single worker on small/free instances to avoid OOM with TF/MediaPipe.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "180"))