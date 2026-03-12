# Swaram Web

Browser-based sign language recognition web app built with Flask, MediaPipe, and TensorFlow Lite.
The app captures camera frames in the browser, sends them to the Flask backend, extracts pose and hand keypoints, and predicts one of the trained gesture labels in real time.

## Included in this repository

- `app.py` Flask server and TFLite inference pipeline
- `templates/index.html` web UI for camera capture and prediction display
- `model.tflite` exported inference model
- `scaler.json` feature scaler parameters
- `labels.json` class labels used by the model
- `requirements.txt` pinned Python dependencies

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Notes

- The server keeps a rolling 30-frame buffer per browser client before returning a prediction.
- Predictions are accepted when confidence is at least `0.85`.
- The repository intentionally excludes training data, model training scripts, local virtual environments, and mobile app sources.

## Hosting

This project cannot run on GitHub Pages because it needs a Python backend for Flask, MediaPipe, and TensorFlow Lite. GitHub stores the code, but it does not run this server-side app for you.

Python is pinned to `3.11.11` for compatibility with `tensorflow==2.13.1` via `.python-version` and `runtime.txt`.

The simplest option is Render:

1. Push the repository to GitHub.
2. Go to Render and create a new `Web Service` from this repository.
3. Render will detect `render.yaml` automatically.
4. Deploy the service.

If you configure it manually on Render, use:

```bash
Build command: pip install -r requirements.txt
Start command: gunicorn --bind 0.0.0.0:$PORT app:app
```

Docker option on Render:

1. Create service using `Environment: Docker`.
2. Render will use `Dockerfile` in this repository.
3. If needed, override start command with:

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

Other hosts that can run this app:

- Railway
- Fly.io
- A Linux VPS with Nginx + Gunicorn

GitHub Pages only works if you rewrite the project as a static-only frontend and move inference elsewhere.
