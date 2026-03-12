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
