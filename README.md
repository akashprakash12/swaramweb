# 👐 Swaram: A Sign Language Recognition System with Lip-Reading Support

> **Bachelor of Technology Project**  
> *Department of Computer Science & Engineering*  
> *Government Engineering College Idukki*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-red.svg)](https://mediapipe.dev/)

🔗 **GitHub Repository:** [akashprakash12/swaramweb](https://github.com/akashprakash12/swaramweb)

## 📖 Overview

**Swaram** is an intelligent assistive communication system designed to help individuals with hearing or speech impairments by converting visual communication into understandable text and speech. The system focuses on **real-time sign language recognition** with an experimental lip-reading module as a supplementary feature.

The core system uses **MediaPipe** for hand landmark detection and an **LSTM-based model** for gesture classification, achieving reliable sign language interpretation. An experimental lip-reading module was implemented using **3D CNN** and **BiLSTM** networks, though it showed limited accuracy due to challenges in visually distinguishing speech patterns, especially in Malayalam.

Recognized outputs are processed through an **NLP module** for sentence generation and converted to speech via **Text-to-Speech (TTS)**, all deployable as both web and mobile applications.

## 🎯 Motivation

Communication barriers significantly affect social inclusion and quality of life for individuals with hearing or speech impairments. While sign language is their primary means of communication, most people are not trained to understand it.

Recent advances in computer vision and deep learning—such as CNNs, RNNs, and real-time tracking frameworks—make it possible to build intelligent systems that interpret visual communication. Swaram aims to bridge this gap by providing a reliable, real-time sign language recognition system with accessible text and speech output.

## ⚠️ Problem Statement

Individuals with hearing and speech impairments face significant communication barriers in daily interactions, including public services, education, and workplaces. Most existing solutions focus on either sign language recognition or lip-reading independently—not as an integrated system.

Lip-reading systems face particular challenges in accurately interpreting speech due to limited visual information, especially for languages like Malayalam where many sounds are not clearly distinguishable through lip movements alone.

**Swaram addresses this by:**
- ✅ Effectively recognizing sign language gestures
- ✅ Converting them into meaningful textual and spoken output
- ✅ Exploring lip-reading as a supplementary feature for future integration

## Project Structure

Main application files:

- `app2.py`: Main Flask web app (sign + lip + LLM + TTS)
- `app.py`: Older/simpler Flask web app variant
- `templates/index.html`: Template used by `app.py`
- `static/`: Static JS/CSS used by template-based UI

Model and metadata artifacts:

- `model.tflite`, `scaler.json`, `labels.json`: Sign model artifacts
- `lip_model.tflite`, `lip_scaler.json`, `lip_labels.json`: Lip model artifacts

Training and pipeline scripts:

- `step1_collect.py`: Collect hand-sign dataset (`dataset/`)
- `step2_train.py`: Train sign model and export artifacts
- `step3_test_desktop.py`: Desktop test for sign inference
- `step4_export_android.py`: Copy model artifacts to `mobile-app/`
- `lip_collect.py`, `lip_train.py`, `lip_test.py`: Lip-reading dataset/train/test pipeline
- `check_tflite.py`, `debug_model_predictions.py`: Model validation/debug utilities

Mobile app folders:

- `mobile-app/`: React Native app source
- `mobile-app-export/`: Export/support assets

## Requirements

- Linux/macOS/Windows
- Python `3.11` recommended
- Webcam access

Install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install web/LLM extras if not already present:

```bash
pip install flask requests gtts groq google-genai google-generativeai
```

## Quick Start (Web App)

Run the main app:

```bash
python app2.py
```

Open:

- `http://127.0.0.1:5000`

By default, the app runs with:

- Sequence length: `30`
- Sign threshold: `0.85`
- Lip threshold: `0.80`

## API Endpoints (`app2.py`)

- `GET /`: Main UI
- `GET /status`: Runtime/model/provider status
- `POST /predict`: Hand-sign prediction from frame image
- `POST /lip_predict`: Lip-reading prediction from frame image
- `POST /reset`: Reset sign client buffer
- `POST /lip_reset`: Reset lip client buffer
- `POST /llm`: Build English sentence + Malayalam translation + optional audio
- `POST /tts`: TTS generation (`audio_b64`)

## Training Workflow

### 1) Collect sign dataset

```bash
python step1_collect.py
```

Optional example:

```bash
python step1_collect.py --signs "Hello,Thanks,Help" --samples 30
```

### 2) Train sign model

```bash
python step2_train.py --data dataset --epochs 120
```

### 3) Test sign model locally

```bash
python step3_test_desktop.py
```

### 4) Export sign artifacts to mobile app

```bash
python step4_export_android.py --app-dir mobile-app
```

## Lip-Reading Workflow

Collect lip dataset:

```bash
python lip_collect.py
```

Train lip model:

```bash
python lip_train.py --data lip_dataset --epochs 120 --arch bilstm
```

Test lip model:

```bash
python lip_test.py --arch bilstm
```

Supported lip model architectures:

- `lstm`
- `bilstm`
- `3dcnn`

## Environment Variables

`app2.py` supports configuration through environment variables:

- `PORT` (default `5000`)
- `LIP_MODEL_PATH`
- `LIP_SCALER_JSON`
- `LIP_LABELS_JSON`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `COHERE_API_KEY`

Example:

```bash
export GROQ_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
python app2.py
```

## Troubleshooting

- If webcam does not open: close other camera apps and retry.
- If predictions stay in `collecting`: ensure good lighting and hand/face visibility.
- If LLM is unavailable: app falls back to local word mapping.
- If TTS fails: ensure `gtts` is installed and internet access is available.
- If MediaPipe/TensorFlow conflict appears: keep versions pinned in `requirements.txt`.



