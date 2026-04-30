# ParkinScan — AI-Powered Parkinson's Detection System

> A graduation project that uses facial expression analysis and deep learning to provide an early indicator of Parkinson's disease.

🌐 **Live App:** [https://parkinscan.netlify.app](https://parkinscan.netlify.app)

---

## What is ParkinScan?

ParkinScan is a web-based application that analyzes subtle facial movement patterns to screen for early signs of Parkinson's disease. Using your camera, the system tracks 468 facial landmarks in real time across 6 guided expressions, then sends the data to an AI model that was trained on 244 YouTube videos of Parkinson's patients and healthy individuals.

The entire test takes under 2 minutes and runs directly in the browser — no installation required for the user.

---

## Live Deployment

| Service              | URL                                 |
| -------------------- | ----------------------------------- |
| Frontend (Netlify)   | https://parkinscan.netlify.app      |
| Backend API (Render) | https://parkinscan-api.onrender.com |

> **Note:** The backend runs on Render's free tier and may take up to 50 seconds to respond after a period of inactivity. Open the API URL once before a demo to wake it up.

---

## How It Works

1. **Consent** — The user is informed about camera usage and asked whether to share anonymized data for research
2. **Camera Access** — The browser requests webcam access; MediaPipe FaceMesh initializes in the background
3. **Facial Tracking** — MediaPipe FaceMesh tracks 468 landmarks on the user's face in real time, entirely in the browser
4. **6 Expressions** — The user performs Neutral, Happiness, Sadness, Anger, Surprise, and Disgust (5 seconds each) with a 3-second get-ready countdown between each
5. **AI Analysis** — 30 frames of landmark data (5 per expression × 6 expressions) are sent to the backend API
6. **Result** — A CNN+LSTM neural network returns a diagnosis with probability scores and severity level, downloadable as a PDF report

---

## Project Structure

```
parkinson_web_app/
├── backend/
│   ├── api.py                  # FastAPI server with prediction endpoint
│   ├── parkinson_model.h5      # Trained CNN+LSTM model
│   └── requirements.txt        # Python dependencies
└── frontend/
    └── index.html              # Complete single-page web application
```

---

## Tech Stack

| Layer            | Technology                                            |
| ---------------- | ----------------------------------------------------- |
| Frontend         | HTML, CSS, JavaScript (single-page app)               |
| Face Tracking    | MediaPipe FaceMesh (browser-based, JS CDN)            |
| PDF Generation   | jsPDF (browser-based, JS CDN)                         |
| Backend          | FastAPI (Python)                                      |
| AI Model         | TensorFlow 2.21.0 / Keras — CNN + LSTM                |
| Server           | Uvicorn (ASGI)                                        |
| Frontend Hosting | Netlify (auto-deploys from GitHub)                    |
| Backend Hosting  | Render (Python web service, auto-deploys from GitHub) |

---

## Model Architecture

The model was trained on the **YouTubePD dataset** (244 videos).

```
Input: (30 frames, 1404 features)  ← 468 landmarks × x,y,z per frame
  → Conv1D (64 filters) + BatchNorm + MaxPool
  → Conv1D (128 filters) + BatchNorm + MaxPool
  → LSTM (64 units) + Dropout (0.4)
  → Dense (64) + Dropout (0.3)
  → Dense (2, softmax)  ← [Healthy, Parkinson's]
```

**Training details:**

- Dataset: 244 YouTube videos, 30-frame sequences
- Train/Test split: 80/20 (stratified)
- Input normalization: Z-score (mean=0.307, std=0.266)
- Optimizer: Adam
- Loss: Categorical crossentropy
- Early stopping: patience=5

---

## Running the Project Locally

### Prerequisites

- Python 3.11
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/sinemnagod/parkinson_web_app.git
cd parkinson_web_app
```

### Step 2 — Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **macOS (Apple Silicon):** If `tensorflow` fails to install, replace it in `requirements.txt` with `tensorflow-macos` and run again.

### Step 3 — Start the backend server

```bash
uvicorn api:app --reload
```

You should see:

```
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

### Step 4 — Open the frontend

Open `frontend/index.html` using VS Code Live Server, or any local HTTP server.

> **Note:** The frontend must be served over HTTP (not opened as a raw file) for camera access to work in Chrome.

---

## API Reference

### `POST /predict`

Accepts a sequence of facial landmarks and returns a diagnosis.

**Request body:**

```json
{
  "sequence": [[1404 values], [1404 values], ...]
}
```

Shape: `(30, 1404)` — 30 frames, 468 landmarks × 3 (x, y, z)

**Response:**

```json
{
  "diagnosis": "Healthy",
  "healthy_probability": 97.9,
  "parkinson_probability": 2.1,
  "severity": 2.1,
  "stage": "Mild"
}
```

The backend applies z-score normalization (mean=0.307, std=0.266) before inference, matching the training pipeline.

---

## Features

- **Real-time face tracking** — MediaPipe runs entirely in the browser; no video is sent to the server
- **Guided expression protocol** — 6 expressions with countdown timers, get-ready overlays, and a face-positioning guide
- **Face detection warning** — Real-time alert if no face is detected during recording
- **Record Again** — Users can redo any individual expression without restarting the test
- **PDF report** — Downloadable report with diagnosis, probabilities, severity, and medical disclaimer
- **Privacy-first** — Only numerical landmark coordinates are transmitted; no video data leaves the device

---

## Known Limitations

### Class Imbalance

The training dataset is imbalanced — 77% healthy vs 23% Parkinson's subjects. Class balancing was not applied during training, which introduces a bias toward predicting "Healthy."

**Planned fix:** Retrain the model with `class_weight` using scikit-learn's `compute_class_weight`.

### Lighting Sensitivity

MediaPipe landmark detection degrades in poor lighting. Users should perform the test in a well-lit environment.

### Camera Quality

Lower resolution webcams may produce less accurate landmark coordinates, affecting predictions.

### Not a Medical Device

ParkinScan is a research prototype developed as a graduation project. It is **not** a certified medical device and should not be used as a substitute for professional neurological diagnosis.

---

## Team

| Name           | Role                              |
| -------------- | --------------------------------- |
| Sinem Doğan    | Frontend Development & Deployment |
| Teammate 2     | Data Preprocessing & Pipeline     |
| Teammate 3 & 4 | Model Architecture & Training     |

---

## Academic Context

This project was developed as a graduation project for the Software Engineering program at Istanbul Aydin University. It demonstrates:

- End-to-end AI application development
- Real-time computer vision in the browser
- REST API design and integration
- Deep learning model deployment
- Full-stack web development
- Cloud deployment (Netlify + Render)

---

## References

- **YouTubePD Dataset** — Used for model training
- **MediaPipe FaceMesh** — Google's real-time facial landmark detection library
- **FastAPI** — Modern Python web framework
- **TensorFlow / Keras** — Deep learning framework
- **jsPDF** — Client-side PDF generation library

---

> ⚠️ **Medical Disclaimer:** ParkinScan is a research and educational tool. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis. Always consult a qualified neurologist for any health concerns.
