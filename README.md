# ParkinScan — AI-Powered Parkinson's Detection System

> A graduation project that uses facial expression analysis and deep learning to provide an early indicator of Parkinson's disease.

---

## What is ParkinScan?

ParkinScan is a web-based application that analyzes subtle facial movement patterns to screen for early signs of Parkinson's disease. Using your camera, the system tracks 468 facial landmarks in real time across 6 guided expressions, then sends the data to an AI model that was trained on 244 YouTube videos of Parkinson's patients and healthy individuals.

The entire test takes under 2 minutes and runs directly in the browser — no installation required for the user.

---

## How It Works

1. **Camera Access** — The user allows webcam access in the browser
2. **Facial Tracking** — MediaPipe FaceMesh tracks 468 landmarks on the user's face in real time
3. **6 Expressions** — The user performs Neutral, Happiness, Sadness, Anger, Surprise, and Disgust (5 seconds each)
4. **AI Analysis** — 30 frames of landmark data (5 per expression × 6 expressions) are sent to the backend
5. **Result** — A CNN+LSTM neural network returns a diagnosis with probability scores and severity level

---

## Project Structure

```
parkinson_web_app/
├── backend/
│   ├── api.py                  # FastAPI server with prediction endpoint
│   └── parkinson_model.h5      # Trained CNN+LSTM model
└── frontend/
    └── index.html              # Complete single-page web application
```

---

## Tech Stack

| Layer          | Technology                         |
| -------------- | ---------------------------------- |
| Frontend       | HTML, CSS, JavaScript              |
| Face Tracking  | MediaPipe FaceMesh (browser-based) |
| Backend        | FastAPI (Python)                   |
| AI Model       | TensorFlow / Keras — CNN + LSTM    |
| PDF Generation | jsPDF                              |
| Server         | Uvicorn                            |

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

- Python 3.9+
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/sinemnagod/parkinson_web_app.git
cd parkinson_web_app
```

### Step 2 — Install backend dependencies

```bash
cd backend
pip install fastapi uvicorn numpy opencv-python
pip install tensorflow-macos  # macOS (Apple Silicon)
# OR
pip install tensorflow        # Windows / Linux / Intel Mac
```

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

Open `frontend/index.html` in your browser using a local server such as VS Code Live Server, or navigate to it directly.

> **Note:** The frontend must be served over HTTP (not opened as a file) for camera access to work properly in Chrome.

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

---

## Known Limitations

### Class Imbalance

The training dataset is imbalanced — 77% healthy vs 23% Parkinson's subjects. Class balancing was not applied during training, which means the model has a bias toward predicting "Healthy." This is a known limitation and a direction for future improvement.

**Planned fix:** Retrain the model with `class_weight` parameter using scikit-learn's `compute_class_weight`.

### Lighting Sensitivity

MediaPipe facial landmark detection degrades significantly in poor lighting. Users should perform the test in a well-lit environment facing a light source.

### Camera Quality

Lower resolution webcams may produce less accurate landmark coordinates, affecting model predictions.

### Not a Medical Device

ParkinScan is a research prototype developed as a graduation project. It is **not** a certified medical device and should not be used as a substitute for professional neurological diagnosis.

---

## Results Screen

The results page displays:

- **Diagnosis** — Healthy or Parkinson's Detected
- **Healthy Probability** — percentage with animated bar
- **Parkinson's Probability** — percentage with animated bar
- **Severity Level** — Mild / Moderate / Severe (if Parkinson's detected)
- **Save Results** — downloads a professionally formatted PDF report

---

## Team

| Name           | Role                              |
| -------------- | --------------------------------- |
| Sinem Doğan    | Frontend Development & Deployment |
| Teammate 2     | Data Preprocessing & Pipeline     |
| Teammate 3 & 4 | Model Architecture & Training     |

---

## Academic Context

This project was developed as a graduation project for the Software Engineering program. It demonstrates:

- End-to-end AI application development
- Real-time computer vision in the browser
- REST API design and integration
- Deep learning model deployment
- Full-stack web development

---

## References

- **YouTubePD Dataset** — Used for model training
- **MediaPipe FaceMesh** — Google's facial landmark detection library
- **FastAPI** — Modern Python web framework
- **TensorFlow/Keras** — Deep learning framework

---

> ⚠️ **Medical Disclaimer:** ParkinScan is a research and educational tool. It is not a certified medical device and should not be used as a substitute for professional medical diagnosis. Always consult a qualified neurologist for any health concerns.
