from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("parkinson_model.h5")

app = FastAPI(title="Parkinson Detection API")

# =========================
# REQUEST FORMAT
# =========================
class PredictionInput(BaseModel):
    sequence: list

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(data: PredictionInput):

    try:
        # Convert input to numpy
        sequence = np.array(data.sequence)

        # Ensure correct shape
        sequence = sequence.reshape(1, 30, 1404)

        # Prediction
        prediction = model.predict(sequence, verbose=0)

        healthy_prob = float(prediction[0][0]) * 100
        parkinson_prob = float(prediction[0][1]) * 100

        diagnosis = "Parkinson" if parkinson_prob > healthy_prob else "Healthy"

        severity = parkinson_prob if diagnosis == "Parkinson" else (100 - healthy_prob)

        # Stage estimation
        if severity < 40:
            stage = "Mild"
        elif severity < 70:
            stage = "Moderate"
        else:
            stage = "Severe"

        return {
            "diagnosis": diagnosis,
            "healthy_probability": round(healthy_prob, 2),
            "parkinson_probability": round(parkinson_prob, 2),
            "severity": round(severity, 2),
            "stage": stage
        }

    except Exception as e:
        return {"error": str(e)}