from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM

class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

model = tf.keras.models.load_model(
    "parkinson_model_balanced.h5",
    custom_objects={'LSTM': CompatibleLSTM},
    compile=False
)

app = FastAPI(title="Parkinson Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEAN = 0.3069034922764374
STD  = 0.26552994273652963

class PredictionInput(BaseModel):
    sequence: list

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        sequence = np.array(data.sequence)
        sequence = sequence.reshape(1, 30, 1404)
        sequence = (sequence - MEAN) / STD
        prediction = model.predict(sequence, verbose=0)
        healthy_prob = float(prediction[0][0]) * 100
        parkinson_prob = float(prediction[0][1]) * 100
        diagnosis = "Parkinson" if parkinson_prob > healthy_prob else "Healthy"
        severity = parkinson_prob if diagnosis == "Parkinson" else (100 - healthy_prob)
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