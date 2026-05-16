from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras

# =========================
# LSTM COMPATIBILITY PATCH
# =========================
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    "parkinson_model_balanced.h5",
    custom_objects={'LSTM': CompatibleLSTM},
    compile=False
)

app = FastAPI(title="Parkinson Detection API")

# =========================
# CORS MIDDLEWARE
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# NORMALIZATION VALUES
# =========================
MEAN = 0.3069034922764374
STD  = 0.26552994273652963

# =========================
# AUTH SETTINGS
# =========================
SECRET_KEY = "parkinscan-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

# =========================
# DATABASE CONNECTION
# =========================
def get_db():
    import os
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        conn = psycopg2.connect(database_url)
    else:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="parkinscan",
            user="sinemdogan"
        )
    return conn

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class PredictionInput(BaseModel):
    sequence: list

class RegisterInput(BaseModel):
    name: str
    surname: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    name: str
    surname: str
    email: str

# =========================
# AUTH HELPERS
# =========================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None

# =========================
# REGISTER ENDPOINT
# =========================
@app.post("/register")
def register(data: RegisterInput):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Check if email already exists
        cur.execute("SELECT id FROM users WHERE email = %s", (data.email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password and save user
        hashed = hash_password(data.password)
        cur.execute(
            "INSERT INTO users (name, surname, email, password) VALUES (%s, %s, %s, %s) RETURNING id",
            (data.name, data.surname, data.email, hashed)
        )
        conn.commit()
        cur.close()
        conn.close()

        return {"message": "Account created successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# LOGIN ENDPOINT
# =========================
@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user or not verify_password(form_data.password, user["password"]):
            raise HTTPException(status_code=401, detail="Incorrect email or password")

        token = create_token({"sub": user["email"]})

        return {
            "access_token": token,
            "token_type": "bearer",
            "name": user["name"],
            "surname": user["surname"],
            "email": user["email"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: PredictionInput, current_user: str = Depends(get_current_user)):
    try:
        sequence = np.array(data.sequence)
        sequence = sequence.reshape(1, 30, 1404)
        sequence = (sequence - MEAN) / STD

        prediction = model.predict(sequence, verbose=0)

        healthy_prob   = float(prediction[0][0]) * 100
        parkinson_prob = float(prediction[0][1]) * 100

        diagnosis = "Parkinson" if parkinson_prob > healthy_prob else "Healthy"
        severity  = parkinson_prob if diagnosis == "Parkinson" else (100 - healthy_prob)

        if severity < 40:
            stage = "Mild"
        elif severity < 70:
            stage = "Moderate"
        else:
            stage = "Severe"

        result = {
            "diagnosis": diagnosis,
            "healthy_probability": round(healthy_prob, 2),
            "parkinson_probability": round(parkinson_prob, 2),
            "severity": round(severity, 2),
            "stage": stage
        }

        # Save result if user is logged in
        if current_user:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = %s", (current_user,))
            user = cur.fetchone()
            if user:
                cur.execute(
                    """INSERT INTO results
                    (user_id, diagnosis, healthy_probability, parkinson_probability, severity, stage)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                    (user[0], diagnosis, round(healthy_prob, 2),
                     round(parkinson_prob, 2), round(severity, 2), stage)
                )
                conn.commit()
            cur.close()
            conn.close()

        return result

    except Exception as e:
        return {"error": str(e)}

# =========================
# HISTORY ENDPOINT
# =========================
@app.get("/history")
def history(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT id FROM users WHERE email = %s", (current_user,))
        user = cur.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        cur.execute(
            """SELECT diagnosis, healthy_probability, parkinson_probability,
               severity, stage, created_at
               FROM results WHERE user_id = %s
               ORDER BY created_at DESC""",
            (user["id"],)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()

        return {"results": [dict(r) for r in results]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))