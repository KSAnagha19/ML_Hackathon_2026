
from pathlib import Path

import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference_service import (
    predict_trip,
    get_available_trips,
    BETA,
    TRANSITION_WEIGHT
)


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgboost_tuned.joblib"

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching using XGBoost + HMM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class PredictionRequest(BaseModel):
    trip_id: str


@app.get("/")
def home():
    return {
        "application": "PathSnap",
        "message": "Backend is running successfully!",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "hmm_connected": True
    }


@app.get("/model-info")
def model_info():
    bundle = joblib.load(MODEL_PATH)

    return {
        "model_name": type(
            bundle["model"]
        ).__name__,
        "model_loaded": True,
        "features": list(bundle["features"]),
        "feature_count": len(bundle["features"]),
        "hmm_beta": BETA,
        "hmm_transition_weight": TRANSITION_WEIGHT,
        "hmm_connected": True
    }


@app.get("/trips")
def list_trips():
    trips = get_available_trips()

    return {
        "total_trips": len(trips),
        "trip_ids": trips
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_trip(request.trip_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}"
        )
