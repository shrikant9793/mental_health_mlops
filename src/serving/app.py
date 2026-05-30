import os
import time
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.serving.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    BatchPredictRequest,
    BatchPredictResponse
)
from src.serving.model_loader import (
    load_config,
    load_model_from_registry,
    get_model_info
)


# ─── Global State ────────────────────────────────────────────
MODEL      = None
MODEL_INFO = {}
CONFIG     = {}
MODEL_NAME = "MentalHealthClassifier"
ALIAS      = "Production"


# ─── Lifespan — Load model at startup ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when API starts — release when API stops."""
    global MODEL, MODEL_INFO, CONFIG

    print("⏳ Starting API — loading model...")
    CONFIG     = load_config()
    MODEL      = load_model_from_registry(MODEL_NAME, ALIAS, CONFIG)
    MODEL_INFO = get_model_info(MODEL_NAME, ALIAS, CONFIG)
    print("✅ Model loaded — API ready!")

    yield  # API runs here

    print("🛑 Shutting down API...")


# ─── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Classifier API",
    description="Depression detection from text using MLOps pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# ─── CORS Middleware ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─── Helper ───────────────────────────────────────────────────
def get_label(prediction: int) -> str:
    """Convert prediction integer to human readable label."""
    return "Depression" if prediction == 1 else "Non-Depression"


def get_confidence(model, text: str) -> float:
    """
    Get confidence score.
    LinearSVC uses decision function — convert to 0-1 range.
    """
    try:
        proba = model.predict_proba([text])[0]
        return round(float(max(proba)), 4)
    except AttributeError:
        # LinearSVC — use decision function
        decision = model.decision_function([text])[0]
        # Normalize to 0-1 using sigmoid
        confidence = 1 / (1 + (2.718 ** -decision))
        return round(float(confidence), 4)


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message" : "Mental Health Classifier API",
        "docs"    : "/docs",
        "health"  : "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    Returns API status and loaded model info.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return HealthResponse(
        status          = "healthy",
        model_name      = MODEL_INFO.get("model_name", MODEL_NAME),
        model_version   = str(MODEL_INFO.get("model_version", "unknown")),
        experiment_name = CONFIG["mlflow"]["experiment_name"]
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Single text prediction endpoint.
    Returns depression prediction with confidence score.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # Predict
        prediction = int(MODEL.predict([request.text])[0])
        confidence = get_confidence(MODEL, request.text)
        label      = get_label(prediction)

        return PredictResponse(
            text       = request.text,
            prediction = prediction,
            label      = label,
            confidence = confidence
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction endpoint.
    Accepts list of texts — returns predictions for all.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        predictions = []
        for text in request.texts:
            prediction = int(MODEL.predict([text])[0])
            confidence = get_confidence(MODEL, text)
            label      = get_label(prediction)

            predictions.append(PredictResponse(
                text       = text,
                prediction = prediction,
                label      = label,
                confidence = confidence
            ))

        return BatchPredictResponse(
            predictions = predictions,
            total       = len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )
        