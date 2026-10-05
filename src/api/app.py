import time
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.models.predict import FraudPredictor
from src.schemas.transaction import (
    BatchTransactionInferenceRequest,
    PredictionOutput,
    TransactionInferenceRequest,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("fastapi_service")
predictor: FraudPredictor = None

# Custom Observability Metrics
INFERENCE_REQUEST_COUNT = Counter(
    "fraud_inference_requests_total",
    "Total number of inference requests received",
    ["status"]
)
FRAUD_DETECTIONS_COUNT = Counter(
    "fraud_detections_total",
    "Total count of flagged fraudulent transactions",
    ["decision"]
)
INFERENCE_LATENCY = Histogram(
    "fraud_inference_latency_seconds",
    "Latency of inference execution in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        config = load_config()
        predictor = FraudPredictor(config, threshold=0.5)
        logger.info("FraudPredictor successfully loaded at server startup.")
    except Exception as exc:
        logger.error(f"Failed to load model artifacts: {exc}")
        raise exc
    yield
    logger.info("Shutting down Fraud Detection API service.")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/metrics")
async def metrics():
    """Scrape endpoint for Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
    return {"status": "healthy", "service": "credit-card-fraud-detection"}

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict_single(payload: TransactionInferenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference engine unavailable.")

    start_time = time.perf_counter()
    try:
        results = predictor.predict([payload])
        res = results[0]
        decision = "fraud" if res["is_fraud"] else "legitimate"
        
        INFERENCE_REQUEST_COUNT.labels(status="success").inc()
        FRAUD_DETECTIONS_COUNT.labels(decision=decision).inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

        return PredictionOutput(
            transaction_id=res["transaction_id"],
            is_fraud=int(res["is_fraud"]),
            fraud_probability=res["fraud_probability"],
        )
    except Exception as exc:
        INFERENCE_REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/predict/batch", response_model=List[PredictionOutput], status_code=status.HTTP_200_OK)
async def predict_batch(payload: BatchTransactionInferenceRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference engine unavailable.")

    start_time = time.perf_counter()
    try:
        results = predictor.predict(payload.instances)
        for r in results:
            decision = "fraud" if r["is_fraud"] else "legitimate"
            FRAUD_DETECTIONS_COUNT.labels(decision=decision).inc()

        INFERENCE_REQUEST_COUNT.labels(status="success").inc()
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

        return [
            PredictionOutput(
                transaction_id=r["transaction_id"],
                is_fraud=int(r["is_fraud"]),
                fraud_probability=r["fraud_probability"],
            )
            for r in results
        ]
    except Exception as exc:
        INFERENCE_REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Batch prediction failure: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))