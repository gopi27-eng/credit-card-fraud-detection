from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd

from src.schemas.transaction import TransactionInput
from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger("inference_engine")


class FraudPredictor:
    """Loads preprocessor and model artifacts to run low-latency batch/real-time inference."""

    def __init__(self, config: AppConfig, threshold: float = 0.5):
        self.config = config
        self.threshold = threshold
        self.preprocessor_path = Path(config.data.preprocessor_path)
        self.model_path = Path(config.model.artifact_path)
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(f"Preprocessor artifact missing at: {self.preprocessor_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model artifact missing at: {self.model_path}")

        logger.info("Loading preprocessor and model artifacts...")
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        logger.info("Artifacts successfully loaded into memory.")

    def predict(self, transactions: List[TransactionInput]) -> List[Dict[str, Any]]:
        """Processes transaction payloads and outputs fraud probability and binary decision."""
        if not transactions:
            return []

        payload_dicts = [t.model_dump() for t in transactions]
        df = pd.DataFrame(payload_dicts)

        transaction_ids = df[self.config.data.id_column].tolist()

        # Transform raw features using persisted ColumnTransformer
        X_transformed = self.preprocessor.transform(df)

        # Predict fraud probability (class 1)
        probabilities = self.model.predict_proba(X_transformed)[:, 1]

        results = []
        for tx_id, prob in zip(transaction_ids, probabilities):
            is_fraud = bool(prob >= self.threshold)
            results.append(
                {
                    "transaction_id": tx_id,
                    "fraud_probability": round(float(prob), 4),
                    "is_fraud": is_fraud,
                }
            )
        return results