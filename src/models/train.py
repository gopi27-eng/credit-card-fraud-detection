import os
from pathlib import Path
from typing import Dict
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger("model_training")


class ModelTrainer:
  """Trains fraud detection models with class imbalance compensation and MLflow logging."""

  def __init__(self, config: AppConfig, log_to_mlflow: bool = True):
    self.config = config
    self.model_cfg = config.model
    self.data_cfg = config.data
    self.log_to_mlflow = log_to_mlflow

    if self.log_to_mlflow:
      if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
      else:
        import dagshub

        dagshub.init(
            repo_owner="gopi27-eng",
            repo_name="credit-card-fraud-detection",
            mlflow=True,
        )

  def calculate_scale_pos_weight(self, y: np.ndarray) -> float:
    """Calculates ratio of negative samples to positive samples for XGBoost."""
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    if pos_count == 0:
      return 1.0
    return float(neg_count / pos_count)

  def train(self) -> Dict[str, float]:
    """Loads transformed data, trains XGBoost, evaluates metrics, and logs run."""
    train_path = Path(self.data_cfg.transformed_train_path)
    test_path = Path(self.data_cfg.transformed_test_path)

    if not train_path.is_file() or not test_path.is_file():
      raise FileNotFoundError(
          "Transformed parquet files not found. Run transformation first."
      )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = self.data_cfg.target_column
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].to_numpy()

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].to_numpy()

    scale_pos_weight = self.calculate_scale_pos_weight(y_train)
    logger.info(
        f"Class imbalance scale_pos_weight calculated: {scale_pos_weight:.2f}"
    )

    model_params = dict(self.model_cfg.params)
    model_params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**model_params)
    logger.info(f"Training {self.model_cfg.algorithm} model...")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }

    logger.info(f"Evaluation Metrics: {metrics}")

    artifact_path = Path(self.model_cfg.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    logger.info(f"Model saved to {artifact_path}")

    if self.log_to_mlflow:
      mlflow.set_experiment(self.config.mlflow.experiment_name)
      with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        logger.info("Logged model, parameters, and metrics to DagsHub MLflow.")

    return metrics


if __name__ == "__main__":
  cfg = load_config()
  trainer = ModelTrainer(cfg, log_to_mlflow=True)
  trainer.train()