from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.models.train import ModelTrainer
from src.utils.config import AppConfig, BaseConfig, DataConfig, MLflowConfig, ModelConfig, SplitConfig


@pytest.fixture
def mock_transformed_data(tmp_path: Path):
    """Generates synthetic transformed parquet splits with target column."""
    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    # Synthetic dataset with imbalanced targets (10% positive)
    n_train, n_test, n_feats = 100, 20, 5
    feature_cols = [f"f_{i}" for i in range(n_feats)]

    X_train = np.random.randn(n_train, n_feats)
    y_train = (np.random.rand(n_train) < 0.1).astype(int)

    X_test = np.random.randn(n_test, n_feats)
    y_test = (np.random.rand(n_test) < 0.1).astype(int)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["is_fraud"] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["is_fraud"] = y_test

    train_path = data_dir / "train_transformed.parquet"
    test_path = data_dir / "test_transformed.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    config = AppConfig(
        base=BaseConfig(project_name="fraud-test", random_state=42),
        data=DataConfig(
            raw_data_path=tmp_path / "raw.csv",
            processed_train_path=tmp_path / "train.parquet",
            processed_test_path=tmp_path / "test.parquet",
            preprocessor_path=artifacts_dir / "preprocessor.joblib",
            transformed_train_path=train_path,
            transformed_test_path=test_path,
            target_column="is_fraud",
            id_column="id",
            numerical_features=["f_0"],
            categorical_features=[],
            binary_features=[],
            split=SplitConfig(test_size=0.2, stratify=False),
        ),
        model=ModelConfig(
            artifact_path=artifacts_dir / "model.joblib",
            algorithm="xgboost",
            params={
                "n_estimators": 5,
                "max_depth": 2,
                "learning_rate": 0.1,
                "eval_metric": "aucpr",
                "random_state": 42,
            },
        ),
        mlflow=MLflowConfig(experiment_name="test-experiments"),
    )
    return config


def test_model_trainer_execution(mock_transformed_data: AppConfig):
    """Ensures model trains, computes metrics, and produces serialized artifact."""
    trainer = ModelTrainer(mock_transformed_data, log_to_mlflow=False)
    metrics = trainer.train()

    # Verify metrics dictionary keys and ranges
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0

    # Verify artifact was persisted
    model_path = mock_transformed_data.model.artifact_path
    assert model_path.is_file()
    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, XGBClassifier)