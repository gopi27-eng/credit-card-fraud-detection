from pathlib import Path
import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.data.transformation import DataTransformationPipeline
from src.utils.config import AppConfig, BaseConfig, DataConfig, MLflowConfig, SplitConfig


@pytest.fixture
def mock_dataset_splits(tmp_path: Path):
    """Creates synthetic train and test parquet files matching production schema."""
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_data = {
        "transaction_id": [1, 2, 3, 4],
        "amount": [10.0, 50.0, 100.0, 20.0],
        "transaction_hour": [8, 12, 18, 23],
        "device_trust_score": [80, 60, 40, 90],
        "velocity_last_24h": [1, 3, 5, 0],
        "cardholder_age": [25, 40, 65, 30],
        "merchant_category": ["Grocery", "Electronics", "Food", "Grocery"],
        "foreign_transaction": [0, 1, 0, 0],
        "location_mismatch": [0, 0, 1, 0],
        "is_fraud": [0, 1, 0, 0],
    }

    test_data = {
        "transaction_id": [5, 6],
        "amount": [15.0, 200.0],
        "transaction_hour": [9, 21],
        "device_trust_score": [75, 30],
        "velocity_last_24h": [2, 6],
        "cardholder_age": [28, 55],
        "merchant_category": ["Clothing", "Grocery"],  # 'Clothing' not present in train
        "foreign_transaction": [0, 1],
        "location_mismatch": [0, 1],
        "is_fraud": [0, 1],
    }

    train_path = processed_dir / "train.parquet"
    test_path = processed_dir / "test.parquet"

    pd.DataFrame(train_data).to_parquet(train_path, index=False)
    pd.DataFrame(test_data).to_parquet(test_path, index=False)

    config = AppConfig(
        base=BaseConfig(project_name="fraud-test", random_state=42),
        data=DataConfig(
            raw_data_path=tmp_path / "data" / "raw" / "dummy.csv",
            processed_train_path=train_path,
            processed_test_path=test_path,
            preprocessor_path=artifacts_dir / "preprocessor.joblib",
            transformed_train_path=processed_dir / "train_transformed.parquet",
            transformed_test_path=processed_dir / "test_transformed.parquet",
            target_column="is_fraud",
            id_column="transaction_id",
            numerical_features=[
                "amount",
                "transaction_hour",
                "device_trust_score",
                "velocity_last_24h",
                "cardholder_age",
            ],
            categorical_features=["merchant_category"],
            binary_features=["foreign_transaction", "location_mismatch"],
            split=SplitConfig(test_size=0.2, stratify=False),
        ),
        mlflow=MLflowConfig(experiment_name="test-exp"),
    )

    return config


def test_transformation_pipeline_run(mock_dataset_splits: AppConfig):
    """Verifies complete transformation execution, file creation, and schema integrity."""
    pipeline = DataTransformationPipeline(mock_dataset_splits)
    pipeline.run()

    preprocessor_path = mock_dataset_splits.data.preprocessor_path
    train_transformed_path = mock_dataset_splits.data.transformed_train_path
    test_transformed_path = mock_dataset_splits.data.transformed_test_path

    # 1. Assert preprocessor artifact saved and reloadable
    assert preprocessor_path.is_file()
    transformer = joblib.load(preprocessor_path)
    assert isinstance(transformer, ColumnTransformer)

    # 2. Assert transformed parquet files exist
    assert train_transformed_path.is_file()
    assert test_transformed_path.is_file()

    train_trans_df = pd.read_parquet(train_transformed_path)
    test_trans_df = pd.read_parquet(test_transformed_path)

    # 3. Assert shapes and target presence
    assert len(train_trans_df) == 4
    assert len(test_trans_df) == 2
    assert "is_fraud" in train_trans_df.columns
    assert "is_fraud" in test_trans_df.columns

    # 4. Assert no null values produced post-transformation
    assert train_trans_df.isnull().sum().sum() == 0
    assert test_trans_df.isnull().sum().sum() == 0


def test_unknown_category_handling(mock_dataset_splits: AppConfig):
    """Verifies that unobserved categorical levels in test data do not throw errors."""
    pipeline = DataTransformationPipeline(mock_dataset_splits)
    pipeline.run()

    preprocessor = joblib.load(mock_dataset_splits.data.preprocessor_path)

    # Test payload containing entirely novel category
    unseen_payload = pd.DataFrame(
        [
            {
                "amount": 99.0,
                "transaction_hour": 14,
                "device_trust_score": 50,
                "velocity_last_24h": 1,
                "cardholder_age": 35,
                "merchant_category": "LuxuryGoods",  # Novel category
                "foreign_transaction": 0,
                "location_mismatch": 0,
            }
        ]
    )

    # Transform must succeed without throwing an exception
    transformed_vector = preprocessor.transform(unseen_payload)
    assert transformed_vector.shape[0] == 1