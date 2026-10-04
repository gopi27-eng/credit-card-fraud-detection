from pathlib import Path
import pandas as pd
import pytest

from src.data.ingest import DataIngestionPipeline
from src.utils.config import AppConfig, BaseConfig, DataConfig, MLflowConfig, SplitConfig


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Creates a deterministic synthetic dataset matching schema definitions."""
    records = []
    # 20 samples: 16 normal (0), 4 fraud (1)
    for i in range(1, 21):
        is_fraud = 1 if i % 5 == 0 else 0
        records.append(
            {
                "transaction_id": i,
                "amount": 50.0 + (i * 2.5),
                "transaction_hour": i % 24,
                "device_trust_score": 50 + (i * 2),
                "velocity_last_24h": i % 5,
                "cardholder_age": 25 + i,
                "merchant_category": "Grocery" if i % 2 == 0 else "Food",
                "foreign_transaction": 0,
                "location_mismatch": 0,
                "is_fraud": is_fraud,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def mock_pipeline_config(tmp_path: Path, sample_dataframe: pd.DataFrame) -> AppConfig:
    """Configures an AppConfig instance pointing to isolated temporary directories."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = raw_dir / "creditcard.csv"
    sample_dataframe.to_csv(raw_csv_path, index=False)

    return AppConfig(
        base=BaseConfig(project_name="fraud-test", random_state=42),
        data=DataConfig(
            raw_data_path=raw_csv_path,
            processed_train_path=processed_dir / "train.parquet",
            processed_test_path=processed_dir / "test.parquet",
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
            split=SplitConfig(test_size=0.2, stratify=True),
        ),
        mlflow=MLflowConfig(experiment_name="test-exp"),
    )


def test_pipeline_run_success(mock_pipeline_config: AppConfig):
    """Ensures pipeline completes, splits data, and produces valid Parquet outputs."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    pipeline.run()

    train_path = mock_pipeline_config.data.processed_train_path
    test_path = mock_pipeline_config.data.processed_test_path

    assert train_path.is_file()
    assert test_path.is_file()

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 20 samples with test_size=0.2 -> 16 train, 4 test
    assert len(train_df) == 16
    assert len(test_df) == 4

    # Stratification check: 20% fraud in train and test
    assert 1 in train_df["is_fraud"].values
    assert 1 in test_df["is_fraud"].values


def test_pipeline_raw_file_missing(mock_pipeline_config: AppConfig):
    """Ensures FileNotFoundError is raised if raw CSV does not exist."""
    mock_pipeline_config.data.raw_data_path = Path("non_existent_creditcard.csv")
    pipeline = DataIngestionPipeline(mock_pipeline_config)

    with pytest.raises(FileNotFoundError):
        pipeline.run()


def test_validation_empty_dataframe(mock_pipeline_config: AppConfig):
    """Ensures empty dataframes are rejected immediately."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Raw dataset is empty"):
        pipeline.validate_dataset_structure(empty_df)


def test_validation_missing_columns(mock_pipeline_config: AppConfig, sample_dataframe: pd.DataFrame):
    """Ensures missing required columns triggers ValueError."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    corrupted_df = sample_dataframe.drop(columns=["device_trust_score"])

    with pytest.raises(ValueError, match="Dataset missing required columns"):
        pipeline.validate_dataset_structure(corrupted_df)


def test_validation_duplicate_ids(mock_pipeline_config: AppConfig, sample_dataframe: pd.DataFrame):
    """Ensures duplicate transaction IDs trigger ValueError."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    # Duplicate first row's ID
    sample_dataframe.loc[1, "transaction_id"] = sample_dataframe.loc[0, "transaction_id"]

    with pytest.raises(ValueError, match="Found 1 duplicate transaction IDs"):
        pipeline.validate_dataset_structure(sample_dataframe)


def test_validation_single_class_target(mock_pipeline_config: AppConfig, sample_dataframe: pd.DataFrame):
    """Ensures ValueError if dataset contains only one target class."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    sample_dataframe["is_fraud"] = 0  # Eliminate fraud instances

    with pytest.raises(ValueError, match="must contain both binary classes"):
        pipeline.validate_dataset_structure(sample_dataframe)


def test_validation_null_values(mock_pipeline_config: AppConfig, sample_dataframe: pd.DataFrame):
    """Ensures presence of NaN values triggers ValueError."""
    pipeline = DataIngestionPipeline(mock_pipeline_config)
    sample_dataframe.loc[0, "amount"] = None

    with pytest.raises(ValueError, match="unexpected NaN values"):
        pipeline.validate_dataset_structure(sample_dataframe)