from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import AppConfig, load_config


@pytest.fixture
def valid_config_dict() -> dict:
    """Fixture returning a complete and valid configuration mapping."""
    return {
        "base": {
            "project_name": "credit-card-fraud-detection",
            "random_state": 42,
        },
        "data": {
            "raw_data_path": "data/raw/creditcard.csv",
            "processed_train_path": "data/processed/train.parquet",
            "processed_test_path": "data/processed/test.parquet",
            "target_column": "is_fraud",
            "id_column": "transaction_id",
            "numerical_features": [
                "amount",
                "transaction_hour",
                "device_trust_score",
                "velocity_last_24h",
                "cardholder_age",
            ],
            "categorical_features": ["merchant_category"],
            "binary_features": ["foreign_transaction", "location_mismatch"],
            "split": {
                "test_size": 0.2,
                "stratify": True,
            },
        },
        "mlflow": {
            "experiment_name": "credit-card-fraud-experiments",
            "tracking_uri": "https://dagshub.com",
        },
    }


def test_load_default_config_file():
    """Verify that the actual checked-in config/config.yaml loads without error."""
    config = load_config("config/config.yaml")
    assert isinstance(config, AppConfig)
    assert config.base.project_name == "credit-card-fraud-detection"
    assert config.data.split.test_size == 0.2
    assert "amount" in config.data.numerical_features


def test_missing_config_file_raises_error(tmp_path: Path):
    """Ensure non-existent configuration path raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_config.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing_path))


def test_invalid_yaml_format(tmp_path: Path):
    """Ensure a YAML file with non-mapping structure raises ValueError."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n")  # List instead of dict mapping

    with pytest.raises(ValueError):
        load_config(str(bad_yaml))


@pytest.mark.parametrize("invalid_test_size", [-0.1, 0.0, 1.0, 1.5])
def test_split_test_size_boundaries(valid_config_dict: dict, tmp_path: Path, invalid_test_size: float):
    """Verify test_size must strictly satisfy: 0.0 < test_size < 1.0."""
    valid_config_dict["data"]["split"]["test_size"] = invalid_test_size
    temp_config = tmp_path / "invalid_split.yaml"

    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)

    with pytest.raises(ValidationError):
        load_config(str(temp_config))


def test_missing_required_key(valid_config_dict: dict, tmp_path: Path):
    """Ensure omission of critical keys (e.g., target_column) fails immediately."""
    del valid_config_dict["data"]["target_column"]
    temp_config = tmp_path / "missing_key.yaml"

    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)

    with pytest.raises(ValidationError):
        load_config(str(temp_config))


def test_extra_fields_forbidden(valid_config_dict: dict, tmp_path: Path):
    """Ensure passing undeclared keys in YAML triggers ValidationError."""
    valid_config_dict["data"]["unauthorized_setting"] = True
    temp_config = tmp_path / "extra_field.yaml"

    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)

    with pytest.raises(ValidationError):
        load_config(str(temp_config))