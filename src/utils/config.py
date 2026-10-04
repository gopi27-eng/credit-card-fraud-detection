from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field


class SplitConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    stratify: bool = Field(default=True)
    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    """Configuration for data paths, features, and split parameters."""

    raw_data_path: Path
    processed_train_path: Path
    processed_test_path: Path
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    target_column: str
    id_column: str
    numerical_features: List[str]
    categorical_features: List[str]
    binary_features: List[str]
    split: SplitConfig

    model_config = ConfigDict(extra="forbid")

class BaseConfig(BaseModel):
    project_name: str
    random_state: int = Field(default=42)
    model_config = ConfigDict(extra="forbid")


class MLflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    base: BaseConfig
    data: DataConfig
    mlflow: MLflowConfig
    model_config = ConfigDict(extra="forbid")


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Loads and validates application configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML content in {path}. Expected key-value mapping.")

    return AppConfig(**raw_dict)

