from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict
import yaml


class BaseConfig(BaseModel):
    project_name: str
    random_state: int
    model_config = ConfigDict(extra="forbid")


class SplitConfig(BaseModel):
    test_size: float
    stratify: bool
    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
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


class ModelConfig(BaseModel):
    artifact_path: Path
    algorithm: str
    params: Dict[str, Any]
    model_config = ConfigDict(extra="forbid")


class MLflowConfig(BaseModel):
    experiment_name: str
    tracking_uri: str = "https://dagshub.com"
    model_config = ConfigDict(extra="forbid")


class AppConfig(BaseModel):
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
    model_config = ConfigDict(extra="forbid")


def load_config(config_path: Path = Path("config/config.yaml")) -> AppConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    return AppConfig(**raw_cfg)