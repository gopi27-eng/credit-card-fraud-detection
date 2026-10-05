from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
import yaml


class BaseConfig(BaseModel):
    project_name: str
    random_state: int
    model_config = ConfigDict(extra="forbid")


class SplitConfig(BaseModel):
    test_size: float = Field(..., gt=0.0, lt=1.0)
    stratify: bool
    model_config = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    raw_data_path: Path
    processed_train_path: Path
    processed_test_path: Path
    target_column: str
    id_column: str
    numerical_features: List[str]
    categorical_features: List[str]
    binary_features: List[str]
    split: SplitConfig
    preprocessor_path: Optional[Path] = None
    transformed_train_path: Optional[Path] = None
    transformed_test_path: Optional[Path] = None

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
    model: Optional[ModelConfig] = None
    mlflow: MLflowConfig
    model_config = ConfigDict(extra="forbid")


def load_config(config_path: Union[str, Path] = Path("config/config.yaml")) -> AppConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Configuration at {path} must parse into a dictionary mapping.")

    return AppConfig(**raw_cfg)