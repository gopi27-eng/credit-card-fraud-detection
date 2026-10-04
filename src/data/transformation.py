from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger("data_transformation")


class DataTransformationPipeline:
    """Preprocesses raw features and produces serializable transformation artifacts."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.data_cfg = config.data

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs an sklearn ColumnTransformer with strict unknown-category tolerance."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.data_cfg.numerical_features),
                ("cat", categorical_transformer, self.data_cfg.categorical_features),
                ("bin", "passthrough", self.data_cfg.binary_features),
            ],
            remainder="drop",
        )
        return preprocessor

    def run(self) -> None:
        """Fits preprocessor on training data, transforms train & test, and persists artifacts."""
        train_path = Path(self.data_cfg.processed_train_path)
        test_path = Path(self.data_cfg.processed_test_path)

        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError("Processed train or test parquet file is missing. Run ingest first.")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        logger.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) parquet splits.")

        preprocessor = self.build_preprocessor()

        # Separate features from target
        y_train = train_df[self.data_cfg.target_column].to_numpy()
        y_test = test_df[self.data_cfg.target_column].to_numpy()

        # Fit strictly on training data
        logger.info("Fitting ColumnTransformer on train features...")
        X_train_transformed = preprocessor.fit_transform(train_df)
        X_test_transformed = preprocessor.transform(test_df)

        # Extract generated column names
        cat_encoder = preprocessor.named_transformers_["cat"]
        encoded_cat_cols = cat_encoder.get_feature_names_out(self.data_cfg.categorical_features).tolist()
        feature_names = self.data_cfg.numerical_features + encoded_cat_cols + self.data_cfg.binary_features

        # Reconstruct transformed DataFrames with target
        train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_transformed_df[self.data_cfg.target_column] = y_train

        test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_transformed_df[self.data_cfg.target_column] = y_test

        # Save artifacts
        preprocessor_file = Path(self.data_cfg.preprocessor_path)
        preprocessor_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_file)
        logger.info(f"Preprocessor artifact persisted at: {preprocessor_file}")

        out_train_path = Path(self.data_cfg.transformed_train_path)
        out_test_path = Path(self.data_cfg.transformed_test_path)
        out_train_path.parent.mkdir(parents=True, exist_ok=True)

        train_transformed_df.to_parquet(out_train_path, index=False)
        test_transformed_df.to_parquet(out_test_path, index=False)

        logger.info(
            f"Transformed data saved successfully. Train shape: {train_transformed_df.shape}, "
            f"Test shape: {test_transformed_df.shape}"
        )


if __name__ == "__main__":
    app_config = load_config()
    pipeline = DataTransformationPipeline(app_config)
    pipeline.run()