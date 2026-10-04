from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import AppConfig, load_config
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")


class DataIngestionPipeline:
    """Enterprise ingestion and validation pipeline for transaction datasets."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.data_cfg = config.data
        self.base_cfg = config.base

    def validate_dataset_structure(self, df: pd.DataFrame) -> None:
        """Runs mandatory integrity assertions on the raw dataframe."""
        if df.empty:
            raise ValueError("Raw dataset is empty. Cannot continue pipeline.")

        # 1. Column conformity check
        all_expected_cols = (
            [self.data_cfg.id_column]
            + self.data_cfg.numerical_features
            + self.data_cfg.categorical_features
            + self.data_cfg.binary_features
            + [self.data_cfg.target_column]
        )
        missing_cols = set(all_expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {sorted(list(missing_cols))}")

        # 2. Check for duplicate IDs
        duplicate_ids = df[self.data_cfg.id_column].duplicated().sum()
        if duplicate_ids > 0:
            raise ValueError(f"Integrity failure: Found {duplicate_ids} duplicate transaction IDs.")

        # 3. Target class check
        target_counts = df[self.data_cfg.target_column].value_counts().to_dict()
        logger.info(f"Target distribution observed: {target_counts}")
        if len(target_counts) < 2:
            raise ValueError(
                f"Target column '{self.data_cfg.target_column}' must contain both binary classes (0 and 1)."
            )

        # 4. Check critical null values
        null_counts = df[all_expected_cols].isnull().sum()
        if null_counts.sum() > 0:
            offending = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Dataset contains unexpected NaN values: {offending}")

    def run(self) -> None:
        """Executes full ingestion, validation, and stratified partitioning."""
        raw_path = Path(self.data_cfg.raw_data_path)
        logger.info(f"Initiating ingestion from: {raw_path}")

        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw data file missing at: {raw_path.resolve()}")

        df = pd.read_csv(raw_path)
        logger.info(f"Raw data loaded successfully. Shape: {df.shape}")

        self.validate_dataset_structure(df)
        logger.info("Data integrity assertions passed successfully.")

        # Stratified train/test split
        logger.info(
            f"Performing train/test split: test_size={self.data_cfg.split.test_size}, "
            f"stratify={self.data_cfg.split.stratify}"
        )

        stratify_col = df[self.data_cfg.target_column] if self.data_cfg.split.stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=self.data_cfg.split.test_size,
            random_state=self.base_cfg.random_state,
            stratify=stratify_col,
        )

        # Ensure processed directory exists
        train_path = Path(self.data_cfg.processed_train_path)
        test_path = Path(self.data_cfg.processed_test_path)
        train_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to compressed parquet format
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(
            f"Ingestion completed. Train shape: {train_df.shape} written to {train_path}. "
            f"Test shape: {test_df.shape} written to {test_path}."
        )


if __name__ == "__main__":
    app_config = load_config()
    pipeline = DataIngestionPipeline(app_config)
    pipeline.run()