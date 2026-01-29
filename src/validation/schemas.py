# src/validation/schemas.py
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from pandera.errors import SchemaError
from src.utils.logger import logger  # Imported to track validation steps


# -----------------------------
# Custom Exception
# -----------------------------
class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""


# -----------------------------
# Core Model Features
# -----------------------------
FEATURE_COLUMNS = {
    "Type": Column(str, nullable=False, checks=Check.isin(["L", "M", "H"])),
    "Air temperature [K]": Column(float, nullable=False),
    "Process temperature [K]": Column(float, nullable=False),
    "Rotational speed [rpm]": Column(int, nullable=False),
    "Torque [Nm]": Column(float, nullable=False),
    "Tool wear [min]": Column(int, nullable=False),
}


# -----------------------------
# Training Schema
# -----------------------------
AI4I_TRAIN_SCHEMA = DataFrameSchema(
    {
        "id": Column(int, nullable=False),
        "Product ID": Column(str, nullable=False),
        **FEATURE_COLUMNS,
        "TWF": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "HDF": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "PWF": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "OSF": Column(int, nullable=False, checks=Check.isin([0, 1])),
        "RNF": Column(int, nullable=False, checks=Check.isin([0, 1])),
    },
    strict="filter", 
    coerce=True,
)


# -----------------------------
# Inference Schema
# -----------------------------
AI4I_INFERENCE_SCHEMA = DataFrameSchema(
    {**FEATURE_COLUMNS},
    strict="filter", 
    coerce=True,
)


# -----------------------------
# Validation Utilities
# -----------------------------
def validate_dataset(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate and clean dataset with detailed logging."""
    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    # 1. Handle Duplicates
    if df.duplicated().any():
        initial_count = len(df)
        df = df.drop_duplicates()
        dropped_count = initial_count - len(df)
        logger.info(f"Dropped {dropped_count} duplicate rows.")

    # 2. Check for Extra Columns (Before Pandera filters them)
    expected_cols = set(schema.columns.keys())
    actual_cols = set(df.columns)
    extra_cols = actual_cols - expected_cols

    if extra_cols:
        logger.debug(f"Extra columns detected and will be filtered: {sorted(list(extra_cols))}")

    try:
        # 3. Perform Pandera Validation
        # strict="filter" will now remove those extra_cols
        validated_df = schema.validate(df, lazy=True)
        return validated_df

    except SchemaError as e:
        logger.error(f"Schema validation failed technical details:\n{e}")
        raise DatasetValidationError(f"Schema validation failed: {e}")


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validation for ingestion/training pipeline."""
    return validate_dataset(df, AI4I_TRAIN_SCHEMA)


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validation for prediction/API inputs."""
    return validate_dataset(df, AI4I_INFERENCE_SCHEMA)