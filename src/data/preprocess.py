# src/data/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split


FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}
RENAME_MAPPING = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw dataset from disk.
    """
    return pd.read_csv(path)


def preprocess_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mandatory preprocessing applied consistently for training and inference.
    - Drop identifiers
    - Construct binary target (if failure columns exist)
    - Remove leakage columns
    - Encode Type
    - Rename columns
    """

    df = df.copy()

    # Drop non-informative identifiers
    df = df.drop(columns=['id', 'Product ID'], errors='ignore')

    # Construct target if failure columns exist (training scenario)
    if set(FAILURE_COLS).issubset(df.columns):
        df['machine_failure'] = (df[FAILURE_COLS].sum(axis=1) > 0).astype(int)
        df = df.drop(columns=FAILURE_COLS, errors='ignore')

    # Encode Type
    if 'Type' in df.columns:
        df['Type'] = df['Type'].map(TYPE_MAPPING)

    # Rename columns
    df = df.rename(columns=RENAME_MAPPING)

    return df


def train_test_split_data(
    df: pd.DataFrame,
    target_col: str = "machine_failure",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Perform stratified train-test split.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
