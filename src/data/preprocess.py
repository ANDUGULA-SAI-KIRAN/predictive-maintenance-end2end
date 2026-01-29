# # src/data/preprocess.py
# import pandas as pd
# from sklearn.model_selection import train_test_split


# FAILURE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
# TYPE_MAPPING = {'L': 0, 'M': 1, 'H': 2}
# RENAME_MAPPING = {
#     "Type": "type",
#     "Air temperature [K]": "air_temp",
#     "Process temperature [K]": "process_temp",
#     "Rotational speed [rpm]": "rpm",
#     "Torque [Nm]": "torque",
#     "Tool wear [min]": "tool_wear",
# }


# def load_data(path: str) -> pd.DataFrame:
#     """
#     Load raw dataset from disk.
#     """
#     return pd.read_csv(path)


# def preprocess_base_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Mandatory preprocessing applied consistently for training and inference.
#     - Drop identifiers
#     - Construct binary target (if failure columns exist)
#     - Remove leakage columns
#     - Encode Type
#     - Rename columns
#     """

#     df = df.copy()

#     # Drop non-informative identifiers
#     df = df.drop(columns=['id', 'Product ID'], errors='ignore')

#     # Construct target if failure columns exist (training scenario)
#     if set(FAILURE_COLS).issubset(df.columns):
#         df['machine_failure'] = (df[FAILURE_COLS].sum(axis=1) > 0).astype(int)
#         df = df.drop(columns=FAILURE_COLS, errors='ignore')

#     # Encode Type
#     if 'Type' in df.columns:
#         df['Type'] = df['Type'].map(TYPE_MAPPING)

#     # Rename columns
#     df = df.rename(columns=RENAME_MAPPING)

#     return df


# def train_test_split_data(
#     df: pd.DataFrame,
#     target_col: str = "machine_failure",
#     test_size: float = 0.2,
#     random_state: int = 42
# ):
#     """
#     Perform stratified train-test split.
#     """

#     X = df.drop(columns=[target_col])
#     y = df[target_col]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=y
#     )

#     return X_train, X_test, y_train, y_test




# src/data/preprocess.py
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger

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

def preprocess_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=['id', 'Product ID'], errors='ignore')
    if set(FAILURE_COLS).issubset(df.columns):
        df['machine_failure'] = (df[FAILURE_COLS].sum(axis=1) > 0).astype(int)
        df = df.drop(columns=FAILURE_COLS, errors='ignore')
    if 'Type' in df.columns:
        df['Type'] = df['Type'].map(TYPE_MAPPING)
    df = df.rename(columns=RENAME_MAPPING)
    return df

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    config = params["preprocess"]
    
    logger.info(f"Loading raw data from {config['input_path']}")
    df = pd.read_csv(config['input_path'])
    
    df_cleaned = preprocess_base_features(df)

    # This ensures consistency even if missing values appear later
    int_cols = df_cleaned.select_dtypes(include=['int64', 'int32']).columns
    df_cleaned[int_cols] = df_cleaned[int_cols].astype('float64')
    logger.info(f"Casted columns {list(int_cols)} to float64 for MLflow compatibility")
    
    # Stratified Split
    train_df, test_df = train_test_split(
        df_cleaned, 
        test_size=config['test_size'], 
        random_state=params['base']['random_state'],
        stratify=df_cleaned['machine_failure']
    )
    
    os.makedirs(config['output_dir'], exist_ok=True)
    train_df.to_csv(os.path.join(config['output_dir'], "train.csv"), index=False)
    test_df.to_csv(os.path.join(config['output_dir'], "test.csv"), index=False)
    logger.info("Preprocessing complete. Train/Test files saved.")