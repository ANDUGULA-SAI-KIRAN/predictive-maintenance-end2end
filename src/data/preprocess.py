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
    df_cleaned[int_cols] = df_cleaned[int_cols].astype('Int64')
    logger.info(f"Casted columns {list(int_cols)} to nullable int type for MLflow compatibility")
    
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