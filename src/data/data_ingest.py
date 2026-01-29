# src/data/data_ingest.py
import os
import pandas as pd
import dagshub
import yaml
from dotenv import load_dotenv
from src.utils.logger import logger

from src.validation.schemas import validate_training_data


def ingest_data():
    try:
        # --- 1. LOAD ENV + PARAMS ---
        load_dotenv()
        token = os.getenv("DAGSHUB_USER_TOKEN")
        if not token:
            raise ValueError("DAGSHUB_USER_TOKEN not found in .env file.")

        # Repo config from ENV (best practice)
        REPO_OWNER = os.getenv("REPO_OWNER")
        REPO_NAME = os.getenv("REPO_NAME")

        if not REPO_OWNER or not REPO_NAME:
            raise ValueError("REPO_OWNER or REPO_NAME missing in .env file.")

        # Load params.yaml (DVC controlled)
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)

        ingest_params = params["ingest"]

        REMOTE_FILE_NAME = ingest_params["remote_file"]
        LOCAL_FILE_PATH = ingest_params["local_file"]
        OUTPUT_DIR = ingest_params["output_dir"]
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, ingest_params["output_file"])

        # --- 2. CONNECT TO DAGSHUB STORAGE ---
        REPO_ID = f"{REPO_OWNER}/{REPO_NAME}"
        logger.info(f"Connecting to DagsHub storage: {REPO_ID}")

        try:
            s3fs_client = dagshub.get_repo_bucket_client(REPO_ID, flavor="s3fs")  # type: ignore
        except Exception as e:
            raise ConnectionError(f"Failed to initialize DagsHub client: {e}")

        # --- 3. READ REMOTE DATA ---
        remote_s3_path = f"{REPO_NAME}/{REMOTE_FILE_NAME}"
        logger.info(f"Reading remote file: {remote_s3_path}")

        try:
            with s3fs_client.open(remote_s3_path, "rb") as f:
                df_remote = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find '{REMOTE_FILE_NAME}' in DagsHub storage root."
            )

        # Validate remote dataset
        logger.info("Validating remote dataset schema...")
        df_remote = validate_training_data(df_remote)

        # --- 4. READ LOCAL DATA ---
        logger.info(f"Reading local file: {LOCAL_FILE_PATH}")

        if not os.path.exists(LOCAL_FILE_PATH):
            raise FileNotFoundError(
                f"Local file '{LOCAL_FILE_PATH}' not found."
            )

        df_local = pd.read_csv(LOCAL_FILE_PATH)

        # Validate local dataset
        logger.info("Validating local dataset schema...")
        df_local = validate_training_data(df_local)

        # ---  MERGE & SAVE ---
        logger.info("Merging datasets...")
        df_combined = pd.concat([df_remote, df_local], ignore_index=True)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)

        logger.info(f"Successfully saved merged data to: {OUTPUT_FILE}")
        print("\n--- INGESTION COMPLETE ---")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise e

if __name__ == "__main__":
    ingest_data()