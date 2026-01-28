import os
import pandas as pd
import dagshub
from dotenv import load_dotenv
from utils.logger import logger

def ingest_data():
    try:
        # --- 1. SETUP & AUTH ---
        load_dotenv()
        
        # DagsHub looks for DAGSHUB_USER_TOKEN for automatic S3 auth
        token = os.getenv("DAGSHUB_USER_TOKEN")
        if not token:
            raise ValueError("DAGSHUB_USER_TOKEN not found in .env file.")
        

        REPO_OWNER = "ANDUGULA-SAI-KIRAN"
        REPO_NAME = "predictive-maintenance-end2end"
        REPO_ID = f"{REPO_OWNER}/{REPO_NAME}"
        
        REMOTE_FILE_NAME = "test.csv"  # File in DagsHub Bucket root
        LOCAL_FILE_PATH = "test.csv"   # Local file in project root
        OUTPUT_DIR = "data/raw"
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combined.csv")

        # --- 2. CONNECT TO DAGSHUB STORAGE ---
        logger.info(f"Connecting to DagsHub storage: {REPO_ID}")
        
        # 's3fs' flavor makes it act like a Python filesystem
        try:
            s3fs_client = dagshub.get_repo_bucket_client(f"{REPO_OWNER}/{REPO_NAME}", flavor="s3fs") # type: ignore
        except Exception as e:
            raise ConnectionError(f"Failed to initialize DagsHub client: {e}")

        # --- 3. READ REMOTE DATA ---
        # Path MUST start with repo_name: "repo_name/file.csv"
        remote_s3_path = f"{REPO_NAME}/{REMOTE_FILE_NAME}"
        
        logger.info(f"Reading remote file: {remote_s3_path}")
        try:
            with s3fs_client.open(remote_s3_path, "rb") as f:
                df_remote = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find '{REMOTE_FILE_NAME}' in DagsHub storage root.")

        # --- 4. READ LOCAL DATA ---
        logger.info(f"Reading local file: {LOCAL_FILE_PATH}")
        if not os.path.exists(LOCAL_FILE_PATH):
            raise FileNotFoundError(f"Local file '{LOCAL_FILE_PATH}' not found in project root.")
        
        df_local = pd.read_csv(LOCAL_FILE_PATH)

        # --- 5. MERGE & SAVE ---
        logger.info("Merging datasets...")
        df_combined = pd.concat([df_remote, df_local], ignore_index=True)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)
        
        logger.info(f"Successfully saved merged data to: {OUTPUT_FILE}")
        print("\n--- INGESTION COMPLETE ---")
        print(f"Next steps:\n1. dvc add {OUTPUT_FILE}\n2. dvc push\n3. git add {OUTPUT_FILE}.dvc")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        exit(1)

if __name__ == "__main__":
    ingest_data()