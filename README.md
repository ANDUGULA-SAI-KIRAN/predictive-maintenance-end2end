project/
│
├── notebooks/                 # Exploratory notebooks (Git tracked)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_experiments.ipynb
│   ├── 03_model_experiments.ipynb
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── ingest.py         # Pull & merge from multiple sources
│   │   ├── preprocess.py       # Cleaning & preprocessing
│   │
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   │
│   ├── models/
│   │   ├── train.py            # Training
│   │   ├── evaluate.py         # Evaluation
│   │   ├── predict.py          # Prediction/inference
│   │
│   ├── api/
│   │   └── app.py              # FastAPI / inference service
│   │
│   ├── ui/
│   │   └── streamlit_app.py    # Streamlit dashboard
│
├── data/                       # All DVC-tracked datasets
│   ├── raw/                    # Ingested & merged datasets
│   │   └── combined.csv
│   ├── processed/              # Cleaned / preprocessed datasets
│   │   └── preprocessed.csv
│   └── features/               # Feature-engineered datasets
│       └── features.csv
│
├── models/                     # DVC-tracked trained models
│   └── final_model.pkl
│
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                  # Hyperparameters & config
├── requirements.txt
├── .github/workflows/           # CI/CD
│   └── ci.yaml
├── mlruns/                      # MLflow logs (can be local or remote)
└── README.md






---

# 📔 MLOps Project Notes: Continuous Training (CT) Pipeline

This project implements a **Level 1 MLOps: Continuous Training** pipeline. It is designed to be portable—transitioning from local/GitHub automation to enterprise-grade cloud environments (AWS SageMaker / Azure ML) with minimal architectural changes.

---

## 1. Current System: "The GitHub Actions & CML Setup"
Currently, the pipeline uses **GitHub Actions** as the "Engine" and **DagsHub** as the "Brains."

* **Orchestration:** GitHub Actions triggers on every `git push`.
* **Pipeline Logic:** `dvc.yaml` defines the stages (Ingest → Preprocess → Train → Evaluate).
* **Experiment Tracking:** **MLflow** (hosted on DagsHub) logs metrics (Recall, F1) and parameters.
* **Model Registry:** Models meeting the **$0.7$ Recall** and **$0.6$ F1** thresholds are automatically registered as **Candidates**.
* **Reporting:** **CML** (Continuous Machine Learning) posts a visual report (Confusion Matrix) directly onto the GitHub Pull Request.



---

## 2. Transitioning to Enterprise Cloud (The "Pro" Setup)
In a professional setting, we shift from "Running code on GitHub" to "Ordering Cloud Compute to run a Container."

### The "How-To" Map: Moving to SageMaker / Azure ML

To replicate this setup on **AWS SageMaker** or **Azure ML**, we follow these 4 steps:

| Component | What we have now | What changes in the Cloud |
| :--- | :--- | :--- |
| **Environment** | `pip install -r requirements.txt` | **Docker Image:** We freeze the environment into a container image and store it in a Registry (AWS ECR / Azure ACR). |
| **Compute** | GitHub Hosted Runner (Small CPU) | **Ephemeral Clusters:** GitHub tells the cloud to spin up a high-power instance (e.g., `ml.m5.large`) just for the training duration. |
| **Data Access** | `dvc pull` via DagsHub | **Cloud Buckets:** Data is synced to an S3 Bucket (AWS) or Blob Storage (Azure). The cloud instance "mounts" this data instantly. |
| **Triggers** | GitHub Action runs the code | GitHub Action uses a **Cloud SDK** to "Submit a Job." |

---

## 3. Interview-Ready Explanation (The "Bridge" Answer)
If asked, *"How would you scale your current GitHub pipeline to a multi-terabyte dataset?"*

> "I have implemented a **modular MLOps pipeline** where the logic is decoupled from the infrastructure. Currently, I use GitHub Actions to run the training. To migrate to a cloud instance like **SageMaker**, I would **Containerize** my project using a Dockerfile, modify my GitHub Action to **launch a SageMaker Training Job**, and have the cloud job pull the image and run the same `dvc repro` command. My `evaluate.py` already logs to a remote **MLflow registry**, so tracking remains seamless."


# 🛠️ Phase 2: Migration to Production Inference

To scale the current FastAPI + Streamlit setup to a Cloud environment (SageMaker/Azure), we will perform the following file-level updates.

### A. Update `src/models/predict.py` (The Loader)
Instead of searching for `metrics.pr_auc DESC`, the code will shift to the **Model Registry API**. This allows the Training Pipeline to "promote" a model, and the API will automatically pick it up without a code change.

* **Current:** Search for `run_id` in SQLite.
* **New:** Load via URI `models:/predictive_maintenance_rf/production`.

### B. Update `src/api/app.py` (The Endpoint)
We will implement a **Lifespan Event**. Instead of waiting for a user to click "Load Model," the API will pre-fetch the `@production` model during startup to ensure zero-latency for the first request.

### C. Update `src/ui/streamlit_ui.py` (The Interface)
The UI will change from "Model Selection" (picking RF vs LGBM) to **"System Status"**. In industry, the user doesn't choose the algorithm; the MLOps engineer chooses the "Champion" model in the background.

---

### How the Cloud Job Executes this Flow:
1. **Container Start:** The Docker container launches on SageMaker.
2. **Registry Handshake:** The API connects to the Remote DagsHub MLflow URI.
3. **Model Pull:** It downloads the `best_model.pkl` into the container's RAM.
4. **Ready:** The FastAPI `health` check returns `200`, and the Load Balancer begins sending sensor data.