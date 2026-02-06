# 🤖 Predictive Maintenance – End-to-End MLOps Automation

An end-to-end **Predictive Maintenance** system showcasing **Continuous Machine Learning (CML)** and Model inferencing via **FastAPi with Docker Image** for Deployment.

## Technology Stack
> ML: scikit-learn, LightGBM, SHAP, Optuna
> MLOps: DVC, MLflow, Dagshub
> API: FastAPI, Uvicorn
> UI: Streamlit
> DevOps: Docker, Docker Compose
> CML: GitHub Actions
> Data: pandas, numpy, pyarrow, Pydantic validation

---
## Project Structure

project/
├── notebooks/                      # Analysis & experimentation
│   ├── 01_eda.ipynb                # Failure patterns & imbalance
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
|   └── Readme.md                   # Notebooks inferences and conlcusions of EDA and preprocessing experiments
│
├── src/
│   ├── data/                    # Data pipeline (Phase 1)
│   │   ├── data_ingest.py
│   │   └── preprocess.py
│   ├── features/                # Feature engineering (Phase 1)
│   │   └── feature_engineering.py
│   ├── models/                  # Model training (Phase 1)
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── shap_analysis.py
│   ├── api/                     # FastAPI (Phase 2)
│   │   └── app.py
│   └── ui/                      # Streamlit (Phase 2)
│       └── streamlit_ui.py
│
├── data/                        # DVC-tracked datasets
│   ├── raw/
│   ├── processed/
│   └── feature_engineered/
│
├── models/                      # DVC-tracked models
│   └── final_model.pkl
│
├── dvc.yaml                     # Pipeline definition (Phase 1)
├── params.yaml                  # Hyperparameter config
├── requirements.txt             # Python dependencies
├── Dockerfile.fastapi           # API container (Phase 2)
├── Dockerfile.streamlit         # UI container (Phase 2)
├── docker-compose.yml           # Local orchestration (Phase 2)
└── .github/workflows/           # GitHub Actions (Phase 1)

---

## 🧭 Architectural Overview

The project is divided into **two structured phases**:

- **Phase 1:** Continuous Machine Learning (CI Automation)
- **Phase 2:** Production Deployment (Cloud-Ready)

Phase 1 focuses on automation, reproducibility, and review-time intelligence.  
Phase 2 focuses on model inference using Docker.

---

## 🚀 PHASE 1: Continuous Machine Learning (CI Automation)

> When a Pull Request is opened, **GitHub Actions** automatically runs the entire ML lifecycle.

Developer commits changes
        ↓
   Create Pull Request
        ↓
GitHub Actions Triggered (Automatic)
        ↓
┌─────────────────────────────────────┐
│  AUTOMATED ML PIPELINE RUNS         │
├─────────────────────────────────────┤
│ 1. Data Ingestion (Validate)        │
│ 2. Preprocessing (Clean & Split)    │
│ 3. Feature Engineering              │
│ 4. Hyperparameter Optimization      │
│    ├─ Random Forest (Optuna: 10 trials) │
│    └─ LightGBM (Optuna: 10 trials)  │
│ 5. Model Training (5-fold CV)       │
│ 6. Evaluation & Metrics             │
│ 7. SHAP Analysis                    │
└─────────────────────────────────────┘
        ↓
  ✅ Pipeline Complete
        ↓
┌─────────────────────────────────────┐
│  RESULTS POSTED IN PULL REQUEST     │
├─────────────────────────────────────┤
│ 📊 Performance Metrics              │
│    • Recall | F1 | PR-AUC           │
│    • Confusion Matrix (Visual)      │
│ 📈 Model Comparison (RF vs LightGBM)│
│ 🎯 Threshold Optimization Results   │
│ 🔍 SHAP Explainability Charts       │
│ ✓ Pass/Fail Validation              │
│    (Recall ≥ 0.7, F1 ≥ 0.6)        │
└─────────────────────────────────────┘
        ↓
  Decision: Merge or Request Changes


### Step 1: Developer Creates a Pull Request

```bash
git checkout -b feature/improve-model
# Make changes to code or parameters
git push origin feature/improve-model
# Open Pull Request on GitHub
```

### Step 2: GitHub Actions Auto-Triggers

Workflow trigger configuration:

``` 
on:
  pull_request:
    branches:
      - main
```
✅ Instant: Triggered automatically on PR creation
✅ No Manual Intervention: Zero waiting time
✅ Isolated Environment: Runs on fresh GitHub runners (clean state)


### Step 3: ML Pipeline Executes (Automated)
DVC Pipeline stages run sequentially:

Stage 1: INGEST
└─ Command: python -m src.data.data_ingest
   Output: data/raw/combined.csv

Stage 2: PREPROCESS
└─ Command: python -m src.data.preprocess
   Inputs: data/raw/combined.csv
   Outputs: 
      - data/processed/train.csv (80%)
      - data/processed/test.csv (20%)

Stage 3: FEATURE ENGINEERING
└─ Command: python -m src.features.feature_engineering
   Inputs: train.csv, test.csv
   Outputs:
      - data/feature_engineered/train_enriched.csv
      - data/feature_engineered/test_enriched.csv

Stage 4: HYPERPARAMETER OPTIMIZATION & TRAINING
└─ Command: python -m src.models.train
   
   🔄 MODEL 1: RANDOM FOREST
   ├─ Optuna Search: 10 trials × 5-fold CV
   ├─ Hyperparameters tuned:
   │  ├─ n_estimators: [100, 500]
   │  ├─ max_depth: [5, 30]
   │  ├─ min_samples_split: [2, 20]
   │  ├─ min_samples_leaf: [1, 10]
   │  ├─ max_features: ["sqrt", "log2", 0.3-0.8]
   │  └─ class_weight: ["balanced", null]
   └─ Best Model: Saved & Logged to MLflow
   
   🔄 MODEL 2: LIGHTGBM
   ├─ Optuna Search: 10 trials × 5-fold CV
   ├─ Hyperparameters tuned:
   │  ├─ n_estimators: [100, 500]
   │  ├─ max_depth: [5, 30]
   │  ├─ learning_rate: [0.01, 0.2]
   │  ├─ num_leaves: [20, 150]
   │  ├─ min_child_samples: [5, 50]
   │  ├─ subsample & colsample_bytree: [0.6-1.0]
   │  └─ class_weight: ["balanced", null]
   └─ Best Model: Saved & Logged to MLflow

Stage 5: MODEL EVALUATION
└─ Command: python -m src.models.evaluate
   Metrics calculated:
   ├─ Precision, Recall, F1-Score
   ├─ PR-AUC, ROC-AUC
   ├─ Confusion Matrix
   └─ Threshold optimization analysis

Stage 6: EXPLAINABILITY ANALYSIS
└─ Command: python -m src.models.shap_analysis
   Generate:
   ├─ SHAP Summary Plots
   ├─ Feature Importance Rankings
   └─ Decision explanations


### Step 4: Results Automatically Posted to PR (via CML)
Continuous Machine Learning (CML) Bot Posts:
```
 Random Forest (Baseline)
- Recall: 0.72 
- F1-Score: 0.68 
- Precision: 0.65
- ROC-AUC: 0.79

 LightGBM (Class-Weighted)
- Recall: 0.78 ✅
- F1-Score: 0.75 ✅
- Precision: 0.72
- ROC-AUC: 0.81
```

### Step 5: Decision Point

The reviewer evaluates the Pull Request based on **automated, objective metrics** — no manual model evaluation is required.

**What the reviewer sees in the PR:**
- ✅ Automated evaluation metrics generated by the CI pipeline

**Decision Logic:**
- The newly trained model’s performance is **compared against the current production model**
  stored in **DagsHub–MLflow**.
- Approval is granted **only if the new model outperforms the production model**
  on the defined evaluation metrics (e.g., Recall, F1-score, PR-AUC).

**Actions:**
- ✅ **APPROVE & MERGE**  
  → New model outperforms production  
  → Model is **registered/promoted to production** in MLflow
- ❌ **REQUEST CHANGES**  
  → New model underperforms production  
  → Developer adjusts parameters or code and the pipeline **re-runs automatically**

---
### 🔄 Continuous Automation Benefits

| What Automates | Before (Manual) | After (Phase 1) |
|---------------|------------------|----------------|
| Model Training | Run locally, ~30 minutes, hope it works | Auto-triggered, isolated, reproducible |
| Hyperparameter Tuning | Manual grid search, days of work | Optuna auto-optimizes in minutes |
| Metrics Calculation | Manual notebooks, error-prone | Auto-computed and validated |
| Results Reporting | Screenshots in Slack, unclear | Professional report posted in PR |
| Threshold Decisions | Subjective and inconsistent | Data-driven, automatic validation |
| Model Registry | Manual upload, version confusion | Auto-registered if validation passes |
| Code Review | “Looks good to me” (no data context) | Reviewer sees actual impact with metrics |
| Deployment Decision | Guesswork, high risk | Clear pass/fail criteria |

---

### 1. Clone & Setup (~5 min)
git clone https://github.com/ANDUGULA-SAI-KIRAN/predictive-maintenance-end2end.git
cd predictive-maintenance-end2end

### 2. Make Changes (2 min)
#### Edit params.yaml to tune hyperparameters
params.yaml
#### OR modify src/models/train.py logic

### 3. Commit & Push (1 min)
git add params.yaml src/models/train.py
git commit -m "Improve model: increase LightGBM learning rate"
git push origin feature/tune-lgbm

### 4. Create PR (1 min)
#### Go to GitHub, click "Create Pull Request"
#### Add description: "Testing new hyperparameters for LightGBM or RF"

### 5. WAIT & WATCH (GitHub does all the work now) ⏳
#### GitHub Actions automatically:
✅ Spins up runner
✅ Installs dependencies
✅ Runs complete ML pipeline
✅ Calculates metrics
✅ Generates visualizations
✅ Posts results in PR comments

### 6. Review Results (3 min)
#### Check PR comments for metrics & charts
#### If metrics look good → MERGE
#### If not → Make changes & repeat
---

### Phase 1 Key Automations
- Trigger: PR creation → GitHub Actions fires
- Data Pipeline: Ingest → Preprocess → Feature Engineer (auto)
- Model Training: Random Forest + LightGBM with Optuna (auto)
- Evaluation: Metrics calculated on test set (auto)
- Reporting: CML posts results to PR (auto)
- Validation: Pass/Fail check against thresholds (auto)
- Registration: Models logged to MLflow if pass (auto)
> Result: Developer changes code → Results appear in PR within minutes. No manual ML operations needed.

---

## PHASE 2: PRODUCTION DEPLOYMENT (Cloud-Ready)

Once Phase 1 ✅ passes and model is updated as production in MLflow(Dagshub) stores the model weights:
```
Model in MLflow Registry
        ↓
Docker Containers Built (FastAPI + Streamlit)
        ↓
Deploy to Cloud (AWS SageMaker / Azure ML)
        ↓
Real-time Inference (Automated scaling)
```

> Note: Refer Readme_docker.md for phase 2 implementations

## Conclusion
This project goal is to showcases a robust architecture for continuously integrating machine learning models with GitHub Actions while preparing for seamless deployment in cloud environments. The focus on automation and reliability is key to ensuring machine learning models are always up to date and ready for production use.
