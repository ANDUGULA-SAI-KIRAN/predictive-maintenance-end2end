# predictive-maintenance-end2end
An end-to-end predictive maintenance pipeline covering data preprocessing, machine learning model training, experiment tracking with MLflow, explainable AI, FastAPI serving, and Docker deployment.


predictive-maintenance-end2end/
│
├── notebooks/
│   ├── eda.ipynb         # Data exploration & visualization
│   ├── preprocessing.ipynb  # Data cleaning, feature engineering and feature selection
│   ├── initial-modeling.ipynb  # Optional: small-scale experiments
│   └── README.md
|
├── src/
│   ├── data/                # Scripts for preprocessing
│   │   └── preprocess.py
│   │
│   ├── features/            # Feature engineering functions
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py         # Optuna/Hyperopt tuning + MLflow logging
│   │   ├── select_best.py   # Load top N models from MLflow
│   │   └── predict.py       # Single model prediction function
│   │
│   ├── xai/
│   │   ├── shap_analysis.py # Compute SHAP global & local values
│   │   └── utils.py         # Helper functions for plots or value extraction
│   │
│   └── api/
│       ├── app.py           # FastAPI app serving predictions + SHAP
│       └── utils.py         # Helper functions for API requests/responses
│
├── tests/                   # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_xai.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile           # For API + model + SHAP deployment
│   └── README.md
│
├── requirements.txt         # Python dependencies
├── setup.py                 # Optional: package src as module
├── README.md                # Project description, instructions
└── .gitignore
