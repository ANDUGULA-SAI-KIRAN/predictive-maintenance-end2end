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



Feature Selection Strategy and Rationale
Why Feature Selection is Important in This Project

The dataset used in this project represents a real-world predictive maintenance problem with imbalanced failure events. In such problems, model performance depends more on meaningful and stable features than on the number of features used.

Feature selection is applied to:

Reduce noise and redundancy in the data

Improve model generalization on rare failure cases

Increase model stability and interpretability

Avoid unnecessary model complexity and overfitting

Rather than aggressively removing features, the goal is to validate and refine the most informative features for prediction.

Chosen Feature Selection Approach

This project uses a lightweight hybrid feature selection strategy, combining Filter and Embedded methods.

Heavy wrapper methods were avoided because the dataset has a limited number of features and excessive feature elimination can reduce model reliability.

1. Filter Method: Initial Feature Screening
Why Filter Methods Are Used

Filter methods are used as an initial screening step to quickly analyze the dataset before training any model.

They help answer simple but important questions:

Are any features constant or nearly constant?

Are multiple features carrying the same information?

Do features show a basic relationship with the target variable?

How Filter Methods Help Prediction

Remove obvious noise and redundant information

Simplify the feature space without affecting core predictive signals

Improve model robustness by eliminating unstable features

Filter methods are fast, model-independent, and help establish confidence in the quality of the input data.

2. Embedded Method: Model-Driven Feature Importance
Why Embedded Methods Are Used

Embedded methods allow the model itself to determine which features contribute most to predictions during training.

Tree-based models are particularly useful because they:

Capture non-linear relationships between features

Automatically handle interactions between variables

Provide intuitive feature importance scores

How Embedded Methods Help Prediction

Identify features that consistently influence predictions

Improve generalization by focusing learning on relevant signals

Preserve interpretability, which is important in industrial and maintenance use cases

Instead of removing features aggressively, feature importance is used to validate feature relevance and guide further feature engineering.

Why Wrapper Methods Were Not Emphasized

Wrapper methods repeatedly train models using different feature subsets. While powerful, they are:

Computationally expensive

More prone to overfitting on small or imbalanced datasets

Unnecessary when the feature set is already compact and meaningful

For this project, wrapper methods would increase complexity without providing proportional performance gains.

How This Strategy Improves Prediction Performance

This structured feature selection approach:

Reduces noise while preserving critical failure indicators

Prevents overfitting caused by irrelevant or redundant features

Improves model stability on minority (failure) cases

Ensures that improvements come from better data representation, not repeated tuning cycles

By focusing on feature quality rather than quantity, the model learns more reliable patterns and produces more trustworthy predictions.

Summary

Filter methods are used for quick validation and redundancy removal

Embedded methods allow the model to identify important features naturally

Wrapper methods are avoided to reduce complexity and overfitting

Feature selection is used to support prediction, not to aggressively reduce features

This approach ensures an efficient, interpretable, and generalizable predictive maintenance model.



# Feature Selection Strategy for Predictive Maintenance Modeling

## 1. Motivation

Predictive maintenance datasets are typically characterized by **imbalanced class distributions**, limited feature sets, and strong domain-driven variables. In such settings, model performance is influenced more by **feature quality and stability** than by extensive feature reduction. Improper feature selection can introduce overfitting, reduce generalization to rare failure events, and increase model variance. Therefore, feature selection in this work is designed to **validate feature relevance**, reduce redundancy, and improve model robustness rather than aggressively eliminate features.

---

## 2. Feature Selection Approach

This study adopts a **lightweight hybrid feature selection strategy**, combining **Filter** and **Embedded** methods. Wrapper-based approaches were deliberately excluded due to their computational cost and increased risk of overfitting on small and imbalanced datasets.

---

## 3. Filter-Based Feature Screening

Filter methods are applied as an initial screening step to evaluate features independently of any learning algorithm. Statistical measures such as variance analysis and correlation analysis are used to identify redundant or weakly informative features. This step ensures data consistency, reduces noise, and confirms that input variables exhibit meaningful relationships with the target variable. By operating prior to model training, filter methods provide a fast and model-agnostic assessment of feature relevance.

---

## 4. Embedded Feature Importance Analysis

Embedded feature selection is performed during model training using tree-based learning algorithms. These models naturally assign importance scores to features based on their contribution to reducing prediction error. Embedded methods capture non-linear relationships and feature interactions, allowing the model to prioritize informative variables while learning. Rather than removing features aggressively, feature importance is analyzed to validate feature relevance and guide subsequent feature engineering.

---

## 5. Exclusion of Wrapper Methods

Although wrapper methods evaluate feature subsets using repeated model training, they were not emphasized in this work. Given the limited feature dimensionality and class imbalance of the dataset, wrapper-based selection increases computational complexity and overfitting risk without providing proportional performance improvements. Consequently, wrapper methods were deemed unsuitable for the objectives of this study.

---

## 6. Impact on Model Performance

The proposed feature selection strategy improves model generalization by reducing noise, mitigating redundancy, and stabilizing learning on minority class samples. By prioritizing feature quality over extensive hyperparameter tuning, the approach avoids iterative tuning cycles and ensures that performance gains arise from meaningful data representation rather than repeated optimization.

---

## 7. Conclusion

The combination of filter-based screening and embedded feature importance provides an efficient and robust feature selection framework for predictive maintenance tasks. This approach enhances interpretability, reduces overfitting, and supports reliable prediction of rare failure events, making it well-suited for real-world industrial applications.



----------------
# Feature Selection Strategy for Predictive Maintenance

## Motivation
Predictive maintenance datasets often have **imbalanced failures** and a limited number of features. Model performance depends more on **feature quality** than feature quantity. Poor feature selection can increase overfitting, reduce generalization, and add instability. The goal is to **validate and refine features** rather than remove them aggressively.

---

## Approach
A **lightweight hybrid strategy** combining **Filter** and **Embedded** methods was adopted. Wrapper methods were avoided due to their **high computational cost** and risk of overfitting on small, imbalanced datasets.

---

## Filter-Based Screening
Filter methods evaluate each feature independently using **statistical measures** like variance and correlation. They identify weak or redundant features quickly, ensuring **data quality** and a **clean input space** without requiring model training.

---

## Embedded Feature Importance
Embedded methods allow the model to **learn feature importance** during training (e.g., tree-based models). This captures **non-linear relationships and interactions**, highlighting the features most predictive of rare failures. Features are validated rather than aggressively removed, guiding further feature engineering.

---

## Impact
This approach:
- Reduces noise and redundancy  
- Improves generalization on minority classes  
- Stabilizes learning  
- Avoids repetitive tuning cycles

By prioritizing **feature quality over quantity**, the model achieves **robust and interpretable predictions** suitable for real-world industrial applications.
