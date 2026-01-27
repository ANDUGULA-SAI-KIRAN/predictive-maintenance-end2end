# Notebooks Overview — Predictive Maintenance (AI4I 2020)

- This folder contains the **analysis and decision-making notebooks** that establish the foundation for the end-to-end predictive maintenance pipeline.  
- The notebooks focus on **understanding machine failure behavior, validating modeling strategies, and finalizing deployable model configurations** before transitioning to production-grade training and deployment.

---

## Business Context & Modeling Philosophy

Machine failures in industrial environments are:
- **Rare but high-impact events**
- Often driven by **non-linear interactions** and **operational thresholds**
- Costly when missed, yet disruptive when over-predicted

Accordingly, this project adopts a **risk-based predictive maintenance approach**, where models are evaluated not only on accuracy but on:
- Their ability to **detect failures**
- Their ability to **prioritize high-risk machines**
- Their suitability for **real operational decision-making**

---

## Notebook Structure & Purpose

### 1. `eda.ipynb` — Exploratory Data Analysis

**Objective**
- Build a deep understanding of machine operation and failure behavior

**Key Focus Areas**
- Failure frequency and class imbalance
- Product type distribution and failure composition
- Operational regimes linked to different failure modes
- Correlation and interaction patterns among operational variables

**Key Outcomes**
- Confirmed that failures are **highly imbalanced and localized**
- Identified **non-linear, rule-driven failure mechanisms**
- Established the need for **feature engineering and non-linear models**

---

### 2. `feature_engineering_experiments.ipynb` — Preprocessing & Strategy Evaluation

**Objective**
- Evaluate preprocessing, feature engineering, and imbalance-handling strategies
- Identify which modeling approaches are most effective and deployable

**Key Activities**
- Mandatory preprocessing and target construction
- Domain-driven feature engineering
- Comparison of imbalance-handling strategies:
  - Baseline (feature-augmented, no imbalance handling)
  - Class-weighted learning
  - SMOTE-Tomek resampling
- Baseline model training across multiple algorithms
- Statistical validation and performance comparison

**Key Outcomes**
- Feature engineering significantly improved model performance
- Imbalance handling is **model-dependent**, not universal
- Decision Trees were excluded due to poor reliability
- LightGBM and Random Forest emerged as top candidates
- SMOTE-Tomek was retained for benchmarking but excluded from final deployment

---

### 3. `model_experiments.ipynb` — Final Model Selection & Calibration

**Objective**
- Consolidate insights and finalize models for production training
- Perform focused imbalance-handling comparison and threshold optimization

**Key Activities**
- Re-evaluation of final candidate models
- Threshold optimization to align predictions with operational risk
- Final performance validation and model positioning

**Finalized Models**
- **LightGBM + Class Weighting**
  - Optimized for **maximum failure detection**
  - Suitable for safety-critical or high-risk environments
- **Random Forest + Baseline**
  - Optimized for **reliable risk ranking**
  - Suitable for maintenance prioritization and planning

**Key Outcome**
- Threshold tuning transformed both models into **operationally viable solutions**
- Final modeling decisions are evidence-backed and business-aligned

---

## Transition to Production Pipeline (`src/`)

The notebooks conclude the **analysis and decision phase** of the project.

The following steps are handled in the `src/` directory:
- Reproducible preprocessing pipelines
- Hyperparameter optimization using Optuna
- Data versioning with DVC (dagshub) and Experiment tracking with MLflow
- Model selection, inference, and deployment readiness

This separation ensures:
- Clean experimentation
- Reproducibility
- Scalable production workflows

---

## Summary

Together, these notebooks:
- Translate raw industrial data into actionable insights
- Systematically evaluate modeling strategies
- Finalize deployable machine learning configurations

They provide a **clear, defensible foundation** for the end-to-end predictive maintenance system implemented in the subsequent stages of the project.
