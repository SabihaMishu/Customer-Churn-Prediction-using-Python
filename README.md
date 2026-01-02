# 🚀 Customer Churn Prediction (using Python)

[![GitHub Repo stars](https://img.shields.io/github/stars/SabihaMishu/Customer-Churn-Prediction-using-Python?style=social)](https://github.com/SabihaMishu/Customer-Churn-Prediction-using-Python)
[![Language](https://img.shields.io/github/languages/top/SabihaMishu/Customer-Churn-Prediction-using-Python)](https://github.com/SabihaMishu/Customer-Churn-Prediction-using-Python)
[![License](https://img.shields.io/github/license/SabihaMishu/Customer-Churn-Prediction-using-Python)](https://github.com/SabihaMishu/Customer-Churn-Prediction-using-Python/blob/main/LICENSE)

A friendly, end-to-end project for predicting customer churn using Python — including data cleaning, feature engineering, model training, evaluation and interpretability. Ideal for data scientists, ML engineers, and business stakeholders who want actionable insights to reduce churn.

---

Table of Contents
- [Project Overview](#project-overview)
- [Why it matters](#why-it-matters)
- [What’s in this repo](#whats-in-this-repo)
- [Quickstart](#quickstart)
- [Usage examples](#usage-examples)
- [Modeling & Evaluation](#modeling--evaluation)
- [Results (example run)](#results-example-run)
- [Visualizations & Explainability](#visualizations--explainability)
- [Extend & Deploy](#extend--deploy)
- [Contributing](#contributing)
- [Credits & License](#credits--license)
- [Contact](#contact)

---

## Project Overview
Customer churn — when a customer stops using a company's product or service — is expensive. This project demonstrates how to build a robust churn prediction pipeline using classical ML and modern tooling. The goal is to identify customers at risk of leaving so businesses can proactively intervene.

Key capabilities:
- Clean and preprocess raw customer data
- Feature engineering (tenure, engagement aggregates, categorical encodings)
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Evaluate with business-focused metrics (ROC-AUC, Precision@K, confusion matrix)
- Explain predictions with feature importance and SHAP

---

## Why it matters
- Reducing churn by even a few percentage points can dramatically increase revenue.
- Predictive models allow targeted retention campaigns instead of costly broad discounts.
- Interpretability helps translate model output into concrete business actions.

---

## What’s in this repo
- notebooks/ — EDA and experimentation notebooks
- data/ — dataset samples or instructions to download original data (not included)
- src/
  - data_preparation.py — cleaning & feature engineering
  - train.py — model training pipeline
  - evaluate.py — evaluation utilities & metrics
  - predict.py — inference script
- models/ — saved trained models (if available)
- reports/ — sample visualizations, model reports
- requirements.txt — Python dependencies
- README.md — this file

> Note: Structure may vary slightly — check the repository root for exact filenames.

---

## Quickstart

1. Clone the repo
   ```bash
   git clone https://github.com/SabihaMishu/Customer-Churn-Prediction-using-Python.git
   cd Customer-Churn-Prediction-using-Python
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

3. Download dataset
   - If using the Telco Customer Churn dataset (Kaggle / IBM sample), place `telco_customer_churn.csv` into `data/` or update the path in the notebook/scripts.

4. Run the Jupyter notebook for exploration
   ```bash
   jupyter lab
   # open notebooks/EDA.ipynb
   ```

5. Train a model
   ```bash
   python src/train.py --data-path data/telco_customer_churn.csv --out-dir models/
   ```

6. Run inference (example)
   ```bash
   python src/predict.py --model models/xgboost.pkl --input data/sample_customers.csv --output results/predictions.csv
   ```

---

## Usage examples

- EDA: notebooks/01_EDA.ipynb — understand features, missingness, churn rates by segment
- Modeling: notebooks/02_modeling.ipynb — baseline to advanced models and cross-validation
- Pipelines: src/train.py — handles preprocessing, CV, model selection and saving artifacts
- Inference: src/predict.py — take new customer rows and return churn probabilities + recommended action

Example snippet (predicting churn probability):
```python
from src.predict import load_model, predict
model = load_model("models/xgboost.pkl")
probs = predict(model, df_new_customers)  # returns probability and label
```

---

## Modeling & Evaluation

Preprocessing steps used:
- Missing value handling (imputation)
- Encoding: One-Hot / Target / Ordinal depending on feature
- Scaling (where appropriate)
- Feature generation: tenure buckets, interaction features, aggregated usage stats

Algorithms commonly included:
- Logistic Regression (baseline)
- Random Forest (robust, interpretable via feature importances)
- XGBoost / LightGBM (performance-focused)

Evaluation metrics:
- ROC-AUC — overall ranking ability
- Precision, Recall, F1 — classification balance
- Precision@K / Lift — business operational metric (target top K for campaigns)
- Confusion Matrix — actionable error analysis

---

## Results (example run)
Your results will depend on data splits and hyperparameters. Example (illustrative):

| Model             | ROC-AUC | Precision | Recall | F1    |
|------------------:|--------:|---------:|------:|------:|
| LogisticRegression| 0.78    | 0.62     | 0.55  | 0.58 |
| RandomForest      | 0.83    | 0.70     | 0.62  | 0.66 |
| XGBoost           | 0.86    | 0.73     | 0.66  | 0.69 |

Tip: Use calibration and threshold tuning to match your retention budget (e.g., optimize Precision@Top-5%).

---

## Visualizations & Explainability
- ROC and PR curves for model comparison
- Confusion matrix heatmaps for error analysis
- Feature importance bars for tree models
- SHAP summary & dependence plots for local + global explainability
These artifacts should be in `reports/` or generated by notebooks.

---

## Extend & Deploy
Want to deploy?
- Export your best model (pickle / joblib / ONNX)
- Wrap inference in a lightweight API (Flask / FastAPI)
- Optionally build a dashboard (Streamlit / Dash) for business users
- Containerize with Docker for reproducible deployments

Suggested deployment workflow:
1. Build an API endpoint that accepts customer records and returns churn probability + top contributing features.
2. Integrate with marketing CRM to trigger retention campaigns for high-risk customers.
3. Monitor model performance and data drift in production.

---

## Contributing
Contributions are welcome! Ideas:
- Add hyperparameter tuning (Optuna)
- Add more explainability (SHAP/ELI5 examples)
- Create a Streamlit demo app
- Improve tests and CI

Please open an issue or submit a pull request with clear description and tests/examples.

---

## Credits & Acknowledgements
- Many churn datasets and examples are inspired by Kaggle / IBM Telco Customer Churn dataset.
- Libraries used: scikit-learn, pandas, numpy, matplotlib/seaborn, xgboost/lightgbm, shap

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact
Maintainer: SabihaMishu  
GitHub: https://github.com/SabihaMishu/Customer-Churn-Prediction-using-Python

If you build something cool with this project, open an issue or send a link — I’d love to see it!
