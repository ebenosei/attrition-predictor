# Employee Attrition Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ebenosei-attrition-predictor.streamlit.app)
[![CI](https://github.com/ebenosei/attrition-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ebenosei/attrition-predictor/actions)

A people analytics tool that predicts employee flight risk using machine learning. Built with scikit-learn, XGBoost, and Streamlit.

## Business Context

Employee attrition costs organizations 50-200% of an employee's annual salary in replacement costs. This tool helps HR practitioners proactively identify at-risk employees and take targeted retention actions before it's too late.

The model is trained on the IBM HR Analytics Employee Attrition dataset and evaluates three classifiers (Logistic Regression, Random Forest, XGBoost) to select the best performer by ROC-AUC.

## Live Demo

[Launch the app](https://ebenosei-attrition-predictor.streamlit.app)

## Features

- **Workforce Overview** — Interactive charts showing attrition patterns across departments, roles, age groups, and more
- **Model Performance** — ROC curves, confusion matrices, feature importance, and side-by-side model comparison
- **Flight Risk Scorer** — Enter individual employee attributes and get a risk score (0-100), risk tier, and SHAP-based explanations
- **Workforce Risk Report** — Upload a CSV of employees, get a ranked risk table, and download an Excel report with conditional formatting

## Screenshots

<!-- Add screenshots here -->

## Setup

### Prerequisites

- Python 3.11+

### Installation

```bash
git clone https://github.com/ebenosei/attrition-predictor.git
cd attrition-predictor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Train the Model

```bash
python src/train.py
```

This downloads the dataset, trains three models, selects the best by ROC-AUC, and saves it to `models/best_model.joblib`.

### Run the App

```bash
streamlit run app/main.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (including `models/best_model.joblib`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Set main file path to `app/main.py`
5. Deploy

## Retraining the Model

To retrain with updated data:

```bash
python src/train.py
```

Then commit and push the updated model:

```bash
git add models/best_model.joblib models/metrics.json
git commit -m "chore: retrain model"
git push
```

The Streamlit Cloud app will automatically pick up the new model on the next deploy.

You can also retrain from the Streamlit sidebar when running locally (the button is hidden on Streamlit Cloud).

## People Analytics Insights

Key findings the model surfaced:

1. **Overtime is the strongest attrition driver.** Employees working overtime leave at roughly 3x the rate of those who don't, making workload management the single highest-leverage retention tool.

2. **Early-career employees are most at risk.** Workers with fewer than 3 years at the company and those in entry-level roles show significantly elevated flight risk, pointing to the importance of onboarding and early engagement programs.

3. **Compensation matters — but it's not everything.** While low monthly income correlates with higher attrition, job satisfaction and environment satisfaction scores are equally predictive. Employees with the lowest satisfaction ratings leave at more than double the baseline rate.

4. **Job role predicts attrition more than department.** Sales Representatives and Laboratory Technicians show the highest attrition rates, regardless of department-level trends, suggesting that role-specific interventions outperform blanket policies.

5. **Stock options and promotions create retention gravity.** Employees with zero stock options and those who haven't been promoted in 5+ years show compounding risk factors that amplify when combined with other dissatisfaction signals.

## Project Structure

```
attrition-predictor/
├── .github/workflows/ci.yml    # CI pipeline
├── .streamlit/config.toml       # Theme configuration
├── app/main.py                  # Streamlit dashboard
├── data/raw/                    # Raw CSV data
├── models/                      # Serialized model
├── notebooks/eda.ipynb          # Exploratory analysis
├── src/
│   ├── ingest.py                # Data download/generation
│   ├── preprocess.py            # Cleaning + feature engineering
│   ├── train.py                 # Model training + evaluation
│   └── predict.py               # Prediction + SHAP logic
├── tests/
│   ├── test_preprocess.py       # Preprocessing tests
│   └── test_predict.py          # Prediction tests
├── requirements.txt             # Pinned dependencies
├── packages.txt                 # System-level deps (Streamlit Cloud)
├── setup.py                     # Package setup
└── README.md
```

## Tech Stack

- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP
- **App:** Streamlit, Plotly
- **Data:** pandas, NumPy
- **CI:** GitHub Actions, pytest
