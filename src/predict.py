"""Prediction and SHAP explainability logic."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


def load_model(model_path: Path = MODEL_PATH) -> dict:
    """Load the serialized model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Please run `python src/train.py` and commit the model file."
        )
    return joblib.load(model_path)


def predict_single(employee: dict, artifact: dict | None = None) -> dict:
    """Predict attrition risk for a single employee.

    Returns:
        dict with keys: probability, risk_score, risk_tier, shap_values
    """
    if artifact is None:
        artifact = load_model()

    pipeline = artifact["pipeline"]
    columns = artifact["columns"]

    # Build a DataFrame with expected columns
    input_df = pd.DataFrame([employee])

    # Ensure all expected columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    # Predict
    probability = float(pipeline.predict_proba(input_df)[:, 1][0])
    risk_score = round(probability * 100, 1)
    risk_tier = _get_risk_tier(risk_score)

    # SHAP explanation
    shap_result = explain_prediction(input_df, artifact)

    return {
        "probability": probability,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "shap_values": shap_result,
    }


def predict_batch(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Predict attrition risk for a batch of employees.

    Returns the input DataFrame with added risk_score and risk_tier columns.
    """
    if artifact is None:
        artifact = load_model()

    pipeline = artifact["pipeline"]
    columns = artifact["columns"]

    # Ensure columns match
    input_df = df.copy()
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    model_input = input_df[columns]

    probabilities = pipeline.predict_proba(model_input)[:, 1]
    result = df.copy()
    result["risk_score"] = (probabilities * 100).round(1)
    result["risk_tier"] = result["risk_score"].apply(_get_risk_tier)

    # Sort by risk score descending
    result = result.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return result


def explain_prediction(
    input_df: pd.DataFrame, artifact: dict | None = None
) -> list[dict]:
    """Generate SHAP values for a prediction.

    Returns list of dicts with feature name, shap_value, and feature_value.
    """
    if artifact is None:
        artifact = load_model()

    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    # Transform through preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(input_df)

    # Get the classifier from the pipeline
    classifier = pipeline.named_steps["classifier"]

    # Use appropriate SHAP explainer
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        # For binary classification, TreeExplainer may return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = attrition
    except Exception:
        explainer = shap.LinearExplainer(classifier, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    shap_values = np.array(shap_values).flatten()

    # Pair with feature names
    n_features = min(len(feature_names), len(shap_values))
    feature_importance = []
    for i in range(n_features):
        feature_importance.append({
            "feature": feature_names[i],
            "shap_value": float(shap_values[i]),
            "abs_shap": abs(float(shap_values[i])),
        })

    # Sort by absolute SHAP value
    feature_importance.sort(key=lambda x: x["abs_shap"], reverse=True)

    return feature_importance


def _get_risk_tier(score: float) -> str:
    """Classify risk score into tiers."""
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"
