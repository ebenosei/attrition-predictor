"""Train models, evaluate, select best by ROC-AUC, and serialize."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.ingest import download_dataset
from src.preprocess import build_preprocessor, get_feature_names, split_features_target

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def train_and_evaluate() -> dict:
    """Train all models, evaluate, and save the best one."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = download_dataset()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42, eval_metric="logloss",
        ),
    }

    results = {}
    best_auc = -1
    best_name = None
    best_pipeline = None

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

        # Fit on full training set
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        results[name] = {
            "roc_auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "cv_mean_auc": float(cv_scores.mean()),
            "cv_std_auc": float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        print(f"  ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_name} (ROC-AUC: {best_auc:.4f})")

    # Get feature names for the best model
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor, X_train)

    # Save model artifacts
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_names": feature_names,
        "results": results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "columns": X_train.columns.tolist(),
    }

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(artifact, model_path, compress=3)
    print(f"Model saved to {model_path}")

    # Save metrics JSON for reference
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    train_and_evaluate()
