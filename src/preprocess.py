"""Data cleaning, feature engineering, and preprocessing pipeline."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns to drop (constant or non-informative)
DROP_COLS = [
    "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours",
]

# Feature type definitions
CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe: drop non-informative columns and handle missing values."""
    df = df.copy()

    # Drop columns that exist in DROP_COLS
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Fill missing numeric columns with median
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical columns with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into features X and binary target y."""
    df = clean_data(df)
    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"])

    # Keep only known feature columns that exist
    keep_cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    keep_num = [c for c in NUMERIC_FEATURES if c in X.columns]
    X = X[keep_cat + keep_num]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the feature set."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in X.columns]

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()

    return num_cols + cat_feature_names


if __name__ == "__main__":
    from src.ingest import download_dataset

    df = download_dataset()
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    feature_names = get_feature_names(preprocessor, X)
    print(f"Features: {len(feature_names)}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
