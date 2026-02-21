"""Unit tests for preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_data,
    get_feature_names,
    split_features_target,
)


@pytest.fixture
def sample_data():
    """Create a small sample dataset matching the IBM HR schema."""
    return pd.DataFrame({
        "EmployeeNumber": [1, 2, 3, 4, 5],
        "Age": [30, 45, 25, 50, 35],
        "Attrition": ["Yes", "No", "Yes", "No", "No"],
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel", "Travel_Rarely", "Travel_Rarely"],
        "DailyRate": [800, 1200, 500, 1400, 900],
        "Department": ["Sales", "Research & Development", "Human Resources", "Sales", "Research & Development"],
        "DistanceFromHome": [5, 15, 25, 3, 10],
        "Education": [3, 4, 2, 5, 3],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Life Sciences"],
        "EnvironmentSatisfaction": [3, 4, 1, 2, 3],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "HourlyRate": [60, 80, 45, 90, 70],
        "JobInvolvement": [3, 4, 2, 3, 3],
        "JobLevel": [2, 4, 1, 5, 3],
        "JobRole": ["Sales Executive", "Manager", "Laboratory Technician", "Research Director", "Research Scientist"],
        "JobSatisfaction": [3, 4, 1, 2, 4],
        "MaritalStatus": ["Single", "Married", "Divorced", "Married", "Single"],
        "MonthlyIncome": [5000, 15000, 2500, 18000, 8000],
        "MonthlyRate": [10000, 20000, 5000, 25000, 12000],
        "NumCompaniesWorked": [2, 5, 0, 3, 1],
        "OverTime": ["Yes", "No", "Yes", "No", "No"],
        "PercentSalaryHike": [15, 20, 12, 18, 14],
        "PerformanceRating": [3, 4, 3, 4, 3],
        "RelationshipSatisfaction": [3, 2, 4, 1, 3],
        "StockOptionLevel": [1, 2, 0, 3, 1],
        "TotalWorkingYears": [8, 20, 2, 30, 12],
        "TrainingTimesLastYear": [3, 2, 4, 1, 3],
        "WorkLifeBalance": [3, 2, 4, 3, 3],
        "YearsAtCompany": [5, 15, 1, 25, 8],
        "YearsInCurrentRole": [3, 10, 0, 15, 5],
        "YearsSinceLastPromotion": [1, 5, 0, 3, 2],
        "YearsWithCurrManager": [3, 8, 0, 12, 5],
    })


@pytest.fixture
def data_with_missing(sample_data):
    """Sample data with missing values."""
    df = sample_data.copy()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "Department"] = np.nan
    df.loc[2, "MonthlyIncome"] = np.nan
    return df


class TestCleanData:
    def test_drops_non_informative_columns(self, sample_data):
        sample_data["EmployeeCount"] = 1
        sample_data["Over18"] = "Y"
        sample_data["StandardHours"] = 80
        result = clean_data(sample_data)
        assert "EmployeeNumber" not in result.columns
        assert "EmployeeCount" not in result.columns
        assert "Over18" not in result.columns
        assert "StandardHours" not in result.columns

    def test_fills_missing_numeric(self, data_with_missing):
        result = clean_data(data_with_missing)
        assert result["Age"].isna().sum() == 0
        assert result["MonthlyIncome"].isna().sum() == 0

    def test_fills_missing_categorical(self, data_with_missing):
        result = clean_data(data_with_missing)
        assert result["Department"].isna().sum() == 0

    def test_does_not_modify_original(self, sample_data):
        original_shape = sample_data.shape
        clean_data(sample_data)
        assert sample_data.shape == original_shape


class TestSplitFeaturesTarget:
    def test_returns_correct_types(self, sample_data):
        X, y = split_features_target(sample_data)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_is_binary(self, sample_data):
        _, y = split_features_target(sample_data)
        assert set(y.unique()).issubset({0, 1})

    def test_target_values(self, sample_data):
        _, y = split_features_target(sample_data)
        assert y.iloc[0] == 1  # "Yes" -> 1
        assert y.iloc[1] == 0  # "No" -> 0

    def test_no_target_in_features(self, sample_data):
        X, _ = split_features_target(sample_data)
        assert "Attrition" not in X.columns

    def test_no_dropped_columns_in_features(self, sample_data):
        X, _ = split_features_target(sample_data)
        assert "EmployeeNumber" not in X.columns

    def test_only_expected_columns(self, sample_data):
        X, _ = split_features_target(sample_data)
        expected = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        assert set(X.columns).issubset(expected)


class TestBuildPreprocessor:
    def test_preprocessor_fits(self, sample_data):
        X, _ = split_features_target(sample_data)
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[0] == len(X)

    def test_output_is_numeric(self, sample_data):
        X, _ = split_features_target(sample_data)
        preprocessor = build_preprocessor(X)
        transformed = preprocessor.fit_transform(X)
        assert np.issubdtype(transformed.dtype, np.number)


class TestGetFeatureNames:
    def test_returns_list(self, sample_data):
        X, _ = split_features_target(sample_data)
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)
        names = get_feature_names(preprocessor, X)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_matches_transform_shape(self, sample_data):
        X, _ = split_features_target(sample_data)
        preprocessor = build_preprocessor(X)
        transformed = preprocessor.fit_transform(X)
        names = get_feature_names(preprocessor, X)
        assert len(names) == transformed.shape[1]
