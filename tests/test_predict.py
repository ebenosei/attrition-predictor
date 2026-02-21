"""Unit tests for prediction module."""

import numpy as np
import pandas as pd
import pytest

from src.predict import _get_risk_tier, predict_batch, predict_single


class TestGetRiskTier:
    def test_high_risk(self):
        assert _get_risk_tier(60) == "High"
        assert _get_risk_tier(80) == "High"
        assert _get_risk_tier(100) == "High"

    def test_medium_risk(self):
        assert _get_risk_tier(30) == "Medium"
        assert _get_risk_tier(45) == "Medium"
        assert _get_risk_tier(59.9) == "Medium"

    def test_low_risk(self):
        assert _get_risk_tier(0) == "Low"
        assert _get_risk_tier(15) == "Low"
        assert _get_risk_tier(29.9) == "Low"


class TestPredictSingle:
    @pytest.fixture
    def trained_artifact(self):
        """Load the trained model artifact, skip if not available."""
        try:
            from src.predict import load_model
            return load_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet. Run `python src/train.py` first.")

    @pytest.fixture
    def sample_employee(self):
        return {
            "Age": 35,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 800,
            "Department": "Sales",
            "DistanceFromHome": 10,
            "Education": 3,
            "EducationField": "Life Sciences",
            "EnvironmentSatisfaction": 3,
            "Gender": "Male",
            "HourlyRate": 65,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 5000,
            "MonthlyRate": 14000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 5,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 3,
        }

    def test_returns_expected_keys(self, sample_employee, trained_artifact):
        result = predict_single(sample_employee, trained_artifact)
        assert "probability" in result
        assert "risk_score" in result
        assert "risk_tier" in result
        assert "shap_values" in result

    def test_probability_range(self, sample_employee, trained_artifact):
        result = predict_single(sample_employee, trained_artifact)
        assert 0 <= result["probability"] <= 1

    def test_risk_score_range(self, sample_employee, trained_artifact):
        result = predict_single(sample_employee, trained_artifact)
        assert 0 <= result["risk_score"] <= 100

    def test_risk_tier_valid(self, sample_employee, trained_artifact):
        result = predict_single(sample_employee, trained_artifact)
        assert result["risk_tier"] in ["Low", "Medium", "High"]

    def test_shap_values_structure(self, sample_employee, trained_artifact):
        result = predict_single(sample_employee, trained_artifact)
        shap_values = result["shap_values"]
        assert isinstance(shap_values, list)
        assert len(shap_values) > 0
        assert "feature" in shap_values[0]
        assert "shap_value" in shap_values[0]

    def test_high_risk_employee(self, trained_artifact):
        high_risk = {
            "Age": 22,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 300,
            "Department": "Sales",
            "DistanceFromHome": 28,
            "Education": 1,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 35,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 1500,
            "MonthlyRate": 3000,
            "NumCompaniesWorked": 7,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 1,
            "TrainingTimesLastYear": 0,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 0,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
        }
        result = predict_single(high_risk, trained_artifact)
        # A very high-risk profile should score above average
        assert result["risk_score"] > 20


class TestPredictBatch:
    @pytest.fixture
    def trained_artifact(self):
        try:
            from src.predict import load_model
            return load_model()
        except FileNotFoundError:
            pytest.skip("Model not trained yet. Run `python src/train.py` first.")

    @pytest.fixture
    def sample_batch(self):
        return pd.DataFrame([
            {
                "Age": 30, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
                "Department": "Sales", "DistanceFromHome": 5, "Education": 3,
                "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3,
                "Gender": "Male", "HourlyRate": 60, "JobInvolvement": 3,
                "JobLevel": 2, "JobRole": "Sales Executive", "JobSatisfaction": 3,
                "MaritalStatus": "Married", "MonthlyIncome": 5000, "MonthlyRate": 10000,
                "NumCompaniesWorked": 2, "OverTime": "No", "PercentSalaryHike": 15,
                "PerformanceRating": 3, "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1, "TotalWorkingYears": 8, "TrainingTimesLastYear": 3,
                "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
            },
            {
                "Age": 45, "BusinessTravel": "Non-Travel", "DailyRate": 1200,
                "Department": "Research & Development", "DistanceFromHome": 2, "Education": 4,
                "EducationField": "Medical", "EnvironmentSatisfaction": 4,
                "Gender": "Female", "HourlyRate": 80, "JobInvolvement": 4,
                "JobLevel": 4, "JobRole": "Manager", "JobSatisfaction": 4,
                "MaritalStatus": "Married", "MonthlyIncome": 15000, "MonthlyRate": 20000,
                "NumCompaniesWorked": 3, "OverTime": "No", "PercentSalaryHike": 20,
                "PerformanceRating": 4, "RelationshipSatisfaction": 4,
                "StockOptionLevel": 2, "TotalWorkingYears": 20, "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 3, "YearsAtCompany": 15, "YearsInCurrentRole": 10,
                "YearsSinceLastPromotion": 3, "YearsWithCurrManager": 8,
            },
        ])

    def test_returns_dataframe(self, sample_batch, trained_artifact):
        result = predict_batch(sample_batch, trained_artifact)
        assert isinstance(result, pd.DataFrame)

    def test_has_risk_columns(self, sample_batch, trained_artifact):
        result = predict_batch(sample_batch, trained_artifact)
        assert "risk_score" in result.columns
        assert "risk_tier" in result.columns

    def test_sorted_by_risk(self, sample_batch, trained_artifact):
        result = predict_batch(sample_batch, trained_artifact)
        scores = result["risk_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_batch_size_preserved(self, sample_batch, trained_artifact):
        result = predict_batch(sample_batch, trained_artifact)
        assert len(result) == len(sample_batch)
