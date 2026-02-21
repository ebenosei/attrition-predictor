"""Download or generate the IBM HR Analytics Employee Attrition dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-prediction/"
    "master/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_CSV = RAW_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def download_dataset(dest: Path = RAW_CSV) -> pd.DataFrame:
    """Download the IBM dataset. Falls back to synthetic generation on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return pd.read_csv(dest)

    try:
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text)
        return pd.read_csv(dest)
    except (requests.RequestException, Exception):
        print("Download failed. Generating synthetic dataset...")
        return generate_synthetic_dataset(dest)


def generate_synthetic_dataset(dest: Path = RAW_CSV) -> pd.DataFrame:
    """Generate a realistic synthetic dataset matching the IBM HR schema."""
    rng = np.random.default_rng(42)
    n = 1470

    job_roles = [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director",
        "Human Resources",
    ]
    departments = ["Sales", "Research & Development", "Human Resources"]
    education_fields = [
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other",
    ]

    df = pd.DataFrame({
        "EmployeeNumber": np.arange(1, n + 1),
        "Age": rng.integers(18, 61, size=n),
        "BusinessTravel": rng.choice(
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
            size=n, p=[0.71, 0.19, 0.10],
        ),
        "DailyRate": rng.integers(100, 1500, size=n),
        "Department": rng.choice(departments, size=n, p=[0.30, 0.63, 0.07]),
        "DistanceFromHome": rng.integers(1, 30, size=n),
        "Education": rng.integers(1, 6, size=n),
        "EducationField": rng.choice(education_fields, size=n),
        "EnvironmentSatisfaction": rng.integers(1, 5, size=n),
        "Gender": rng.choice(["Male", "Female"], size=n, p=[0.60, 0.40]),
        "HourlyRate": rng.integers(30, 100, size=n),
        "JobInvolvement": rng.integers(1, 5, size=n),
        "JobLevel": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.33, 0.33, 0.20, 0.10, 0.04]),
        "JobRole": rng.choice(job_roles, size=n),
        "JobSatisfaction": rng.integers(1, 5, size=n),
        "MaritalStatus": rng.choice(
            ["Single", "Married", "Divorced"], size=n, p=[0.32, 0.46, 0.22],
        ),
        "MonthlyIncome": rng.integers(1000, 20000, size=n),
        "MonthlyRate": rng.integers(2000, 27000, size=n),
        "NumCompaniesWorked": rng.integers(0, 10, size=n),
        "OverTime": rng.choice(["Yes", "No"], size=n, p=[0.28, 0.72]),
        "PercentSalaryHike": rng.integers(11, 26, size=n),
        "PerformanceRating": rng.choice([3, 4], size=n, p=[0.85, 0.15]),
        "RelationshipSatisfaction": rng.integers(1, 5, size=n),
        "StockOptionLevel": rng.choice([0, 1, 2, 3], size=n, p=[0.40, 0.35, 0.15, 0.10]),
        "TotalWorkingYears": rng.integers(0, 41, size=n),
        "TrainingTimesLastYear": rng.integers(0, 7, size=n),
        "WorkLifeBalance": rng.integers(1, 5, size=n),
        "YearsAtCompany": rng.integers(0, 41, size=n),
        "YearsInCurrentRole": rng.integers(0, 19, size=n),
        "YearsSinceLastPromotion": rng.integers(0, 16, size=n),
        "YearsWithCurrManager": rng.integers(0, 18, size=n),
    })

    # Generate attrition with ~15% rate, influenced by realistic factors
    risk = np.zeros(n)
    risk += (df["OverTime"] == "Yes").astype(float) * 0.15
    risk += (df["JobSatisfaction"] == 1).astype(float) * 0.10
    risk += (df["EnvironmentSatisfaction"] == 1).astype(float) * 0.08
    risk += (df["YearsAtCompany"] < 3).astype(float) * 0.08
    risk += (df["MonthlyIncome"] < 4000).astype(float) * 0.06
    risk += (df["DistanceFromHome"] > 20).astype(float) * 0.05
    risk += (df["MaritalStatus"] == "Single").astype(float) * 0.04
    risk += (df["WorkLifeBalance"] == 1).astype(float) * 0.06
    base_rate = 0.05
    prob = np.clip(base_rate + risk, 0, 0.8)
    df["Attrition"] = rng.binomial(1, prob).astype(str)
    df["Attrition"] = df["Attrition"].map({"1": "Yes", "0": "No"})

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


if __name__ == "__main__":
    data = download_dataset()
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Attrition rate: {(data['Attrition'] == 'Yes').mean():.1%}")
