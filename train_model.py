import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap
import joblib
import json
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

ALL_FEATURES = CATEGORICAL_COLS + NUMERICAL_COLS

label_encoders = {}
scaler = StandardScaler()

def generate_synthetic_data(n=1000):
    np.random.seed(42)
    data = {
        "Age": np.random.randint(18, 60, n),
        "BusinessTravel": np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.7, 0.2, 0.1]),
        "DailyRate": np.random.randint(100, 1500, n),
        "Department": np.random.choice(["Sales", "Research & Development", "Human Resources"], n, p=[0.3, 0.6, 0.1]),
        "DistanceFromHome": np.random.randint(1, 30, n),
        "Education": np.random.randint(1, 5, n),
        "EducationField": np.random.choice(["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"], n),
        "EnvironmentSatisfaction": np.random.randint(1, 5, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "HourlyRate": np.random.randint(30, 100, n),
        "JobInvolvement": np.random.randint(1, 5, n),
        "JobLevel": np.random.randint(1, 6, n),
        "JobRole": np.random.choice(["Sales Executive", "Research Scientist", "Laboratory Technician",
                                      "Manufacturing Director", "Healthcare Representative", "Manager",
                                      "Sales Representative", "Research Director", "Human Resources"], n),
        "JobSatisfaction": np.random.randint(1, 5, n),
        "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], n, p=[0.3, 0.5, 0.2]),
        "MonthlyIncome": np.random.randint(1000, 20000, n),
        "MonthlyRate": np.random.randint(2000, 27000, n),
        "NumCompaniesWorked": np.random.randint(0, 10, n),
        "OverTime": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "PercentSalaryHike": np.random.randint(11, 25, n),
        "PerformanceRating": np.random.choice([3, 4], n, p=[0.85, 0.15]),
        "RelationshipSatisfaction": np.random.randint(1, 5, n),
        "StockOptionLevel": np.random.randint(0, 4, n),
        "TotalWorkingYears": np.random.randint(0, 40, n),
        "TrainingTimesLastYear": np.random.randint(0, 7, n),
        "WorkLifeBalance": np.random.randint(1, 5, n),
        "YearsAtCompany": np.random.randint(0, 40, n),
        "YearsInCurrentRole": np.random.randint(0, 18, n),
        "YearsSinceLastPromotion": np.random.randint(0, 15, n),
        "YearsWithCurrManager": np.random.randint(0, 17, n),
    }
    df = pd.DataFrame(data)
    # Realistic attrition logic
    attrition_prob = (
        0.05 +
        0.15 * (df["OverTime"] == "Yes") +
        0.10 * (df["JobSatisfaction"] <= 2) +
        0.08 * (df["EnvironmentSatisfaction"] <= 2) +
        0.07 * (df["WorkLifeBalance"] <= 2) +
        0.06 * (df["YearsAtCompany"] <= 2) +
        0.05 * (df["NumCompaniesWorked"] >= 5) +
        0.04 * (df["DistanceFromHome"] >= 20) +
        0.05 * (df["MaritalStatus"] == "Single") +
        0.03 * (df["BusinessTravel"] == "Travel_Frequently") -
        0.05 * (df["StockOptionLevel"] >= 2) -
        0.04 * (df["JobLevel"] >= 4)
    )
    attrition_prob = np.clip(attrition_prob, 0.01, 0.95)
    df["Attrition"] = (np.random.random(n) < attrition_prob).astype(int)
    return df

def preprocess(df, fit=False):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
    return df

def train_model():
    print("Generating training data...")
    df = generate_synthetic_data(2000)
    df_processed = preprocess(df, fit=True)

    feature_cols = [c for c in ALL_FEATURES if c in df_processed.columns]
    X = df_processed[feature_cols]
    y = df_processed["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_names": feature_cols,
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist()))
    }

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)

    # Save everything
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoders, MODEL_DIR / "label_encoders.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model trained successfully!")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"   F1 Score:  {metrics['f1_score']:.4f}")
    return model, metrics

if __name__ == "__main__":
    train_model()
