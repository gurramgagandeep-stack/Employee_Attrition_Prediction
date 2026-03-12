import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import List, Dict, Any

MODEL_DIR = Path(__file__).parent.parent / "trained_models"

_model = None
_scaler = None
_label_encoders = None
_explainer = None
_feature_cols = None

def load_artifacts():
    global _model, _scaler, _label_encoders, _explainer, _feature_cols
    try:
        _model = joblib.load(MODEL_DIR / "model.pkl")
        _scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        _label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
        _explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
        _feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

def get_model():
    if _model is None:
        load_artifacts()
    return _model, _scaler, _label_encoders, _explainer, _feature_cols

def preprocess_input(data: Dict, label_encoders: Dict, feature_cols: List) -> np.ndarray:
    df = pd.DataFrame([data])
    categorical_cols = ["BusinessTravel", "Department", "EducationField",
                        "Gender", "JobRole", "MaritalStatus", "OverTime"]
    for col in categorical_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df.values

def predict_single(data: Dict) -> Dict:
    model, scaler, label_encoders, explainer, feature_cols = get_model()
    if model is None:
        raise ValueError("Model not loaded. Please train the model first.")

    X = preprocess_input(data, label_encoders, feature_cols)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= 0.5)

    # SHAP explanation
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_dict = {feat: round(float(val), 4) for feat, val in zip(feature_cols, shap_vals)}
    top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    risk_level = "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"
    risk_color = "#22c55e" if prob < 0.3 else "#f59e0b" if prob < 0.6 else "#ef4444"

    return {
        "attrition_probability": round(float(prob), 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "shap_values": shap_dict,
        "top_factors": [{"feature": k, "impact": v} for k, v in top_factors],
    }

def predict_batch(records: List[Dict]) -> List[Dict]:
    results = []
    for record in records:
        try:
            result = predict_single(record)
            result["employee_id"] = record.get("EmployeeNumber", record.get("employee_id", "N/A"))
            result["name"] = record.get("name", f"Employee {record.get('EmployeeNumber', 'N/A')}")
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "employee_id": record.get("EmployeeNumber", "N/A")})
    return results
