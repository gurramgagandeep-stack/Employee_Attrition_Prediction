from fastapi import APIRouter, HTTPException
import json
from pathlib import Path

router = APIRouter()
MODEL_DIR = Path(__file__).parent.parent / "trained_models"

@router.get("/model-info")
def get_model_info():
    try:
        metrics_path = MODEL_DIR / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="Model not trained yet. Run training script.")
        with open(metrics_path) as f:
            metrics = json.load(f)
        sorted_features = sorted(metrics.get("feature_importance", {}).items(), key=lambda x: x[1], reverse=True)
        return {
            "success": True,
            "metrics": {k: v for k, v in metrics.items() if k != "feature_importance"},
            "top_features": [{"feature": k, "importance": round(v, 4)} for k, v in sorted_features[:15]]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
def trigger_training():
    try:
        from app.models.train_model import train_model
        _, metrics = train_model()
        return {"success": True, "message": "Model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
