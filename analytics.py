from fastapi import APIRouter
import numpy as np

router = APIRouter()

@router.get("/analytics/overview")
def get_overview():
    """Return mock company-wide analytics for dashboard demo"""
    return {
        "success": True,
        "data": {
            "total_employees": 1470,
            "attrition_rate": 16.12,
            "avg_tenure": 7.2,
            "high_risk_count": 89,
            "medium_risk_count": 214,
            "low_risk_count": 1167,
            "departments": [
                {"name": "Sales", "attrition_rate": 20.6, "count": 446},
                {"name": "R&D", "attrition_rate": 13.8, "count": 961},
                {"name": "HR", "attrition_rate": 19.0, "count": 63},
            ],
            "monthly_trend": [
                {"month": "Jan", "attrition": 12}, {"month": "Feb", "attrition": 15},
                {"month": "Mar", "attrition": 10}, {"month": "Apr", "attrition": 18},
                {"month": "May", "attrition": 14}, {"month": "Jun", "attrition": 22},
                {"month": "Jul", "attrition": 16}, {"month": "Aug", "attrition": 19},
                {"month": "Sep", "attrition": 13}, {"month": "Oct", "attrition": 25},
                {"month": "Nov", "attrition": 17}, {"month": "Dec", "attrition": 11},
            ],
            "risk_factors": [
                {"factor": "Overtime", "impact": 0.28},
                {"factor": "Job Satisfaction", "impact": 0.22},
                {"factor": "Work-Life Balance", "impact": 0.18},
                {"factor": "Years at Company", "impact": 0.15},
                {"factor": "Monthly Income", "impact": 0.12},
                {"factor": "Distance From Home", "impact": 0.08},
            ]
        }
    }
