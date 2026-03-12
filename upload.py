from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
from app.services.prediction_service import predict_batch

router = APIRouter()

@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        records = df.to_dict(orient="records")
        if len(records) > 500:
            raise HTTPException(status_code=400, detail="Max 500 records per upload.")
        results = predict_batch(records)

        high_risk = [r for r in results if r.get("risk_level") == "High"]
        medium_risk = [r for r in results if r.get("risk_level") == "Medium"]
        low_risk = [r for r in results if r.get("risk_level") == "Low"]

        avg_prob = sum(r.get("attrition_probability", 0) for r in results if "error" not in r) / max(len(results), 1)

        return {
            "success": True,
            "total_employees": len(records),
            "predictions": results,
            "summary": {
                "high_risk_count": len(high_risk),
                "medium_risk_count": len(medium_risk),
                "low_risk_count": len(low_risk),
                "average_attrition_probability": round(avg_prob, 4),
                "predicted_attrition_rate": round(
                    sum(1 for r in results if r.get("prediction") == 1) / max(len(results), 1) * 100, 2
                )
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/sample-csv")
def get_sample_csv():
    sample = """Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education,EducationField,EnvironmentSatisfaction,Gender,HourlyRate,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,MonthlyRate,NumCompaniesWorked,OverTime,PercentSalaryHike,PerformanceRating,RelationshipSatisfaction,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,WorkLifeBalance,YearsAtCompany,YearsInCurrentRole,YearsSinceLastPromotion,YearsWithCurrManager,EmployeeNumber,name
35,Travel_Rarely,800,Research & Development,5,3,Life Sciences,3,Male,65,3,2,Research Scientist,3,Single,5000,15000,2,No,13,3,3,1,8,3,3,5,3,1,3,1001,Alice Johnson
28,Travel_Frequently,600,Sales,20,2,Marketing,2,Female,45,2,1,Sales Representative,2,Single,3000,10000,4,Yes,12,3,2,0,3,2,2,2,1,0,1,1002,Bob Smith
45,Non-Travel,1200,Human Resources,3,4,Human Resources,4,Male,85,4,4,Manager,4,Married,12000,22000,1,No,18,4,4,3,20,5,4,15,10,5,10,1003,Carol White"""
    return {"csv_content": sample, "filename": "sample_employees.csv"}
