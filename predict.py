from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.prediction_service import predict_single

router = APIRouter()

class EmployeeData(BaseModel):
    Age: int = Field(..., ge=18, le=70)
    BusinessTravel: str = "Travel_Rarely"
    DailyRate: int = 800
    Department: str = "Research & Development"
    DistanceFromHome: int = 5
    Education: int = Field(3, ge=1, le=5)
    EducationField: str = "Life Sciences"
    EnvironmentSatisfaction: int = Field(3, ge=1, le=4)
    Gender: str = "Male"
    HourlyRate: int = 65
    JobInvolvement: int = Field(3, ge=1, le=4)
    JobLevel: int = Field(2, ge=1, le=5)
    JobRole: str = "Research Scientist"
    JobSatisfaction: int = Field(3, ge=1, le=4)
    MaritalStatus: str = "Single"
    MonthlyIncome: int = 5000
    MonthlyRate: int = 15000
    NumCompaniesWorked: int = 2
    OverTime: str = "No"
    PercentSalaryHike: int = 13
    PerformanceRating: int = Field(3, ge=1, le=4)
    RelationshipSatisfaction: int = Field(3, ge=1, le=4)
    StockOptionLevel: int = Field(1, ge=0, le=3)
    TotalWorkingYears: int = 8
    TrainingTimesLastYear: int = 3
    WorkLifeBalance: int = Field(3, ge=1, le=4)
    YearsAtCompany: int = 5
    YearsInCurrentRole: int = 3
    YearsSinceLastPromotion: int = 1
    YearsWithCurrManager: int = 3
    EmployeeNumber: Optional[int] = None
    name: Optional[str] = None

@router.post("/predict")
def predict(employee: EmployeeData):
    try:
        result = predict_single(employee.dict())
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
