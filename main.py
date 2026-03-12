from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, upload, model_info, analytics

app = FastAPI(
    title="Employee Attrition Predictor API",
    description="AI-powered HR analytics for predicting employee attrition",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(model_info.router, prefix="/api/v1", tags=["Model"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])

@app.get("/")
def root():
    return {"message": "Employee Attrition Predictor API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}
