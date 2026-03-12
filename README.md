# AttritionIQ — Employee Attrition Predictor

> AI-powered HR analytics platform built for enterprise workforce intelligence.  
> Predict which employees are at risk of leaving — with full SHAP explainability.

---

## ✨ Features

- **Individual Prediction** — Enter employee data and get instant attrition risk with probability score
- **Bulk CSV Upload** — Upload entire workforce exports (up to 500 employees) for batch analysis
- **SHAP Explainability** — Every prediction shows the top driving factors (positive/negative)
- **Interactive Dashboard** — Company-wide attrition trends, department breakdown, risk distribution
- **Model Insights** — Feature importances, confusion matrix, performance metrics radar
- **Retrain On Demand** — Trigger model retraining via UI with fresh synthetic/real data

---

## 🏗 Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | Scikit-learn Gradient Boosting Classifier |
| Explainability | SHAP (TreeExplainer) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Recharts |
| Styling | Custom CSS Design System (white + yellow theme) |

---

## 📁 Project Structure

```
employee-attrition-predictor/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── predict.py       # Single employee prediction endpoint
│   │   │   ├── upload.py        # CSV bulk upload endpoint
│   │   │   ├── model_info.py    # Model metrics & retraining endpoint
│   │   │   └── analytics.py    # Dashboard analytics endpoint
│   │   ├── models/
│   │   │   └── train_model.py   # Model training script
│   │   └── services/
│   │       └── prediction_service.py  # Core prediction + SHAP logic
│   ├── trained_models/          # Saved model artifacts (auto-created)
│   └── requirements.txt
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── App.js               # Root app + sidebar navigation
    │   ├── App.css              # Global design system CSS
    │   ├── index.js
    │   ├── pages/
    │   │   ├── Dashboard.js     # Overview analytics
    │   │   ├── PredictSingle.js # Individual prediction form
    │   │   ├── BulkUpload.js    # CSV drag-and-drop upload
    │   │   └── ModelInsights.js # Model performance + features
    │   └── utils/
    │       └── api.js           # Axios API client
    └── package.json
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

---

### 1. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate.bat     # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model (required before running)
python -m app.models.train_model

# Start the API server
uvicorn app.main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend will be available at: `http://localhost:3000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/predict` | Single employee prediction |
| `POST` | `/api/v1/upload` | Bulk CSV upload |
| `GET` | `/api/v1/model-info` | Model metrics + feature importances |
| `POST` | `/api/v1/train` | Trigger model retraining |
| `GET` | `/api/v1/analytics/overview` | Dashboard analytics |
| `GET` | `/api/v1/sample-csv` | Download sample CSV |

---

## 📊 CSV Format

Your CSV must include these columns (see sample from the Upload page):

```
Age, BusinessTravel, DailyRate, Department, DistanceFromHome,
Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate,
JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus,
MonthlyIncome, MonthlyRate, NumCompaniesWorked, OverTime, PercentSalaryHike,
PerformanceRating, RelationshipSatisfaction, StockOptionLevel,
TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
```

Optional columns: `EmployeeNumber`, `name`

---

## 🎨 Design System

The frontend uses a **minimalist white + yellow** theme:
- **Primary**: `#FBBF24` (Amber Yellow)
- **Background**: `#FFFFFF` / `#F9FAFB`
- **Accent Dark**: `#0A0A0A`
- **Fonts**: Syne (display) + DM Sans (body)

---

## 🔧 Configuration

Set these environment variables for production:

**Backend** (`backend/.env`):
```
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

**Frontend** (`frontend/.env`):
```
REACT_APP_API_URL=http://localhost:8000/api/v1
```

---

## 📈 Model Performance

The default Gradient Boosting model achieves on synthetic data:

| Metric | Score |
|--------|-------|
| Accuracy | ~87% |
| ROC-AUC | ~91% |
| F1 Score | ~73% |

For production use, replace synthetic training data with real IBM HR Analytics or company-specific data in `train_model.py`.

---

## 🛡 Production Checklist

- [ ] Replace synthetic data with real HR data
- [ ] Add authentication (JWT / OAuth2)
- [ ] Configure proper CORS origins
- [ ] Set up PostgreSQL for prediction logging
- [ ] Deploy backend: Railway / Fly.io / AWS ECS
- [ ] Deploy frontend: Vercel / Netlify

---

Built with ⚡ for enterprise HR analytics use cases.
