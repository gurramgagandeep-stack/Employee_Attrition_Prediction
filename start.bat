@echo off
echo 🚀 AttritionIQ — Starting up...

echo 📦 Setting up Python backend...
cd backend
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo 🤖 Training ML model...
python -m app.models.train_model

echo ⚡ Starting FastAPI on port 8000...
start cmd /k "call venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"

cd ..\frontend
echo 🎨 Installing frontend...
npm install

echo 🌐 Starting React on port 3000...
start cmd /k "npm start"

echo.
echo ✅ AttritionIQ is running!
echo    Frontend: http://localhost:3000
echo    Backend:  http://localhost:8000
echo    API Docs: http://localhost:8000/docs
pause
