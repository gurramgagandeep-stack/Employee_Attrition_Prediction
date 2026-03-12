#!/bin/bash

echo "🚀 AttritionIQ — Starting up..."
echo ""

# Backend
echo "📦 Setting up Python backend..."
cd backend
python -m venv venv 2>/dev/null || true
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

pip install -r requirements.txt -q

echo "🤖 Training ML model..."
python -m app.models.train_model

echo "⚡ Starting FastAPI server on port 8000..."
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

cd ../frontend
echo ""
echo "🎨 Installing frontend dependencies..."
npm install --silent

echo "🌐 Starting React app on port 3000..."
npm start &
FRONTEND_PID=$!

echo ""
echo "✅ AttritionIQ is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
