@echo off
echo Starting Vislona AI Chatbot...
echo.

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags > nul 2>&1
if %errorlevel% neq 0 (
    echo Ollama is not running. Starting Ollama...
    start /b ollama serve
    timeout /t 3 > nul
)

REM Start Streamlit
streamlit run vislona_chatbot.py

pause
