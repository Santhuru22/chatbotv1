#!/bin/bash
echo "Starting Vislona AI Chatbot..."
echo

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 3
fi

# Start Streamlit
streamlit run vislona_chatbot.py
