#!/usr/bin/env python3
"""
Vislona RAG Chatbot Setup Script
This script helps you set up the environment and dependencies for the Vislona chatbot.
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path
import time

def run_command(command, description=""):
    """Run a system command and return success status"""
    print(f"🔄 {description if description else f'Running: {command}'}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Success!")
            return True, result.stdout
        else:
            print(f"❌ Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout: Command took too long")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("💡 Please install Python 3.8 or higher")
        return False

def check_ollama_installation():
    """Check if Ollama is installed"""
    print("🦙 Checking Ollama installation...")
    success, output = run_command("ollama --version", "Checking Ollama version")

    if success:
        print(f"✅ Ollama is installed: {output.strip()}")
        return True
    else:
        print("❌ Ollama is not installed")
        print("💡 Install Ollama from: https://ollama.ai/download")
        print("💡 Or use the following commands:")
        print("   • Windows: Download from https://ollama.ai/download")
        print("   • macOS: brew install ollama")
        print("   • Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return False

def check_ollama_running():
    """Check if Ollama server is running"""
    print("🌐 Checking Ollama server status...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama server is running with {len(models)} models")
            return True, models
        else:
            print("❌ Ollama server is not responding correctly")
            return False, []
    except requests.exceptions.RequestException:
        print("❌ Ollama server is not running")
        print("💡 Start Ollama with: ollama serve")
        return False, []

def install_python_packages():
    """Install required Python packages"""
    print("📦 Installing Python packages...")

    packages = [
        "streamlit",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "faiss-cpu",
        "requests",
        "pathlib"
    ]

    for package in packages:
        print(f"📥 Installing {package}...")
        success, output = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"❌ Failed to install {package}")
            return False

    print("✅ All Python packages installed successfully!")
    return True

def download_ollama_models():
    """Download recommended Ollama models"""
    print("🎯 Setting up Ollama models...")

    # Check if Ollama is running
    running, existing_models = check_ollama_running()
    if not running:
        print("❌ Please start Ollama server first: ollama serve")
        return False

    existing_model_names = [model['name'] for model in existing_models] if existing_models else []

    # Recommended models
    models_to_install = [
        ("nomic-embed-text", "Embedding model for vector search"),
        ("llama3.2:1b", "Lightweight chat model (recommended for testing)"),
    ]

    # Optional models
    optional_models = [
        ("llama3.2:3b", "Better quality chat model"),
        ("llama3.1:8b", "High quality chat model (requires more resources)"),
    ]

    # Install essential models
    for model, description in models_to_install:
        if model in existing_model_names:
            print(f"✅ {model} already installed")
            continue

        print(f"📥 Installing {model} - {description}")
        print("⏱️ This may take a few minutes...")

        success, output = run_command(f"ollama pull {model}", f"Downloading {model}")
        if success:
            print(f"✅ {model} installed successfully!")
        else:
            print(f"❌ Failed to install {model}")
            print("💡 You can install it manually later with: ollama pull", model)

    # Ask about optional models
    print("\n🤔 Optional models (better quality but larger size):")
    for model, description in optional_models:
        if model in existing_model_names:
            print(f"✅ {model} already installed")
            continue

        response = input(f"Install {model} - {description}? (y/N): ").strip().lower()
        if response == 'y':
            print(f"📥 Installing {model}...")
            success, output = run_command(f"ollama pull {model}", f"Downloading {model}")
            if success:
                print(f"✅ {model} installed successfully!")
            else:
                print(f"❌ Failed to install {model}")
        else:
            print(f"⏭️ Skipping {model}")

    return True

def create_project_structure():
    """Create necessary project directories and files"""
    print("📁 Creating project structure...")

    # Create directories
    directories = [
        "data",
        "models",
        "logs",
        "vector_stores"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create requirements.txt
    requirements = """streamlit>=1.28.0
langchain>=0.3.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0
faiss-cpu>=1.7.4
requests>=2.31.0
pathlib
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")

    # Create .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# Streamlit
.streamlit/

# Vector stores
faiss_index_*
*.faiss
*.pkl

# Logs
logs/
*.log

# Data
data/
models/

# IDE
.vscode/
.idea/
*.swp
*.swo
"""

    with open(".gitignore", "w") as f:
        f.write(gitignore)
    print("✅ Created .gitignore")

    return True

def create_launch_scripts():
    """Create convenient launch scripts"""
    print("🚀 Creating launch scripts...")

    # Windows batch script
    windows_script = """@echo off
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
"""

    with open("start_chatbot.bat", "w") as f:
        f.write(windows_script)
    print("✅ Created start_chatbot.bat (Windows)")

    # Unix shell script
    unix_script = """#!/bin/bash
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
"""

    with open("start_chatbot.sh", "w") as f:
        f.write(unix_script)

    # Make shell script executable
    try:
        os.chmod("start_chatbot.sh", 0o755)
        print("✅ Created start_chatbot.sh (Linux/macOS)")
    except:
        print("✅ Created start_chatbot.sh (set executable permissions manually)")

    return True

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running installation tests...")

    # Test Python imports
    test_imports = [
        "streamlit",
        "langchain",
        "langchain_community.vectorstores",
        "langchain_ollama",
        "faiss",
        "requests"
    ]

    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} import successful")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            return False

    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection successful")
        else:
            print("❌ Ollama connection failed")
            return False
    except:
        print("⚠️ Ollama connection test skipped (server not running)")

    print("✅ All tests passed!")
    return True

def main():
    """Main setup function"""
    print("🤖 Vislona RAG Chatbot Setup")
    print("=" * 50)

    # Step 1: Check Python version
    if not check_python_version():
        return False

    # Step 2: Check Ollama installation
    if not check_ollama_installation():
        print("\n⏹️ Setup paused. Please install Ollama first.")
        return False

    # Step 3: Install Python packages
    print("\n" + "=" * 30)
    install_packages = input("📦 Install Python packages? (Y/n): ").strip().lower()
    if install_packages != 'n':
        if not install_python_packages():
            print("❌ Package installation failed")
            return False

    # Step 4: Create project structure
    print("\n" + "=" * 30)
    create_structure = input("📁 Create project structure? (Y/n): ").strip().lower()
    if create_structure != 'n':
        create_project_structure()

    # Step 5: Download Ollama models
    print("\n" + "=" * 30)
    download_models = input("🎯 Download Ollama models? (Y/n): ").strip().lower()
    if download_models != 'n':
        download_ollama_models()

    # Step 6: Create launch scripts
    print("\n" + "=" * 30)
    create_scripts = input("🚀 Create launch scripts? (Y/n): ").strip().lower()
    if create_scripts != 'n':
        create_launch_scripts()

    # Step 7: Run tests
    print("\n" + "=" * 30)
    run_test = input("🧪 Run installation tests? (Y/n): ").strip().lower()
    if run_test != 'n':
        run_tests()

    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Save the chatbot code as 'vislona_chatbot.py'")
    print("2. Start Ollama server: ollama serve")
    print("3. Run the chatbot: streamlit run vislona_chatbot.py")
    print("4. Or use launch scripts: ./start_chatbot.sh (Linux/macOS) or start_chatbot.bat (Windows)")
    print("\n💡 Tips:")
    print("• The chatbot will create a demo vector store on first run")
    print("• You can add your own documents to create a custom knowledge base")
    print("• Check the sidebar for configuration options")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup encountered issues. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your Python environment and try again")