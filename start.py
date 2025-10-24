#!/usr/bin/env python3
"""
Startup script for Clash Royale Win Predictor AI
"""
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import tensorflow
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def check_environment():
    """Check if environment variables are set"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        subprocess.run(["cp", ".env.example", ".env"])
        print("📝 Please edit .env file with your configuration")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        "CLASH_ROYALE_API_KEY",
        "DATABASE_URL", 
        "SECRET_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("📝 Please set these in your .env file")
        return False
    
    print("✅ Environment variables configured")
    return True

def setup_database():
    """Setup database with migrations"""
    try:
        print("🗄️  Setting up database...")
        # Try with python -m alembic first
        try:
            subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to direct alembic command
            subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("✅ Database migrations completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Database setup failed: {e}")
        print("📝 Continuing without database migrations...")
        return True  # Continue anyway
    except Exception as e:
        print(f"⚠️  Database setup error: {e}")
        print("📝 Continuing without database migrations...")
        return True  # Continue anyway

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend server...")
    
    # Start uvicorn server
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    return subprocess.Popen(cmd)

def wait_for_backend():
    """Wait for backend to be ready"""
    print("⏳ Waiting for backend to start...")
    
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(f"   Attempt {i+1}/30...")
    
    print("❌ Backend failed to start")
    return False

def open_frontend():
    """Open the frontend in browser"""
    import webbrowser
    frontend_url = "http://localhost:8000/static/index.html"
    
    print(f"🌐 Opening frontend at {frontend_url}")
    webbrowser.open(frontend_url)

def main():
    """Main startup function"""
    print("🎮 Clash Royale Win Predictor AI - Startup")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\n📋 Setup Instructions:")
        print("1. Copy .env.example to .env")
        print("2. Get a Clash Royale API key from https://developer.clashroyale.com/")
        print("3. Set up a PostgreSQL database")
        print("4. Configure your .env file with the required values")
        print("5. Run this script again")
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    
    try:
        # Wait for backend to be ready
        if wait_for_backend():
            # Open frontend
            open_frontend()
            
            print("\n🎉 Application started successfully!")
            print("📊 Backend API: http://localhost:8000")
            print("🎮 Frontend: http://localhost:8000/static/index.html")
            print("📚 API Docs: http://localhost:8000/docs")
            print("\nPress Ctrl+C to stop the server")
            
            # Keep the script running
            backend_process.wait()
        else:
            backend_process.terminate()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()
