"""
Quick run script for the algorithmic trading system
This script ensures all dependencies are met and runs the system
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import yfinance
        import pandas
        import numpy
        import sklearn
        import yaml
        print(" All required packages are installed")
        return True
    except ImportError as e:
        print(f" Missing package: {e}")
        return False

def install_missing_packages():
    """Install missing packages"""
    try:
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install packages automatically")
        return False

def create_minimal_structure():
    """Create minimal directory structure if missing"""
    directories = ['src', 'config', 'logs', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/strategy/__init__.py',
        'src/ml/__init__.py',
        'src/automation/__init__.py',
        'src/utils/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch()

def run_trading_system():
    """Run the main trading system"""
    try:
        # Import and run main
        from main import run_trading_system
        run_trading_system()
    except Exception as e:
        print(f"Error running trading system: {e}")
        print("\nTrying alternative import...")
        try:
            exec(open('main.py').read())
        except Exception as e2:
            print(f"Failed to run: {e2}")
            return False
    return True

def main():
    """Main execution function"""
    print("Algorithmic Trading System - Quick Start")
    print("="*50)
    
    # Check if requirements are met
    if not check_requirements():
        print("Installing missing packages...")
        if not install_missing_packages():
            print("Please install requirements manually:")
            print("pip install yfinance pandas numpy scikit-learn pyyaml")
            return
    
    # Create minimal structure
    create_minimal_structure()
    
    # Run the system
    print("\nStarting trading system...")
    run_trading_system()

if __name__ == "__main__":
    main()