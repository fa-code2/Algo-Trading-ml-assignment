"""
Script to automatically fix relative imports in the trading system
"""

import os
import re
import glob
from pathlib import Path

def fix_relative_imports_in_file(file_path):
    """Fix relative imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Common relative import patterns to fix
        patterns = [
            # from .config import -> from utils.config import
            (r'from \.config import', 'from utils.config import'),
            (r'from \.logger import', 'from utils.logger import'),
            
            # from .data_fetcher import -> from data.data_fetcher import
            (r'from \.data_fetcher import', 'from data.data_fetcher import'),
            (r'from \.indicators import', 'from data.indicators import'),
            
            # from .rsi_ma_strategy import -> from strategy.rsi_ma_strategy import
            (r'from \.rsi_ma_strategy import', 'from strategy.rsi_ma_strategy import'),
            (r'from \.backtester import', 'from strategy.backtester import'),
            
            # from .model_trainer import -> from ml.model_trainer import
            (r'from \.model_trainer import', 'from ml.model_trainer import'),
            
            # from .sheets_manager import -> from automation.sheets_manager import
            (r'from \.sheets_manager import', 'from automation.sheets_manager import'),
            
            # Cross-directory imports
            (r'from \.\.utils import', 'from utils import'),
            (r'from \.\.utils\.config import', 'from utils.config import'),
            (r'from \.\.utils\.logger import', 'from utils.logger import'),
            
            (r'from \.\.data import', 'from data import'),
            (r'from \.\.data\.data_fetcher import', 'from data.data_fetcher import'),
            (r'from \.\.data\.indicators import', 'from data.indicators import'),
            
            (r'from \.\.strategy import', 'from strategy import'),
            (r'from \.\.strategy\.rsi_ma_strategy import', 'from strategy.rsi_ma_strategy import'),
            (r'from \.\.strategy\.backtester import', 'from strategy.backtester import'),
            
            (r'from \.\.ml import', 'from ml import'),
            (r'from \.\.ml\.model_trainer import', 'from ml.model_trainer import'),
            
            (r'from \.\.automation import', 'from automation import'),
            (r'from \.\.automation\.sheets_manager import', 'from automation.sheets_manager import'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"  {pattern} -> {replacement}")
        
        # Save if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes_made
        
        return []
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def fix_all_imports():
    """Fix relative imports in all Python files"""
    print("Fixing relative imports in Python files...")
    print("="*50)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    
    if not os.path.exists(src_dir):
        print(f"❌ Source directory not found: {src_dir}")
        return False
    
    # Find all Python files in src directory
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                python_files.append(os.path.join(root, file))
    
    if not python_files:
        print("No Python files found in src directory")
        return False
    
    total_changes = 0
    files_modified = 0
    
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, script_dir)
        changes = fix_relative_imports_in_file(file_path)
        
        if changes:
            print(f"\n✓ Modified: {relative_path}")
            for change in changes:
                print(change)
            files_modified += 1
            total_changes += len(changes)
        else:
            print(f"  No changes needed: {relative_path}")
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"Files processed: {len(python_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Total changes: {total_changes}")
    
    if files_modified > 0:
        print(f"\n Successfully fixed relative imports!")
        print("You can now run the test script again:")
        print("python test_system.py")
    else:
        print("\n No relative imports found to fix.")
        print("The import issues might be due to other problems.")
    
    return files_modified > 0

def backup_files():
    """Create backup of Python files before modification"""
    print("Creating backup of Python files...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    backup_dir = os.path.join(script_dir, 'backup_src')
    
    if not os.path.exists(src_dir):
        print("No src directory found to backup")
        return False
    
    if os.path.exists(backup_dir):
        print(f"Backup directory already exists: {backup_dir}")
        return True
    
    try:
        import shutil
        shutil.copytree(src_dir, backup_dir)
        print(f" Backup created: {backup_dir}")
        return True
    except Exception as e:
        print(f" Failed to create backup: {e}")
        return False

def analyze_imports():
    """Analyze import patterns in the source files"""
    print("Analyzing import patterns...")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')
    
    if not os.path.exists(src_dir):
        print(f" Source directory not found: {src_dir}")
        return
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    relative_imports = []
    absolute_imports = []
    
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, script_dir)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('from .') or line.startswith('from ..'):
                    relative_imports.append(f"{relative_path}:{i} - {line}")
                elif line.startswith('from ') and 'import' in line:
                    absolute_imports.append(f"{relative_path}:{i} - {line}")
                    
        except Exception as e:
            print(f"Error reading {relative_path}: {e}")
    
    print(f"Found {len(relative_imports)} relative imports:")
    for imp in relative_imports:
        print(f"  {imp}")
    
    print(f"\nFound {len(absolute_imports)} absolute imports:")
    for imp in absolute_imports[:10]:  # Show first 10
        print(f"  {imp}")
    
    if len(absolute_imports) > 10:
        print(f"  ... and {len(absolute_imports) - 10} more")

def main():
    """Main function"""
    print("Python Import Fixer for Algorithmic Trading System")
    print("="*60)
    
    # Analyze current imports
    analyze_imports()
    
    print("\nOptions:")
    print("1. Create backup and fix imports")
    print("2. Fix imports without backup (risky)")
    print("3. Only analyze (no changes)")
    print("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == '1':
        if backup_files():
            fix_all_imports()
        else:
            print(" Backup failed. Not proceeding with fixes.")
    elif choice == '2':
        print("  Proceeding without backup...")
        fix_all_imports()
    elif choice == '3':
        print("Analysis completed. No changes made.")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()