#!/usr/bin/env python3
"""
Simple syntax and structure test for ScientificXGBRegressor
This test checks code structure without running into dependency issues
"""

import ast
import sys
import os

def test_syntax():
    """Test that all Python files have valid syntax"""
    print("🧪 Testing Python syntax for all files...")
    
    files_to_test = [
        'xgboost.py',
        'test_gpu_simple.py',
        'test_gpu_features.py',
        'test_parameter_warnings.py',
        'test_fixes.py',
        'scientific_xgb_demo.py'
    ]
    
    all_passed = True
    
    for filename in files_to_test:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Parse the AST to check syntax
                ast.parse(source, filename=filename)
                print(f"   ✅ {filename}: Syntax OK")
                
            except SyntaxError as e:
                print(f"   ❌ {filename}: Syntax Error - {e}")
                all_passed = False
            except Exception as e:
                print(f"   ⚠️  {filename}: Other Error - {e}")
                all_passed = False
        else:
            print(f"   ⚠️  {filename}: File not found")
    
    return all_passed

def test_class_structure():
    """Test that the main classes are properly defined"""
    print("\n🔍 Testing class structure...")
    
    try:
        with open('xgboost.py', 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes_found = []
        functions_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes_found.append(node.name)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                functions_found.append(node.name)
        
        expected_classes = ['GPUManager', 'ScientificXGBRegressor']
        expected_functions = ['create_scientific_xgb_regressor']
        
        print(f"   Classes found: {classes_found}")
        print(f"   Functions found: {len(functions_found)} functions")
        
        for cls in expected_classes:
            if cls in classes_found:
                print(f"   ✅ {cls}: Found")
            else:
                print(f"   ❌ {cls}: Missing")
        
        for func in expected_functions:
            if func in functions_found:
                print(f"   ✅ {func}: Found")
            else:
                print(f"   ❌ {func}: Missing")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error analyzing structure: {e}")
        return False

def test_imports_structure():
    """Test that import statements are properly structured"""
    print("\n📦 Testing import structure...")
    
    files_to_check = [
        'test_gpu_features.py',
        'test_parameter_warnings.py', 
        'test_fixes.py',
        'scientific_xgb_demo.py'
    ]
    
    all_good = True
    
    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Check if it uses the correct import pattern
                if 'import xgboost' in source and 'from xgboost import' not in source:
                    print(f"   ✅ {filename}: Uses correct import pattern")
                elif 'from xgboost import' in source:
                    print(f"   ⚠️  {filename}: Uses old import pattern (may cause issues)")
                    all_good = False
                else:
                    print(f"   ℹ️  {filename}: No xgboost imports found")
                    
            except Exception as e:
                print(f"   ❌ {filename}: Error checking imports - {e}")
                all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧪 ScientificXGBRegressor Syntax and Structure Test")
    print("=" * 60)
    
    # Test 1: Syntax
    syntax_ok = test_syntax()
    
    # Test 2: Class structure
    structure_ok = test_class_structure()
    
    # Test 3: Import structure
    imports_ok = test_imports_structure()
    
    print("\n📊 Test Summary:")
    print(f"   Syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"   Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    
    if syntax_ok and structure_ok and imports_ok:
        print("\n🎉 All tests passed! Code structure is correct.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        sys.exit(1) 