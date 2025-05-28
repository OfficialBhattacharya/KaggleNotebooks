"""
Example usage of the ModelExplainer class
This script demonstrates how to use the ModelExplainer with sample data
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import sys
import os

# Add the dataModeller directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelExplainer import ModelExplainer

def create_sample_data_and_model(task_type='classification', save_model=True):
    """
    Create sample data and train a model for demonstration.
    
    Parameters:
    -----------
    task_type : str
        'classification' or 'regression'
    save_model : bool
        Whether to save the model to disk
    
    Returns:
    --------
    tuple : (model, X_train, X_test, y_train, y_test, fitted_values, predictions)
    """
    print(f"\n🔄 Creating sample {task_type} data and model...")
    
    if task_type == 'classification':
        # Create classification dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        fitted_values = model.predict_proba(X_train)[:, 1]  # Probabilities for positive class
        predictions = model.predict_proba(X_test)[:, 1]
        
    else:  # regression
        # Create regression dataset
        X, y = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        fitted_values = model.predict(X_train)
        predictions = model.predict(X_test)
    
    # Save model if requested
    if save_model:
        model_path = f"sample_{task_type}_model.joblib"
        joblib.dump(model, model_path)
        print(f"✅ Model saved to: {model_path}")
    
    print(f"✅ Sample {task_type} data and model created")
    print(f"   📊 Training samples: {len(X_train)}")
    print(f"   📊 Test samples: {len(X_test)}")
    print(f"   📊 Features: {len(feature_names)}")
    
    return model, X_train, X_test, y_train, y_test, fitted_values, predictions

def demonstrate_model_explainer(task_type='classification'):
    """
    Demonstrate the ModelExplainer class functionality.
    
    Parameters:
    -----------
    task_type : str
        'classification' or 'regression'
    """
    print(f"\n{'='*80}")
    print(f"DEMONSTRATING MODEL EXPLAINER - {task_type.upper()}")
    print(f"{'='*80}")
    
    # Create sample data and model
    model, X_train, X_test, y_train, y_test, fitted_values, predictions = create_sample_data_and_model(
        task_type=task_type, save_model=True
    )
    
    # Example 1: Initialize with pre-loaded model
    print(f"\n📋 Example 1: Using pre-loaded model")
    explainer = ModelExplainer(
        model=model,
        sample_fraction=0.5,  # Use 50% of data for memory efficiency
        verbose=True
    )
    
    # Prepare data
    explainer.prepare_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fitted_values=fitted_values,
        predictions=predictions
    )
    
    # Setup explainers
    print(f"\n🔧 Setting up explainers...")
    shap_success = explainer.setup_shap_explainer()
    lime_success = explainer.setup_lime_explainer()
    
    if not shap_success and not lime_success:
        print("❌ Neither SHAP nor LIME could be initialized. Please install them:")
        print("   pip install shap lime")
        return
    
    # Model performance summary
    summary = explainer.model_performance_summary()
    
    # Explain specific instances
    if shap_success:
        print(f"\n🔍 SHAP explanation for instance 0...")
        shap_explanation = explainer.explain_instance_shap(instance_index=0)
        
        if shap_explanation:
            print(f"   📊 Top 3 SHAP features: {shap_explanation['top_features'][:3]}")
    
    if lime_success:
        print(f"\n🔍 LIME explanation for instance 0...")
        lime_explanation = explainer.explain_instance_lime(instance_index=0, num_features=10)
        
        if lime_explanation:
            print(f"   📊 Top 3 LIME features: {lime_explanation['top_features'][:3]}")
    
    # Compare explanations if both are available
    if shap_success and lime_success:
        print(f"\n🔄 Comparing SHAP and LIME explanations...")
        comparison = explainer.compare_explanations(instance_index=0, num_features=10)
        
        if comparison:
            print(f"   📊 Agreement score: {comparison['agreement_score']:.3f}")
            if comparison['correlation'] is not None:
                print(f"   📊 Importance correlation: {comparison['correlation']:.3f}")
    
    # Global feature importance
    if shap_success:
        print(f"\n🌍 Calculating global feature importance...")
        global_importance = explainer.global_feature_importance()
        
        if global_importance:
            print(f"   📊 Top 5 globally important features:")
            for i, (feature, importance) in enumerate(global_importance['top_10_features'][:5]):
                print(f"      {i+1}. {feature}: {importance:.4f}")
    
    # Example 2: Initialize with model file
    print(f"\n📋 Example 2: Loading model from file")
    model_path = f"sample_{task_type}_model.joblib"
    
    try:
        explainer_from_file = ModelExplainer(
            model_path=model_path,
            sample_fraction=0.3,  # Use 30% of data
            verbose=True
        )
        
        # Prepare data
        explainer_from_file.prepare_data(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            fitted_values=fitted_values,
            predictions=predictions
        )
        
        print(f"✅ Successfully loaded model from file and prepared data")
        
        # Quick explanation
        if explainer_from_file.setup_shap_explainer():
            explanation = explainer_from_file.explain_instance_shap(instance_index=5, plot=False)
            if explanation:
                print(f"   📊 Instance 5 top feature: {explanation['top_features'][0]}")
        
    except Exception as e:
        print(f"❌ Error loading model from file: {str(e)}")
    
    # Clean up
    try:
        os.remove(model_path)
        print(f"🧹 Cleaned up: {model_path}")
    except:
        pass
    
    print(f"\n✅ {task_type.title()} demonstration completed!")

def main():
    """
    Main function to run demonstrations.
    """
    print("MODEL EXPLAINER DEMONSTRATION")
    print("="*80)
    
    # Check if required packages are available
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
    
    try:
        import lime
        lime_available = True
    except ImportError:
        lime_available = False
    
    if not shap_available and not lime_available:
        print("❌ Neither SHAP nor LIME is installed.")
        print("Please install them using:")
        print("   pip install shap lime")
        return
    elif not shap_available:
        print("⚠️  SHAP not installed. Only LIME demonstrations will work.")
        print("Install SHAP with: pip install shap")
    elif not lime_available:
        print("⚠️  LIME not installed. Only SHAP demonstrations will work.")
        print("Install LIME with: pip install lime")
    
    # Run demonstrations
    try:
        # Classification example
        demonstrate_model_explainer('classification')
        
        # Regression example
        demonstrate_model_explainer('regression')
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 All demonstrations completed!")

if __name__ == "__main__":
    main() 