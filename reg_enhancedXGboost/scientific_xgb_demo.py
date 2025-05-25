"""
ScientificXGBRegressor Demonstration
===================================

This script demonstrates the advanced features of the ScientificXGBRegressor
using the calorie prediction dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our ScientificXGBRegressor
import xgboost
ScientificXGBRegressor = xgboost.ScientificXGBRegressor
create_scientific_xgb_regressor = xgboost.create_scientific_xgb_regressor

def load_sample_data():
    """
    Create sample calorie prediction data for demonstration.
    If you have the actual dataset, you can replace this function.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic calorie prediction features
    age = np.random.normal(35, 12, n_samples)
    height = np.random.normal(170, 10, n_samples)  # cm
    weight = np.random.normal(70, 15, n_samples)   # kg
    duration = np.random.normal(30, 10, n_samples)  # minutes
    heart_rate = np.random.normal(120, 20, n_samples)
    body_temp = np.random.normal(37, 0.5, n_samples)
    sex = np.random.choice([0, 1], n_samples)  # 0=female, 1=male
    
    # Create realistic calorie burn formula
    # Base metabolic rate calculation
    bmr = np.where(sex == 1, 
                   88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age),
                   447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age))
    
    # Calorie burn based on duration, heart rate, and BMR
    calories = (bmr * duration / 1440) * (heart_rate / 70) * np.random.normal(1.0, 0.1, n_samples)
    calories = np.maximum(calories, 50)  # Minimum 50 calories
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'Sex': sex,
        'Calories': calories
    })
    
    return df

def demonstrate_scientific_xgb():
    """
    Comprehensive demonstration of ScientificXGBRegressor features.
    """
    print("🧪 ScientificXGBRegressor Comprehensive Demonstration")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    df = load_sample_data()
    
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f} calories")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 2. Create ScientificXGBRegressor
    print("\n🔬 Step 2: Creating ScientificXGBRegressor...")
    model = create_scientific_xgb_regressor(
        cv_folds=5,
        auto_tune=True,
        verbose=True,
        n_estimators=500,
        learning_rate=0.1
    )
    
    # 3. Automated parameterization
    print("\n⚙️ Step 3: Automated Parameterization...")
    params = model.automated_parameterization(X_train.values, y_train.values)
    print(f"Automated parameters applied: {len(params)} parameters optimized")
    
    # 4. Fit the model
    print("\n🔄 Step 4: Fitting the model...")
    model.fit(X_train, y_train)
    
    # 5. Generate predictions
    print("\n📈 Step 5: Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f} calories")
    print(f"Test MAE: {test_mae:.2f} calories")
    
    # 6. Diagnostic plots
    print("\n📊 Step 6: Generating diagnostic plots...")
    try:
        fig = model.diagnostic_plots(X_train.values, y_train.values, figsize=(16, 12))
        plt.tight_layout()
        plt.savefig('scientific_xgb_diagnostics.png', dpi=300, bbox_inches='tight')
        print("✅ Diagnostic plots saved as 'scientific_xgb_diagnostics.png'")
        # plt.show()  # Uncomment to display plots
    except Exception as e:
        print(f"⚠️ Could not generate diagnostic plots: {e}")
    
    # 7. Elbow tuning demonstration
    print("\n🔧 Step 7: Elbow method hyperparameter tuning...")
    try:
        elbow_results = model.elbow_tune(
            X_train.values, y_train.values,
            param_name='n_estimators',
            param_range=[100, 200, 300, 500, 750, 1000],
            cv_folds=3
        )
        print(f"Optimal n_estimators: {elbow_results['elbow_value']}")
        print(f"Elbow CV score: {elbow_results['elbow_score']:.4f}")
    except Exception as e:
        print(f"⚠️ Elbow tuning failed: {e}")
    
    # 8. Nested cross-validation
    print("\n🔄 Step 8: Nested Cross-Validation...")
    try:
        cv_results = model.nested_cross_validate(
            X_train.values, y_train.values,
            inner_cv=3, outer_cv=3,  # Reduced for demo speed
            param_grid={
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'n_estimators': [200, 500]
            }
        )
        
        print("Nested CV Results:")
        for metric, mean_val in cv_results['mean_scores'].items():
            std_val = cv_results['std_scores'][metric]
            print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    except Exception as e:
        print(f"⚠️ Nested CV failed: {e}")
    
    # 9. Incremental learning demonstration
    print("\n📚 Step 9: Incremental Learning...")
    try:
        # Create some "new" data
        new_data = load_sample_data().sample(100, random_state=123)
        X_new = new_data.drop('Calories', axis=1)
        y_new = new_data['Calories']
        
        incremental_results = model.incremental_learn(
            X_new.values, y_new.values,
            n_new_estimators=100
        )
        print(f"Incremental learning: {incremental_results['status']}")
        print(f"New estimators: {incremental_results.get('original_n_estimators', 0)} → {incremental_results.get('new_n_estimators', 0)}")
    except Exception as e:
        print(f"⚠️ Incremental learning failed: {e}")
    
    # 10. Save complete model package
    print("\n💾 Step 10: Saving model package...")
    try:
        package_path = model.save_model_package(
            save_dir="./scientific_xgb_calorie_model",
            include_diagnostics=True,
            include_data=True,
            X_sample=X_train.values[:100],
            y_sample=y_train.values[:100]
        )
        print(f"✅ Complete model package saved to: {package_path}")
    except Exception as e:
        print(f"⚠️ Model package saving failed: {e}")
    
    # 11. Feature importance analysis
    print("\n🎯 Step 11: Feature Importance Analysis...")
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 Most Important Features:")
        for idx, row in importance_df.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 12. Model loading demonstration
    print("\n📂 Step 12: Model Loading Demonstration...")
    try:
        # Load the saved model
        loaded_model = ScientificXGBRegressor.load_model_package("./scientific_xgb_calorie_model")
        
        # Test predictions with loaded model
        loaded_predictions = loaded_model.predict(X_test.values[:10])
        original_predictions = model.predict(X_test.values[:10])
        
        # Check if predictions match
        predictions_match = np.allclose(loaded_predictions, original_predictions)
        print(f"✅ Model loaded successfully. Predictions match: {predictions_match}")
        
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
    
    print("\n🎉 ScientificXGBRegressor demonstration completed!")
    print("=" * 60)
    
    return model, X_test, y_test

def quick_demo():
    """
    Quick demonstration for rapid testing.
    """
    print("🚀 Quick ScientificXGBRegressor Demo")
    print("-" * 40)
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(500)
    
    # Create and fit model
    model = create_scientific_xgb_regressor(
        cv_folds=3,
        auto_tune=True,
        verbose=True
    )
    
    # Automated parameterization and fit
    model.automated_parameterization(X, y)
    model.fit(X, y)
    
    # Quick evaluation
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"✅ Quick demo R²: {r2:.4f}")
    
    return model

if __name__ == "__main__":
    # Choose demo type
    demo_type = input("Choose demo type (full/quick) [default: quick]: ").strip().lower()
    
    if demo_type == "full":
        # Run comprehensive demonstration
        model, X_test, y_test = demonstrate_scientific_xgb()
    else:
        # Run quick demonstration
        model = quick_demo()
    
    print("\n📚 Next steps:")
    print("- Explore the saved model package")
    print("- Check the diagnostic plots")
    print("- Try different hyperparameter optimization strategies")
    print("- Experiment with incremental learning") 