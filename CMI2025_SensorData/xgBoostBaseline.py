import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Import evaluation API
import kaggle_evaluation.cmi_inference_server
pd.set_option('display.max_rows', 300)
random_seed = 42
train = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
train = train.drop(["sequence_type","orientation","behavior","phase"])
data = train.join(train_demo,on="subject",how="left")
def feature_engineering(data:pl.DataFrame):
    demographic_cols = [
    "adult_child", "age", "sex", "handedness",
    "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
    ]
    target_col = "gesture"
    
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in demographic_cols + [target_col, "sequence_id", "row_id","sequence_counter","subject"]
    ]
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            # pl.col(c).mode().list.first().alias(f"{c}_mode"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
        ]
    
    # first() for demographics and target
    agg_exprs += [
        pl.col(c).first().alias(c) for c in demographic_cols + [target_col]
    ]
    
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data
cleaned_data = feature_engineering(data)
target_col = "gesture"
pdf = cleaned_data.to_pandas()  # keeps nullable dtypes

le = LabelEncoder()
y = le.fit_transform(pdf[target_col])
X = pdf.drop(columns=[target_col, "sequence_id"])         # drop id + label
def competition_metric(y_true, y_pred, le_instance, all_original_gestures):
    """
    Competition metric calculation
    """
    bfrb_gestures = [g for g in all_original_gestures if g in le_instance.classes_]
    
    # Binary F1: All are Target in this filtered dataset
    y_true_binary = np.ones_like(y_true, dtype=int)
    y_pred_binary = np.ones_like(y_pred, dtype=int)
    binary_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', pos_label=1, zero_division=0)
    
    # Macro F1: specific gesture classification
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    final_score = (binary_f1 + macro_f1) / 2
    return final_score, binary_f1, macro_f1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

all_original_gestures_in_train = pdf['gesture'].unique()

callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=100)]

# LightGBM model with cross-validation
print("\nTraining LightGBM models with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold + 1}/5")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # LightGBM model with GPU acceleration
    model = lgb.LGBMClassifier(
        objective='multiclass',
        n_estimators= 1000,
        learning_rate= 0.08,
        max_depth= 15,
        reg_alpha= 0.8,
        lambda_l2= 4.0,  
        num_leaves=31, 
        min_child_samples= 32,
        colsample_bytree= 0.3,
        subsample= 0.5,
        subsample_freq=0,
        cat_smooth=30.0,
        is_unbalance=True,
        max_bin=127,
        verbose=-1,  
        metric='multi_logloss',   
        device='gpu',  
    )
    
    # Train model with verbose output
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],  
        eval_metric='multi_logloss',  
        callbacks=callbacks
    )
    
    # Predict
    y_pred_fold = model.predict(X_val_fold)
    
    # Calculate score
    score, binary_f1, macro_f1 = competition_metric(
        y_val_fold, y_pred_fold, le, all_original_gestures_in_train
    )
    
    cv_scores.append(score)
    models.append(model)
    
    print(f"Fold {fold + 1} - Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f})")

print(f"\nCross-validation results:")
print(f"Mean CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print(f"Individual fold scores: {cv_scores}")

# Train final model on all data with GPU acceleration
print("\nTraining final model on all training data...")
import kaggle_evaluation.cmi_inference_server
import os

def feature_engineering_inf(data:pl.DataFrame):
    demographic_cols = [
    "adult_child", "age", "sex", "handedness",
    "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"
    ]
    
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in demographic_cols + ["sequence_id", "row_id","sequence_counter","subject"]
    ]
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            # pl.col(c).mode().list.first().alias(f"{c}_mode"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
        ]
    
    # first() for demographics and target
    agg_exprs += [
        pl.col(c).first().alias(c) for c in demographic_cols
    ]
    
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    data = sequence.join(demographics,on="subject",how="left")
    cleaned_data = feature_engineering_inf(data)
    pdf = cleaned_data.to_pandas().drop(columns=["sequence_id"])
    # Predict using ensemble of CV models
    predictions = []
    for model in models:
        pred = model.predict(pdf)
        # Ensure we get a scalar value
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        predictions.append(int(pred))
    
    # Use majority vote or most confident prediction
    predicted_label_id = max(set(predictions), key=predictions.count)
    
    # Convert back to gesture string
    predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
    
    return predicted_gesture_str
import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )