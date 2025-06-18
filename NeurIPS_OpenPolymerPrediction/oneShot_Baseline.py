# Load datasets
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Filter out rows where FFV is not missing
ffv_data = train_df[train_df['FFV'].notnull()].copy()

# Text featurization from SMILES
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
X = tfidf.fit_transform(ffv_data['SMILES'])
y = ffv_data['FFV'].values

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"FFV RMSE: {rmse:.4f}")
test_tfidf = tfidf.transform(test_df['SMILES'])
ffv_preds = model.predict(test_tfidf)
# Create a new DataFrame for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tg': 0,           # placeholder
    'FFV': ffv_preds,  # your predicted values
    'Tc': 0,
    'Density': 0,
    'Rg': 0
})

# Save to CSV in the correct location
submission.to_csv('/kaggle/working/submission.csv', index=False)