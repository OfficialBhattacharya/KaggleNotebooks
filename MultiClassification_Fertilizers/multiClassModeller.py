import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
SEED = 42
N_SPLITS = 5
EARLY_STOPPING_PATIENCE = 29
MAX_EPOCHS = 30 
BATCH_SIZE = 4096

# --- Hyperparameter Tuning ---
MODEL_PARAMS = {
    'embed_dim': 128,
    'num_heads': 8,
    'num_transformer_layers': 5,
    'ff_hidden_dim': 512,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
}
# --- Feature Engineering ---
print("\nPerforming Feature Engineering...")

original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Function to apply feature engineering for base features
def apply_base_feature_engineering(df):
    df_copy = df.copy()
    df_copy['Temp_Humidity_Interaction'] = df_copy['Temparature'] * df_copy['Humidity']
    df_copy['N_P_Ratio'] = df_copy['Nitrogen'] / (df_copy['Phosphorous'].replace(0, 1e-6))
    df_copy['K_P_Ratio'] = df_copy['Potassium'] / (df_copy['Phosphorous'].replace(0, 1e-6))
    df_copy['Soil_Crop_Combination'] = df_copy['Soil Type'].astype(str) + '_' + df_copy['Crop Type'].astype(str)
    df_copy['Total_Nutrients'] = df_copy['Nitrogen'] + df_copy['Potassium'] + df_copy['Phosphorous']

    df_copy['Temp_Nitrogen_Ratio'] = df_copy['Temparature'] / (df_copy['Nitrogen'].replace(0, 1e-6))
    df_copy['Humidity_Phosphorous_Ratio'] = df_copy['Humidity'] / (df_copy['Phosphorous'].replace(0, 1e-6))
    df_copy['Moisture_Potassium_Ratio'] = df_copy['Moisture'] / (df_copy['Potassium'].replace(0, 1e-6))
    
    df_copy['N_P_K_Ratio_Combined'] = df_copy['Nitrogen'] / (df_copy['Potassium'].replace(0, 1e-6) + df_copy['Phosphorous'].replace(0, 1e-6) + 1e-6)

    df_copy['Nutrient_Mean'] = df_copy[['Nitrogen', 'Potassium', 'Phosphorous']].mean(axis=1)
    df_copy['Nutrient_Imbalance_N'] = (df_copy['Nitrogen'] - df_copy['Nutrient_Mean']).abs()
    df_copy['Nutrient_Imbalance_P'] = (df_copy['Phosphorous'] - df_copy['Nutrient_Mean']).abs()
    df_copy['Nutrient_Imbalance_K'] = (df_copy['Potassium'] - df_copy['Nutrient_Mean']).abs()
    
    df_copy = df_copy.drop('Nutrient_Mean', axis=1)

    # Binning numerical features (as strings for categorical handling)
    for col in original_numerical_cols:
        df_copy[f'{col}_Binned'] = df_copy[col].astype(str)
    return df_copy

# Apply base FE to all datasets
X_original_temp = apply_base_feature_engineering(X_original)
X_additional_temp = apply_base_feature_engineering(X_additional)
X_test_temp = apply_base_feature_engineering(X_test)

print("Feature Engineering complete.")

# --- Define feature lists after base FE ---
new_numerical_features = [
    'Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio', 'Total_Nutrients',
    'Temp_Nitrogen_Ratio', 'Humidity_Phosphorous_Ratio', 'Moisture_Potassium_Ratio',
    'N_P_K_Ratio_Combined', 'Nutrient_Imbalance_N', 'Nutrient_Imbalance_P', 'Nutrient_Imbalance_K'
]

base_categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination']
base_categorical_features.extend([f'{col}_Binned' for col in original_numerical_cols])


# Polynomial Features (fit on original numerical columns, transform all)
poly = PolynomialFeatures(degree=2, include_bias=False)

# Fit on original numerical features from original_numerical_cols
X_original_poly_transformed = poly.fit_transform(X_original_temp[original_numerical_cols])
X_additional_poly_transformed = poly.transform(X_additional_temp[original_numerical_cols])
X_test_poly_transformed = poly.transform(X_test_temp[original_numerical_cols])

poly_feature_names = poly.get_feature_names_out(original_numerical_cols)

# Create separate DataFrames for polynomial features
df_poly_original = pd.DataFrame(X_original_poly_transformed, columns=poly_feature_names, index=X_original_temp.index)
df_poly_additional = pd.DataFrame(X_additional_poly_transformed, columns=poly_feature_names, index=X_additional_temp.index)
df_poly_test = pd.DataFrame(X_test_poly_transformed, columns=poly_feature_names, index=X_test_temp.index)


# --- Constructing Final DataFrames with correct column sets ---
# Final numerical features will be the `new_numerical_features` + all `poly_feature_names`
final_numerical_features = sorted(list(set(new_numerical_features) | set(poly_feature_names)))

# Final categorical features are simply the base_categorical_features
final_categorical_features = sorted(list(set(base_categorical_features)))

# Ensure no overlap
assert set(final_numerical_features).isdisjoint(set(final_categorical_features)), \
    "FATAL: Overlap detected between numerical and categorical features!"


# Build the final DataFrames by concatenating the exact numerical and categorical parts
X_original_fe = pd.concat([
    X_original_temp[new_numerical_features],
    df_poly_original,
    X_original_temp[final_categorical_features]
], axis=1)[final_numerical_features + final_categorical_features].copy()

X_additional_fe = pd.concat([
    X_additional_temp[new_numerical_features],
    df_poly_additional,
    X_additional_temp[final_categorical_features]
], axis=1)[final_numerical_features + final_categorical_features].copy()

X_test_fe = pd.concat([
    X_test_temp[new_numerical_features],
    df_poly_test,
    X_test_temp[final_categorical_features]
], axis=1)[final_numerical_features + final_categorical_features].copy()


print("\n--- Feature List Verification (POST-CONSTRUCTION) ---")
print(f"Number of final numerical features: {len(final_numerical_features)}")
print(f"Number of final categorical features: {len(final_categorical_features)}")
print(f"Total features (numerical + categorical): {len(final_numerical_features) + len(final_categorical_features)}")
print(f"X_original_fe.shape after final construction: {X_original_fe.shape}")
print(f"X_original_fe numerical columns (derived): {X_original_fe[final_numerical_features].shape[1]}")
print(f"X_original_fe categorical columns (derived): {X_original_fe[final_categorical_features].shape[1]}")
assert len(final_numerical_features) == X_original_fe[final_numerical_features].shape[1], "Numerical feature count mismatch!"
assert len(final_categorical_features) == X_original_fe[final_categorical_features].shape[1], "Categorical feature count mismatch!"
assert X_original_fe.shape[1] == len(final_numerical_features) + len(final_categorical_features), "Total column count mismatch!"
print("--- End Feature List Verification (POST-CONSTRUCTION) ---\n")


# --- Categorical & Numerical Preprocessing for TabTransformer ---
print("\nPreprocessing features for TabTransformer...")

# Ordinal Encoding for Categorical Features
categorical_dims = {} # Store num unique categories for embedding layers
for col in final_categorical_features:
    combined_series = pd.concat([
        X_original_fe[col].astype(str),
        X_additional_fe[col].astype(str),
        X_test_fe[col].astype(str)
    ], axis=0)
    
    # Get all unique categories from the combined data
    all_unique_cats_for_col = sorted(combined_series.astype('category').cat.categories.tolist())
    
    oenc = OrdinalEncoder(dtype=np.int64, categories=[all_unique_cats_for_col]) # Use int64 for embeddings, specify categories
    
    # Transform all dataframes. This will give 0-based indices.
    X_original_fe[col] = oenc.fit_transform(X_original_fe[col].astype(str).values.reshape(-1, 1)).flatten()
    X_additional_fe[col] = oenc.transform(X_additional_fe[col].astype(str).values.reshape(-1, 1)).flatten()
    X_test_fe[col] = oenc.transform(X_test_fe[col].astype(str).values.reshape(-1, 1)).flatten()
    
    X_original_fe[col] = X_original_fe[col] + 1
    X_additional_fe[col] = X_additional_fe[col] + 1
    X_test_fe[col] = X_test_fe[col] + 1

    categorical_dims[col] = len(all_unique_cats_for_col) + 1

# Numerical Scaling
scaler = StandardScaler()
# Fit scaler on combined original and additional training data
scaler.fit(pd.concat([X_original_fe[final_numerical_features], X_additional_fe[final_numerical_features]], axis=0))

X_original_fe[final_numerical_features] = scaler.transform(X_original_fe[final_numerical_features])
X_additional_fe[final_numerical_features] = scaler.transform(X_additional_fe[final_numerical_features])
X_test_fe[final_numerical_features] = scaler.transform(X_test_fe[final_numerical_features])

print("Feature preprocessing complete.")
print("Processed X_original_fe shape:", X_original_fe.shape)
print("Processed X_additional_fe shape:", X_additional_fe.shape)
print("Processed X_test_fe shape:", X_test_fe.shape)
print(f"Number of final numerical features: {len(final_numerical_features)}")
print(f"Number of final categorical features: {len(final_categorical_features)}")


# --- Target Encoding ---
label_encoder = LabelEncoder()
y_encoded_all_train = label_encoder.fit_transform(pd.concat([y_original, y_additional]).values)
y_original_encoded = label_encoder.transform(y_original.values)
y_additional_encoded = label_encoder.transform(y_additional.values) 

fertilizer_classes = label_encoder.classes_
NUM_CLASSES = len(fertilizer_classes)
print("\nTarget encoding complete. Fertilizer classes (order):", fertilizer_classes)
print(f"Number of classes: {NUM_CLASSES}")

# --- Check and Apply Class Weights for Imbalance ---
print("\nChecking target class distribution...")
class_counts = pd.Series(y_encoded_all_train).value_counts(normalize=True).sort_index()
print("Normalized class frequencies (encoded):")
for i, count in enumerate(class_counts):
    print(f"  Class {label_encoder.inverse_transform([i])[0]} ({i}): {count:.4f}")


min_freq = class_counts.min()
max_freq = class_counts.max()

if max_freq / min_freq > 1.5:
    print("\nClass imbalance detected. Calculating balanced class weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded_all_train), y=y_encoded_all_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("Class weights (encoded order):", class_weights)
else:
    print("\nClass distribution appears relatively balanced. No class weights applied.")
    class_weights_tensor = None # No weights

# --- MAP@3 Calculation Functions ---
def apk(actual, predicted, k=3):
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# --- TabTransformer Model Definition (PyTorch) ---
class TabTransformer(nn.Module):
    def __init__(self, numerical_features_count, categorical_dims_dict, 
                 embed_dim=32, num_heads=4, num_transformer_layers=3, 
                 ff_hidden_dim=128, dropout_rate=0.1, num_classes=7):
        super().__init__()
        
        self.numerical_dim = numerical_features_count
        self.categorical_features_count = len(categorical_dims_dict)
        
        # Categorical Embeddings
        # Ensure order is consistent by iterating through sorted keys when creating embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim) for col, num_categories in sorted(categorical_dims_dict.items())
        ])
        
        # Linear layer for numerical features
        self.numerical_proj = nn.Linear(self.numerical_dim, embed_dim)
        
        # Transformer Encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_hidden_dim, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        self.mlp_input_dim = embed_dim * (1 + self.categorical_features_count) 
        
        # --- Deeper MLP Head with Batch Normalization ---
        self.mlp_head = nn.Sequential(
            nn.Linear(self.mlp_input_dim, ff_hidden_dim),
            nn.BatchNorm1d(ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.BatchNorm1d(ff_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim // 2, num_classes)
        )
        
    def forward(self, x_numerical, x_categorical):
 
        cat_embeddings = [emb(x_categorical[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeddings = torch.stack(cat_embeddings, dim=1)
        
        # Process numerical features
        num_proj = self.numerical_proj(x_numerical).unsqueeze(1)
        
        # Concatenate for transformer input
        transformer_input = torch.cat([num_proj, cat_embeddings], dim=1)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Flatten for MLP
        flattened_output = transformer_output.view(transformer_output.size(0), -1) 
        
        # Final classification
        logits = self.mlp_head(flattened_output)
        return logits


# --- Training and Evaluation Function ---
def train_and_evaluate_model(model, train_loader, val_loader, optimizer, criterion, 
                             epochs, device, early_stopping_patience, fertilizer_classes, 
                             y_val_original_labels):
    
    best_map3 = -1.0
    epochs_no_improve = 0
    
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.")

    # --- Advanced Learning Rate Scheduler ---
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, verbose=True)
    
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for num_batch, cat_batch, labels_batch in train_loader:
            num_batch, cat_batch, labels_batch = num_batch.to(device), cat_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            
            # --- Mixed Precision Training ---
            with autocast():
                outputs = model(num_batch, cat_batch)
                loss = criterion(outputs, labels_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_preds_proba = []
        val_true_labels_encoded = []
        with torch.no_grad():
            for num_batch, cat_batch, labels_batch in val_loader:
                num_batch, cat_batch, labels_batch = num_batch.to(device), cat_batch.to(device), labels_batch.to(device)
                
                with autocast():
                    outputs = model(num_batch, cat_batch)
                
                val_preds_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                val_true_labels_encoded.extend(labels_batch.cpu().numpy())
        
        val_preds_proba = np.array(val_preds_proba)
        
        val_ranked_labels = []
        for i in range(len(val_preds_proba)):
            top_3_indices = np.argsort(val_preds_proba[i])[-3:][::-1]
            val_ranked_labels.append([fertilizer_classes[idx] for idx in top_3_indices])

        current_map3 = mapk([[label] for label in y_val_original_labels], val_ranked_labels, k=3)
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val MAP@3: {current_map3:.5f}")

        scheduler.step()

        if current_map3 > best_map3:
            best_map3 = current_map3
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} as validation MAP@3 did not improve for {early_stopping_patience} epochs.")
                break
    
    # Load best model weights before returning
    model.load_state_dict(torch.load('best_model.pth'))
    return best_map3


# --- Main Cross-Validation Loop ---
oof_preds_total = np.zeros((len(X_original_fe), NUM_CLASSES))
test_preds_list_total = []
fold_map3_scores_list = []

print(f"\nStarting {N_SPLITS}-Fold Cross-Validation for TabTransformer...")

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx_original, val_idx_original) in enumerate(skf.split(X_original_fe, y_original_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    # Use the final_numerical_features and final_categorical_features to select columns
    X_train_fold_original_num = X_original_fe.iloc[train_idx_original][final_numerical_features].values 
    X_train_fold_original_cat = X_original_fe.iloc[train_idx_original][final_categorical_features].values
    y_train_fold_original = y_original_encoded[train_idx_original]

    X_val_fold_num = X_original_fe.iloc[val_idx_original][final_numerical_features].values
    X_val_fold_cat = X_original_fe.iloc[val_idx_original][final_categorical_features].values
    y_val_fold_original_labels = y_original.iloc[val_idx_original].values 
    y_val_fold_encoded = y_original_encoded[val_idx_original]

    X_train_final_num = np.concatenate([X_train_fold_original_num, X_additional_fe[final_numerical_features].values])
    X_train_final_cat = np.concatenate([X_train_fold_original_cat, X_additional_fe[final_categorical_features].values])
    y_train_final = np.concatenate([y_train_fold_original, y_additional_encoded])

    print(f"Fold {fold+1} training data shape (numerical part): {X_train_final_num.shape}")
    print(f"Fold {fold+1} training data shape (categorical part): {X_train_final_cat.shape}")
    print(f"Fold {fold+1} validation data shape (numerical part): {X_val_fold_num.shape}")
    print(f"Fold {fold+1} validation data shape (categorical part): {X_val_fold_cat.shape}")

    # Convert to PyTorch Tensors
    train_num_tensor = torch.tensor(X_train_final_num, dtype=torch.float32)
    train_cat_tensor = torch.tensor(X_train_final_cat, dtype=torch.long)
    train_labels_tensor = torch.tensor(y_train_final, dtype=torch.long)

    val_num_tensor = torch.tensor(X_val_fold_num, dtype=torch.float32)
    val_cat_tensor = torch.tensor(X_val_fold_cat, dtype=torch.long)
    val_labels_tensor = torch.tensor(y_val_fold_encoded, dtype=torch.long)
    
    test_num_tensor = torch.tensor(X_test_fe[final_numerical_features].values, dtype=torch.float32) 
    test_cat_tensor = torch.tensor(X_test_fe[final_categorical_features].values, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(train_num_tensor, train_cat_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=os.cpu_count())

    val_dataset = TensorDataset(val_num_tensor, val_cat_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=os.cpu_count())

    test_dataset = TensorDataset(test_num_tensor, test_cat_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=os.cpu_count())

    # Initialize model, optimizer, loss
    model = TabTransformer(
        numerical_features_count=len(final_numerical_features),
        categorical_dims_dict=categorical_dims,
        embed_dim=MODEL_PARAMS['embed_dim'],
        num_heads=MODEL_PARAMS['num_heads'],
        num_transformer_layers=MODEL_PARAMS['num_transformer_layers'],
        ff_hidden_dim=MODEL_PARAMS['ff_hidden_dim'],
        dropout_rate=MODEL_PARAMS['dropout_rate'],
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=MODEL_PARAMS['learning_rate'], weight_decay=MODEL_PARAMS['weight_decay'])
    
    # --- Use class_weights_tensor if calculated, otherwise no weights ---
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Train and evaluate
    fold_best_map3 = train_and_evaluate_model(
        model, train_loader, val_loader, optimizer, criterion, 
        MAX_EPOCHS, DEVICE, EARLY_STOPPING_PATIENCE, fertilizer_classes, 
        y_val_fold_original_labels
    )
    fold_map3_scores_list.append(fold_best_map3)

    model.eval()
    fold_oof_preds_proba = []
    with torch.no_grad():
        for num_batch, cat_batch, _ in val_loader:
            num_batch, cat_batch = num_batch.to(DEVICE), cat_batch.to(DEVICE)
            outputs = model(num_batch, cat_batch)
            fold_oof_preds_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    oof_preds_total[val_idx_original] = np.array(fold_oof_preds_proba)

    fold_test_preds_proba = []
    with torch.no_grad():
        for num_batch, cat_batch in test_loader:
            num_batch, cat_batch = num_batch.to(DEVICE), cat_batch.to(DEVICE)
            outputs = model(num_batch, cat_batch)
            fold_test_preds_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    test_preds_list_total.append(np.array(fold_test_preds_proba))
    
    del model, train_loader, val_loader, test_loader, train_num_tensor, train_cat_tensor, train_labels_tensor, val_num_tensor, val_cat_tensor, val_labels_tensor, test_num_tensor, test_cat_tensor
    gc.collect()
    torch.cuda.empty_cache()


print("\nCross-validation complete.")
print("\nIndividual Fold MAP@3 scores:", [f"{s:.5f}" for s in fold_map3_scores_list])
print(f"Average Fold MAP@3: {np.mean(fold_map3_scores_list):.5f}")


# --- Overall OOF MAP@3 Calculation (on OOF predictions) ---
y_true_labels_for_map = [[label] for label in y_original.values]

oof_ranked_labels = []
for i in range(len(oof_preds_total)):
    top_3_indices = np.argsort(oof_preds_total[i])[-3:][::-1]
    oof_ranked_labels.append([fertilizer_classes[idx] for idx in top_3_indices])

print("\nCalculating Overall OOF MAP@3 score...")
oof_map3_score = mapk(y_true_labels_for_map, oof_ranked_labels, k=3)
print(f"Overall OOF MAP@3: {oof_map3_score:.5f}")


# --- Generate Submission File ---
final_test_preds = np.mean(test_preds_list_total, axis=0)

test_ranked_labels = []
for i in range(len(final_test_preds)):
    top_3_indices = np.argsort(final_test_preds[i])[-3:][::-1]
    top_3_fertilizers = [fertilizer_classes[idx] for idx in top_3_indices]
    test_ranked_labels.append(" ".join(top_3_fertilizers))

submission_df = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': test_ranked_labels
})

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())