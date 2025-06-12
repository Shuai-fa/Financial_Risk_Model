# file: src/train.py

import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from data_processing import create_features 
import config

def train_model(df):
    """Trains the XGBoost model and saves it."""
    if df is None:
        print("ERROR: Input DataFrame is None. Skipping training.")
        return

    print("--> Preparing data for model training...")
    y = df[config.TARGET_COL]
    
    # Define columns to drop before training
    # These are either targets, identifiers, or time components
    cols_to_drop = [
        config.TARGET_COL, config.COMPANY_ID_COL, 'year', 'month', 
        'default_year', 'default_month', 'report_period'
    ]
    X = df.drop(columns=cols_to_drop, errors='ignore')

    print(f"--> Creating models directory if not exists: '{config.MODELS_DIR}'")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save feature names for consistent use in evaluation
    with open(config.TRAINING_COLUMNS_PATH, 'w') as f:
        for col in X.columns: f.write(f"{col}\n")
    print(f"    Feature names saved to {config.TRAINING_COLUMNS_PATH}")

    # Calculate scale_pos_weight for handling class imbalance
    if (y == 1).sum() > 0:
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    else:
        scale_pos_weight = 1
    print(f"    Class imbalance ratio (scale_pos_weight) set to: {scale_pos_weight:.2f}")

    print("--> Training XGBoost model...")
    xgb_model = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', 
        scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False
    )
    xgb_model.fit(X, y)
    
    joblib.dump(xgb_model, config.XGB_MODEL_SAVE_PATH)
    print(f"--> XGBoost model successfully trained and saved to: {config.XGB_MODEL_SAVE_PATH}\n")

def main():
    """Main function to run the training pipeline."""
    print("\n======== STARTING TRAINING PIPELINE ========")
    try:
        # Assumes the CSV has a proper header
        dataframe = pd.read_csv(config.SAMPLE_DATA_PATH) 
        print(f"Successfully loaded data from {config.SAMPLE_DATA_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {config.SAMPLE_DATA_PATH}. Please run 1_create_sample_data.py first.")
        return
    
    featured_dataframe = create_features(dataframe)
    train_model(featured_dataframe)
    print("======== TRAINING PIPELINE FINISHED ========")

if __name__ == '__main__':
    main()