# file: src/evaluate.py

import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data_processing import create_features 
import config

def main_evaluate():
    print("\n======== STARTING EVALUATION PIPELINE ========")
    try:
        with open(config.TRAINING_COLUMNS_PATH, 'r') as f:
            training_columns = [line.strip() for line in f]
        model = joblib.load(config.XGB_MODEL_SAVE_PATH)
        
        # Load the same data used for training/demo
        df_eval = pd.read_csv(config.SAMPLE_DATA_PATH)
        print(f"Successfully loaded data for evaluation from {config.SAMPLE_DATA_PATH}")
        
    except FileNotFoundError:
        print("ERROR: Model or data file not found. Please run train.py first.")
        return

    # Apply the exact same feature engineering
    df_eval = create_features(df_eval)
    
    # Prepare test set
    true_labels = df_eval[config.TARGET_COL]
    cols_to_drop = [
        config.TARGET_COL, config.COMPANY_ID_COL, 'year', 'month', 
        'default_year', 'default_month', 'report_period'
    ]
    X_eval = df_eval.drop(columns=cols_to_drop, errors='ignore')
    X_test = X_eval.reindex(columns=training_columns, fill_value=0)
    
    print("--> Loading model and making predictions...")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= config.DECISION_THRESHOLD).astype(int)

    # --- Generate Report and Plot ---
    print("\n--- Classification Report ---")
    print(f"(Using threshold = {config.DECISION_THRESHOLD})")
    print(classification_report(true_labels, predictions, target_names=['Non-Default (0)', 'Default (1)'], zero_division=0))
    
    print("\n--> Generating Confusion Matrix plot...")
    try:
        plt.rcParams['font.sans-serif'] = ['Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("Warning: Font not found, labels might not display correctly.")

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Threshold={config.DECISION_THRESHOLD})', fontsize=16)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.xticks(ticks=[0.5, 1.5], labels=['Non-Default', 'Default'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Non-Default', 'Default'], rotation=0)
    
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    save_path = os.path.join(config.IMAGES_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"\n--> Confusion Matrix plot saved to: {save_path}")
    print("\n======== EVALUATION PIPELINE FINISHED ========")

if __name__ == '__main__':
    main_evaluate()