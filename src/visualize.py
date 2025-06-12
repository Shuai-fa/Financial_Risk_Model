# file: src/visualize.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from data_processing import create_features 
import config

def plot_feature_importance(model, feature_names, save_path):
    """Creates and saves the feature importance plot."""
    print("--> Generating Feature Importance plot...")
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    top_20 = importances.head(20)

    plt.figure(figsize=(12, 10))
    plt.barh(top_20['feature'], top_20['importance'], color='cornflowerblue')
    plt.xlabel('Feature Importance (F-score)')
    plt.ylabel('Features')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"    Feature Importance plot saved to: {save_path}")

def plot_pr_curve(true_labels, probabilities, save_path):
    """Creates and saves the Precision-Recall curve."""
    print("--> Generating Precision-Recall curve...")

    precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'PR Curve (AUC = {pr_auc:0.2f})')
    plt.xlabel('Recall (How many actual defaulters did we find?)', fontsize=14)
    plt.ylabel('Precision (How accurate are our predictions?)', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=18)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    # Annotate a reference point
    try:
        target_precision = 0.60
        idx = np.min(np.where(precision >= target_precision))
        threshold_value = thresholds[idx-1] if idx > 0 else thresholds[0]
        plt.plot(recall[idx], precision[idx], 'ro', markersize=8, label=f'Example Threshold ≈ {threshold_value:.2f}')
        plt.text(recall[idx], precision[idx] - 0.05, 
                 f'Precision: {precision[idx]:.2f}\nRecall: {recall[idx]:.2f}',
                 ha='center', va='top', fontsize=12, color='red',
                 bbox=dict(boxstyle="round,pad=0.3", fc="ivory", alpha=0.8))
    except (ValueError, IndexError):
        print("    Could not find an example point for annotation automatically.")

    plt.savefig(save_path)
    print(f"    Precision-Recall curve saved to: {save_path}")

def main():
    """Main function to run the visualization pipeline."""
    print("\n======== STARTING VISUALIZATION PIPELINE ========")
    try:
        with open(config.TRAINING_COLUMNS_PATH, 'r') as f:
            training_columns = [line.strip() for line in f]
        model = joblib.load(config.XGB_MODEL_SAVE_PATH)
        df_eval = pd.read_csv(config.SAMPLE_DATA_PATH)
    except FileNotFoundError:
        print("ERROR: Model or data files not found. Please run train.py first.")
        return

    # Set font for plotting
    try:
        plt.rcParams['font.sans-serif'] = ['Helvetica']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("Warning: Font 'Helvetica' not found. Using default.")

    # --- 1. Generate Feature Importance Plot ---
    plot_feature_importance(
        model=model, 
        feature_names=training_columns, 
        save_path=os.path.join(config.IMAGES_DIR, 'feature_importance.png')
    )

    # --- 2. Generate Precision-Recall Curve ---
    # We need to re-create the test set to get true labels and probabilities
    df_eval_featured = create_features(df_eval.copy())
    true_labels = df_eval_featured[config.TARGET_COL]
    
    cols_to_drop = [
        config.TARGET_COL, config.COMPANY_ID_COL, 'year', 'month', 
        'default_year', 'default_month', 'report_period'
    ]
    X_eval = df_eval_featured.drop(columns=cols_to_drop, errors='ignore')
    X_test = X_eval.reindex(columns=training_columns, fill_value=0)
    probabilities = model.predict_proba(X_test)[:, 1]

    plot_pr_curve(
        true_labels=true_labels, 
        probabilities=probabilities, 
        save_path=os.path.join(config.IMAGES_DIR, 'precision_recall_curve.png')
    )
    
    print("\n======== VISUALIZATION PIPELINE FINISHED ========")

if __name__ == '__main__':
    main()