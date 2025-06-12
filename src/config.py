# file: src/config.py
import os

# --- Directory Paths ---
# Using relative paths makes the project portable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

# --- File Paths ---
# For demonstration purposes, we use the safe sample data
# For your actual full run, you would swap these with your real data paths
# Ex: REAL_TRAIN_PATH = '/path/to/your/real/train.csv'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
ALL_DATA_PATH = os.path.join(DATA_DIR, 'all_data.csv')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_data.csv')


# --- Model & Column Artifact Paths ---
XGB_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
TRAINING_COLUMNS_PATH = os.path.join(MODELS_DIR, 'training_columns.txt')

# --- Feature & Target Names ---
COMPANY_ID_COL = 'company_id' 
TARGET_COL = 'default_status'
YEAR_COL = 'year'
MONTH_COL = 'month'
DEFAULT_YEAR_COL = 'default_year'
DEFAULT_MONTH_COL = 'default_month'

# --- Model Parameters ---
# This is the optimal threshold you find from the P-R curve for business use
# We start with 0.5 as a default for evaluation
DECISION_THRESHOLD = 0.5