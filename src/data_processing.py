# file: src/data_processing.py

import pandas as pd
import numpy as np
import config

def create_features(df):
    """
    Engineers new features from a pre-cleaned DataFrame.
    Assumes the input DataFrame already has English headers and correct data types.
    
    Args:
        df (pandas.DataFrame): The cleaned DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with new engineered features.
    """
    print("--> Creating new features (Feature Engineering)...")
    
    # 1. Financial Ratios
    # Use .div() for safe division, replacing potential infinity with 0
    df['debt_to_equity_ratio'] = df['total_assets'].div(df['total_equity']).replace([np.inf, -np.inf], 0).fillna(0)
    df['current_ratio'] = df['total_current_assets'].div(df['short_term_debt']).replace([np.inf, -np.inf], 0).fillna(0)
    print("    Financial ratios created.")

    # 2. Year-over-Year Growth Rates
    # Sort values to ensure correct calculation order
    df.sort_values(by=[config.COMPANY_ID_COL, 'year', 'month'], inplace=True)
    
    growth_cols = ['revenue', 'net_profit', 'total_assets']
    for col in growth_cols:
        df[f'{col}_yoy_growth'] = df.groupby(config.COMPANY_ID_COL)[col].pct_change(periods=1)
    print("    Year-over-Year growth rates created.")
    
    # 3. Final cleanup of any NaN values created by pct_change
    df.fillna(0, inplace=True)
    print("    Feature engineering complete.")
    
    return df