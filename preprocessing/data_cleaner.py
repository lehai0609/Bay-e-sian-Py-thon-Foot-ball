import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional

def handle_missing_values(df: pd.DataFrame,
                          event_cols_pattern: str = '_cum_',
                          strength_cols: List[str] = ['home_team_strength', 'away_team_strength'],
                          impute_strategy: str = 'mean') -> pd.DataFrame:
    """
    Handles missing values in the feature matrix.
    - Fills NaN in event count columns with 0.
    - Imputes NaN in strength columns using specified strategy.

    Args:
        df: The feature matrix DataFrame.
        event_cols_pattern: String pattern to identify event count columns.
        strength_cols: List of team strength column names.
        impute_strategy: 'mean', 'median', or 'zero' for strength imputation.

    Returns:
        DataFrame with missing values handled.
    """
    df_cleaned = df.copy()

    # Fill event count NaNs with 0
    event_cols = [col for col in df_cleaned.columns if event_cols_pattern in col]
    df_cleaned[event_cols] = df_cleaned[event_cols].fillna(0)

    # Impute strength columns
    for col in strength_cols:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                if impute_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif impute_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif impute_strategy == 'zero':
                     fill_value = 0
                else:
                    fill_value = 0 # Default to zero if strategy unknown
                print(f"Imputing NaNs in '{col}' with {impute_strategy} value: {fill_value:.2f}")
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)

    # Check for any remaining NaNs (optional)
    if df_cleaned.isnull().any().any():
         print("Warning: NaNs still remain after cleaning:")
         print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])

    return df_cleaned


def scale_features(df: pd.DataFrame, cols_to_scale: List[str]) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scales specified numerical features using StandardScaler.

    Args:
        df: The feature matrix DataFrame.
        cols_to_scale: List of column names to scale.

    Returns:
        A tuple containing:
          - The DataFrame with scaled features.
          - The fitted StandardScaler object (or None if no columns scaled).
    """
    df_scaled = df.copy()
    scaler = None
    
    # Identify columns that actually exist in the DataFrame
    actual_cols_to_scale = [col for col in cols_to_scale if col in df_scaled.columns]

    if not actual_cols_to_scale:
        print("No columns specified or found for scaling.")
        return df_scaled, None

    scaler = StandardScaler()
    
    # Ensure columns are numeric before scaling
    for col in actual_cols_to_scale:
        df_scaled[col] = pd.to_numeric(df_scaled[col], errors='coerce')
    
    # Drop rows with NaNs introduced by coercion if any, or handle them
    df_scaled.dropna(subset=actual_cols_to_scale, inplace=True) 
    
    if df_scaled.empty:
        print("DataFrame became empty after handling non-numeric values in scaling columns.")
        return df_scaled, None

    try:
        df_scaled[actual_cols_to_scale] = scaler.fit_transform(df_scaled[actual_cols_to_scale])
        print(f"Scaled columns: {actual_cols_to_scale}")
    except ValueError as e:
        print(f"Error during scaling: {e}. Check if columns contain non-numeric data despite coercion.")
        return df, None # Return original df on error

    return df_scaled, scaler


def ensure_data_types(df: pd.DataFrame,
                      event_cols_pattern: str = '_cum_',
                      outcome_col: str = 'Outcome',
                      intercept_col: str = 'intercept') -> pd.DataFrame:
    """
    Ensures columns have appropriate data types.

    Args:
        df: DataFrame to process.
        event_cols_pattern: Pattern for event columns (should be int).
        outcome_col: Name of the outcome column (should be category or int).
        intercept_col: Name of the intercept column (should be int).

    Returns:
        DataFrame with corrected data types.
    """
    df_typed = df.copy()

    for col in df_typed.columns:
        if event_cols_pattern in col or col == intercept_col:
            try:
                # Use Int64 to handle potential NaNs if cleaning wasn't perfect
                df_typed[col] = df_typed[col].astype(pd.Int64Dtype())
            except (TypeError, ValueError):
                 print(f"Could not convert column {col} to integer.")
        elif col == outcome_col:
             # Categorical might be better if using statsmodels later
             try:
                 df_typed[col] = df_typed[col].astype('category')
             except (TypeError, ValueError):
                  print(f"Could not convert column {col} to category.")
        # Scaled features are typically float, leave others as is unless specified

    print("Data types ensured (Event counts/Intercept: Int64, Outcome: Category).")
    return df_typed