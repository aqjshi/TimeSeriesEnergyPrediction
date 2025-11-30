import pandas as pd
import numpy as np
import os

def load_and_clean_data(filename: str = "individual_household_electric_power_consumption.csv", resample_freq: str = None):
    """
    Loads, cleans, and prepares the power consumption dataset.
    
    Args:
        filename: Path to CSV.
        resample_freq: None for raw data (minute-level), 'H' for Hourly, 'D' for Daily.
                       Recommended: 'H' or None for feature extraction.
    """
    print(f"Loading {filename}...")
    
    power_df = pd.read_csv(
        filename,
        sep=';', 
        na_values=['?', 'nan'],
        low_memory=False,
        infer_datetime_format=True,
        parse_dates={'DateTime': ['Date', 'Time']},
        index_col='DateTime'
    )

    # 2. Ensure Numeric Types
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]
    
    # Coerce to numeric (handles any remaining non-numeric issues)
    for col in numeric_cols:
        power_df[col] = pd.to_numeric(power_df[col], errors='coerce')

    # 3. Handle Missing Values
    # Interpolation is preferred over dropping for Time Series to maintain frequency
    print("Interpolating missing values...")
    power_df.interpolate(method='linear', inplace=True)
    
    # Drop any remaining NAs (usually at the very start of the file)
    power_df.dropna(inplace=True)

    # 4. Resampling (Optional but recommended for noise reduction)
    if resample_freq:
        print(f"Resampling data to frequency: {resample_freq}")
        power_df = power_df.resample(resample_freq).mean()
        # Interpolate again in case resampling introduced gaps
        power_df.interpolate(method='linear', inplace=True)

    print(f"Data Cleaned. Shape: {power_df.shape}")
    return power_df, numeric_cols

def extract_features(df, target_col='Global_active_power'):
    """
    Example of preparing data for univariate prediction while 
    using other columns for feature extraction.
    """
    df_features = df.copy()
    
    # --- Feature Engineering Example ---
    
    # 1. Time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    
    # 2. Lag features (Autocorrelation features)
    # Valid for prediction: "What was the power 24 hours ago?"
    df_features['lag_24h'] = df_features[target_col].shift(24) 
    
    # 3. Rolling Window features (Moving Averages)
    # Valid for prediction: "What was the average power over the last 6 hours?"
    df_features['rolling_mean_6h'] = df_features[target_col].rolling(window=6).mean()
    
    # 4. Cross-variable features (Using the other columns)
    # Example: Intensity to Power ratio
    df_features['intensity_to_power_ratio'] = df_features['Global_intensity'] / (df_features['Global_active_power'] + 1e-5)

    # Drop rows with NaNs created by lagging/rolling
    df_features.dropna(inplace=True)
    
    return df_features

if __name__ == "__main__":
    # Settings
    FILE_NAME = "individual_household_electric_power_consumption.csv"
    

    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='D')
    
    # Prepare for Prediction
    final_df = extract_features(clean_df, target_col='Global_active_power')
    
    # Separate Target and Features
    target = final_df['Global_active_power']
    features = final_df.drop(columns=['Global_active_power'])
    
    print("\n--- Readiness for Prediction ---")
    print(f"Target Series Shape: {target.shape}")
    print(f"Feature Matrix Shape: {features.shape}")
    print(f"Features available: {list(features.columns)}")
    print("\nFirst 5 rows of cleaned data:")
    print(final_df.head())
        
    
