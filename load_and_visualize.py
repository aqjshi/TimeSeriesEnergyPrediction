import pandas as pd
import numpy as np
import os

def load_and_clean_data(filename: str = "individual_household_electric_power_consumption.csv", resample_freq: str = None):
    print(f"Loading {filename}...")

    power_df = pd.read_csv(filename, sep=',', na_values=['?', 'nan'], low_memory=False)
    power_df['DateTime'] = pd.to_datetime(
        power_df['Date'] + ' ' + power_df['Time'], 
        format='%d/%m/%Y %H:%M:%S', 
        errors='coerce'
    )
    

    power_df.drop(columns=['Date', 'Time'], inplace=True)
    power_df.set_index('DateTime', inplace=True)
    
    # Drop rows where DateTime parsing failed
    power_df = power_df[power_df.index.notnull()]
    
    # Sort index to ensure time order
    power_df.sort_index(inplace=True)

    # 3. Ensure Numeric Types
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        power_df[col] = pd.to_numeric(power_df[col], errors='coerce')

    # 4. Handle Missing Values
    print("Interpolating missing values...")
    power_df.interpolate(method='linear', inplace=True)
    power_df.dropna(inplace=True)

    # 5. Resampling
    if resample_freq:
        print(f"Resampling data to frequency: {resample_freq}")
        power_df = power_df.resample(resample_freq).mean()
        # Interpolate again in case resampling introduced gaps
        power_df.interpolate(method='linear', inplace=True)
        power_df.dropna(inplace=True)

    print(f"Data Cleaned. Shape: {power_df.shape}")
    return power_df, numeric_cols

def extract_features(df, target_col='Global_active_power'):

    df_features = df.copy()
    


    # Drop rows with NaNs created by lagging/rolling
    df_features.dropna(inplace=True)
    
    return df_features

if __name__ == "__main__":
    # Settings
    FILE_NAME = "individual_household_electric_power_consumption.csv"

    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='D')
    
    # Prepare for Prediction
    final_df = extract_features(clean_df, target_col='Global_active_power')
    
    target = final_df['Global_active_power']
    features = final_df.drop(columns=['Global_active_power'])
    
    print("\n--- Readiness for Prediction ---")
    print(f"Target Series Shape: {target.shape}")
    print(f"Feature Matrix Shape: {features.shape}")
    print(f"Features available: {list(features.columns)}")
    print("\nFirst 5 rows of cleaned data:")
    print(final_df.head())
    
