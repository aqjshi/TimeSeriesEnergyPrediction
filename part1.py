import json
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
from load_and_visualize import   load_and_clean_data  
FILE_NAME = "individual_household_electric_power_consumption.csv"

ORIGINAL_COL = 'Global_active_power'
STATIONARY_COL = 'detrended_deseasonalized'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20




def apply_sarima_transform(
        df: pd.DataFrame,
        start_index: int,
        slope: float,
        intercept: float,
        seasonal_period: int,
        prev_univariate_history: pd.Series = None
    ) -> None:

    time_index = np.arange(start_index, start_index + len(df))
    df['detrended_univariate'] = df[ORIGINAL_COL] - (slope * time_index + intercept)

    # 2. Seasonal Differencing with History Stitching
    if prev_univariate_history is not None:
        combined = pd.concat([prev_univariate_history, df['detrended_univariate']])
        df[STATIONARY_COL] = combined.diff(periods=seasonal_period).iloc[-len(df):]
    else:
        df[STATIONARY_COL] = df['detrended_univariate'].diff(periods=seasonal_period)






def calculate_seasonal_period_from_acf(series, max_lag=336, skip_lags=24):
    acf_values = acf(
        series.dropna(), # Use .dropna() just in case, though detrended should be clean
        nlags=max_lag,
        alpha=None, # Only need the values, not confidence intervals
        fft=True
    )
    seasonal_acf_range = acf_values[skip_lags + 1 : max_lag + 1] 

    max_corr_index = np.argmax(np.abs(seasonal_acf_range))
    
    calculated_period = max_corr_index + (skip_lags + 1)
    
    return calculated_period

def process_data_and_get_stationary_splits(power_df, seasonal_period) -> Tuple[Dict[str, pd.DataFrame], float, float, pd.Series]: 

    time_series_data = power_df[ORIGINAL_COL]
    N = len(time_series_data)
    train_end_index = int(N * TRAIN_RATIO)
    val_end_index = int(N * (TRAIN_RATIO + VAL_RATIO))

    # Initial Split
    train_df = time_series_data.iloc[0:train_end_index].copy().to_frame(ORIGINAL_COL)
    val_df = time_series_data.iloc[train_end_index:val_end_index].copy().to_frame(ORIGINAL_COL)
    test_df = time_series_data.iloc[val_end_index:].copy().to_frame(ORIGINAL_COL)

    # 2. FIT TREND ON TRAIN
    time_index_train = np.arange(len(train_df))
    slope, intercept, _, _, _ = stats.linregress(time_index_train, train_df[ORIGINAL_COL])


    apply_sarima_transform(train_df, 0, slope, intercept, seasonal_period)

 
    val_history = train_df['detrended_univariate'].iloc[-seasonal_period:]
    apply_sarima_transform(
        val_df, len(train_df), slope, intercept, seasonal_period, prev_univariate_history=val_history
    )

    combined_univariate_history_source = pd.concat([train_df['detrended_univariate'], val_df['detrended_univariate']])
    test_history = combined_univariate_history_source.iloc[-seasonal_period:]
    apply_sarima_transform(
        test_df, len(train_df) + len(val_df), slope, intercept, seasonal_period, prev_univariate_history=test_history
    )


    train_df_stationary = train_df[[STATIONARY_COL]].dropna()
    val_df_stationary = val_df[[STATIONARY_COL]].dropna()
    test_df_stationary = test_df[[STATIONARY_COL]].dropna()

    stationary_splits = {
        "train": train_df_stationary,
        "val": val_df_stationary,
        "test": test_df_stationary
    }


    full_detrended_series = pd.concat([
        train_df['detrended_univariate'],
        val_df['detrended_univariate'],
        test_df['detrended_univariate']
    ])

    print(f"Data processing complete. Trend (slope: {slope:.4f}) fitted on Train.")

    return stationary_splits, slope, intercept, full_detrended_series



def part1():

    # --- 1. Initial Load, Split, and Trend Removal on TRAIN ---
    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='15min')

    time_series_data = clean_df[ORIGINAL_COL]
    N = len(time_series_data)
    train_end_index = int(N * TRAIN_RATIO)
    
    # 1.1 Split the raw series to isolate the training data
    train_data = time_series_data.iloc[0:train_end_index]
    
    # 1.2 Trend fitting on TRAIN DATA ONLY
    time_index_train = np.arange(len(train_data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time_index_train, train_data
    )
    linear_trend_train = slope * time_index_train + intercept
    detrended_series = train_data - linear_trend_train 
    nlags_to_check = 336
    skip_lags = 24 
    
    # Calculate period from the DETRENDED TRAIN series
    calculated_period = calculate_seasonal_period_from_acf(
        detrended_series, 
        max_lag=nlags_to_check, 
        skip_lags=skip_lags
    )



    print(f"Most likely Seasonal Period Detected (from detrended train): {calculated_period}") 
    acf_values_find, confint_find = acf(
        detrended_series.dropna(), # Only need the detrended train data here
        nlags=nlags_to_check,
        alpha=0.05,
        fft=True
    )
    N_acf_find = len(detrended_series.dropna())
    conf_bound_find = 1.96 / np.sqrt(N_acf_find)
    lags = np.arange(len(acf_values_find))

    plt.figure(figsize=(12, 6))
    plt.stem(lags, acf_values_find, markerfmt="o", linefmt="red", basefmt="k-")
    plt.axhspan(-conf_bound_find, conf_bound_find, alpha=0.1, color='red', label='99% Confidence Interval')
    plt.axvline(calculated_period, color='green', linestyle='--', linewidth=2, label=f'Detected Period S={calculated_period}')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'ACF of Detrended Training Series (Peak at S={calculated_period})')
    plt.xlabel('Lag (15-min Intervals)')
    plt.ylabel('Autocorrelation')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xlim(-0.5, nlags_to_check + 0.5)
    plt.tight_layout()
    plt.savefig('seasonal_acf_plot_finding.png')
    plt.close()
    print("Saved ACF plot (to FIND period) to seasonal_acf_plot_finding.png")
    # ------------------------------------------

    # --- 3. Run the complete data processing pipeline with the detected period ---
    
    # Load data again (if necessary, though it's already loaded, it's safer for this function call)
    clean_df_orig, cols = load_and_clean_data(FILE_NAME, resample_freq='15min')

    # This function uses the trend fitted only on the train set (internally) and applies the transformation
    stationary_splits, slope, intercept, full_detrended_series = process_data_and_get_stationary_splits(
        clean_df_orig, seasonal_period=calculated_period
    )
    
    # 4. Use the stationary training split for final ACF verification
    final_stationary_series = stationary_splits['train']
    
    # Calculate ACF on the fully transformed train series (for plotting only)
    nlags=nlags_to_check
    stationary_acf_values, confint = acf(
        final_stationary_series.iloc[:, 0], # Use the first column (the detrended/deseasonalized value)
        nlags=nlags,
        alpha=0.05,
        fft=True
    )

    N_acf = len(final_stationary_series)
    conf_bound = 1.96 / np.sqrt(N_acf)
    lags = np.arange(len(stationary_acf_values))

    # --- PLOT 2: ACF to VERIFY Stationarity (on Train set) ---
    plt.figure(figsize=(12, 6))
    plt.stem(lags, stationary_acf_values, markerfmt="o", linefmt="blue", basefmt="k-")
    plt.axhspan(-conf_bound, conf_bound, alpha=0.1, color='blue', label='95% Confidence Interval')
    plt.axvline(calculated_period, color='orange', linestyle='--', linewidth=2, label=f'Seasonal Lag S={calculated_period}')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'ACF of Fully Stationary Training Series (S={calculated_period})')
    plt.xlabel('Lag (15-min Intervals)')
    plt.ylabel('Autocorrelation')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xlim(-0.5, nlags + 0.5)
    plt.tight_layout()
    plt.savefig('stationary_acf_plot_verifying.png') 
    plt.close() # Close plot
    
    sarima_params = {
        "slope": slope,
        "intercept": intercept,
        "seasonal_period": (int(calculated_period))
    }

    # Define the filename
    param_file_path = "sarima_params.json"

    # Write the dictionary to a JSON file
    with open(param_file_path, 'w') as f:
        json.dump(sarima_params, f, indent=4)
    print(f"\nSaved SARIMA parameters to {param_file_path}")

    # --- Part 3: Summary Printouts ---
    # Recalculate N_train, N_val, N_test using the original clean_df size for total length
    N_train = int(len(clean_df_orig) * TRAIN_RATIO)
    N_val = int(len(clean_df_orig) * (TRAIN_RATIO + VAL_RATIO))
    N_test = len(clean_df_orig)
    print(f"\nSplit Indices (Total length {N_test}): Train end={N_train}, Val end={N_val}")

    print("\n--- Summary of Transformations ---")
    print(f"Trend Fit: Slope={slope:.4f}, Intercept={intercept:.4f} (Fitted on Train)")
    print(f"Seasonality Period: {calculated_period}")
    print(f"Train Usable Samples: {len(stationary_splits['train'])}")
    print(f"Val Usable Samples: {len(stationary_splits['val'])}")
    print(f"Test Usable Samples: {len(stationary_splits['test'])}")
    
    return clean_df_orig, calculated_period
