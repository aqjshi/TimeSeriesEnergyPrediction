import json
import os
import secrets
from typing import Any, Dict, List, Tuple
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse
from darts.models import BlockRNNModel, LinearRegressionModel, NaiveDrift
from load_and_visualize import   load_and_clean_data  

from scipy.signal import periodogram # <--- Add this import at the top



FILE_NAME = "individual_household_electric_power_consumption.csv"

ORIGINAL_COL = 'Global_active_power'
STATIONARY_COL = 'detrended_deseasonalized'
SEASONAL_PERIOD = 0
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


def process_data_and_get_stationary_splits(
    airport_df: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], float, float, pd.Series]: # <-- MODIFIED: Added pd.Series to output
    """Splits data, fits linear trend on train, transforms all sets, and returns
    stationary splits, trend parameters, and the full detrended series."""

    time_series_data = airport_df[ORIGINAL_COL]
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


    apply_sarima_transform(train_df, 0, slope, intercept, SEASONAL_PERIOD)

 
    val_history = train_df['detrended_univariate'].iloc[-SEASONAL_PERIOD:]
    apply_sarima_transform(
        val_df, len(train_df), slope, intercept, SEASONAL_PERIOD, prev_univariate_history=val_history
    )

    combined_univariate_history_source = pd.concat([train_df['detrended_univariate'], val_df['detrended_univariate']])
    test_history = combined_univariate_history_source.iloc[-SEASONAL_PERIOD:]
    apply_sarima_transform(
        test_df, len(train_df) + len(val_df), slope, intercept, SEASONAL_PERIOD, prev_univariate_history=test_history
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
    """
    Main function to run the time series analysis pipeline.
    """
    global SEASONAL_PERIOD # Use the global variable so it updates for the whole script

    # --- Part 1: Initial Load and ACF Plotting ---
    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='H')
    
    time_series = clean_df[ORIGINAL_COL]

    plt.figure(figsize=(10, 5)) 
    plt.plot(time_series.index, time_series)
    plt.xlabel('Index')
    plt.ylabel('Global_active_power')
    plt.title('Global_active_power Over Time')
    plt.savefig("Global_active_power.png")
    plt.close()
    
    time_series_data = clean_df[ORIGINAL_COL]
    time_index = np.arange(len(clean_df))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        time_index, time_series_data
    )
    linear_trend = slope * time_index + intercept
    clean_df['detrended_univariate'] = time_series_data - linear_trend

    # ------------------------------------------------------------------
    # NEW: SPECTRAL ANALYSIS TO DETECT SEASONALITY
    # ------------------------------------------------------------------
    print("Performing Spectral Analysis...")
    
    # 1. Get the detrended series (drop NaNs if any exist)
    data_for_spectrum = clean_df['detrended_univariate'].dropna().values
    
    # 2. Compute the Periodogram (Frequencies vs Power)
    frequencies, spectrum = periodogram(data_for_spectrum)
    
    # 3. Apply Offset: specific request to offset by 24.
    # We zero out the first 24 components (low frequencies/long periods) 
    # to avoid picking up residual trends or very long cycles.
    spectrum[0:24] = 0
    
    # 4. Find the index with the maximum power
    max_idx = np.argmax(spectrum)
    dominant_freq = frequencies[max_idx]
    
    # 5. Convert Frequency to Period (T = 1 / f)
    # We define a local variable 'seasonal_period' and update the Global
    if dominant_freq > 0:
        calculated_period = int(round(1 / dominant_freq))
    else:
        calculated_period = 1 # Fallback if frequency is 0 (unlikely after offset)

    # 6. Update the Global Variable
    SEASONAL_PERIOD = calculated_period
    
    # 7. Print the result
    print(f"Spectral Analysis Complete. Offset: 24.")
    print(f"Most likely Seasonal Period Detected: {SEASONAL_PERIOD}")
    # ------------------------------------------------------------------

    # Continue with the rest of the script using the detected period
    clean_df['detrended_deseasonalized'] = (
        clean_df['detrended_univariate'].diff(periods=SEASONAL_PERIOD)
    )
    print(f"Seasonality (period={SEASONAL_PERIOD}) removed using Seasonal Differencing.")

    # Drop the NaNs created by the differencing step for the ACF calculation
    final_stationary_series = clean_df['detrended_deseasonalized'].dropna()
    nlags = min(2 * SEASONAL_PERIOD, len(final_stationary_series) // 2 - 1) # Safety check for nlags

    # Calculate ACF on the fully transformed series
    stationary_acf_values, confint = acf(
        final_stationary_series,
        nlags=nlags,
        alpha=0.05,
        fft=True
    )

    N_acf = len(final_stationary_series)
    conf_bound = 1.96 / np.sqrt(N_acf)
    lags = np.arange(len(stationary_acf_values))

    # Plot the new ACF
    plt.figure(figsize=(12, 6))
    plt.stem(lags, stationary_acf_values, markerfmt="o", linefmt="blue", basefmt="k-")
    plt.axhspan(-conf_bound, conf_bound, alpha=0.1, color='blue', label='95% Confidence Interval')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'ACF of Fully Stationary Series (Detrended & Deseasonalized, S={SEASONAL_PERIOD})')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.xlim(-0.5, nlags + 0.5)
    plt.tight_layout()
    plt.savefig('stationary_acf_plot_corrected.png')
    plt.close() # Close plot
    print("Saved stationary ACF plot to stationary_acf_plot_corrected.png")
    
    # Reload original for splitting process
    clean_df_orig, cols = load_and_clean_data(FILE_NAME, resample_freq='H')

    # Step 1: Process and get stationary data and parameters
    # Note: process_data_and_get_stationary_splits uses the global SEASONAL_PERIOD we just set
    stationary_splits, slope, intercept, full_detrended_series = process_data_and_get_stationary_splits(clean_df_orig)

    sarima_params = {
        "slope": slope,
        "intercept": intercept,
        "seasonal_period": SEASONAL_PERIOD
    }

    # Define the filename
    param_file_path = "sarima_params.json"

    # Write the dictionary to a JSON file
    with open(param_file_path, 'w') as f:
        json.dump(sarima_params, f, indent=4)
    print(f"\nSaved SARIMA parameters to {param_file_path}")

    # --- Part 3: Summary Printouts ---
    N_train = int(len(clean_df_orig) * TRAIN_RATIO)
    N_val = int(len(clean_df_orig) * VAL_RATIO + N_train)
    N_test = int(len(clean_df_orig))
    print(f"\nSplit Indices: Train end={N_train}, Val end={N_val}, Test end={N_test}")

    print("\n--- Summary of Transformations ---")
    print(f"Trend Fit: Slope={slope:.4f}, Intercept={intercept:.4f}")
    print(f"Seasonality Period: {SEASONAL_PERIOD}")
    print(f"Train Usable Samples: {len(stationary_splits['train'])} (Loss: {N_train - len(stationary_splits['train'])})")
    print(f"Val Usable Samples: {len(stationary_splits['val'])}")
    print(f"Test Usable Samples: {len(stationary_splits['test'])}")



def create_fourier_ar_features(data_series: pd.Series, period: int, lag_step: int = 1, n_fourier_pairs: int = 1) -> pd.DataFrame:
    features = pd.DataFrame({'Y_target': data_series.values}, index=data_series.index)
    
    # Y_target starts after the first 'lag_step' entries, as they have no AR feature
    Y_target = features['Y_target'].iloc[lag_step:]
    
    X_df = pd.DataFrame(index=Y_target.index)
    
    # 1. Time trend feature
    # Note: Using the integer index value of the time series for 't'
    X_df['t'] = Y_target.index.values - 1 
    
    # 2. Fourier features
    omega = 2 * np.pi / period
    for k in range(1, n_fourier_pairs + 1):
        X_df[f'sin_t_{k}'] = np.sin(k * omega * X_df['t'])
        X_df[f'cos_t_{k}'] = np.cos(k * omega * X_df['t'])
        
    # 3. Autoregressive (AR) feature
    # The AR feature uses the value L steps prior: Y_t-L
    X_df[f'Y_t-{lag_step}'] = data_series.shift(lag_step).loc[Y_target.index]
    
    return X_df, Y_target

def evaluate_set(X_set, Y_set, name, history_series, lag, model):
    """
    Evaluates model predictions against a naive baseline for a given dataset.
    """
    Y_pred = model.predict(X_set)
    
    # Naive L-step prediction: Y_t = Y_t-L
    naive_preds = Y_set.shift(lag)
    
    # Correctly seed the first prediction using the history series
    # Find the index 'lag' steps before the start of Y_set
    seed_index = Y_set.index[0] - lag
    naive_preds.iloc[0] = history_series.loc[seed_index]
    
    # Get indices where naive forecast is valid (after the initial seed)
    valid_indices = naive_preds.dropna().index
    
    Y_set_eval = Y_set.loc[valid_indices]
    
    # Y_pred is already aligned to Y_set, so we filter it based on valid_indices
    # This ensures we are comparing the exact same set of predictions
    Y_pred_eval = Y_pred[Y_set.index.isin(valid_indices)]
    naive_preds_eval = naive_preds.loc[valid_indices]
    
    # Model Metrics
    model_mae = mean_absolute_error(Y_set_eval, Y_pred_eval)
    model_mse = mean_squared_error(Y_set_eval, Y_pred_eval)
    model_rmse = np.sqrt(model_mse)
    
    # Naive Metrics
    naive_mae = mean_absolute_error(Y_set_eval, naive_preds_eval)
    naive_mse = mean_squared_error(Y_set_eval, naive_preds_eval)
    naive_rmse = np.sqrt(naive_mse)
    
    metrics = {
        'model_mae': model_mae, 'model_mse': model_mse, 'model_rmse': model_rmse,
        'naive_mae': naive_mae, 'naive_mse': naive_mse, 'naive_rmse': naive_rmse
    }
    return metrics, Y_pred_eval, Y_set_eval, naive_preds_eval

def part2():
    """
    Main function to run the Fourier-AR model evaluation pipeline.
    """
    # --- 1. Load Data and Parameters ---
    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='H')
    
    original_series = clean_df[ORIGINAL_COL]

    with open("sarima_params.json", 'r') as f:
        sarima_params = json.load(f)
    SARIMA_PERIOD = sarima_params.get('seasonal_period', 753)

    # --- 2. Split Data ---
    N_total = len(original_series)
    TRAIN_RATIO = .7
    VAL_RATIO = .2
    N_train_end = int(N_total * TRAIN_RATIO)
    N_val_end = int(N_total * (TRAIN_RATIO + VAL_RATIO))
    
    train_data = original_series.iloc[0:N_train_end].copy()
    val_data = original_series.iloc[N_train_end:N_val_end].copy()
    test_data = original_series.iloc[N_val_end:].copy()

    # --- 3. Run Experiment Loop ---
    FORECAST_HORIZONS = [1]
    MAX_K =5
    final_results = []

    print(f"Running evaluation for horizons {FORECAST_HORIZONS} with K=1 to {MAX_K}...")

    for L in FORECAST_HORIZONS:
        # Set the current LAG equal to the forecast horizon L
        current_LAG = L
        
        results_val = []
        results_test = []

        # 1. Hyperparameter tuning loop for K (1 to 5)
        for K in range(1, MAX_K + 1):
            # --- Feature Engineering ---
            
            # Train Set
            X_train, Y_train = create_fourier_ar_features(train_data, SARIMA_PERIOD, current_LAG, K)
            
            # Validation Set (needs LAG history from train)
            val_and_history = original_series.loc[train_data.index[-current_LAG]:]
            X_val, Y_val = create_fourier_ar_features(val_and_history, SARIMA_PERIOD, current_LAG, K)
            # Filter to only the validation set indices
            X_val = X_val.loc[val_data.index]
            Y_val = Y_val.loc[val_data.index]
            
            # Test Set (needs LAG history from val)
            test_and_history = original_series.loc[val_data.index[-current_LAG]:]
            X_test, Y_test = create_fourier_ar_features(test_and_history, SARIMA_PERIOD, current_LAG, K)
            # Filter to only the test set indices
            X_test = X_test.loc[test_data.index]
            Y_test = Y_test.loc[test_data.index]
            
            # --- Model Training and Evaluation ---
            model = Ridge(alpha=1.0) 
            model.fit(X_train, Y_train)
            
            # Evaluate on Val
            # For evaluation, the history for the naive forecast is the preceding set (train_data)
            val_metrics, _, _, _ = evaluate_set(X_val, Y_val, "", train_data, current_LAG, model)
            val_metrics['K'] = K
            results_val.append(val_metrics)
            
            # Evaluate on Test
            # For evaluation, the history for the naive forecast is the preceding set (val_data)
            test_metrics, _, _, _ = evaluate_set(X_test, Y_test, "", val_data, current_LAG, model)
            test_metrics['K'] = K
            results_test.append(test_metrics)

        val_df = pd.DataFrame(results_val)
        test_df = pd.DataFrame(results_test)
        
        # 2. Find the best K based on Validation RMSE
        best_k_row = val_df.iloc[val_df['model_rmse'].argmin()]
        best_k = int(best_k_row['K'])

        # 3. Retrieve the metrics for the best K from the Test set
        best_test_metrics = test_df[test_df['K'] == best_k].iloc[0].to_dict()
        
        # Also get the Naive Baseline (which is the same across all K for a given L)
        naive_metrics = test_df.iloc[0].to_dict()
        
        final_results.append({
            'Forecast_Horizon_L': L,
            'Best_K_by_Val': best_k,
            'Model_MAE': round(best_test_metrics['model_mae'], 2),
            'Model_MSE': round(best_test_metrics['model_mse'], 2),
            'Model_RMSE': round(best_test_metrics['model_rmse'], 2),
            'Naive_MAE': round(naive_metrics['naive_mae'], 2),
            'Naive_MSE': round(naive_metrics['naive_mse'], 2),
            'Naive_RMSE': round(naive_metrics['naive_rmse'], 2)
        })

    # --- 4. Display Final Results ---
    results_df = pd.DataFrame(final_results).set_index('Forecast_Horizon_L')
    print("\n--- Final Results Table ---")
    print(results_df)

    # Display best K values for clarity
    best_k_df = results_df[['Best_K_by_Val']].copy().rename(columns={'Best_K_by_Val': 'Optimal Fourier Pairs (K)'})
    print("\n--- Optimal K per Horizon ---")
    print(best_k_df)




if __name__ == "__main__":
    part1()
    # part2()
    # part3()
    # part4()




# def create_fourier_covariates(series: TimeSeries, period: float, K: int) -> TimeSeries:
#     """
#     Generates Fourier series covariates based on the series' time index.
#     """
#     time_index = series.time_index
#     t = np.arange(len(time_index))
#     fourier_df = pd.DataFrame(index=time_index)
    
#     for k in range(1, K + 1):
#         omega = 2 * np.pi * k / period
#         sin_col = np.sin(omega * t)
#         cos_col = np.cos(omega * t)
#         fourier_df[f'sin_{k}_p{period}'] = sin_col
#         fourier_df[f'cos_{k}_p{period}'] = cos_col
        
#     return TimeSeries.from_dataframe(fourier_df)

# def create_trend_covariate(series: TimeSeries) -> TimeSeries:
#     """
#     Generates a raw, unscaled linear trend covariate.
#     """
#     time_index = series.time_index
#     trend = np.arange(len(time_index)) 
#     trend_df = pd.DataFrame(index=time_index, data={'linear_trend': trend})
#     return TimeSeries.from_dataframe(trend_df)


# def part3():
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(torch.cuda.is_available())
#     print(f"device: {device}")

#     config = {
#         'output_chunk_length': 1,
#         'TRAIN_RATIO': .7,
#         'VAL_RATIO': .2,
#     } 
    
#     torch.set_float32_matmul_precision('medium')
    

#     run_name = f"forecast_{secrets.token_hex(4)}"
    


#     airport_df = pd.read_csv("AirportFootfalls_data.csv").set_index('index')
#     series = TimeSeries.from_dataframe(
#         airport_df.reset_index(), 
#         'index', 
#         ORIGINAL_COL
#     )

#     PERIOD = 753 
#     K_FOURIER = 5
#     fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
#     trend_covariate = create_trend_covariate(series)
#     all_covariates = fourier_covariates.stack(trend_covariate)

#     # 2. Split Target Series
#     train_series, val_test_series = series.split_before(config['TRAIN_RATIO'])
#     val_split_pos = int(len(val_test_series) * (config['VAL_RATIO'] / (1 - config['TRAIN_RATIO'])))
#     val_split_point_index_value = val_test_series.time_index[val_split_pos]
#     val_series, test_series = val_test_series.split_before(val_split_point_index_value)

#     # 3. Split Covariates (at the exact same points as target)
#     train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
#     val_cov, test_cov = val_test_cov.split_before(val_series.end_time())


#     print("\n--- Calculating Naive 1-step forecast (Baseline) ---")
#     naive_model = NaiveDrift()
#     naive_model.fit(train_series)
    
#     naive_val_preds = naive_model.historical_forecasts(
#         val_series,
#         start=val_series.start_time(),
#         verbose=False
#     )
#     naive_test_preds = naive_model.historical_forecasts(
#         test_series,
#         start=test_series.start_time(),
#         verbose=False
#     )

#     val_series_aligned = val_series.slice_intersect(naive_val_preds)
#     test_series_aligned = test_series.slice_intersect(naive_test_preds)

#     naive_val_rmse = rmse(val_series_aligned, naive_val_preds)
#     naive_val_mae = mae(val_series_aligned, naive_val_preds)
#     naive_val_mse = mse(val_series_aligned, naive_val_preds)
#     naive_test_rmse = rmse(test_series_aligned, naive_test_preds)
#     naive_test_mae = mae(test_series_aligned, naive_test_preds)
#     naive_test_mse = mse(test_series_aligned, naive_test_preds)
    
#     print(f"Naive Validation RMSE: {naive_val_rmse:.4f}")
#     print(f"Naive Validation MAE:  {naive_val_mae:.4f}")
#     print(f"Naive Validation MSE:  {naive_val_mse:.4f}")
#     print(f"Naive Test RMSE: {naive_test_rmse:.4f}")
#     print(f"Naive Test MAE:  {naive_test_mae:.4f}")
#     print(f"Naive Test MSE:  {naive_test_mse:.4f}")

#     # --- 5. AR+FOURIER MODEL (Scaling) ---
    
#     scaler = Scaler()
#     train_scaled = scaler.fit_transform(train_series)
#     val_scaled = scaler.transform(val_series)
#     test_scaled = scaler.transform(test_series)
    
#     cov_scaler = Scaler() #SEPARATE scaler
#     train_cov_scaled = cov_scaler.fit_transform(train_cov)
#     val_cov_scaled = cov_scaler.transform(val_cov)
#     test_cov_scaled = cov_scaler.transform(test_cov)

#     all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
    
#     # --- 6. AR+FOURIER MODEL (Training & Prediction) ---

#     print("\n--- Training Fourier-AR-Trend model (LinearRegressionModel) ---")
#     fourier_model = LinearRegressionModel(
#         lags=1, 
#         lags_future_covariates=[0],
#         output_chunk_length=1
#     )

#     fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)
    
#     print("Predicting with Fourier-AR-Trend model...")
#     val_preds_fourier_scaled = fourier_model.predict(
#         n=len(val_scaled),
#         series=train_scaled,
#         future_covariates=all_covariates_scaled, 
#         verbose=False
#     )
#     test_preds_fourier_scaled = fourier_model.predict(
#         n=len(test_scaled),
#         series=train_scaled.append(val_scaled), 
#         future_covariates=all_covariates_scaled, 
#         verbose=False
#     )

#     # Unscale final predictions and actuals
#     val_preds_unscaled = scaler.inverse_transform(val_preds_fourier_scaled)
#     val_unscaled = scaler.inverse_transform(val_scaled)
    
#     preds_unscaled = scaler.inverse_transform(test_preds_fourier_scaled)
#     test_unscaled = scaler.inverse_transform(test_scaled)


#     final_val_rmse = rmse(val_unscaled, val_preds_unscaled)
#     final_val_mae = mae(val_unscaled, val_preds_unscaled)
#     final_val_mse = mse(val_unscaled, val_preds_unscaled)
    
#     final_test_rmse = rmse(test_unscaled, preds_unscaled)
#     final_test_mae = mae(test_unscaled, preds_unscaled)
#     final_test_mse = mse(test_unscaled, preds_unscaled)


#     all_actuals = val_unscaled.append(test_unscaled)
#     all_predictions = val_preds_unscaled.append(preds_unscaled)
    
#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)
    
#     csv_filename = os.path.join(output_dir, f"{run_name}.csv")
#     print(f"\nSaving combined predictions and actuals to {csv_filename}...")
    
#     ACTUAL_COL_NAME = ORIGINAL_COL 
#     PREDICTED_COL_NAME = 'Predicted_Footfalls'

#     df_actuals = all_actuals.to_dataframe()
#     df_predictions = all_predictions.to_dataframe().rename(
#         columns={ACTUAL_COL_NAME: PREDICTED_COL_NAME}
#     )
#     results_df = pd.concat([df_actuals, df_predictions], axis=1)

#     val_end_date = val_unscaled.end_time()
#     results_df['Data_Split'] = np.where(results_df.index <= val_end_date, 'Validation', 'Test')
    
#     results_df.to_csv(csv_filename, index_label='index')

#     plot_filename = os.path.join(output_dir, f"{run_name}.png")
#     print(f"Generating and saving plot to {plot_filename}...")
    
#     plt.figure(figsize=(14, 6))
    
#     val_df = results_df[results_df['Data_Split'] == 'Validation']
#     plt.plot(val_df.index, val_df[ACTUAL_COL_NAME], label='Validation Actual', color='tab:blue', linewidth=2)
#     plt.plot(val_df.index, val_df[PREDICTED_COL_NAME], label='Validation Forecast', color='tab:orange', linestyle='--', linewidth=1.5)
    
#     test_df = results_df[results_df['Data_Split'] == 'Test']
#     plt.plot(test_df.index, test_df[ACTUAL_COL_NAME], label='Test Actual', color='tab:green', linewidth=2)
#     plt.plot(test_df.index, test_df[PREDICTED_COL_NAME], label='Test Forecast', color='tab:red', linestyle='--', linewidth=1)
    
#     plt.title('Footfalls Forecast (Fourier-AR-Trend Model)')
#     plt.xlabel('Index')
#     plt.ylabel(ACTUAL_COL_NAME)
#     plt.legend()
#     plt.grid(True, which='both', linestyle=':', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     plt.close() 
    
#     print("\n--- Final Model Metrics (AR-Fourier-Trend) ---")
#     print(f"Final Validation RMSE: {final_val_rmse:.4f}")
#     print(f"Final Validation MAE:  {final_val_mae:.4f}")
#     print(f"Final Validation MSE:  {final_val_mse:.4f}")
#     print(f"Final Test RMSE: {final_test_rmse:.4f}")
#     print(f"Final Test MAE:  {final_test_mae:.4f}")
#     print(f"Final Test MSE:  {final_test_mse:.4f}")
    
#     print(f"\n--- Process Complete ---")


# def create_fourier_covariates(series: TimeSeries, period: float, K: int) -> TimeSeries:
#     """
#     Generates Fourier series covariates based on the series' time index.
#     """
#     time_index = series.time_index
#     t = np.arange(len(time_index))
#     fourier_df = pd.DataFrame(index=time_index)
    
#     for k in range(1, K + 1):
#         omega = 2 * np.pi * k / period
#         sin_col = np.sin(omega * t)
#         cos_col = np.cos(omega * t)
#         fourier_df[f'sin_{k}_p{period}'] = sin_col
#         fourier_df[f'cos_{k}_p{period}'] = cos_col
        
#     return TimeSeries.from_dataframe(fourier_df)

# def create_trend_covariate(series: TimeSeries) -> TimeSeries:

#     time_index = series.time_index
#     trend = np.arange(len(time_index)) 
#     trend_df = pd.DataFrame(index=time_index, data={'linear_trend': trend})
#     return TimeSeries.from_dataframe(trend_df)


# def part4():
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(torch.cuda.is_available())
#     print(f"device: {device}")

#     optimal_config_values = {
#         'input_chunk_length': 120, 
#         'output_chunk_length': 1,
#         'TRAIN_RATIO': .7,
#         'VAL_RATIO': .2,
#         'batch_size': 64,
#         'model_type': 'LSTM',
#         'hidden_dim': 32, 
#         'n_layers': 1,
#         'dropout_rate': 0.4,
#         'epochs': 14, 
#         'learning_rate': 0.004,
#         'weight_decay': 0.001,
#         'grad_clip': 2
#     } 
    
#     torch.set_float32_matmul_precision('medium')
#     project_name = f'DSCC275P2'
#     run_name = f"{secrets.token_hex(4)}"
    
#     wandb.init(project=project_name, config=optimal_config_values, name=run_name)
#     config = wandb.config 
#     wandb_logger = WandbLogger(project=project_name, name=run_name, log_model=False)

#     airport_df = pd.read_csv("AirportFootfalls_data.csv").set_index('index')
#     series = TimeSeries.from_dataframe(
#         airport_df.reset_index(), 
#         'index', 
#         ORIGINAL_COL
#     )

#     PERIOD = 753 
#     K_FOURIER = 5
#     fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
#     trend_covariate = create_trend_covariate(series) # Uses the MODIFIED function
#     all_covariates = fourier_covariates.stack(trend_covariate)

#     # 2. Split Target Series
#     print("\n---  shapes: Initial Data Shapes ---")
#     print(f"Original 'series' shape:      (len={len(series)}, width={series.width})")
#     print(f"Original 'all_covariates' shape: (len={len(all_covariates)}, width={all_covariates.width})")
#     train_series, val_test_series = series.split_before(config.TRAIN_RATIO)
#     val_split_pos = int(len(val_test_series) * (config.VAL_RATIO / (1 - config.TRAIN_RATIO)))
#     val_split_point_index_value = val_test_series.time_index[val_split_pos]
#     val_series, test_series = val_test_series.split_before(val_split_point_index_value)

#     # 3. Split Covariates (at the exact same points as target)
#     print("\n---  shapes: After Target Split ---")
#     print(f"'train_series' shape: (len={len(train_series)}, width={train_series.width})")
#     print(f"'val_series' shape:   (len={len(val_series)}, width={val_series.width})")
#     print(f"'test_series' shape:  (len={len(test_series)}, width={test_series.width})")

#     train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
#     val_cov, test_cov = val_test_cov.split_before(val_series.end_time())
#     print("\n---  shapes: After Covariate Split ---")
#     print(f"'train_cov' shape: (len={len(train_cov)}, width={train_cov.width})")
#     print(f"'val_cov' shape:   (len={len(val_cov)}, width={val_cov.width})")
#     print(f"'test_cov' shape:  (len={len(test_cov)}, width={test_cov.width})")
#     # 4. Scale Target Series (Fit ONLY on train_series)
#     scaler = Scaler()
#     train_scaled = scaler.fit_transform(train_series)
#     val_scaled = scaler.transform(val_series)
#     test_scaled = scaler.transform(test_series)
    
#     # 5. Scale Covariates 
#     cov_scaler = Scaler() #SEPARATE scaler
#     train_cov_scaled = cov_scaler.fit_transform(train_cov)
#     val_cov_scaled = cov_scaler.transform(val_cov)
#     test_cov_scaled = cov_scaler.transform(test_cov)

#     # 6. Recombine Scaled Covariates for Darts
#     all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
    

#     print("\n---  shapes: AR-Fourier Model Inputs (to .fit()) ---")
#     print(f"Target (train_scaled):          (len={len(train_scaled)}, width={train_scaled.width})")
#     print(f"Covariates (all_covariates_scaled): (len={len(all_covariates_scaled)}, width={all_covariates_scaled.width})")
#     print("Training Fourier-AR-Trend model (LinearRegressionModel)...")
#     fourier_model = LinearRegressionModel(
#         lags=1, 
#         lags_future_covariates=[0],
#         output_chunk_length=1
#     )

#     fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)

    
#     print("Calculating training set residuals...")
#     fitted_train_values = fourier_model.historical_forecasts(
#         train_scaled,
#         future_covariates=all_covariates_scaled,
#         start=train_scaled.time_index[1],
#         verbose=False
#     )
#     train_residuals = train_scaled.slice_intersect(fitted_train_values) - fitted_train_values

#     print("Calculating validation set residuals...")
#     fitted_val_values = fourier_model.historical_forecasts(
#         train_scaled.append(val_scaled),
#         future_covariates=all_covariates_scaled, 
#         start=val_scaled.start_time(), 
#         verbose=False
#     )
#     val_residuals = val_scaled.slice_intersect(fitted_val_values) - fitted_val_values
    
    
#     # Standardize the residuals 
#     sklearn_scaler = StandardScaler()
#     residual_scaler = Scaler(sklearn_scaler)
    
#     # Fit residual scaler ONLY on training residuals
#     scaled_train_residuals = residual_scaler.fit_transform(train_residuals)
#     scaled_val_residuals = residual_scaler.transform(val_residuals)

#     print("\n---  shapes: LSTM Model Inputs (to .fit()) ---")
#     print(f"Target (scaled_train_residuals): (len={len(scaled_train_residuals)}, width={scaled_train_residuals.width})")
#     print(f"Validation (scaled_val_residuals): (len={len(scaled_val_residuals)}, width={scaled_val_residuals.width})")
#     print(f"   -> Darts will chunk these into samples of (input_chunk_length={config.input_chunk_length}, width={scaled_train_residuals.width})")

#     model = BlockRNNModel(
#         input_chunk_length=config.input_chunk_length,
#         output_chunk_length=config.output_chunk_length,
#         model=config.model_type,
#         hidden_dim=config.hidden_dim,
#         n_rnn_layers=config.n_layers,
#         dropout=config.dropout_rate,
#         batch_size=config.batch_size,
#         n_epochs=config.epochs,
#         optimizer_kwargs={'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        
#         pl_trainer_kwargs={
#             "accelerator": "gpu" if device.type == "cuda" else "cpu",
#             "devices": 1 if device.type == "cuda" else "auto",
#             "gradient_clip_val": config.grad_clip,
#             "logger": wandb_logger, 
#             "enable_model_summary": False 
#         },
        
#         force_reset=True
#     )

#     # Train RNN on scaled residuals
#     model.fit(
#         scaled_train_residuals, 
#         val_series=scaled_val_residuals,
#         verbose=True
#     )

#     print("\n---  shapes: LSTM Prediction Inputs (to .predict()) ---")
#     print(f"Val history (scaled_train_residuals): (len={len(scaled_train_residuals)}, width={scaled_train_residuals.width})")
#     print(f"   -> Darts will use the last (input_chunk_length={config.input_chunk_length}, width={scaled_train_residuals.width}) of this history to predict")
#     val_preds_lstm_scaled = model.predict(
#         n=len(val_scaled), 
#         series=scaled_train_residuals,
#         verbose=False
#     )
#     test_history_series = scaled_train_residuals.append(scaled_val_residuals)
#     print("\n---  shapes: LSTM Prediction Inputs (to .predict()) ---")
#     print(f"Test history (train+val residuals): (len={len(test_history_series)}, width={test_history_series.width})")
#     print(f"   -> Darts will use the last (input_chunk_length={config.input_chunk_length}, width={test_history_series.width}) of this history to predict")
#     test_preds_lstm_scaled = model.predict(
#         n=len(test_scaled), 
#         series=scaled_train_residuals.append(scaled_val_residuals), 
#         verbose=False
#     )
    
#     # 2. Un-standardize the residual predictions
#     val_preds_residuals = residual_scaler.inverse_transform(val_preds_lstm_scaled)
#     test_preds_residuals = residual_scaler.inverse_transform(test_preds_lstm_scaled)

#     print("Predicting seasonality/trend from Fourier-AR-Trend model")
#     val_preds_fourier_scaled = fourier_model.predict(
#         n=len(val_scaled),
#         series=train_scaled,
#         future_covariates=all_covariates_scaled, 
#         verbose=False
#     )
#     test_preds_fourier_scaled = fourier_model.predict(
#         n=len(test_scaled),
#         series=train_scaled.append(val_scaled),
#         future_covariates=all_covariates_scaled, 
#         verbose=False
#     )

#     # 4. Add predictions together (still in scaled terms of original series)
#     val_preds_total_scaled = val_preds_fourier_scaled + val_preds_residuals
#     test_preds_total_scaled = test_preds_fourier_scaled + test_preds_residuals

#     # 5. Unscale final predictions and actuals (using the *target* scaler)
#     val_preds_unscaled = scaler.inverse_transform(val_preds_total_scaled)
#     val_unscaled = scaler.inverse_transform(val_scaled)
    
#     preds_unscaled = scaler.inverse_transform(test_preds_total_scaled)
#     test_unscaled = scaler.inverse_transform(test_scaled)

#     # --- METRICS AND LOGGING ---
 
#     final_val_rmse = rmse(val_unscaled, val_preds_unscaled)
#     final_val_mae = mae(val_unscaled, val_preds_unscaled)
#     final_val_mse = mse(val_unscaled, val_preds_unscaled)
    
#     final_test_rmse = rmse(test_unscaled, preds_unscaled)
#     final_test_mae = mae(test_unscaled, preds_unscaled)
#     final_test_mse = mse(test_unscaled, preds_unscaled)

#     wandb.log({
#         'final/val_rmse': final_val_rmse,
#         'final/val_mae': final_val_mae,
#         'final/val_mse': final_val_mse,
#         'final/test_rmse': final_test_rmse,
#         'final/test_mae': final_test_mae,
#         'final/test_mse': final_test_mse 
#     })
    
    
#     all_actuals = val_unscaled.append(test_unscaled)
#     all_predictions = val_preds_unscaled.append(preds_unscaled)
    
#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)
    
#     print(f"Saving combined predictions and actuals to {output_dir}/predictions.csv...")
    
#     ACTUAL_COL_NAME = ORIGINAL_COL 
#     PREDICTED_COL_NAME = 'Predicted_Footfalls'

#     df_actuals = all_actuals.to_dataframe()
#     df_predictions = all_predictions.to_dataframe().rename(
#         columns={ACTUAL_COL_NAME: PREDICTED_COL_NAME}
#     )
#     results_df = pd.concat([df_actuals, df_predictions], axis=1)

#     val_end_date = val_unscaled.end_time()
#     results_df['Data_Split'] = np.where(results_df.index <= val_end_date, 'Validation', 'Test')
    
#     results_df.to_csv(os.path.join(output_dir, f"{run_name}.csv"), index_label='index')

#     print(f"Generating and saving plot to {output_dir}/{run_name}.png...")
    
#     plt.figure(figsize=(14, 6))
    
#     val_df = results_df[results_df['Data_Split'] == 'Validation']
#     plt.plot(val_df.index, val_df[ACTUAL_COL_NAME], label='Validation Actual', color='tab:blue', linewidth=2)
#     plt.plot(val_df.index, val_df[PREDICTED_COL_NAME], label='Validation Forecast', color='tab:orange', linestyle='--', linewidth=1.5)
    
#     test_df = results_df[results_df['Data_Split'] == 'Test']
#     plt.plot(test_df.index, test_df[ACTUAL_COL_NAME], label='Test Actual', color='tab:green', linewidth=2)
#     plt.plot(test_df.index, test_df[PREDICTED_COL_NAME], label='Test Forecast', color='tab:red', linestyle='--', linewidth=1)
    
#     plt.title('Hybrid Footfalls Forecast (Fourier-AR-Trend + LSTM on Scaled Residuals)')
#     plt.xlabel('Index')
#     plt.ylabel(ACTUAL_COL_NAME)
#     plt.legend()
#     plt.grid(True, which='both', linestyle=':', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{run_name}.png"))
#     plt.close() 
    
#     print("\n--- Darts + PyTorch Lightning + WandB Complete (Hybrid Model) ---")
#     print(f"Final Validation RMSE: {final_val_rmse:.4f}")
#     print(f"Final Test RMSE: {final_test_rmse:.4f}")
#     print(f"WandB run logged under: {run_name}")
    
#     wandb.finish()


    
