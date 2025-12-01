import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import rmse
from darts.models import  LinearRegressionModel, NaiveSeasonal 



FILE_NAME = "individual_household_electric_power_consumption.csv"

ORIGINAL_COL = 'Global_active_power'
STATIONARY_COL = 'detrended_deseasonalized'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20



def create_fourier_covariates(series: TimeSeries, period: float, K: int) -> TimeSeries:
    """
    Generates Fourier series covariates based on the series' time index.
    (Required by Darts models in part2, part3, part4)
    """
    time_index = series.time_index
    t = np.arange(len(time_index))
    fourier_df = pd.DataFrame(index=time_index)
    
    for k in range(1, K + 1):
        omega = 2 * np.pi * k / period
        sin_col = np.sin(omega * t)
        cos_col = np.cos(omega * t)
        fourier_df[f'sin_{k}_p{period}'] = sin_col
        fourier_df[f'cos_{k}_p{period}'] = cos_col
        
    return TimeSeries.from_dataframe(fourier_df)

def part2(clean_df, calculated_period):
    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    
    param_file_path = "sarima_params.json"
    
    # --- 1. Split Data (Darts style) ---
    N_total = len(series)
    N_train = int(N_total * TRAIN_RATIO)
    N_val = int(N_total * VAL_RATIO)
    
    # Split into Train, Val, Test series
    train_series, val_test_series = series.split_before(TRAIN_RATIO)
    val_split_point_index_value = val_test_series.time_index[N_val]
    val_series, test_series = val_test_series.split_before(val_split_point_index_value)
    
    print(f"Using Seasonal Period (calculated_period) for Fourier/Lag: {calculated_period}")
    print(f"Train size: {len(train_series)}, Val size: {len(val_series)}")
    
    # --- 2. Set Experiment Parameters ---
    current_LAG = calculated_period # The lag used for the AR feature
    MAX_K = 20 
    
    print(f"Running evaluation for {current_LAG}-step forecast with K=1 to {MAX_K}...")

    results_val = []
    
    # 3. Hyperparameter tuning loop for K (1 to 10)
    for K in range(1, MAX_K + 1):
        fourier_covariates = create_fourier_covariates(series, period=calculated_period, K=K)
        

        train_cov, val_test_cov = fourier_covariates.split_before(train_series.end_time())
        val_cov, test_cov = val_test_cov.split_before(val_series.end_time())
        
        all_covariates = train_cov.append(val_cov).append(test_cov)
            
        # --- Model Training ---
        # Use Darts LinearRegressionModel which handles the AR lag automatically
        model = LinearRegressionModel(
            lags=int(current_LAG),            # Auto-regressive lag (seasonal)
            lags_future_covariates=[0],      # Use covariates at time t to predict y_t
            output_chunk_length=1            # Predict one step at a time
        )
        
        # We only fit on the training data
        model.fit(train_series, future_covariates=all_covariates)
        
        # --- Model Evaluation (Validation Set) ---
        # Generate N_val predictions starting from the end of the train set
        val_preds = model.predict(
            n=len(val_series),
            series=train_series, # History series
            future_covariates=all_covariates
        )
        
        # Naive Seasonal Baseline (consistent with Naive Seasonal in part3)
        naive_model = NaiveSeasonal(K=int(current_LAG)) 
        naive_model.fit(train_series)
        naive_preds = naive_model.predict(
                n=len(val_series) 
            )
        
        # Ensure alignment (Darts handles this well, but good practice)
        val_series_aligned = val_series.slice_intersect(val_preds)
        val_preds_aligned = val_preds.slice_intersect(val_series)
        naive_preds_aligned = naive_preds.slice_intersect(val_series)


        # Calculate Darts Metrics
        model_rmse = rmse(val_series_aligned, val_preds_aligned)
        naive_rmse = rmse(val_series_aligned, naive_preds_aligned)
        
        val_metrics = {
            'K': K,
            'model_rmse': model_rmse, 
            'naive_rmse': naive_rmse,
        }
        results_val.append(val_metrics)
        
        print(f" K={K}: Val RMSE = {model_rmse:.4f}, Naive RMSE = {naive_rmse:.4f}")
            
    val_df = pd.DataFrame(results_val)
    
    # 4. Find the best K based on Validation RMSE
    best_k_row = val_df.iloc[val_df['model_rmse'].argmin()]
    best_k = int(best_k_row['K'])

    # Log and print the best K
    best_val_rmse = best_k_row['model_rmse']
    print(f"\n--- Sweep Complete (L={current_LAG}) ---")
    print(f"Optimal K found based on Validation RMSE: K={best_k} (RMSE: {best_val_rmse:.4f})")
    print(val_df) 
    
    # 5. Save results
    sarima_params = {}
    if os.path.exists(param_file_path):
        with open(param_file_path, 'r') as f:
            sarima_params = json.load(f)
            
    sarima_params['seasonal_period'] = int(calculated_period) 
    sarima_params['fourier_k'] = int(best_k)
    with open(param_file_path, 'w') as f:
        json.dump(sarima_params, f, indent=4)
        
    best_k_df = pd.DataFrame([{'Optimal Fourier Pairs (K)': best_k}])
    print(best_k_df)
    
    return best_k
