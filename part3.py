import json
import os
import secrets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import  rmse
from darts.models import LinearRegressionModel, NaiveSeasonal 
from darts import concatenate

from part2 import  create_fourier_covariates

# --- Constants ---
FILE_NAME = "individual_household_electric_power_consumption.csv"
ORIGINAL_COL = 'Global_active_power'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
PERIOD_STEPS = 96  # 24 hours * 4 quarters = 96 steps

def create_trend_covariate(series: TimeSeries) -> TimeSeries:
    """Creates a simple linear trend covariate."""
    time_index = series.time_index
    trend = np.arange(len(time_index)) 
    trend_df = pd.DataFrame(index=time_index, data={'linear_trend': trend})
    return TimeSeries.from_dataframe(trend_df)

def part3(clean_df, calculated_period):
    print("\n=== PART 3: Fourier-AR-Trend Rolling Forecast ===")
    
    # Ensure no gaps in index
    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    
    # 1. Configuration 
    run_name = f"forecast_{secrets.token_hex(4)}"
  
    # Better logic: Use passed period, fallback to JSON, fallback to Constant
    try:
        with open("sarima_params.json", 'r') as f:
            sarima_params = json.load(f)
            # Prioritize JSON if exists, otherwise use calculated_period
            PERIOD = sarima_params.get('seasonal_period', calculated_period) 
            K_FOURIER = sarima_params.get('fourier_k', 10)
    except FileNotFoundError:
        PERIOD = calculated_period
        K_FOURIER = 10
    
    print(f"Using Seasonal Period: {PERIOD}, Fourier K: {K_FOURIER}")

    # 1.2 Create Covariates
    fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
    trend_covariate = create_trend_covariate(series)
    all_covariates = fourier_covariates.stack(trend_covariate)

    # 2. Split Target Series
    train_series, val_test_series = series.split_before(TRAIN_RATIO)
    
    # Calculate split point ensuring we don't break strict time indices
    val_len = int(len(val_test_series) * (VAL_RATIO / (1 - TRAIN_RATIO)))
    val_series, test_series = val_test_series.split_after(val_len) # split_after is often cleaner here

    # 3. Split Covariates (Aligning with target splits)
    train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
    val_cov, test_cov = val_test_cov.split_before(val_series.end_time())

    # 4. Naive Baseline
    print("\n--- Calculating Naive Seasonal Baseline ---")
    naive_model = NaiveSeasonal(K=PERIOD) 
    naive_model.fit(train_series)
    
    naive_test_preds = naive_model.historical_forecasts(
        test_series,
        start=test_series.start_time(),
        forecast_horizon=1,
        stride=1,
        verbose=False
    )
    
    # Align for metric calculation
    test_series_aligned_naive = test_series.slice_intersect(naive_test_preds)
    naive_test_rmse = rmse(test_series_aligned_naive, naive_test_preds)
    print(f"Naive Baseline Test RMSE: {naive_test_rmse:.4f}")

    # --- 5. Scaling ---
    # Fit ONLY on train
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)
    
    cov_scaler = Scaler()
    train_cov_scaled = cov_scaler.fit_transform(train_cov)
    val_cov_scaled = cov_scaler.transform(val_cov)
    test_cov_scaled = cov_scaler.transform(test_cov)

    # Recombine scaled covariates for the forecasting context
    all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
    
    # --- 6. AR+FOURIER MODEL ---
    print("\n--- Training Fourier-AR-Trend model ---")
    
    fourier_model = LinearRegressionModel(
        lags=[-PERIOD], # Looking back exactly one period (e.g. 24hrs)
        lags_future_covariates=[0], 
        output_chunk_length=PERIOD 
    )
    fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)
    
    print(f"Calculating {PERIOD}-step rolling forecast (Validation)...")
    
    # Helper to concat safely
    train_val_scaled = train_scaled.append(val_scaled)
    train_val_test_scaled = train_val_scaled.append(test_scaled)

    # 1. Validation Forecast
    val_preds_fourier_scaled = fourier_model.historical_forecasts(
        series=train_val_scaled, 
        future_covariates=all_covariates_scaled,
        start=val_scaled.start_time(), 
        forecast_horizon=PERIOD, 
        stride=PERIOD, 
        retrain=False,
        last_points_only=False, 
        verbose=False
    )

    if isinstance(val_preds_fourier_scaled, list):
         val_preds_fourier_scaled = concatenate(val_preds_fourier_scaled, ignore_time_axis=True)
    
    print(f"Calculating {PERIOD}-step rolling forecast (Test)...")
    
    # 2. Test Forecast
    test_preds_fourier_scaled = fourier_model.historical_forecasts(
        series=train_val_test_scaled, 
        future_covariates=all_covariates_scaled, 
        start=test_scaled.start_time(), 
        forecast_horizon=PERIOD,
        stride=PERIOD, 
        retrain=False, 
        last_points_only=False,
        verbose=False
    )
    if isinstance(test_preds_fourier_scaled, list):
        test_preds_fourier_scaled = concatenate(test_preds_fourier_scaled, ignore_time_axis=True)

    # Align predictions with actuals
    val_preds_fourier_scaled = val_preds_fourier_scaled.slice_intersect(val_scaled)
    val_scaled_aligned = val_scaled.slice_intersect(val_preds_fourier_scaled)
    
    test_preds_fourier_scaled = test_preds_fourier_scaled.slice_intersect(test_scaled)
    test_scaled_aligned = test_scaled.slice_intersect(test_preds_fourier_scaled)

    # Unscale
    val_preds_unscaled = scaler.inverse_transform(val_preds_fourier_scaled)
    val_unscaled_aligned = scaler.inverse_transform(val_scaled_aligned)
    
    test_preds_unscaled = scaler.inverse_transform(test_preds_fourier_scaled)
    test_unscaled_aligned = scaler.inverse_transform(test_scaled_aligned)

    # Metrics
    final_val_rmse = rmse(val_unscaled_aligned, val_preds_unscaled)
    final_test_rmse = rmse(test_unscaled_aligned, test_preds_unscaled)

    # --- 7. Save Results and Plot ---
    # Convert to Pandas DataFrames individually first to avoid continuity errors
    val_act_df = val_unscaled_aligned.to_dataframe()
    test_act_df = test_unscaled_aligned.to_dataframe()
    
    val_pred_df = val_preds_unscaled.to_dataframe()
    test_pred_df = test_preds_unscaled.to_dataframe()

    # Concatenate using Pandas (tolerant of gaps)
    df_actuals = pd.concat([val_act_df, test_act_df])
    df_predictions = pd.concat([val_pred_df, test_pred_df]).rename(columns={ORIGINAL_COL: 'Predicted'})
    
    results_df = pd.concat([df_actuals, df_predictions], axis=1)
    
    # Add logic to label Split
    results_df['Data_Split'] = np.where(results_df.index <= val_unscaled_aligned.end_time(), 'Validation', 'Test')
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{run_name}.csv")
    results_df.to_csv(csv_filename, index_label='index')

    # Plot
    plot_filename = os.path.join(output_dir, f"{run_name}.png")
    plt.figure(figsize=(14, 6))
    
    val_df = results_df[results_df['Data_Split'] == 'Validation']
    plt.plot(val_df.index, val_df[ORIGINAL_COL], label='Validation Actual', color='tab:blue', alpha=0.7)
    plt.plot(val_df.index, val_df['Predicted'], label='Validation Forecast', color='orange', linestyle='--', alpha=0.8)
    
    test_df = results_df[results_df['Data_Split'] == 'Test']
    plt.plot(test_df.index, test_df[ORIGINAL_COL], label='Test Actual', color='green', alpha=0.7)
    plt.plot(test_df.index, test_df['Predicted'], label='Test Forecast', color='red', linestyle='--', alpha=0.8)
    
    plt.title(f'Rolling {PERIOD_STEPS}-step Forecast (Linear Regression w/ Fourier)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    
    print("\n--- Final Metrics ---")
    print(f"Validation RMSE: {final_val_rmse:.4f}")
    print(f"Test RMSE:       {final_test_rmse:.4f}")
    if final_test_rmse < naive_test_rmse:
        print(f"-> Improvement over Naive: {100 * (naive_test_rmse - final_test_rmse) / naive_test_rmse:.2f}%")
    print(f"Plot saved to {plot_filename}")
