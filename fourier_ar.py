import json
import os
import secrets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse
from darts.models import  LinearRegressionModel
from tqdm import tqdm
from load_and_visualize import load


def create_fourier_covariates(series: TimeSeries, period: float, K: int) -> TimeSeries:
    """
    Generates Fourier series covariates based on the series' time index.
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

def create_trend_covariate(series: TimeSeries) -> TimeSeries:

    time_index = series.time_index
    trend = np.arange(len(time_index)) 
    trend_df = pd.DataFrame(index=time_index, data={'linear_trend': trend})
    return TimeSeries.from_dataframe(trend_df)


def run_analysis():
    output_dir = "fourier_ar"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Metadata
    json_path = "load_and_visualize/output.json"
    meta_data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
    column_stats = meta_data.get('column_stats', {})
    
    # 2. Load Data
    print("Loading data...")
    whole_power_df, numeric_cols = load()
    
    # Pre-processing
    whole_power_df = whole_power_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    whole_power_df = whole_power_df.resample('min').mean()
    whole_power_df = whole_power_df.interpolate(method='linear')
    
    # Config
    config = {'TRAIN_RATIO': .7, 'VAL_RATIO': .2} 
    torch.set_float32_matmul_precision('medium')
    
    eval_horizons = [1, 5, 15, 60, 300]
    max_horizon = max(eval_horizons) # 300
    
    # Master list to store metrics for all columns
    all_metrics_report = []

    for target_col in tqdm(numeric_cols, desc="Forecasting Targets", position=0, leave=True):
  
        # Generate a unique ID for this column run
        run_name = f"{target_col}_{secrets.token_hex(2)}"
        
        print(f"\nProcessing target: {target_col}")
        
        series = TimeSeries.from_dataframe(
            whole_power_df, 
            value_cols=target_col,
            freq='min'
        )

        PERIOD = column_stats.get('preiodgram_seasonal_lag', 1440)
        K_FOURIER = 5
        fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
        trend_covariate = create_trend_covariate(series)
        all_covariates = fourier_covariates.stack(trend_covariate)

        # 2. Split Target Series
        train_series, val_test_series = series.split_before(config['TRAIN_RATIO'])
        val_split_pos = int(len(val_test_series) * (config['VAL_RATIO'] / (1 - config['TRAIN_RATIO'])))
        val_split_point_index_value = val_test_series.time_index[val_split_pos]
        val_series, test_series = val_test_series.split_before(val_split_point_index_value)

        # 3. Split Covariates
        train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
        val_cov, test_cov = val_test_cov.split_before(val_series.end_time())

        # 4. Scaling
        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_series)
        val_scaled = scaler.transform(val_series)
        test_scaled = scaler.transform(test_series)
        
        cov_scaler = Scaler() 
        train_cov_scaled = cov_scaler.fit_transform(train_cov)
        val_cov_scaled = cov_scaler.transform(val_cov)
        test_cov_scaled = cov_scaler.transform(test_cov)

        all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
        
        # --- 5. TRAINING ---
        print("\n--- Training Fourier-AR-Trend model ---")
        fourier_model = LinearRegressionModel(
            lags=1, 
            lags_future_covariates=[0],
            output_chunk_length=60, 
            n_jobs=8
        )

        fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)
        
        # --- 6. PREDICTION (Optimization applied here) ---
        
        # A. Predict Validation (Necessary for the plot)
        # We need to predict 'len(val)' steps starting from end of train
        print("\n--- Pred Val Fourier-AR-Trend model ---")
        val_preds_scaled = fourier_model.predict(
            n=len(val_series),
            series=train_scaled,
            future_covariates=all_covariates_scaled
            )
        
        # B. Predict Test (Max Horizon)
        # We predict 300 steps starting from end of validation
        print("\n--- Pred Test Fourier-AR-Trend model ---")
        test_preds_scaled_full = fourier_model.predict(
            n=max_horizon,
            series=train_scaled.append(val_scaled),
            future_covariates=all_covariates_scaled
            )

        # Unscale
        val_preds_unscaled = scaler.inverse_transform(val_preds_scaled)
        val_unscaled = scaler.inverse_transform(val_scaled)
        
        preds_unscaled_full = scaler.inverse_transform(test_preds_scaled_full)
        test_unscaled_full = scaler.inverse_transform(test_scaled) # Full test set

        # --- 7. METRICS SLICING ---
        for h in eval_horizons:
            print(h)
            # Slice the prediction and actuals to length h
            # (Handle edge case if test set is shorter than 300)
            slice_len = min(h, len(test_unscaled_full))
            
            pred_h = preds_unscaled_full[:slice_len]
            actual_h = test_unscaled_full[:slice_len]
            
            rmse_val = rmse(actual_h, pred_h)
            mae_val = mae(actual_h, pred_h)
            mse_val = mse(actual_h, pred_h)
            
            all_metrics_report.append({
                'target': target_col,
                'horizon': h,
                'rmse': rmse_val,
                'mae': mae_val,
                'mse': mse_val
            })
        
        # --- 8. SAVING CSV & PLOT ---
        
        # Combine actuals
        # Note: We only append the *full* test prediction (max horizon) to the CSV
        # to keep the file clean.
        all_actuals = val_unscaled.append(test_unscaled_full)
        all_predictions = val_preds_unscaled.append(preds_unscaled_full)
        
        csv_filename = os.path.join(output_dir, f"forecast_{run_name}.csv")
        plot_filename = os.path.join(output_dir, f"forecast_{run_name}.png")
        
        ACTUAL_COL_NAME = target_col 
        PREDICTED_COL_NAME = 'Predicted_target_col'

        df_actuals = all_actuals.to_dataframe()
        df_predictions = all_predictions.to_dataframe().rename(
            columns={ACTUAL_COL_NAME: PREDICTED_COL_NAME}
        )
        results_df = pd.concat([df_actuals, df_predictions], axis=1)

        val_end_date = val_unscaled.end_time()
        results_df['Data_Split'] = np.where(results_df.index <= val_end_date, 'Validation', 'Test')
        
        results_df.to_csv(csv_filename, index_label='index')

        # Plotting
        plt.figure(figsize=(14, 6))
        
        val_df = results_df[results_df['Data_Split'] == 'Validation']
        plt.plot(val_df.index, val_df[ACTUAL_COL_NAME], label='Validation Actual', color='tab:blue', linewidth=2)
        plt.plot(val_df.index, val_df[PREDICTED_COL_NAME], label='Validation Forecast', color='tab:orange', linestyle='--', linewidth=1.5)
        
        test_df = results_df[results_df['Data_Split'] == 'Test']
        # Only plot up to max_horizon
        test_df = test_df.iloc[:max_horizon] 
        
        plt.plot(test_df.index, test_df[ACTUAL_COL_NAME], label='Test Actual', color='tab:green', linewidth=2)
        plt.plot(test_df.index, test_df[PREDICTED_COL_NAME], label='Test Forecast', color='tab:red', linestyle='--', linewidth=1)
        
        plt.title(f'{target_col} Forecast (Fourier-AR-Trend)')
        plt.xlabel('Index')
        plt.ylabel(ACTUAL_COL_NAME)
        plt.legend()
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close() 

    # --- 9. SAVE FINAL METRICS REPORT ---
    metrics_df = pd.DataFrame(all_metrics_report)
    metrics_path = os.path.join(output_dir, "all_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n--- Processing Complete. Metrics saved to {metrics_path} ---")

if __name__ == "__main__":
    run_analysis()