import json
import os
import secrets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, rmse, mape
from darts.models import BlockRNNModel, LinearRegressionModel
from darts import concatenate
from part1 import part1
from part2 import part2, create_fourier_covariates
from part3 import create_trend_covariate, part3

# --- Constants ---
FILE_NAME = "individual_household_electric_power_consumption.csv"
ORIGINAL_COL = 'Global_active_power'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
PERIOD_STEPS = 96  

def part4(clean_df, calculated_period):
    print("\n=== PART 4: Residual Learning (LSTM) ===")
    
    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = {
        'input_chunk_length': 96,
        'output_chunk_length': 96,
        'TRAIN_RATIO': .7,
        'VAL_RATIO': .2,
        'batch_size': 32,
        'model_type': 'LSTM',
        'hidden_dim': 32,
        'n_layers': 1,
        'dropout_rate': 0.0,
        'epochs': 15,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grad_clip': 0.5
    }

    # Setup WandB
    project_name = f'DSCC275FINAL'
    run_name = f"LSTM_Resid_{secrets.token_hex(4)}"
    wandb.init(project=project_name, config=config, name=run_name, reinit=True)
    
    with open("sarima_params.json", 'r') as f:
        sarima_params = json.load(f)
    PERIOD = sarima_params.get('seasonal_period', 96)
    K_FOURIER = sarima_params.get('fourier_k', 10)

    fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
    trend_covariate = create_trend_covariate(series)
    all_covariates = fourier_covariates.stack(trend_covariate)

    # Splits
    train_series, val_test_series = series.split_before(config['TRAIN_RATIO'])
    val_split_pos = int(len(val_test_series) * (config['VAL_RATIO'] / (1 - config['TRAIN_RATIO'])))
    val_split_point = val_test_series.time_index[val_split_pos]
    val_series, test_series = val_test_series.split_before(val_split_point)

    # Scalers (Main Series)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)
    
    cov_scaler = Scaler()
    all_covariates_scaled = cov_scaler.fit_transform(all_covariates)

    # 1. Train Linear Model (Base)
    print("Training Base Linear Model...")
    fourier_model = LinearRegressionModel(
        lags=[-PERIOD], 
        lags_future_covariates=[0], 
        output_chunk_length=PERIOD
    )
    fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)

    # 2. Calculate Residuals (In-Sample)
    print("Calculating Residuals...")
    
    # Train Residuals
    fitted_train_list = fourier_model.historical_forecasts(
        train_scaled,
        future_covariates=all_covariates_scaled,
        start=train_scaled.start_time() + pd.Timedelta(hours=24), 
        stride=PERIOD,
        retrain=False,
        last_points_only=False,
        verbose=False
    )
    fitted_train = concatenate(fitted_train_list, ignore_time_axis=True)
    train_residuals = train_scaled.slice_intersect(fitted_train) - fitted_train

    # Val Residuals
    fitted_val_list = fourier_model.historical_forecasts(
        train_scaled.append(val_scaled),
        future_covariates=all_covariates_scaled,
        start=val_scaled.start_time(),
        stride=PERIOD,
        retrain=False,
        last_points_only=False,
        verbose=False
    )
    fitted_val = concatenate(fitted_val_list, ignore_time_axis=True)
    val_residuals = val_scaled.slice_intersect(fitted_val) - fitted_val

    # Scale Residuals (StandardScaler)
    res_scaler = Scaler(scaler=StandardScaler()) 
    train_res_scaled = res_scaler.fit_transform(train_residuals)
    val_res_scaled = res_scaler.transform(val_residuals)

    # 3. Train LSTM on Residuals
    print("Training LSTM on Residuals...")
    wandb_logger = WandbLogger(project=project_name, name=run_name, log_model=False)
    
    lstm_model = BlockRNNModel(
        input_chunk_length=config['input_chunk_length'],
        output_chunk_length=config['output_chunk_length'],
        model='LSTM',
        hidden_dim=config['hidden_dim'],
        n_rnn_layers=config['n_layers'],
        dropout=config['dropout_rate'],
        batch_size=config['batch_size'],
        n_epochs=config['epochs'],
        optimizer_kwargs={'lr': config['learning_rate']},
        pl_trainer_kwargs={
            "accelerator": "gpu" if device.type == "cuda" else "cpu",
            "devices": 1,
            "gradient_clip_val": config['grad_clip'],
            "logger": wandb_logger,
            "enable_model_summary": False
        },
        random_state=42
    )

    lstm_model.fit(
        train_res_scaled, 
        val_series=val_res_scaled, 
        verbose=True
    )

    # --- Helper: Hybrid Prediction Logic ---
    def predict_hybrid(target_series, history_series, res_history_series):
        # A. Base Forecast
        base_pred_list = fourier_model.historical_forecasts(
            series=history_series.append(target_series),
            future_covariates=all_covariates_scaled,
            start=target_series.start_time(),
            forecast_horizon=PERIOD,
            stride=PERIOD,
            retrain=False,
            last_points_only=False,
            verbose=False
        )
        base_pred = concatenate(base_pred_list, ignore_time_axis=True)
        
        # B. Residual Forecast
        n_steps = len(base_pred)
        res_pred = lstm_model.predict(n=n_steps, series=res_history_series)
        
        # Unscale residuals
        res_pred_unscaled = res_scaler.inverse_transform(res_pred)
        
        # Combine
        base_values = base_pred.values()
        res_values = res_pred_unscaled.values()
        min_len = min(len(base_values), len(res_values))
        total_values = base_values[:min_len] + res_values[:min_len]
        
        total_pred_scaled = TimeSeries.from_times_and_values(
            times=base_pred.time_index[:min_len],
            values=total_values,
            columns=base_pred.columns
        )
        return scaler.inverse_transform(total_pred_scaled)

    # --- Helper: Save Plot ---
    def save_plot(actual, pred, title, filename):
        plt.figure(figsize=(15, 6))
        actual.plot(label='Actual', color='black', alpha=0.7)
        pred.plot(label='Hybrid Prediction', color='blue', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close()

    # --- Helper: Save CSV ---
    def save_csv(actual, pred, filename):
        intersect = actual.slice_intersect(pred)
        pred_intersect = pred.slice_intersect(actual)
        
        df = pd.DataFrame({
            'datetime': intersect.time_index,
            'actual': intersect.values().flatten(),
            'prediction': pred_intersect.values().flatten()
        })
        df.to_csv(filename, index=False)
        print(f"CSV saved to {filename}")

    # --- UPDATED Helper: Print & Log Metrics ---
    def print_metrics(actual, pred, stage_name):
        _rmse = rmse(actual, pred)
        _mae = mae(actual, pred)
        _mse = mse(actual, pred)
        _mape = mape(actual, pred)
        
        print(f"\n--- {stage_name} Metrics ---")
        print(f"RMSE: {_rmse:.4f}")
        print(f"MAE:  {_mae:.4f}")
        print(f"MSE:  {_mse:.4f}")
        print(f"MAPE: {_mape:.4f}%")
        
        # --- NEW: Log to WandB ---
        # We lowercase the stage_name (e.g., 'validation' or 'test') for cleaner charts
        wandb_metrics = {
            f"final/{stage_name.lower()}_rmse": _rmse,
            f"final/{stage_name.lower()}_mae": _mae,
            f"final/{stage_name.lower()}_mse": _mse,
            f"final/{stage_name.lower()}_mape": _mape
        }
        wandb.log(wandb_metrics)
        print(f"Logged {stage_name} metrics to WandB")

        return {"RMSE": _rmse, "MAE": _mae, "MSE": _mse, "MAPE": _mape}

    # ==========================================
    # 4. Generate & Save Validation Predictions
    # ==========================================
    print("\nGenerating Validation Predictions...")
    val_pred_hybrid = predict_hybrid(val_scaled, train_scaled, train_res_scaled)
    
    val_pred_hybrid = val_pred_hybrid.slice_intersect(val_series)
    val_actuals = val_series.slice_intersect(val_pred_hybrid)

    print_metrics(val_actuals, val_pred_hybrid, "val")
    save_plot(val_actuals, val_pred_hybrid, "Validation: Actual vs Hybrid Prediction", "val_prediction_plot.png")
    save_csv(val_actuals, val_pred_hybrid, "val_predictions.csv")

    # ==========================================
    # 5. Generate & Save Test Predictions
    # ==========================================
    print("\nGenerating Test Predictions...")
    full_res_history = concatenate([train_res_scaled, val_res_scaled], ignore_time_axis=True)
    train_val_history = concatenate([train_scaled, val_scaled], ignore_time_axis=True)
    
    test_pred_hybrid = predict_hybrid(test_scaled, train_val_history, full_res_history)
    
    test_pred_hybrid = test_pred_hybrid.slice_intersect(test_series)
    test_actuals = test_series.slice_intersect(test_pred_hybrid)

    print_metrics(test_actuals, test_pred_hybrid, "test")
    save_plot(test_actuals, test_pred_hybrid, "Test: Actual vs Hybrid Prediction", "test_prediction_plot.png")
    save_csv(test_actuals, test_pred_hybrid, "test_predictions.csv")
    
    wandb.finish()

if __name__ == "__main__":
    full_data_df, calculated_period = part1() 
    part2(full_data_df, calculated_period)
    part3(full_data_df, calculated_period)
    part4(full_data_df, calculated_period)
