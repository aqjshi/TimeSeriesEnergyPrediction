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
from darts.metrics import mae, mse, rmse,  smape
from darts.models import BlockRNNModel, LinearRegressionModel
from darts import concatenate
from part1 import part1
from part2 import part2, create_fourier_covariates
from part3 import create_trend_covariate, part3
from pytorch_lightning.callbacks import EarlyStopping
# --- Constants ---
FILE_NAME = "individual_household_electric_power_consumption.csv"
ORIGINAL_COL = 'Global_active_power'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
PERIOD_STEPS = 96  


def part4(clean_df, calculated_period):
    print("\n=== PART 4: Pure LSTM MIMO (Paper Implementation) ===")
    
    # 1. Prepare Series
    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Config (Paper-aligned)
    config = {
            'input_chunk_length': 96,   # Lookback: 24 hours
            'output_chunk_length': 96,  # Forecast: 24 hours (MIMO)
            'TRAIN_RATIO': 0.7,
            'VAL_RATIO': 0.2,
            'batch_size': 64,
            'model_type': 'LSTM',
            'hidden_dim': 64,
            'n_layers': 2,
            'dropout_rate': 0.1, 
            'epochs': 6,       
            'learning_rate': 0.0001,
            'grad_clip': 0.5
        }

    # Setup WandB
    project_name = f'DSCC275FINAL'
    run_name = f"Pure_LSTM_MIMO_{secrets.token_hex(4)}"
    wandb.init(project=project_name, config=config, name=run_name, reinit=True)
    
    # 3. Covariates (Essential for LSTMs to "know" time)
    # Even "Pure" LSTMs need time embeddings to capture seasonality
    with open("sarima_params.json", 'r') as f:
        sarima_params = json.load(f)
    PERIOD = sarima_params.get('seasonal_period', 96)
    K_FOURIER = sarima_params.get('fourier_k', 10)

    fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
    trend_covariate = create_trend_covariate(series)
    all_covariates = fourier_covariates.stack(trend_covariate)

    # 4. Splits
    train_series, val_test_series = series.split_before(config['TRAIN_RATIO'])
    val_split_pos = int(len(val_test_series) * (config['VAL_RATIO'] / (1 - config['TRAIN_RATIO'])))
    val_split_point = val_test_series.time_index[val_split_pos]
    val_series, test_series = val_test_series.split_before(val_split_point)

    # 5. Scaling (Standardization is crucial for LSTMs)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)
    
    cov_scaler = Scaler()
    all_covariates_scaled = cov_scaler.fit_transform(all_covariates)

    # 6. Initialize LSTM Model
    # BlockRNNModel creates the "Many-to-Many" (MIMO) structure
    wandb_logger = WandbLogger(project=project_name, name=run_name, log_model=False)
    early_stopper = EarlyStopping(
        monitor="val_loss",     # Watch the validation loss
        patience=3,             # Stop if it doesn't improve for 3 epochs
        min_delta=0.0001,       # Minimum change to count as an improvement
        mode='min'              # We want the loss to be minimized
    )
    lstm_model = BlockRNNModel(
       input_chunk_length=config['input_chunk_length'],
        output_chunk_length=config['output_chunk_length'],
        model='LSTM',
        hidden_dim=config['hidden_dim'],
        n_rnn_layers=config['n_layers'],
        dropout=0.2,            # <--- INCREASE THIS (See point #2 below)
        batch_size=config['batch_size'],
        n_epochs=50,            # You can set this high now, EarlyStopping will cut it short
        optimizer_kwargs={'lr': config['learning_rate']},
        pl_trainer_kwargs={
            "accelerator": "gpu" if device.type == "cuda" else "cpu",
            "devices": 1,
            "gradient_clip_val": config['grad_clip'],
            "logger": wandb_logger,
            "callbacks": [early_stopper]  # <--- ADD THIS LINE
        },
        random_state=42
    )
    # 7. Train on RAW DATA (No Residuals)
    print(f"Training Pure LSTM on {len(train_scaled)} samples...")
    lstm_model.fit(
        train_scaled, 
        future_covariates=all_covariates_scaled,
        val_series=val_scaled, 
        val_future_covariates=all_covariates_scaled,
        verbose=True
    )

    # --- Helper: Save Plot ---
    def save_plot(actual, pred, title, filename):
        plt.figure(figsize=(15, 6))
        actual.plot(label='Actual', color='black', alpha=0.5)
        pred.plot(label='LSTM Prediction', color='blue', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close()

    # --- Helper: Calculate & Log Metrics ---
    def log_metrics(actual, pred, stage_name):
        _rmse = rmse(actual, pred)
        _mae = mae(actual, pred)
        _mse = mse(actual, pred)
        _smape = smape(actual, pred)
        
        print(f"\n--- {stage_name} Metrics ---")
        print(f"RMSE: {_rmse:.4f}")
        print(f"MAE:  {_mae:.4f}")
        print(f"SMAPE: {_smape:.4f}%")
        
        wandb.log({
            f"final/{stage_name.lower()}_rmse": _rmse,
            f"final/{stage_name.lower()}_mae": _mae,
            f"final/{stage_name.lower()}_mse": _mse,
            f"final/{stage_name.lower()}_smape": _smape
        })
        return _rmse

    # ==========================================
    # 8. Rolling Validation Forecast
    # ==========================================
    print("\nGenerating Validation Predictions (Rolling Window)...")
    # This matches the "Day-Ahead" scenario:
    # We predict 96 steps, then step forward 96 steps, exposing the actuals to the model
    val_pred = lstm_model.historical_forecasts(
        series=train_scaled.append(val_scaled), # Full history available up to current step
        future_covariates=all_covariates_scaled,
        start=val_scaled.start_time(),
        forecast_horizon=config['output_chunk_length'], # Predict 24h
        stride=config['output_chunk_length'],           # Move forward 24h
        retrain=False,
        verbose=True,
        last_points_only=False
    )
    # Concatenate the chunks into one continuous series
    if isinstance(val_pred, list):
        val_pred = concatenate(val_pred, ignore_time_axis=True)
    
    val_pred = scaler.inverse_transform(val_pred)
    val_actuals = val_series.slice_intersect(val_pred)
    
    log_metrics(val_actuals, val_pred, "val")
    save_plot(val_actuals, val_pred, "Validation: Pure LSTM MIMO", "val_prediction_plot.png")

    # ==========================================
    # 9. Rolling Test Forecast
    # ==========================================
    print("\nGenerating Test Predictions (Rolling Window)...")
    
    # We combine Train + Val to serve as history for the Test set
    full_history = train_scaled.append(val_scaled).append(test_scaled)
    
    test_pred = lstm_model.historical_forecasts(
        series=full_history,
        future_covariates=all_covariates_scaled,
        start=test_scaled.start_time(),
        forecast_horizon=config['output_chunk_length'],
        stride=config['output_chunk_length'],
        retrain=False,
        verbose=True,
        last_points_only=False
    )
    if isinstance(test_pred, list):
        test_pred = concatenate(test_pred, ignore_time_axis=True)
        
    test_pred = scaler.inverse_transform(test_pred)
    test_actuals = test_series.slice_intersect(test_pred)

    log_metrics(test_actuals, test_pred, "test")
    save_plot(test_actuals, test_pred, "Test: Pure LSTM MIMO", "test_prediction_plot.png")
    
    # Save CSV
    df = pd.DataFrame({
        'datetime': test_actuals.time_index,
        'actual': test_actuals.values().flatten(),
        'prediction': test_pred.values().flatten()
    })
    df.to_csv("test_predictions.csv", index=False)
    print("CSV saved to test_predictions.csv")

    wandb.finish()

if __name__ == "__main__":
    # Ensure you have these imported or available from your other files
    full_data_df, calculated_period = part1() 
    part2(full_data_df, calculated_period)
    part3(full_data_df, calculated_period)
    part4(full_data_df, calculated_period)
