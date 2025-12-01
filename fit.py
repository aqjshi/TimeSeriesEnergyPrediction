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
from darts.metrics import mae, mse, rmse
from darts.models import BlockRNNModel, LinearRegressionModel, NaiveSeasonal 
from darts import concatenate
from part1 import part1
from part2 import part2, create_fourier_covariates

FILE_NAME = "individual_household_electric_power_consumption.csv"

ORIGINAL_COL = 'Global_active_power'
STATIONARY_COL = 'detrended_deseasonalized'
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
PERIOD_STEPS = 96


def create_trend_covariate(series: TimeSeries) -> TimeSeries:
    """
    Creates a simple linear trend covariate (0, 1, 2, ...) 
    based on the time index length. (Defined once, here or in part2)
    """
    time_index = series.time_index
    trend = np.arange(len(time_index)) 
    trend_df = pd.DataFrame(index=time_index, data={'linear_trend': trend})
    return TimeSeries.from_dataframe(trend_df)

# --- part3 Function (Fixed) ---

def part3(clean_df, calculated_period):
    # The 'calculated_period' is now used as a fallback if JSON loading fails.
    # However, for consistency, we rely on the JSON/default PERIOD_STEPS.

    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    
    # 1. Configuration & Parameter Loading
    run_name = f"forecast_{secrets.token_hex(4)}"
    
    try:
        with open("sarima_params.json", 'r') as f:
            sarima_params = json.load(f)
        PERIOD = sarima_params.get('seasonal_period', PERIOD_STEPS) 
        K_FOURIER = sarima_params.get('fourier_k', 10) 
    except FileNotFoundError:
        print("Warning: sarima_params.json not found. Using default PERIOD=96, K_FOURIER=10.")
        PERIOD = PERIOD_STEPS
        K_FOURIER = 10
    
    # 1.2 Create Covariates (using imported/locally defined functions)
    fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
    trend_covariate = create_trend_covariate(series)
    all_covariates = fourier_covariates.stack(trend_covariate)

    # 2. Split Target Series
    train_series, val_test_series = series.split_before(TRAIN_RATIO)
    # Calculate split index based on VAL_RATIO relative to the remaining data
    val_split_pos = int(len(val_test_series) * (VAL_RATIO / (1 - TRAIN_RATIO)))
    val_split_point_index_value = val_test_series.time_index[val_split_pos]
    val_series, test_series = val_test_series.split_before(val_split_point_index_value)

    # 3. Split Covariates (at the exact same points as target)
    train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
    val_cov, test_cov = val_test_cov.split_before(val_series.end_time())


    # 4. Naive Baseline (Unchanged 1-step forecast for *traditional* baseline)
    print("\n--- Calculating Naive 1-step forecast (Traditional Baseline) ---")
    naive_model = NaiveSeasonal(K=PERIOD) 
    naive_model.fit(train_series)
    
    naive_val_preds = naive_model.historical_forecasts(
        val_series,
        start=val_series.start_time(),
        verbose=False
    )
    naive_test_preds = naive_model.historical_forecasts(
        test_series,
        start=test_series.start_time(),
        verbose=False
    )

    val_series_aligned = val_series.slice_intersect(naive_val_preds)
    test_series_aligned = test_series.slice_intersect(naive_test_preds)

    naive_val_rmse = rmse(val_series_aligned, naive_val_preds)
    naive_test_rmse = rmse(test_series_aligned, naive_test_preds)
    print(f"Naive 1-step Validation RMSE: {naive_val_rmse:.4f}")
    print(f"Naive 1-step Test RMSE: {naive_test_rmse:.4f}")


    # --- 5. AR+FOURIER MODEL (Scaling) ---
    
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)
    
    cov_scaler = Scaler() 
    train_cov_scaled = cov_scaler.fit_transform(train_cov)
    val_cov_scaled = cov_scaler.transform(val_cov)
    test_cov_scaled = cov_scaler.transform(test_cov)

    all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
    
    # --- 6. AR+FOURIER MODEL (Training & Prediction) ---

    print("\n--- Training Fourier-AR-Trend model (LinearRegressionModel) ---")
    # FIX: Using LAG_CHOICE for better feature capture
    fourier_model = LinearRegressionModel(
        lags=PERIOD_STEPS, 
        lags_future_covariates=[0], 
        output_chunk_length=PERIOD_STEPS 
    )
    fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)
    
    print(f"Calculating {PERIOD_STEPS}-step rolling forecast...")
    
    # FIX: Use historical_forecasts for rolling multi-step evaluation on validation set
    val_preds_fourier_scaled_list = fourier_model.historical_forecasts(
        series=train_scaled.append(val_scaled), # Pass the whole series (train+val) for correct history
        future_covariates=all_covariates_scaled,
        start=val_scaled.start_time(), # Start prediction at beginning of validation set
        forecast_horizon=PERIOD_STEPS, # Predict 96 steps out
        stride=PERIOD_STEPS, # Predict a new 96-step chunk every 96 steps
        retrain=False, 
        verbose=False
    )
    
    # FIX: Concatenate the list of TimeSeries objects resulting from historical_forecasts
    val_preds_fourier_scaled = concatenate(val_preds_fourier_scaled_list, ignore_time_axis=True)
    
    # FIX: Use historical_forecasts for rolling multi-step evaluation on test set
    test_preds_fourier_scaled_list = fourier_model.historical_forecasts(
        series=train_scaled.append(val_scaled).append(test_scaled), 
        future_covariates=all_covariates_scaled, 
        start=test_scaled.start_time(), 
        forecast_horizon=PERIOD_STEPS,
        stride=PERIOD_STEPS, 
        retrain=False, 
        verbose=False
    )
    
    test_preds_fourier_scaled = concatenate(test_preds_fourier_scaled_list, ignore_time_axis=True)


    # Align actuals and predictions for accurate metric calculation
    # Note: The alignment must be done *before* unscaling if using the scaled actuals (val_scaled/test_scaled)
    val_preds_fourier_scaled, val_scaled_aligned = val_preds_fourier_scaled.align(val_scaled)
    test_preds_fourier_scaled, test_scaled_aligned = test_preds_fourier_scaled.align(test_scaled)


    # Unscale final predictions and aligned actuals
    val_preds_unscaled = scaler.inverse_transform(val_preds_fourier_scaled)
    val_unscaled_aligned = scaler.inverse_transform(val_scaled_aligned)
    
    test_preds_unscaled = scaler.inverse_transform(test_preds_fourier_scaled)
    test_unscaled_aligned = scaler.inverse_transform(test_scaled_aligned) # Consistent naming

    # FIX: Using aligned actuals for metric calculation
    final_val_rmse = rmse(val_unscaled_aligned, val_preds_unscaled)
    final_val_mae = mae(val_unscaled_aligned, val_preds_unscaled)
    final_val_mse = mse(val_unscaled_aligned, val_preds_unscaled)
    
    final_test_rmse = rmse(test_unscaled_aligned, test_preds_unscaled)
    final_test_mae = mae(test_unscaled_aligned, test_preds_unscaled)
    final_test_mse = mse(test_unscaled_aligned, test_preds_unscaled)


    # --- 7. Save Results and Plot ---
    # Use the aligned/intersection series for plotting
    all_actuals = val_unscaled_aligned.append(test_unscaled_aligned)
    all_predictions = val_preds_unscaled.append(test_preds_unscaled)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = os.path.join(output_dir, f"{run_name}.csv")
    print(f"\nSaving combined predictions and actuals to {csv_filename}...")
    
    ACTUAL_COL_NAME = ORIGINAL_COL 
    PREDICTED_COL_NAME = 'Predicted_Global_active_power'

    df_actuals = all_actuals.to_dataframe()
    df_predictions = all_predictions.to_dataframe().rename(
        columns={ACTUAL_COL_NAME: PREDICTED_COL_NAME}
    )
    # Ensure alignment before concatenation, though align() above handles this
    results_df = pd.concat([df_actuals, df_predictions], axis=1)

    # Use the end date of the aligned validation actuals for splitting plot data
    val_end_date = val_unscaled_aligned.end_time()
    results_df['Data_Split'] = np.where(results_df.index <= val_end_date, 'Validation', 'Test')
    
    results_df.to_csv(csv_filename, index_label='index')

    plot_filename = os.path.join(output_dir, f"{run_name}.png")
    print(f"Generating and saving plot to {plot_filename}...")
    
    # Plotting logic 
    plt.figure(figsize=(14, 6))
    
    val_df = results_df[results_df['Data_Split'] == 'Validation']
    plt.plot(val_df.index, val_df[ACTUAL_COL_NAME], label='Validation Actual', color='tab:blue', linewidth=2)
    plt.plot(val_df.index, val_df[PREDICTED_COL_NAME], label='Validation Forecast', color='tab:orange', linestyle='--', linewidth=1.5)
    
    test_df = results_df[results_df['Data_Split'] == 'Test']
    plt.plot(test_df.index, test_df[ACTUAL_COL_NAME], label='Test Actual', color='tab:green', linewidth=2)
    plt.plot(test_df.index, test_df[PREDICTED_COL_NAME], label='Test Forecast', color='tab:red', linestyle='--', linewidth=1)
    
    plt.title(f'Global_active_power Rolling {PERIOD_STEPS}-step Forecast (Fourier-AR-Trend)')
    plt.xlabel('Index')
    plt.ylabel(ACTUAL_COL_NAME)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close() 
    
    print("\n--- Final Model Metrics (AR-Fourier-Trend) ---")
    print(f"Naive 1-step Test RMSE: {naive_test_rmse:.4f}")
    print(f"Final Validation RMSE ({PERIOD_STEPS}-step rolling): {final_val_rmse:.4f}")
    print(f"Final Test RMSE ({PERIOD_STEPS}-step rolling): {final_test_rmse:.4f}")
    # FIX: Added context to metrics
    if final_test_rmse < naive_test_rmse:
        print(f"-> Improvement over Naive 1-step: {100 * (naive_test_rmse - final_test_rmse) / naive_test_rmse:.2f}%")
    
    print(f"\n--- Process Complete ---")
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

def part4(clean_df, calculated_period):
    series = TimeSeries.from_dataframe(clean_df, freq='15min', value_cols=ORIGINAL_COL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(f"device: {device}")

    optimal_config_values = {
        # --- CHANGE 3: Update Input/Output lengths ---
        'input_chunk_length': 96,  # 4 Days of history (was 120)
        'output_chunk_length': 96,  # 1 Full Day prediction (was 1)
        
        # ... keep other params ...
        'TRAIN_RATIO': .7,
        'VAL_RATIO': .2,
        'batch_size': 32,          # Recommend lowering batch size slightly as sequences are longer
        'model_type': 'LSTM',
        'hidden_dim': 32, 
        'n_layers': 1,
        'dropout_rate': 0.3,       # Paper used low dropout for MIMO
        'epochs': 20, 
        'learning_rate': 0.01,
        'weight_decay': 0.0001,
        'grad_clip': 2
    }
        
    torch.set_float32_matmul_precision('medium')
    project_name = f'DSCC275FINAL'
    run_name = f"{secrets.token_hex(4)}"
    
    wandb.init(project=project_name, config=optimal_config_values, name=run_name)
    config = wandb.config 
    wandb_logger = WandbLogger(project=project_name, name=run_name, log_model=False)


    with open("sarima_params.json", 'r') as f:
        sarima_params = json.load(f)
    PERIOD = sarima_params.get('seasonal_period', 96) # Default 96
    K_FOURIER = sarima_params.get('fourier_k', 10)    # FIXED: Use 'fourier_k' from part2 (default 10)
    fourier_covariates = create_fourier_covariates(series, period=PERIOD, K=K_FOURIER)
    trend_covariate = create_trend_covariate(series) # Uses the MODIFIED function
    all_covariates = fourier_covariates.stack(trend_covariate)

    # 2. Split Target Series
    print("\n---  shapes: Initial Data Shapes ---")
    print(f"Original 'series' shape:      (len={len(series)}, width={series.width})")
    print(f"Original 'all_covariates' shape: (len={len(all_covariates)}, width={all_covariates.width})")
    train_series, val_test_series = series.split_before(config.TRAIN_RATIO)
    val_split_pos = int(len(val_test_series) * (config.VAL_RATIO / (1 - config.TRAIN_RATIO)))
    val_split_point_index_value = val_test_series.time_index[val_split_pos]
    val_series, test_series = val_test_series.split_before(val_split_point_index_value)

    # 3. Split Covariates (at the exact same points as target)
    print("\n---  shapes: After Target Split ---")
    print(f"'train_series' shape: (len={len(train_series)}, width={train_series.width})")
    print(f"'val_series' shape:   (len={len(val_series)}, width={val_series.width})")
    print(f"'test_series' shape:  (len={len(test_series)}, width={test_series.width})")

    train_cov, val_test_cov = all_covariates.split_before(train_series.end_time())
    val_cov, test_cov = val_test_cov.split_before(val_series.end_time())
    print("\n---  shapes: After Covariate Split ---")
    print(f"'train_cov' shape: (len={len(train_cov)}, width={train_cov.width})")
    print(f"'val_cov' shape:   (len={len(val_cov)}, width={val_cov.width})")
    print(f"'test_cov' shape:  (len={len(test_cov)}, width={test_cov.width})")
    # 4. Scale Target Series (Fit ONLY on train_series)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_series)
    val_scaled = scaler.transform(val_series)
    test_scaled = scaler.transform(test_series)
    
    # 5. Scale Covariates 
    cov_scaler = Scaler() #SEPARATE scaler
    train_cov_scaled = cov_scaler.fit_transform(train_cov)
    val_cov_scaled = cov_scaler.transform(val_cov)
    test_cov_scaled = cov_scaler.transform(test_cov)

    # 6. Recombine Scaled Covariates for Darts
    all_covariates_scaled = train_cov_scaled.append(val_cov_scaled).append(test_cov_scaled)
    

    print("\n---  shapes: AR-Fourier Model Inputs (to .fit()) ---")
    print(f"Target (train_scaled):          (len={len(train_scaled)}, width={train_scaled.width})")
    print(f"Covariates (all_covariates_scaled): (len={len(all_covariates_scaled)}, width={all_covariates_scaled.width})")
    print("Training Fourier-AR-Trend model (LinearRegressionModel)...")
    fourier_model = LinearRegressionModel(
        lags=96,                         # FIXED: Use the 96-step AR lag to capture seasonality
        lags_future_covariates=[0],
        output_chunk_length=96           # FIXED: Consistent 96 steps
    )

    fourier_model.fit(train_scaled, future_covariates=all_covariates_scaled)

    
    print("Calculating training set residuals...")
    fitted_train_values_list = fourier_model.historical_forecasts(
        train_scaled,
        future_covariates=all_covariates_scaled,
        start=train_scaled.time_index[0] + train_scaled.freq * 96, 
        stride=96,      
        retrain=False,   
        last_points_only=False, 
        verbose=True
    )
    

    if isinstance(fitted_train_values_list, list):
        # --- FIX: Add ignore_time_axis=True ---
        fitted_train_values = concatenate(fitted_train_values_list, ignore_time_axis=True)
    else:
        fitted_train_values = fitted_train_values_list

    train_residuals = train_scaled.slice_intersect(fitted_train_values) - fitted_train_values

    fitted_val_values_list = fourier_model.historical_forecasts(
        train_scaled.append(val_scaled), 
        future_covariates=all_covariates_scaled, 
        start=val_scaled.start_time(), 
        stride=96,      
        retrain=False,  
        last_points_only=False,
        verbose=True
    )
    
    if isinstance(fitted_val_values_list, list):
        # --- FIX: Add ignore_time_axis=True ---
        fitted_val_values = concatenate(fitted_val_values_list, ignore_time_axis=True)
    else:
        fitted_val_values = fitted_val_values_list

    val_residuals = val_scaled.slice_intersect(fitted_val_values) - fitted_val_values
    
    # Standardize the residuals 
    sklearn_scaler = StandardScaler()
    residual_scaler = Scaler(sklearn_scaler)
    
    # Fit residual scaler ONLY on training residuals
    scaled_train_residuals = residual_scaler.fit_transform(train_residuals)
    scaled_val_residuals = residual_scaler.transform(val_residuals)

    print("\n---  shapes: LSTM Model Inputs (to .fit()) ---")
    print(f"Target (scaled_train_residuals): (len={len(scaled_train_residuals)}, width={scaled_train_residuals.width})")
    print(f"Validation (scaled_val_residuals): (len={len(scaled_val_residuals)}, width={scaled_val_residuals.width})")
    print(f"   -> Darts will chunk these into samples of (input_chunk_length={config.input_chunk_length}, width={scaled_train_residuals.width})")
    full_series = concatenate([scaled_train_residuals, scaled_val_residuals], ignore_time_axis=True)
    

    n_val = len(scaled_val_residuals)
    
    # We take the last (n_val + input_chunk) steps from the reconstructed full series
    required_len = n_val + config.input_chunk_length
    
    val_series_with_history = full_series[-required_len:]
    model = BlockRNNModel(
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        model=config.model_type,
        hidden_dim=config.hidden_dim,
        n_rnn_layers=config.n_layers,
        dropout=config.dropout_rate,
        batch_size=config.batch_size,
        n_epochs=config.epochs,
        optimizer_kwargs={'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        
        pl_trainer_kwargs={
            "accelerator": "gpu" if device.type == "cuda" else "cpu",
            "devices": 1 if device.type == "cuda" else "auto",
            "gradient_clip_val": config.grad_clip,
            "logger": wandb_logger, 
            "enable_model_summary": False 
        },
        
        force_reset=True
    )

    # Train RNN on scaled residuals
    model.fit(
        scaled_train_residuals, 
        val_series=val_series_with_history, # <--- Use the stitched series here
        verbose=True
    )

    print("\n---  shapes: LSTM Prediction Inputs (to .predict()) ---")
    
    # 1. Predict Validation (History is just training data)
    print(f"Val history (scaled_train_residuals): (len={len(scaled_train_residuals)}, width={scaled_train_residuals.width})")
    val_preds_lstm_scaled = model.predict(
        n=len(val_scaled), 
        series=scaled_train_residuals,
        verbose=False
    )


    test_history_series = concatenate([scaled_train_residuals, scaled_val_residuals], ignore_time_axis=True)

    print(f"Test history (train+val residuals): (len={len(test_history_series)}, width={test_history_series.width})")
    test_preds_lstm_scaled = model.predict(
        n=len(test_scaled), 
        series=test_history_series, 
        verbose=False
    )


    val_preds_residuals = residual_scaler.inverse_transform(val_preds_lstm_scaled)
    test_preds_residuals = residual_scaler.inverse_transform(test_preds_lstm_scaled)

    print("Predicting seasonality/trend from Fourier-AR-Trend model")
    val_preds_fourier_scaled = fourier_model.predict(
        n=len(val_scaled),
        series=train_scaled,
        future_covariates=all_covariates_scaled, 
        verbose=False
    )
    test_preds_fourier_scaled = fourier_model.predict(
        n=len(test_scaled),
        series=train_scaled.append(val_scaled),
        future_covariates=all_covariates_scaled, 
        verbose=False
    )

    # 4. Add predictions together (still in scaled terms of original series)
    val_preds_total_scaled = val_preds_fourier_scaled + val_preds_residuals
    test_preds_total_scaled = test_preds_fourier_scaled + test_preds_residuals

    # 5. Unscale final predictions and actuals (using the *target* scaler)
    val_preds_unscaled = scaler.inverse_transform(val_preds_total_scaled)
    val_unscaled = scaler.inverse_transform(val_scaled)
    
    preds_unscaled = scaler.inverse_transform(test_preds_total_scaled)
    test_unscaled = scaler.inverse_transform(test_scaled)


 
    final_val_rmse = rmse(val_unscaled, val_preds_unscaled)
    final_val_mae = mae(val_unscaled, val_preds_unscaled)
    final_val_mse = mse(val_unscaled, val_preds_unscaled)
    
    final_test_rmse = rmse(test_unscaled, preds_unscaled)
    final_test_mae = mae(test_unscaled, preds_unscaled)
    final_test_mse = mse(test_unscaled, preds_unscaled)

    wandb.log({
        'final/val_rmse': final_val_rmse,
        'final/val_mae': final_val_mae,
        'final/val_mse': final_val_mse,
        'final/test_rmse': final_test_rmse,
        'final/test_mae': final_test_mae,
        'final/test_mse': final_test_mse 
    })
    
    
    all_actuals = val_unscaled.append(test_unscaled)
    all_predictions = val_preds_unscaled.append(preds_unscaled)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving combined predictions and actuals to {output_dir}/predictions.csv...")
    
    ACTUAL_COL_NAME = ORIGINAL_COL 
    PREDICTED_COL_NAME = 'Predicted_Global_active_power'

    df_actuals = all_actuals.to_dataframe()
    df_predictions = all_predictions.to_dataframe().rename(
        columns={ACTUAL_COL_NAME: PREDICTED_COL_NAME}
    )
    results_df = pd.concat([df_actuals, df_predictions], axis=1)

    val_end_date = val_unscaled.end_time()
    results_df['Data_Split'] = np.where(results_df.index <= val_end_date, 'Validation', 'Test')
    
    results_df.to_csv(os.path.join(output_dir, f"{run_name}.csv"), index_label='index')

    print(f"Generating and saving plot to {output_dir}/{run_name}.png...")
    
    plt.figure(figsize=(14, 6))
    
    val_df = results_df[results_df['Data_Split'] == 'Validation']
    plt.plot(val_df.index, val_df[ACTUAL_COL_NAME], label='Validation Actual', color='tab:blue', linewidth=2)
    plt.plot(val_df.index, val_df[PREDICTED_COL_NAME], label='Validation Forecast', color='tab:orange', linestyle='--', linewidth=1.5)
    
    test_df = results_df[results_df['Data_Split'] == 'Test']
    plt.plot(test_df.index, test_df[ACTUAL_COL_NAME], label='Test Actual', color='tab:green', linewidth=2)
    plt.plot(test_df.index, test_df[PREDICTED_COL_NAME], label='Test Forecast', color='tab:red', linestyle='--', linewidth=1)
    
    plt.title('Hybrid Global_active_power Forecast (Fourier-AR-Trend + LSTM on Scaled Residuals)')
    plt.xlabel('Index')
    plt.ylabel(ACTUAL_COL_NAME)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_name}.png"))
    plt.close() 
    
    print("\n--- Darts + PyTorch Lightning + WandB Complete (Hybrid Model) ---")
    print(f"Final Validation RMSE: {final_val_rmse:.4f}")
    print(f"Final Test RMSE: {final_test_rmse:.4f}")
    print(f"WandB run logged under: {run_name}")
    
    wandb.finish()


    
if __name__ == "__main__":
    
    # Load and process data once in part1
    full_data_df, calculated_period = part1() 
    
    part2(full_data_df, calculated_period)
    part3(full_data_df, calculated_period)
    part4(full_data_df, calculated_period)

