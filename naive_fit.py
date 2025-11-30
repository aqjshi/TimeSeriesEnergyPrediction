import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from load_and_visualize import load

# --- 1. VECTORIZED METRIC CALCULATIONS ---
def fast_calculate_metrics(y_true, y_pred):
    """
    Calculates metrics using pure Numpy for speed.
    """
    # Error vector
    e = y_true - y_pred
    
    # Standard Metrics
    mae = np.mean(np.abs(e))
    mse = np.mean(e**2)
    rmse = np.sqrt(mse)
    
    # MAPE (Handle division by zero safely)
    # Mask where true is 0 to avoid inf
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs(e[mask] / y_true[mask])) * 100
    else:
        mape = 0.0

    # Vector Metric (Euclidean Norm of the whole error vector)
    euclidean_error = np.linalg.norm(e)
    
    # Cumulative Metric
    cumulative_absolute_error = np.abs(np.sum(y_pred) - np.sum(y_true))

    return {
        "step_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "MAPE": float(mape),
            "RMSE": float(rmse)
        },
        "vector_metrics": {
            "Euclidean_Error": float(euclidean_error)
        },
        "cumulative_metrics": {
            "Cumulative_Absolute_Error": float(cumulative_absolute_error)
        }
    }

# --- 2. VECTORIZED FORECAST ENGINE ---
def vectorized_naive_forecast(series_array, seasonality_k, horizon, split_indices):
    """
    Performs Naive forecasting without loops using array shifting.
    Prediction for time t = Value at time (t - K)
    """
    # The 'prediction' array is just the original array shifted by K (seasonality)
    # We pad the beginning with NaNs because we can't predict before time K
    preds_full = np.full_like(series_array, np.nan)
    
    # Shift logic: preds[t] = series[t - K]
    # If K=1 (Persistence), preds[1] = series[0]
    # If K=1440 (Daily), preds[1440] = series[0]
    
    shift = seasonality_k
    
    # Safety check: ensure shift isn't larger than data
    if shift < len(series_array):
        preds_full[shift:] = series_array[:-shift]
    
    # Extract Test Set
    train_end, val_end = split_indices
    
    # Train: 0 to train_end
    # Val: train_end to val_end
    # Test: val_end to end
    
    # We only care about aligning the Test set for final metrics
    # Note: Naive Multi-step is just repeating the same value if K=1, 
    # but strictly speaking, Seasonal Naive at step h is y_{t+h-m*(k+1)}. 
    # For simple "Standard" Naive (K=1), y_pred[t] is just y[t-1].
    
    y_true_test = series_array[val_end:]
    y_pred_test = preds_full[val_end:]
    
    # Handle NaNs (if test set starts before K, which shouldn't happen with 70% split)
    valid_mask = ~np.isnan(y_pred_test)
    y_true_test = y_true_test[valid_mask]
    y_pred_test = y_pred_test[valid_mask]
    
    return y_true_test, y_pred_test, preds_full

def run_fast_naive_evaluation():
    output_dir = "naive_fit"
    os.makedirs(output_dir, exist_ok=True)
    

    with open("load_and_visualize/output.json", 'r') as f:
        part1_data = json.load(f)
    column_stats = part1_data.get('column_stats', {})
    print("Loaded seasonality data from Part 1.")


    # 2. Load Data
    whole_power_df, numeric_cols = load()
    
    # Pre-processing (ensure numeric)
    whole_power_df = whole_power_df[numeric_cols]
    for col in numeric_cols:
        whole_power_df[col] = pd.to_numeric(whole_power_df[col], errors='coerce')
    
    # Resample (Crucial for consistent steps)
    whole_power_df = whole_power_df.resample('min').mean().interpolate(method='linear')
    
    # Calculate Split Indices
    n_rows = len(whole_power_df)
    train_end = int(n_rows * 0.7)
    val_end = int(n_rows * 0.9) # 0.7 + 0.2
    
    split_indices = (train_end, val_end)
    
    # 3. Horizons to sweep
    n_steps = [1, 5, 15, 60, 300]
    
    results_json = { "Model_A_Naive": {} }
    
    # 4. Main Execution Loop
    for step in tqdm(n_steps, desc="Horizon Sweep"):
        step_key = f"horizon_{step}_steps" if step > 1 else "horizon_1_step"
        results_json["Model_A_Naive"][step_key] = {}
        
        for col in numeric_cols:
            k = column_stats.get(col, {}).get('seasonal_lag_detected', 1)
            csv_data = {}
            valid_indices = whole_power_df.index[val_end:]
            
            # The 'True' target values for the test set
            y_true_base = whole_power_df[col].values[val_end:]
            
            # Loop through 1 to current horizon (e.g., 1 to 5)
            for h in range(1, step + 1):
                shift_amount = h 
                
                pred_full = whole_power_df[col].shift(shift_amount)
                y_pred_h = pred_full.values[val_end:]
                
                # Add pair to dictionary: True_th, Pred_th
                csv_data[f"True_t{h}"] = y_true_base
                csv_data[f"Pred_t{h}"] = y_pred_h

            # Create DataFrame
            df_res = pd.DataFrame(csv_data, index=valid_indices)
            
            # Save CSV
            csv_path = os.path.join(output_dir, f"forecast_{col}_h{step}.csv")
            df_res.to_csv(csv_path)

    
            y_true_final = df_res[f"True_t{step}"].values
            y_pred_final = df_res[f"Pred_t{step}"].values
            
            # Handle NaNs created by shifting
            mask = ~np.isnan(y_pred_final)
            metrics = fast_calculate_metrics(y_true_final[mask], y_pred_final[mask])
            results_json["Model_A_Naive"][step_key][col] = metrics

            plt.figure(figsize=(12, 6))
            
            # Use only the last 1000 points for visibility
            plot_len = min(1000, len(y_true_final))
            
            # Plot True Data
            plt.plot(df_res.index[:plot_len], y_true_final[:plot_len], 
                     label='True', color='black', alpha=0.6)
            
            # Plot the Prediction from the LONGEST horizon (step)
            # This aligns the prediction made 'step' ago with the current True value
            plt.plot(df_res.index[:plot_len], y_pred_final[:plot_len], 
                     label=f'Naive Pred (Horizon={step})', color='red', alpha=0.6, linestyle='--')
            
            plt.title(f"{col} - Horizon {step} (Prediction made {step} steps ago)")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"plot_{col}_h{step}.png"))
            plt.close()

    # 5. Output JSON
    out_file = os.path.join(output_dir, "output.json")
    with open(out_file, 'w') as f:
        json.dump(results_json, f, indent=4)
        
    print(f"Fast analysis complete. Saved to {out_file}")

if __name__ == "__main__":
    run_fast_naive_evaluation()