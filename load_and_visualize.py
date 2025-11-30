import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import imageio.v3 as iio
import os
import glob 
import json
from statsmodels.tsa.stattools import acf
from tqdm import tqdm


def load(filename: str = "individual_household_electric_power_consumption.csv"):
    power_df = pd.read_csv(
        filename,
        sep=',',
        na_values=['?'] 
    )

    power_df['DateTime'] = pd.to_datetime(
        power_df['Date'] + ' ' + power_df['Time'], 
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce' 
    )
    
    power_df.set_index('DateTime', inplace=True)
    
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        power_df[col] = pd.to_numeric(power_df[col], errors='coerce')

    power_df.interpolate(method='linear', inplace=True) 
    power_df.dropna(subset=numeric_cols, inplace=True)
    
    return power_df, numeric_cols


def create_heatmap_animation(frame_dir):
    filenames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))

    images = [iio.imread(filename) for filename in filenames]
    output_gif_path = "load_and_visualize/ewma_covariance_animation.gif"
    
    iio.imwrite(
        output_gif_path, 
        images, 
        duration=150, 
        loop=0 
    )
def plot_ewma_covariance(power_df, resample_freq: int =300):
    columns_to_analyze = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
        'Sub_metering_3'
    ]
    
    df_metrics = power_df[columns_to_analyze].copy()

    df_daily = df_metrics
    df_daily.interpolate(method='linear', inplace=True)
    
    # Calculation of returns and EWMA Covariance
    returns_df = df_daily.pct_change().dropna() 
    EWMA_SPAN = 32
    ewma_cov_series = returns_df.ewm(span=EWMA_SPAN).cov().stack()
    col1 = 'Global_active_power'
    col2 = 'Global_intensity'

    cov_col1_only = ewma_cov_series.xs(col1, level=1)
    cov_time_series = cov_col1_only.xs(col2, level=1)

    plt.figure(figsize=(14, 6))
    cov_time_series.plot(
        title=f'EWMA Covariance (Daily, Span={EWMA_SPAN}):\n{col1} vs {col2}',
        label='EWMA Covariance',
        color='teal',
        linewidth=1.5
    )
    plt.xlabel('Date Time')
    plt.ylabel('Covariance Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('load_and_visualize/ewma_covariance_gap_gi_linechart.png')
    plt.close()


    # Prepare for Heatmap Generation
    full_cov_df = ewma_cov_series.unstack(level=2)
    time_points = full_cov_df.index.get_level_values(0).unique()
    sample_timestamps = time_points[::resample_freq] 

    frame_dir = "ewma_heatmap_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)


    print(f"\nGenerating {len(sample_timestamps)} heatmaps...")
    for i, ts in tqdm(enumerate(sample_timestamps), total=len(sample_timestamps), desc="Heatmap Frames"):
        cov_matrix = full_cov_df.loc[ts]
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            cov_matrix, 
            annot=True, fmt=".2e", 
            cmap='coolwarm', 
            linewidths=.5, 
            linecolor='lightgray',
            cbar_kws={'label': 'EWMA Covariance Value'}
        )
        plt.title(f'EWMA Covariance Matrix | Time: {ts.strftime("%Y-%m-%d")}')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Ensure the frame_dir path is correctly prepended
        filename = f"{frame_dir}/frame_{i:04d}_{ts.strftime('%Y%m%d')}.png" 
        plt.savefig(filename)
        plt.close()


    create_heatmap_animation(frame_dir)
    
    return returns_df, ewma_cov_series

def A_extract_and_plot_volatility(ewma_cov_series, metric_name='Global_active_power', EWMA_SPAN=32):

    var_col1_only = ewma_cov_series.xs(metric_name, level=1)
    
    ewma_variance_series = var_col1_only.xs(metric_name, level=1)

    ewma_volatility_series = np.sqrt(ewma_variance_series)

    plt.figure(figsize=(14, 6))
    ewma_volatility_series.plot(
        title=f'EWMA Volatility (Daily, Span={EWMA_SPAN}) of {metric_name}',
        label='EWMA Volatility',
        color='darkred',
        linewidth=1.5
    )
    plt.xlabel('Date Time')
    plt.ylabel('Volatility (Std. Dev. of Daily % Change)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'load_and_visualize/ewma_volatility_{metric_name.lower().replace("_", "-")}_linechart.png')
    plt.close()
    print(f"Saved EWMA Volatility line chart to ewma_volatility_{metric_name.lower().replace('_', '-')}_linechart.png")
    
    return ewma_volatility_series

def B_calculate_and_plot_correlation(ewma_cov_series, metric_i='Global_active_power', metric_j='Global_intensity', EWMA_SPAN=32):

    cov_ij_series = ewma_cov_series.xs(metric_i, level=1).xs(metric_j, level=1)

    var_i_series = ewma_cov_series.xs(metric_i, level=1).xs(metric_i, level=1)

    var_j_series = ewma_cov_series.xs(metric_j, level=1).xs(metric_j, level=1)

    ewma_correlation_series = cov_ij_series / np.sqrt(var_i_series * var_j_series)

    plt.figure(figsize=(14, 6))
    ewma_correlation_series.plot(
        title=f'EWMA Correlation (Daily, Span={EWMA_SPAN}):\n{metric_i} vs {metric_j}',
        label='EWMA Correlation',
        color='orange',
        linewidth=1.5
    )
    plt.axhline(0.8, color='red', linestyle='--', alpha=0.7) # Example threshold
    plt.axhline(0, color='gray', linestyle='-')
    plt.xlabel('Date Time')
    plt.ylabel('Correlation Coefficient (ρ)')
    plt.ylim(-0.1, 1.0) # Correlation is typically high for these metrics
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'load_and_visualize/ewma_correlation_{metric_i.lower().replace("_", "-")}_{metric_j.lower().replace("_", "-")}_linechart.png')
    plt.close()
    print(f"Saved EWMA Correlation line chart to ewma_correlation_{metric_i.lower().replace('_', '-')}_{metric_j.lower().replace('_', '-')}_linechart.png")
    
    return ewma_correlation_series
       
def plot_acf_analysis(series, nlags=1440, plot_name='acf_analysis_plot.png', LAG_STEP=10, offset=300):
    acf_values, confint = acf(series.values.flatten(), nlags=nlags, alpha=0.05, fft=True)
    
    start_offset = offset
    
    # Downsample the ACF, starting from the new offset
    downsampled_acf = acf_values[start_offset::LAG_STEP] 
    
    # Find the index of the highest peak in the DOWN-SAMPLED array
    downsampled_peak_index = np.argmax(downsampled_acf)
    
    # --- FIX 2: Use the new start offset in the final calculation ---
    # True Lag = (Index in Downsampled Array * Step Size) + Start Offset
    seasonal_lag = (downsampled_peak_index * LAG_STEP) + start_offset
    plt.figure(figsize=(12, 5))
    
    plt.plot(acf_values, marker='o', linestyle='--')
    
    plt.plot(confint[:, 0], color='gray', linestyle=':')
    plt.plot(confint[:, 1], color='gray', linestyle=':')
    
    plt.title(f'Autocorrelation Function (ACF) - Lag up to {nlags} minutes')
    plt.xlabel('Lag (Minutes)')
    plt.ylabel('Autocorrelation')
    plt.axhline(0, color='black', linewidth=0.5)
    
    plt.axvline(1440, color='red', linestyle='-', linewidth=1, label='K=1440 (24h Cycle)')
    
    plt.plot(seasonal_lag, acf_values[seasonal_lag], 
             marker='x', color='green', markersize=10, 
             label=f'ACF Peak Lag (K={seasonal_lag})')

    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.close()
    print(f"Saved ACF plot to {plot_name}")

    return int(seasonal_lag)

from scipy.signal import periodogram

def find_seasonal_lag_via_periodogram(series, fs_rate=1.0, max_lag_minutes=None):
    """
    Finds the most dominant non-zero period (lag) in a series.
    
    Args:
        series: The time series data (e.g., power_df['Global_active_power']).
        fs_rate: The sampling frequency (1.0 since data is sampled every minute).
        max_lag_minutes: The maximum period (in minutes) to consider.
    """
    
    # 1. Compute the Periodogram (Power Spectrum Density)
    # The output 'f' is frequency, 'Pxx' is power/magnitude.
    # The data is already at 1 sample/minute, so fs=1.0
    f, Pxx = periodogram(series.values, fs=fs_rate, scaling='spectrum')

    # 2. Filter out high-frequency (low-lag) noise and known non-seasonal features
    # Start searching after the first 10 minutes (to ignore immediate noise/decay)
    min_freq_index = 10 
    
    # Optional: Filter out periods longer than, say, 7 days (7*1440 minutes)
    if max_lag_minutes is not None:
        # Frequency = 1 / Period (in minutes)
        max_freq_to_consider = 1.0 / max_lag_minutes
        f_mask = (f >= max_freq_to_consider)
        f_search, Pxx_search = f[f_mask], Pxx[f_mask]
    else:
        f_search, Pxx_search = f[min_freq_index:], Pxx[min_freq_index:]

    # 3. Find the peak power index
    peak_power_index = np.argmax(Pxx_search)
    
    # 4. Extract the dominant frequency
    dominant_frequency = f_search[peak_power_index]

    # 5. Convert Frequency back to Lag/Period (Period = 1 / Frequency)
    # The resulting lag is in the original time unit (minutes).
    dominant_period_minutes = 1.0 / dominant_frequency
    
    return int(round(dominant_period_minutes))

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("load_and_visualize", exist_ok=True)

    print("Loading data...")
    whole_power_df, numeric_cols = load()
    num_rows = len(whole_power_df)
    truncate_index = num_rows * 70 // 100 

    # 2. Truncate the rows to the first 70% and select the specified columns
    # The slice `[:truncate_index]` selects from the start (index 0) up to (but not including) the calculated index.
    # The list `numeric_cols` selects only those columns.
    power_df = whole_power_df.iloc[:truncate_index, :][numeric_cols]

    # 3. Optional: Print information about the new DataFrame
    print(f"Original number of rows: {num_rows}")
    print(f"Truncated index (70%): {truncate_index}")
    print(f"New DataFrame shape: {power_df.shape}")
    print(f"Columns selected: {power_df.columns.tolist()}")
    # --- 0. PRE-CALCULATE HEAVY MATRICES (Do this ONCE, not in loop) ---
    print("Running Global EWMA analysis (this may take a moment)...")
    returns_df, ewma_cov_series = plot_ewma_covariance(power_df)

    column_metadata = {
        'Global_active_power': {'unit': 'Power (kW)', 'color': 'red'},
        'Global_reactive_power': {'unit': 'Power (kW)', 'color': 'orange'},
        'Voltage': {'unit': 'Voltage (V)', 'color': 'blue'},
        'Global_intensity': {'unit': 'Current (A)', 'color': 'purple'},
        'Sub_metering_1': {'unit': 'Energy (Wh)', 'color': 'green'},
        'Sub_metering_2': {'unit': 'Energy (Wh)', 'color': 'brown'},
        'Sub_metering_3': {'unit': 'Energy (Wh)', 'color': 'cyan'}
    }

    stats_output = {}
    
    # --- 1. FULL SWEEP LOOP ---
    print(f"Starting sweep of {len(numeric_cols)} columns...")
    
    for col in tqdm(numeric_cols):
        print(f"Analyzing {col}...")
        col_data = power_df[col]
        meta = column_metadata.get(col, {'unit': 'Value', 'color': 'black'})
        
        # A. Basic Plot
        plt.figure(figsize=(15, 6))
        plt.plot(col_data.index, col_data, label=f'{col}', color=meta["color"], linewidth=0.8)
        plt.title(f'{col} Over Time')
        plt.tight_layout()
        plt.savefig(f"load_and_visualize/{col.lower()}.png")
        plt.close() 

        # B. ACF Analysis
        
        seasonal_lag = plot_acf_analysis(
            col_data, 
            nlags=10080, 
            plot_name=f"load_and_visualize/acf_{col.lower()}.png", 
            LAG_STEP=60
        )
        preiodgram_seasonal_lag = find_seasonal_lag_via_periodogram(col_data, max_lag_minutes=10080)
        # C. Volatility Analysis
        # Extract specific volatility for this column from the pre-calculated matrix
        ewma_volatility_series = A_extract_and_plot_volatility(
            ewma_cov_series, 
            metric_name=col, # Pass the STRING name, not the data
            EWMA_SPAN=32
        )

        # D. Correlation Sweep (One vs All)
        correlations_summary = {}
        
        for other_col in numeric_cols:
            if col == other_col:
                continue # Skip correlation with self (always 1.0)
            
            # Extract correlation series
            corr_series = B_calculate_and_plot_correlation(
                ewma_cov_series, 
                metric_i=col,
                metric_j=other_col,
                EWMA_SPAN=32
            )
            
            # Store summary stat (Mean Correlation) in JSON 
            # (Storing the whole series makes JSON too big)
            correlations_summary[f"{other_col}"] = float(corr_series.mean())
        std_dev_val = col_data.std()
        mean_raw_val = col_data.mean()
        
        # Calculate CV = std / mean, handling division by zero (result -> 0.0)
        # We use np.nan_to_num to clean the result of the division
        if mean_raw_val != 0:
            coefficient_of_variation = std_dev_val / mean_raw_val
        else:
            coefficient_of_variation = 0.0 # Cannot calculate CV if mean is zero
        # E. Aggregate Stats for JSON
        stats_output[col] = {
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'mean_raw': float(col_data.mean()),
            'std_dev': float(col_data.std()),
            'drift_returns_mean': float(np.nan_to_num(returns_df[col].mean(), nan=0.0, posinf=0.0, neginf=0.0)) if col in returns_df.columns else 0.0,
            'seasonal_lag_detected': int(seasonal_lag),
            'preiodgram_seasonal_lag': int(preiodgram_seasonal_lag),
            'avg_ewma_volatility': float(ewma_volatility_series.mean()),
            'avg_correlations': correlations_summary,
            'coefficient_of_variation': float(np.nan_to_num(coefficient_of_variation, nan=0.0, posinf=0.0, neginf=0.0)),

        }

    # --- 2. SAVE OUTPUT ---
    part1_output = {
        'column_stats': stats_output,
        'analysis_metadata': {
            'ewma_span_used': 32,
            'total_rows': len(power_df),
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    part1_json = "load_and_visualize/output.json"

    with open(part1_json, 'w') as f:
        json.dump(part1_output, f, indent=4)
        
    print(f"Full sweep complete. Stats saved to {part1_json}")