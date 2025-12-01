import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_clean_data(filename: str = "individual_household_electric_power_consumption.csv", resample_freq: str = 'D'):
    print(f"Loading {filename}...")

    power_df = pd.read_csv(filename, sep=',', na_values=['?', 'nan'], low_memory=False)
    power_df['DateTime'] = pd.to_datetime(
        power_df['Date'] + ' ' + power_df['Time'], 
        format='%d/%m/%Y %H:%M:%S', 
        errors='coerce'
    )
    
    power_df.drop(columns=['Date', 'Time'], inplace=True)
    power_df.set_index('DateTime', inplace=True)
    power_df = power_df[power_df.index.notnull()]
    power_df.sort_index(inplace=True)

    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 
        'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        power_df[col] = pd.to_numeric(power_df[col], errors='coerce')

    print("Interpolating missing values...")
    power_df.interpolate(method='linear', inplace=True)
    power_df.dropna(inplace=True)

    # 2. Resampling
    if resample_freq:
        print(f"Resampling data to frequency: {resample_freq}")
        power_df = power_df.resample(resample_freq).median()
        power_df.interpolate(method='linear', inplace=True)
        power_df.dropna(inplace=True)

    print(f"Data Cleaned. Shape: {power_df.shape}")
    

    plot_filename = f"power_consumption_plot_{resample_freq}.png"
    
    plt.figure(figsize=(16, 6))
    
    # Plot the main variable (Global_active_power)
    power_df['Global_active_power'].plot(
        title=f'Global Active Power ({resample_freq} Median)',
        xlabel='Date',
        ylabel='Global Active Power (kW)'
    )
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend() 
    plt.tight_layout()
    
    # Save the figure as a PNG file
    plt.savefig(plot_filename)
    plt.close() # Close the figure to free up memory
    
    print(f"Plot successfully saved to {plot_filename}")
    
    return power_df, numeric_cols

if __name__ == "__main__":
    # Settings
    FILE_NAME = "individual_household_electric_power_consumption.csv"

    clean_df, cols = load_and_clean_data(FILE_NAME, resample_freq='D')
    


