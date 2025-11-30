import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
individual_household_electric_power_consumption = fetch_ucirepo(id=235)

# data (as pandas dataframes)
X = individual_household_electric_power_consumption.data.features
y = individual_household_electric_power_consumption.data.targets

df_combined = pd.concat([X, y], axis=1)

# Save the combined DataFrame to a CSV file
# index=False prevents the Pandas row numbers from being written as a column in the CSV file
df_combined.to_csv('individual_household_electric_power_consumption.csv', index=False)

# metadata 
print("--- Metadata ---")
print(individual_household_electric_power_consumption.metadata)

# variable information 
print("\n--- Variable Information ---")
print(individual_household_electric_power_consumption.variables)

print("\nSuccessfully saved dataset to 'individual_household_electric_power_consumption.csv'")