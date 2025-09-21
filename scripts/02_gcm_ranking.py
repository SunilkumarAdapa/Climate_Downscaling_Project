# scripts/02_gcm_ranking.py

"""
GCM Ranking Script

Purpose:
- Evaluate GCM performance against observational data for a historical period.
- Calculate performance metrics like Root Mean Square Error (RMSE).
- Save the ranking scores to a CSV file.

Inputs:
- GCM historical data from 'data/raw/gcm/{GCM_NAME}/'
- Observational data from 'data/raw/observational/'

Outputs:
- A CSV file with ranking scores in 'outputs/tables/gcm_ranking_scores.csv'
"""
import xarray as xr
import numpy as np
import pandas as pd
import os

print("--- Running 02_gcm_ranking.py ---")

# --- Configuration ---
# In a real scenario, you'd have a list of GCMs to loop through.
# Here, we simulate it with just one for demonstration.
GCM_LIST = ['ACCESS-CM2']
OBS_FILE = 'data/raw/observational/obs_daily_data_1980-2014.nc'
VARIABLE = 'tas'  # Temperature variable to rank
HISTORICAL_PERIOD = slice('1980-01-01', '2014-12-31')
OUTPUT_TABLE = 'outputs/tables/gcm_ranking_scores.csv'

# Define a bounding box for the study area (e.g., Hyderabad, India)
LAT_RANGE = slice(17, 18)
LON_RANGE = slice(78, 79)

# --- Functions ---
def calculate_rmse(gcm_data, obs_data):
    """Calculates the Root Mean Square Error between two xarray DataArrays."""
    return np.sqrt(((gcm_data - obs_data)**2).mean()).item()

# --- Main Execution ---
ranking_results = []

try:
    # Load and process observational data
    obs_ds = xr.open_dataset(OBS_FILE).sel(time=HISTORICAL_PERIOD, lat=LAT_RANGE, lon=LON_RANGE)
    # Resample to monthly mean to match many GCM outputs
    obs_monthly = obs_ds[VARIABLE].resample(time='1M').mean()

    # Loop through each GCM
    for gcm_name in GCM_LIST:
        print(f"Processing GCM: {gcm_name}...")
        gcm_file = f'data/raw/gcm/{gcm_name}/tas_historical.nc'
        
        # Load and process GCM data
        gcm_ds = xr.open_dataset(gcm_file).sel(time=HISTORICAL_PERIOD, lat=LAT_RANGE, lon=LON_RANGE)
        
        # Simple unit conversion: Kelvin to Celsius if needed
        if gcm_ds[VARIABLE].attrs.get('units', '').lower() == 'k':
            gcm_ds[VARIABLE] = gcm_ds[VARIABLE] - 273.15
        
        gcm_monthly = gcm_ds[VARIABLE].resample(time='1M').mean()
        
        # Align timestamps (important!)
        gcm_aligned, obs_aligned = xr.align(gcm_monthly, obs_monthly, join='inner')

        # Calculate metrics
        rmse = calculate_rmse(gcm_aligned, obs_aligned)
        mean_bias = (gcm_aligned - obs_aligned).mean().item()
        
        ranking_results.append({
            'gcm': gcm_name,
            'rmse': rmse,
            'mean_bias': mean_bias
        })
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - Mean Bias: {mean_bias:.4f}")

    # Create a DataFrame and save the results
    ranking_df = pd.DataFrame(ranking_results).sort_values(by='rmse', ascending=True)
    
    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    ranking_df.to_csv(OUTPUT_TABLE, index=False)
    
    print(f"\nRanking complete. Scores saved to {OUTPUT_TABLE}")
    print(ranking_df)

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the raw data exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Finished 02_gcm_ranking.py ---")