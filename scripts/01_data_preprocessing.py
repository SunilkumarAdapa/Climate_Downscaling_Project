# scripts/01_data_preprocessing.py

"""
Data Preprocessing Script

Purpose:
- Perform initial data cleaning and preparation.
- This could include clipping data to a study area, reformatting time, etc.
- For this template, we will just check if the files can be opened.

Inputs:
- Raw GCM and observational data from 'data/raw/'

Outputs:
- (Optional) Preprocessed data saved to 'data/processed/'
"""

import xarray as xr
import os

print("--- Running 01_data_preprocessing.py ---")

# --- Configuration ---
GCM_NAME = 'ACCESS-CM2'
OBS_FILE = 'data/raw/observational/obs_daily_data_1980-2014.nc'
GCM_HIST_FILE = f'data/raw/gcm/{GCM_NAME}/tas_historical.nc'
GCM_FUTURE_FILE = f'data/raw/gcm/{GCM_NAME}/pr_ssp585.nc'

# --- Main Execution ---
try:
    # Try to open the datasets to ensure they are accessible and not corrupt
    obs_ds = xr.open_dataset(OBS_FILE)
    print(f"Successfully opened observational data: {OBS_FILE}")
    print(obs_ds.head())

    gcm_hist_ds = xr.open_dataset(GCM_HIST_FILE)
    print(f"\nSuccessfully opened GCM historical data: {GCM_HIST_FILE}")
    print(gcm_hist_ds.head())

    gcm_future_ds = xr.open_dataset(GCM_FUTURE_FILE)
    print(f"\nSuccessfully opened GCM future data: {GCM_FUTURE_FILE}")
    print(gcm_future_ds.head())

    print("\nPreprocessing check complete. All raw data files are accessible.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure your raw data is placed correctly in the 'data/raw/' directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Finished 01_data_preprocessing.py ---")