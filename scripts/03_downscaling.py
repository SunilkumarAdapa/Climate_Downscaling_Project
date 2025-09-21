# scripts/03_downscaling.py

"""
Statistical Downscaling Script

Purpose:
- Downscale coarse GCM projections to a finer spatial resolution.
- This example uses simple bilinear interpolation.
- The target grid is derived from the high-resolution observational data.

Inputs:
- Coarse-resolution GCM future data from 'data/raw/gcm/{GCM_NAME}/'
- High-resolution observational data (for target grid)

Outputs:
- Downscaled GCM data in 'data/processed/downscaled/{GCM_NAME}/'
"""
import xarray as xr
import os

print("--- Running 03_downscaling.py ---")

# --- Configuration ---
GCM_NAME = 'ACCESS-CM2'
VARIABLE = 'pr'
GCM_FUTURE_FILE = f'data/raw/gcm/{GCM_NAME}/{VARIABLE}_ssp585.nc'
OBS_FILE = 'data/raw/observational/obs_daily_data_1980-2014.nc'  # Used for grid reference
OUTPUT_DOWNSCALED_FILE = f'data/processed/downscaled/{GCM_NAME}/{VARIABLE}_ssp585_downscaled.nc'

# --- Main Execution ---
try:
    # Load the coarse GCM data to be downscaled
    gcm_ds_coarse = xr.open_dataset(GCM_FUTURE_FILE)
    print(f"Loaded coarse GCM data. Original grid size: lat={len(gcm_ds_coarse.lat)}, lon={len(gcm_ds_coarse.lon)}")

    # Load observational data to define the high-resolution target grid
    obs_ds = xr.open_dataset(OBS_FILE)
    print(f"Loaded observational data. Target grid size: lat={len(obs_ds.lat)}, lon={len(obs_ds.lon)}")

    # Perform downscaling using interpolation
    print("Performing bilinear interpolation...")
    gcm_ds_downscaled = gcm_ds_coarse.interp(
        lat=obs_ds.lat,
        lon=obs_ds.lon,
        method='linear'  # Bilinear interpolation
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_DOWNSCALED_FILE), exist_ok=True)

    # Save the downscaled data
    gcm_ds_downscaled.to_netcdf(OUTPUT_DOWNSCALED_FILE)

    print(f"\nDownscaling complete. New grid size: lat={len(gcm_ds_downscaled.lat)}, lon={len(gcm_ds_downscaled.lon)}")
    print(f"Downscaled file saved to: {OUTPUT_DOWNSCALED_FILE}")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the raw data files exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Finished 03_downscaling.py ---")