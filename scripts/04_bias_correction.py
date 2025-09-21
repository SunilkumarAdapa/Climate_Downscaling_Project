# scripts/04_bias_correction.py

"""
Bias Correction Script

Purpose:
- Correct systematic biases in the downscaled GCM projections.
- This example uses Quantile Delta Mapping (QDM), a standard technique.
- It "trains" the correction on a historical period and "applies" it to the future.

Inputs:
- Historical observational data from 'data/raw/observational/'
- Historical GCM data from 'data/raw/gcm/{GCM_NAME}/'
- Downscaled future GCM data from 'data/processed/downscaled/{GCM_NAME}/'

Outputs:
- Bias-corrected GCM data in 'data/processed/bias_corrected/{GCM_NAME}/'
"""
import xarray as xr
from xclim import sdba
import os

print("--- Running 04_bias_correction.py ---")

# --- Configuration ---
GCM_NAME = 'ACCESS-CM2'
VARIABLE = 'pr'
HISTORICAL_PERIOD = slice('1980-01-01', '2014-12-31')

# Input files
OBS_FILE = 'data/raw/observational/obs_daily_data_1980-2014.nc'
GCM_HIST_RAW_FILE = f'data/raw/gcm/{GCM_NAME}/{VARIABLE}_historical.nc' # Assuming you have this file
DOWNSCALED_FUTURE_FILE = f'data/processed/downscaled/{GCM_NAME}/{VARIABLE}_ssp585_downscaled.nc'

# Output file
OUTPUT_CORRECTED_FILE = f'data/processed/bias_corrected/{GCM_NAME}/{VARIABLE}_ssp585_corrected.nc'

# --- Main Execution ---
try:
    # 1. Load data
    print("Loading datasets...")
    # Reference data for training (historical observations)
    obs_hist = xr.open_dataset(OBS_FILE).sel(time=HISTORICAL_PERIOD)[VARIABLE]

    # Model data for training (historical simulation)
    # Note: For best results, this should also be downscaled first.
    # For simplicity, we interpolate it here.
    gcm_hist_raw = xr.open_dataset(GCM_HIST_RAW_FILE).sel(time=HISTORICAL_PERIOD)[VARIABLE]
    gcm_hist = gcm_hist_raw.interp(lat=obs_hist.lat, lon=obs_hist.lon, method='linear')

    # Model data to be corrected (future downscaled simulation)
    gcm_future = xr.open_dataset(DOWNSCALED_FUTURE_FILE)[VARIABLE]
    
    # Unit conversion for precipitation: kg m-2 s-1 to mm/day
    # 1 kg/m²/s = 86400 mm/day
    if gcm_hist.attrs.get('units', '').lower() == 'kg m-2 s-1':
        print("Converting GCM units from 'kg m-2 s-1' to 'mm/day'")
        gcm_hist = gcm_hist * 86400
        gcm_future = gcm_future * 86400
        gcm_hist.attrs['units'] = 'mm/day'
        gcm_future.attrs['units'] = 'mm/day'

    # 2. Perform Bias Correction
    print("Performing Quantile Delta Mapping bias correction...")
    # The 'QDM' method adjusts the quantiles of the model to match obs
    # while preserving the model's projected trend.
    qdm = sdba.QuantileDeltaMapping.train(
        ref=obs_hist,
        hist=gcm_hist,
        sim=gcm_future,
        nquantiles=100,
        group='time.month' # Apply correction separately for each month
    )
    corrected_data = qdm.adjust(sim=gcm_future)
    
    # 3. Save the output
    print(f"Saving corrected data to {OUTPUT_CORRECTED_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_CORRECTED_FILE), exist_ok=True)
    corrected_data.to_netcdf(OUTPUT_CORRECTED_FILE)
    
    print("\nBias correction complete.")

except FileNotFoundError as e:
    print(f"Error: {e}. One of the required input files is missing.")
    print("NOTE: This script requires a historical GCM precipitation file (e.g., 'pr_historical.nc').")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- Finished 04_bias_correction.py ---")