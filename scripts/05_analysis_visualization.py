# scripts/05_analysis_visualization.py

"""
Analysis and Visualization Script

Purpose:
- Analyze the final bias-corrected data.
- Create visualizations, such as time series plots and maps.
- Save figures to the 'outputs/figures/' directory.

Inputs:
- Bias-corrected data from 'data/processed/bias_corrected/{GCM_NAME}/'
- (Optional) Raw GCM data for comparison.

Outputs:
- PNG image files in 'outputs/figures/'
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

print("--- Running 05_analysis_visualization.py ---")

# --- Configuration ---
GCM_NAME = 'ACCESS-CM2'
VARIABLE = 'pr'
CORRECTED_FILE = f'data/processed/bias_corrected/{GCM_NAME}/{VARIABLE}_ssp585_corrected.nc'
RAW_FUTURE_FILE = f'data/raw/gcm/{GCM_NAME}/{VARIABLE}_ssp585.nc' # For comparison
FIG_TIMESERIES_PATH = f'outputs/figures/{GCM_NAME}_precip_timeseries.png'
FIG_MAP_PATH = f'outputs/figures/{GCM_NAME}_precip_map_2040-2060.png'

# --- Main Execution ---
try:
    # Load the final corrected dataset
    ds_corrected = xr.open_dataset(CORRECTED_FILE)
    print("Loaded bias-corrected data.")

    # --- 1. Create a Time Series Plot ---
    print("Generating time series plot...")
    
    # Calculate annual mean precipitation over the entire area
    annual_mean_corrected = ds_corrected[VARIABLE].mean(dim=['lat', 'lon']).resample(time='1Y').mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    annual_mean_corrected.plot(ax=ax, label='Bias-Corrected')
    
    ax.set_title(f'Projected Annual Mean Precipitation for {GCM_NAME} (SSP5-8.5)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Precipitation (mm/day)')
    ax.legend()
    ax.grid(True)
    
    os.makedirs(os.path.dirname(FIG_TIMESERIES_PATH), exist_ok=True)
    plt.savefig(FIG_TIMESERIES_PATH, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to: {FIG_TIMESERIES_PATH}")
    plt.close(fig)

    # --- 2. Create a Map ---
    print("Generating spatial map for a future period (2040-2060)...")
    
    # Select a future time slice and calculate the mean
    future_period = slice('2040-01-01', '2060-12-31')
    mean_future_precip = ds_corrected[VARIABLE].sel(time=future_period).mean(dim='time')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Plot the data
    mean_future_precip.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        cbar_kwargs={'label': 'Mean Precipitation (mm/day)'}
    )
    
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_title(f'Mean Projected Precipitation (2040-2060)\n{GCM_NAME} (SSP5-8.5)')
    
    os.makedirs(os.path.dirname(FIG_MAP_PATH), exist_ok=True)
    plt.savefig(FIG_MAP_PATH, dpi=300, bbox_inches='tight')
    print(f"Map saved to: {FIG_MAP_PATH}")
    plt.close(fig)

except FileNotFoundError as e:
    print(f"Error: {e}. Could not find the bias-corrected data file.")
except Exception as e:
    print(f"An unexpected error occurred during visualization: {e}")

print("--- Finished 05_analysis_visualization.py ---")