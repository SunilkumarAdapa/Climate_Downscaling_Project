import xarray as xr
from xsdba.adjustment import EmpiricalQuantileMapping
import os
from glob import glob

# ----------------------------
# CONFIGURATION
# ----------------------------

BASE_DIR = r"C:\Users\ADAPA SUNIL KUMAR\Climate_Downscaling_Project\data\temperature_Gcm_data"

# Observed data (IMD)
OBS_DIR = os.path.join(BASE_DIR, "observed_data/IMD_AVERAGE_TEMP_0.25")
OBS_PATTERN = "obs_tas_*.nc"  # adjust if needed

# GCM data
GCM_DIR = os.path.join(BASE_DIR, "INM-CM4-8_YEARLY-TEMP-DATA")
GCM_HIST_PATTERN = "INM-CM4-8_*1975-2000*.nc"   # historical period
GCM_FUT_PATTERN  = "INM-CM4-8_*2001-2014*.nc"   # future period

OUTPUT_DIR = os.path.join(BASE_DIR, "bias_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAR_NAME = "tas"  # variable name in NetCDF files (tasmax/tasmin/tas)

# ----------------------------
# HELPER FUNCTION: Merge yearly files
# ----------------------------
def merge_yearly_files(folder_pattern, var_name):
    """Merge multiple yearly NetCDF files into a single DataArray."""
    files = sorted(glob(os.path.join(folder_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {folder_pattern}")
    ds = xr.open_mfdataset(files, combine='by_coords')
    return ds[var_name]

# ----------------------------
# LOAD OBSERVED DATA (1975-2000)
# ----------------------------
print("Merging observed files (1975-2000)...")
obs = merge_yearly_files(os.path.join(OBS_DIR, OBS_PATTERN), VAR_NAME)
print(f"  -> Observed data shape: {obs.shape}")

# ----------------------------
# MERGE GCM HISTORICAL AND FUTURE DATA
# ----------------------------
print("Merging GCM historical (1975-2000)...")
gcm_hist = merge_yearly_files(os.path.join(GCM_DIR, GCM_HIST_PATTERN), VAR_NAME)
print(f"  -> GCM historical shape: {gcm_hist.shape}")

print("Merging GCM future (2001-2014)...")
gcm_fut = merge_yearly_files(os.path.join(GCM_DIR, GCM_FUT_PATTERN), VAR_NAME)
print(f"  -> GCM future shape: {gcm_fut.shape}")

# ----------------------------
# BIAS CORRECTION
# ----------------------------
print("\nTraining Empirical Quantile Mapping...")
eqm = EmpiricalQuantileMapping.train(gcm_hist, obs)

print("Adjusting historical data...")
hist_bc = eqm.adjust(gcm_hist)

print("Adjusting future data...")
fut_bc = eqm.adjust(gcm_fut)

# ----------------------------
# SAVE OUTPUT
# ----------------------------
hist_outfile = os.path.join(OUTPUT_DIR, "INM-CM4-8_hist_bias_corrected.nc")
fut_outfile  = os.path.join(OUTPUT_DIR, "INM-CM4-8_future_bias_corrected.nc")

hist_bc.to_netcdf(hist_outfile)
fut_bc.to_netcdf(fut_outfile)

print(f"\nSaved bias-corrected historical: {hist_outfile}")
print(f"Saved bias-corrected future: {fut_outfile}")
print("\nBias correction completed successfully!")
