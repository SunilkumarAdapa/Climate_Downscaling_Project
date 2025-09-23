import xarray as xr
import os
import glob
import re
from xsdba.adjustment import EmpiricalQuantileMapping

# --- CONFIGURATION: UPDATE YOUR FOLDER PATHS BELOW ---

# 1. Directory Setup
# Paste the FULL PATH to the folder containing your yearly IMD files
INPUT_DIR_OBSERVED = r"C:\Users\ADAPA SUNIL KUMAR\Climate_Downscaling_Project\data\temperature_Gcm_data\observed_data"
# Paste the FULL PATH to the folder containing all your GCM subfolders
INPUT_DIR_GCM_BASE = r"C:\Users\ADAPA SUNIL KUMAR\Climate_Downscaling_Project\data\temperature_Gcm_data\observed_data\IMD_AVERAGE_TEMP_0.25"
OUTPUT_DIR_BASE = r"C:\Users\ADAPA SUNIL KUMAR\Climate_Downscaling_Project\data\processed\bias_corrected\INM-CM4-8"

# 2. File and NetCDF Variable Naming (Set based on your data info)
OBSERVED_FILE_PATTERN = "IMD_*_avg_temp_0.25.nc" 
GCM_FILE_PATTERN = "*.nc"
TIME_VAR = "time"          # Based on your data info
OBS_DATA_VAR = "avg_temp"  # Based on your observed data info
GCM_DATA_VAR = "data"      # Based on your GCM data info

# 3. Time Periods for analysis
TRAINING_START_DATE = "1975-01-01"
TRAINING_END_DATE = "2000-12-31"
APPLICATION_START_DATE = "2001-01-01"
APPLICATION_END_DATE = "2014-12-31"

# --- END OF CONFIGURATION ---


def main():
    """Main script to run the bias correction process for all GCMs."""
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

    print("Step 1: Loading and preparing observed data...")
    obs_filepath_pattern = os.path.join(INPUT_DIR_OBSERVED, OBSERVED_FILE_PATTERN)
    try:
        obs_ds = xr.open_mfdataset(obs_filepath_pattern, combine='by_coords')
        obs_da = obs_ds[OBS_DATA_VAR]
        obs_da.attrs['units'] = 'degC' 
        obs_train = obs_da.sel({TIME_VAR: slice(TRAINING_START_DATE, TRAINING_END_DATE)}).load()
        print(" -> Observed data loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load observed data. Check path and pattern.")
        print(f" -> Path tried: {obs_filepath_pattern}")
        print(f" -> Error details: {e}")
        return

    gcm_folders = [f.path for f in os.scandir(INPUT_DIR_GCM_BASE) if f.is_dir()]
    if not gcm_folders:
        print(f"FATAL: No GCM subfolders found in '{INPUT_DIR_GCM_BASE}'. Please check the path.")
        return

    print(f"\nStep 2: Found {len(gcm_folders)} GCM(s) to process.")
    
    for gcm_folder_path in gcm_folders:
        gcm_name = os.path.basename(gcm_folder_path)
        print(f"\n--- Processing GCM: {gcm_name} ---")

        gcm_output_dir = os.path.join(OUTPUT_DIR_BASE, gcm_name)
        os.makedirs(gcm_output_dir, exist_ok=True)

        try:
            gcm_pattern = os.path.join(gcm_folder_path, GCM_FILE_PATTERN)
            gcm_ds = xr.open_mfdataset(gcm_pattern, combine='by_coords')
            gcm_da = gcm_ds[GCM_DATA_VAR]
            gcm_da.attrs['units'] = 'degC'

            # --- Handle Coordinate Mismatches ---
            # Rename GCM coordinates to match Observed data for consistency
            rename_dict = {}
            if 'lat' in gcm_da.coords and 'LATITUDE' in obs_da.coords:
                rename_dict['lat'] = 'LATITUDE'
            if 'lon' in gcm_da.coords and 'LONGITUDE' in obs_da.coords:
                rename_dict['lon'] = 'LONGITUDE'
            if rename_dict:
                print(f"  -> Harmonizing coordinate names: {rename_dict}")
                gcm_da = gcm_da.rename(rename_dict)
            # ------------------------------------

            gcm_train = gcm_da.sel({TIME_VAR: slice(TRAINING_START_DATE, TRAINING_END_DATE)}).load()
            gcm_apply = gcm_da.sel({TIME_VAR: slice(APPLICATION_START_DATE, APPLICATION_END_DATE)})

            if gcm_apply.time.size == 0:
                print("  -> WARNING: No data in application period. Skipping.")
                continue

            # Apply Empirical Quantile Mapping
            print("  -> Training EQM model and applying correction...")
            eqm = EmpiricalQuantileMapping.train(gcm_train, obs_train)
            corrected_eqm = eqm.adjust(gcm_apply)
            
            output_eqm_path = os.path.join(gcm_output_dir, f"{gcm_name}_EQM_corrected.nc")
            corrected_eqm.to_netcdf(output_eqm_path)
            print(f"  -> Saved: {os.path.basename(output_eqm_path)}")

        except Exception as e:
            print(f"  -> ERROR: Failed to process {gcm_name}. Skipping. Reason: {e}")
            continue

    print("\n✅ All GCMs processed successfully!")

if __name__ == "__main__":
    main()