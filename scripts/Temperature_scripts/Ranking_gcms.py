import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from typing import Tuple

# --- CONFIGURATION: EDIT THESE VALUES CAREFULLY ---

# 1. Directory and File Configuration
INPUT_DATA_DIR = "data/temperature_rank"
OUTPUT_DIR = "results"
PLOTS_DIR = "plots"

# 2. Column Names in your Excel files (MUST match exactly)
GCM_NAME_COLUMN = "GCM"
VARIABLE_NAME_COLUMN = "Variable"
PCC_COLUMN = "PCC"
STDNORM_COLUMN = "stdNorm"
RMSE_COLUMN = "RMSE"

# 3. Comprehensive Skill Score Weights (used to calculate the score itself)
CSS_WEIGHTS = {
    'PCC': 0.5,
    'RMSE': 0.3,
    'StdNorm': 0.2
}

# 4. AHP and MCDM Configuration
# CORRECTED: The criteria for formal ranking are now only the 3 fundamental metrics.
MCDM_CRITERIA = ['PCC', 'RMSE', 'StdNorm_Error']
# CORRECTED: Benefit attributes match the 3 fundamental metrics.
BENEFIT_ATTRIBUTES = np.array([True, False, False])

# CORRECTED: The AHP matrix is now a 3x3, comparing only the fundamental metrics.
AHP_COMPARISON_MATRIX = np.array([
    [1, 3, 5],    # PCC vs (PCC, RMSE, StdNorm_Error)
    [1/3, 1, 2],  # RMSE vs (PCC, RMSE, StdNorm_Error)
    [1/5, 1/2, 1]   # StdNorm_Error vs (PCC, RMSE, StdNorm_Error)
])

VIKOR_V = 0.5

# --- END OF CONFIGURATION ---


# --- HELPER FUNCTIONS (No need to edit) ---

def calculate_taylor_score(pcc: pd.Series, stdnorm: pd.Series) -> pd.Series:
    """Calculates the Taylor Score."""
    return ((1 + pcc) ** 4) / (4 * ((stdnorm + 1 / stdnorm) ** 2))

def ahp_weights(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculates criteria weights and consistency ratio from an AHP matrix."""
    n = matrix.shape[0]
    normalized_matrix = matrix / matrix.sum(axis=0)
    weights = normalized_matrix.sum(axis=1) / n
    random_index = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12}
    lambda_max = np.mean(np.dot(matrix, weights) / weights)
    consistency_index = (lambda_max - n) / (n - 1) if n > 1 else 0
    consistency_ratio = consistency_index / random_index.get(n, 1.45)
    return weights, consistency_ratio

def topsis(matrix: pd.DataFrame, weights: np.ndarray, benefit_attrs: np.ndarray) -> pd.DataFrame:
    """Performs TOPSIS ranking."""
    norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    ideal_best = np.where(benefit_attrs, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_worst = np.where(benefit_attrs, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))
    dist_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
    closeness = dist_worst / (dist_best + dist_worst)
    return pd.DataFrame({
        'GCM': matrix.index,
        'TOPSIS_Score': closeness,
        'TOPSIS_Rank': closeness.rank(ascending=False, method='min')
    }).set_index('GCM')

def vikor(matrix: pd.DataFrame, weights: np.ndarray, benefit_attrs: np.ndarray, v: float) -> pd.DataFrame:
    """Performs VIKOR ranking."""
    f_best = np.where(benefit_attrs, matrix.max(axis=0), matrix.min(axis=0))
    f_worst = np.where(benefit_attrs, matrix.min(axis=0), matrix.max(axis=0))
    epsilon = 1e-9
    normalized_diff = (f_best - matrix) / (f_best - f_worst + epsilon)
    weighted_diff = normalized_diff * weights
    S = weighted_diff.sum(axis=1)
    R = weighted_diff.max(axis=1)
    q_values = v * (S - S.min()) / (S.max() - S.min() + epsilon) + (1 - v) * (R - R.min()) / (R.max() - R.min() + epsilon)
    return pd.DataFrame({
        'GCM': matrix.index,
        'VIKOR_Q': q_values,
        'VIKOR_Rank': q_values.rank(ascending=True, method='min')
    }).set_index('GCM')

def plot_step_ranking(df: pd.DataFrame, score_col: str, title: str, xlabel: str, output_path: str, ascending_sort: bool = False):
    """Generates and saves a bar plot for any given score or rank column."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = df.sort_values(score_col, ascending=ascending_sort).head(20)
    sns.barplot(x=top_20[score_col], y=top_20.index, palette="viridis", ax=ax, orient='h')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('GCM', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      -> Plot saved: {os.path.basename(output_path)}")

def plot_final_ranking(ranking_df: pd.DataFrame, score_col: str, output_path: str):
    """Generates and saves the final master plot."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    top_20 = ranking_df.head(20).sort_values(score_col, ascending=False)
    sns.barplot(x=top_20[score_col], y=top_20.index, palette="plasma", ax=ax, orient='h')
    ax.set_xlabel('Final Borda Score (Higher is Better)', fontsize=12)
    ax.set_title('Overall Top 20 GCMs Across All Variables', fontsize=16, weight='bold')
    ax.set_ylabel('GCM', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Final plot saved to {output_path}")

def main():
    """Orchestrates the entire GCM ranking workflow."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Step 1: Calculating AHP weights for MCDM ranking...")
    mcdm_weights, cr = ahp_weights(AHP_COMPARISON_MATRIX)
    print(f"  -> MCDM Weights: {dict(zip(MCDM_CRITERIA, np.round(mcdm_weights, 3)))}")
    print(f"  -> Consistency Ratio (CR): {cr:.3f}{' (Good)' if cr <= 0.1 else ' (Inconsistent!)'}")

    print("\nStep 2: Processing each variable file...")
    excel_files = glob.glob(os.path.join(INPUT_DATA_DIR, "*.xlsx"))
    if not excel_files:
        print(f"FATAL: No Excel (.xlsx) files found in '{INPUT_DATA_DIR}'. Exiting.")
        return

    all_final_ranks = {}
    for filepath in excel_files:
        df = pd.read_excel(filepath)
        
        try:
            variable_name = df[VARIABLE_NAME_COLUMN].iloc[0]
        except (KeyError, IndexError):
            print(f"  -> WARNING: Could not find column '{VARIABLE_NAME_COLUMN}' in {os.path.basename(filepath)}. Using filename instead.")
            variable_name = os.path.splitext(os.path.basename(filepath))[0]
            
        print(f"\n  Processing Variable: {variable_name}")
        df = df.set_index(GCM_NAME_COLUMN)
        
        variable_plot_dir = os.path.join(PLOTS_DIR, variable_name)
        os.makedirs(variable_plot_dir, exist_ok=True)

        # Calculate composite scores for supplementary analysis and plotting
        df['Taylor_Score'] = calculate_taylor_score(df[PCC_COLUMN], df[STDNORM_COLUMN])
        norm_pcc = (df[PCC_COLUMN] - df[PCC_COLUMN].min()) / (df[PCC_COLUMN].max() - df[PCC_COLUMN].min())
        norm_rmse = (df[RMSE_COLUMN].max() - df[RMSE_COLUMN]) / (df[RMSE_COLUMN].max() - df[RMSE_COLUMN].min())
        std_error = abs(df[STDNORM_COLUMN] - 1)
        norm_std = (std_error.max() - std_error) / (std_error.max() - std_error.min())
        df['Comprehensive_Skill_Score'] = (norm_pcc * CSS_WEIGHTS['PCC'] + 
                                           norm_rmse * CSS_WEIGHTS['RMSE'] + 
                                           norm_std * CSS_WEIGHTS['StdNorm']).fillna(0)
        
        plot_step_ranking(df, 'Taylor_Score', f'Ranking by Taylor Score ({variable_name})', 'Taylor Score (Higher is Better)', os.path.join(variable_plot_dir, '1_Taylor_Score_Rank.png'), ascending_sort=False)
        plot_step_ranking(df, 'Comprehensive_Skill_Score', f'Ranking by Comprehensive Skill Score ({variable_name})', 'Comprehensive Skill Score (Higher is Better)', os.path.join(variable_plot_dir, '2_Comprehensive_Skill_Score_Rank.png'), ascending_sort=False)
        
        # CORRECTED: The decision matrix for TOPSIS/VIKOR uses only the fundamental metrics.
        decision_matrix = pd.DataFrame({
            'PCC': df[PCC_COLUMN],
            'RMSE': df[RMSE_COLUMN],
            'StdNorm_Error': abs(df[STDNORM_COLUMN] - 1)
        })

        # Run TOPSIS and VIKOR on the corrected 3-criteria matrix
        topsis_results = topsis(decision_matrix, mcdm_weights, BENEFIT_ATTRIBUTES)
        vikor_results = vikor(decision_matrix, mcdm_weights, BENEFIT_ATTRIBUTES, VIKOR_V)
        
        plot_step_ranking(topsis_results, 'TOPSIS_Score', f'Ranking by TOPSIS ({variable_name})', 'TOPSIS Score (Higher is Better)', os.path.join(variable_plot_dir, '3_TOPSIS_Rank.png'), ascending_sort=False)
        plot_step_ranking(vikor_results, 'VIKOR_Q', f'Ranking by VIKOR ({variable_name})', 'VIKOR Q-Score (Lower is Better)', os.path.join(variable_plot_dir, '4_VIKOR_Rank.png'), ascending_sort=True)

        # Combine results and aggregate ranks for the variable
        results_df = df.join([topsis_results, vikor_results])
        results_df['Aggregated_Rank'] = results_df[['TOPSIS_Rank', 'VIKOR_Rank']].mean(axis=1).rank(method='min')
        all_final_ranks[variable_name] = results_df['Aggregated_Rank']
        
        plot_step_ranking(results_df, 'Aggregated_Rank', f'Final Aggregated Rank ({variable_name})', 'Rank (Lower is Better)', os.path.join(variable_plot_dir, '5_Final_Variable_Rank.png'), ascending_sort=True)

        output_path = os.path.join(OUTPUT_DIR, f"{variable_name}_full_results.csv")
        results_df.to_csv(output_path)
        print(f"  -> Detailed results saved to {os.path.basename(output_path)}")

    # Aggregate ranks across all variables
    print("\nStep 3: Aggregating ranks across all variables...")
    master_ranking_df = pd.DataFrame(all_final_ranks)
    num_gcms = len(master_ranking_df)
    borda_points = num_gcms - master_ranking_df + 1
    master_ranking_df['Final_Borda_Score'] = borda_points.sum(axis=1)
    master_ranking_df.sort_values('Final_Borda_Score', ascending=False, inplace=True)
    master_output_path = os.path.join(OUTPUT_DIR, "MASTER_OVERALL_RANKING.csv")
    master_ranking_df.to_csv(master_output_path)
    print(f"  -> Master ranking saved to {os.path.basename(master_output_path)}")

    # Generate final plot
    print("\nStep 4: Generating final summary plot...")
    plot_path = os.path.join(PLOTS_DIR, "Overall_GCM_Ranking_Plot.png")
    plot_final_ranking(master_ranking_df, "Final_Borda_Score", plot_path)

    print("\n Workflow Completed Successfully!")

if __name__ == "__main__":
    main()