import os
import pandas as pd

# Define your frameworks and their FL results filenames
# (Adjust filenames/paths as needed if they differ in your system)
FL_FILENAMES = {
    "scDLC":   "scDLC_FL_results_pre_train.csv",
    "scCDCG":  "scCDCG_FL_results_pre_train.csv",
    "scSMD":   "scSMD_FL_results_pre_train.csv",
    "ACTINN":  "ACTINN_FL_results.csv"
}

# Define the four datasets (A, B, C, D) and the corresponding subfolders
DATASETS = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]

# The columns in each FL result CSV that contain the per-method F1 and Emission data
# e.g., for scDLC_FL_results_pre_train.csv, we typically see:
# FL_Baseline_F1, FL_Baseline_emissions, etc.
# Adjust if your actual columns differ.
METHODS = ["Baseline", "SmallBatch", "MixedPrecision", "ReduceComplexity"]

# The structure for F1 and Emissions columns for each method, e.g.:
# FL_Baseline_F1, FL_Baseline_emissions, ...
def f1_col(method):
    return f"FL_{method}_F1"

def emission_col(method):
    return f"FL_{method}_emissions"


# Prepare an empty DataFrame for final results
# Index: the frameworks, Columns: the four datasets
best_methods_df = pd.DataFrame(index=FL_FILENAMES.keys(), columns=DATASETS)

for dataset in DATASETS:
    # Path to the folder with the results for this dataset
    # e.g., "../../results/dataset_A", etc.
    dataset_results_path = os.path.join("../methodoloy/results", dataset)
    
    for framework, filename in FL_FILENAMES.items():
        # Full path to the CSV file for this framework's FL results in the current dataset
        csv_path = os.path.join(dataset_results_path, filename)
        
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            # We can skip or set a placeholder
            best_methods_df.loc[framework, dataset] = "N/A"
            continue
        
        # Load the CSV with F1 and emission data
        df = pd.read_csv(csv_path)
        # We assume there's only one row in these summary CSVs
        # that has the aggregated F1 and emission values for each method
        # e.g. the columns: FL_Baseline_F1, FL_Baseline_emissions, etc.
        # If your CSV has multiple rows, you might need to handle that differently.
        if df.empty:
            print(f"Warning: {csv_path} is empty or no data.")
            best_methods_df.loc[framework, dataset] = "N/A"
            continue
        
        # Typically there's only 1 row, so let's take row 0:
        row = df.iloc[0]
        
        # We'll compute EAPS for each method
        best_method = None
        best_eaps = -1.0
        
        for method in METHODS:
            # Extract F1 and emission
            # e.g. row["FL_Baseline_F1"], row["FL_Baseline_emissions"]
            col_f1 = f1_col(method)       # e.g. "FL_Baseline_F1"
            col_emission = emission_col(method)  # e.g. "FL_Baseline_emissions"
            
            if col_f1 not in row or col_emission not in row:
                # skip if not present
                continue
            
            f1_value = row[col_f1]
            emission_value = row[col_emission]
            
            # Edge case: if emission_value == 0, skip or handle carefully to avoid division by zero
            if emission_value <= 0:
                # skip or set eaps to something minimal
                continue
            
            eaps = f1_value / emission_value
            
            # Check if it's the best so far
            if eaps > best_eaps:
                best_eaps = eaps
                best_method = method
        
        if best_method is None:
            # Could happen if no columns found or emission=0
            best_methods_df.loc[framework, dataset] = "N/A"
        else:
            best_methods_df.loc[framework, dataset] = best_method

# Rename columns to "Dataset A", "Dataset B", etc. if desired
best_methods_df.rename(columns={
    "dataset_A": "Dataset A",
    "dataset_B": "Dataset B",
    "dataset_C": "Dataset C",
    "dataset_D": "Dataset D"
}, inplace=True)

# Save the final table as CSV
output_csv_path = "../../Analysis/Analysis Data/EAPS_BestMethods.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
best_methods_df.to_csv(output_csv_path)

print("EAPS best methods table saved to:", output_csv_path)
print(best_methods_df)