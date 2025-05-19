import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

########################################
# Configuration / Setup
########################################

# Base paths (adjust if needed)
results_base = "../methodoloy/results"  # Where your results are stored
analysis_data_folder = "../Analysis/Analysis Data"
png_folder = "../Analysis/png"
pdf_folder = "../Analysis/pdf"

os.makedirs(analysis_data_folder, exist_ok=True)
os.makedirs(png_folder, exist_ok=True)
os.makedirs(pdf_folder, exist_ok=True)

# We have 4 datasets, each with FL results
datasets = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]

# Our frameworks and corresponding FL result file names
framework_files = {
    "scDLC":    "scDLC_FL_results_pre_train.csv",
    "scCDCG":   "scCDCG_FL_results_pre_train.csv",
    "scSMD":    "scSMD_FL_results_pre_train.csv",
    "ACTINN":   "ACTINN_FL_results.csv"
}

# Columns of interest (FL methods) for each metric
fl_accuracy_cols = [
    "FL_Baseline_Accuracy",
    "FL_SmallBatch_Accuracy",
    "FL_MixedPrecision_Accuracy",
    "FL_ReduceComplexity_Accuracy"
]
fl_f1_cols = [
    "FL_Baseline_F1",
    "FL_SmallBatch_F1",
    "FL_MixedPrecision_F1",
    "FL_ReduceComplexity_F1"
]
fl_emission_cols = [
    "FL_Baseline_emissions",
    "FL_SmallBatch_emissions",
    "FL_MixedPrecision_emissions",
    "FL_ReduceComplexity_emissions"
]

########################################
# Helper function to min–max normalize
########################################
def minmax_norm(grp):
    mn, mx = grp.min(), grp.max()
    if mx == mn:
        return grp * 0
    return (grp - mn) / (mx - mn)

########################################
# 1) Load & Reshape Data
########################################

all_rows_accuracy = []
all_rows_f1 = []
all_rows_emissions = []

for ds in datasets:
    for framework, file_name in framework_files.items():
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        
        # ACCURACY
        if any(c in df.columns for c in fl_accuracy_cols):
            df_acc = df[[c for c in fl_accuracy_cols if c in df.columns]].melt(
                var_name="Method", value_name="Accuracy"
            )
            df_acc["Method"] = df_acc["Method"].str.replace("FL_", "").str.replace("_Accuracy", "")
            df_acc["Dataset"] = ds
            df_acc["Framework"] = framework
            all_rows_accuracy.append(df_acc)
        
        # F1
        if any(c in df.columns for c in fl_f1_cols):
            df_f1 = df[[c for c in fl_f1_cols if c in df.columns]].melt(
                var_name="Method", value_name="F1"
            )
            df_f1["Method"] = df_f1["Method"].str.replace("FL_", "").str.replace("_F1", "")
            df_f1["Dataset"] = ds
            df_f1["Framework"] = framework
            all_rows_f1.append(df_f1)
        
        # EMISSIONS
        if any(c in df.columns for c in fl_emission_cols):
            df_em = df[[c for c in fl_emission_cols if c in df.columns]].melt(
                var_name="Method", value_name="Emissions"
            )
            df_em["Method"] = df_em["Method"].str.replace("FL_", "").str.replace("_emissions", "")
            df_em["Dataset"] = ds
            df_em["Framework"] = framework
            all_rows_emissions.append(df_em)

# Concat all
df_accuracy = pd.concat(all_rows_accuracy, ignore_index=True) if all_rows_accuracy else pd.DataFrame()
df_f1 = pd.concat(all_rows_f1, ignore_index=True) if all_rows_f1 else pd.DataFrame()
df_emissions = pd.concat(all_rows_emissions, ignore_index=True) if all_rows_emissions else pd.DataFrame()

########################################
# 2) Normalization for Emissions
########################################
if not df_emissions.empty:
    df_emissions["Normalized_Emissions"] = df_emissions.groupby("Dataset")["Emissions"].transform(minmax_norm)

########################################
# 3) Save Processed Data
########################################
df_accuracy.to_csv(os.path.join(analysis_data_folder, "FL_Accuracy_OnlyFL.csv"), index=False)
df_f1.to_csv(os.path.join(analysis_data_folder, "FL_F1_OnlyFL.csv"), index=False)
df_emissions.to_csv(os.path.join(analysis_data_folder, "FL_Emissions_OnlyFL.csv"), index=False)
print("Processed data saved.")

########################################
# 4) Plotting with larger fonts and adjusted legends
########################################
# Increase all font sizes globally
plt.rcParams.update({'font.size': 20})
sns.set_theme(style="whitegrid")

four_color_palette = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

# Accuracy plot
if not df_accuracy.empty:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_accuracy, 
        x="Framework", 
        y="Accuracy", 
        hue="Method", 
        palette=four_color_palette
    )
    ax.set_title("Federated Learning Accuracy by Framework", fontsize=24)
    ax.set_xlabel("Framework", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    plt.legend(title="FL Method", title_fontsize=20, fontsize=18, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(png_folder, "FL_OnlyMethods_Accuracy.png"), dpi=300)
    plt.savefig(os.path.join(pdf_folder, "FL_OnlyMethods_Accuracy.pdf"))
    plt.close()
    print("Boxplot for FL Accuracy saved.")

# F1 plot with legend bottom-left
if not df_f1.empty:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_f1, 
        x="Framework", 
        y="F1", 
        hue="Method", 
        palette=four_color_palette
    )
    ax.set_title("Federated Learning F1 by Framework", fontsize=24)
    ax.set_xlabel("Framework", fontsize=22)
    ax.set_ylabel("F1 Score", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    plt.legend(title="FL Method", title_fontsize=20, fontsize=18, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(png_folder, "FL_OnlyMethods_F1.png"), dpi=300)
    plt.savefig(os.path.join(pdf_folder, "FL_OnlyMethods_F1.pdf"))
    plt.close()
    print("Boxplot for FL F1 saved.")

# Emissions plot with legend top-left
if not df_emissions.empty:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_emissions, 
        x="Framework", 
        y="Normalized_Emissions", 
        hue="Method", 
        palette=four_color_palette
    )
    ax.set_title("Federated Learning Emissions by Framework (Normalized 0-1)", fontsize=24)
    ax.set_xlabel("Framework", fontsize=22)
    ax.set_ylabel("Normalized Emissions", fontsize=22)
    ax.tick_params(axis='both', labelsize=20)
    plt.legend(title="FL Method", title_fontsize=20, fontsize=18, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(png_folder, "FL_OnlyMethods_Emissions_Normalized.png"), dpi=300)
    plt.savefig(os.path.join(pdf_folder, "FL_OnlyMethods_Emissions_Normalized.pdf"))
    plt.close()
    print("Boxplot for FL Emissions (normalized) saved.")

print("All plots generated successfully!")
