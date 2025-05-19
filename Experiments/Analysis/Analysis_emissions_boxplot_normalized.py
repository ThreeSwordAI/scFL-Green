import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration of Paths ---
results_base     = os.path.join("..", "methodoloy", "results")
analysis_data_dir = os.path.join("..", "Analysis", "Analysis Data")
png_dir          = os.path.join("..", "Analysis", "png")
pdf_dir          = os.path.join("..", "Analysis", "pdf")

os.makedirs(analysis_data_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

datasets   = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
frameworks = ["scDLC", "scCDCG", "scSMD", "ACTINN"]

records = []  # to hold emissions records

# For Deep Learning (DL)
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_DL_results.csv"
            col_name  = "DL_Baseline_emissions"
        else:
            file_name = f"{fw}_DL_results_pre_train.csv"
            col_name  = "DL_Baseline_emissions"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        emission = df.iloc[0][col_name]
        records.append({
            "Dataset": ds,
            "Framework": fw,
            "Learning": "Deep Learning",
            "Emissions": emission
        })

# For Federated Learning (FL)
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_FL_results.csv"
            col_name  = "FL_Baseline_emissions"
        else:
            file_name = f"{fw}_FL_results_pre_train.csv"
            col_name  = "FL_Baseline_emissions"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        emission = df.iloc[0][col_name]
        records.append({
            "Dataset": ds,
            "Framework": fw,
            "Learning": "Federated Learning",
            "Emissions": emission
        })

# Create & normalize DataFrame
em_df = pd.DataFrame(records)
def normalize(group):
    min_val, max_val = group.min(), group.max()
    return (group - min_val) / (max_val - min_val) if max_val != min_val else 0

em_df["Normalized_Emissions"] = em_df.groupby("Dataset")["Emissions"].transform(normalize)

# Save normalized data
norm_csv = os.path.join(analysis_data_dir, "Emissions_BoxPlot_Normalized.csv")
em_df.to_csv(norm_csv, index=False)
print(f"Processed normalized emissions data saved to {norm_csv}")

# --- Plotting ---
# Increase all font sizes
plt.rcParams.update({'font.size': 20})
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 8))
# Boxplot
sns.boxplot(
    data=em_df,
    x="Framework",
    y="Normalized_Emissions",
    hue="Learning",
    ax=ax
)

# Titles & labels with larger fonts
ax.set_title(
    "Normalized Baseline Emissions (0-1) Across Datasets (DL & FL)",
    fontsize=24,
    pad=15
)
ax.set_xlabel("Framework",    fontsize=22, labelpad=10)
ax.set_ylabel("Normalized Emissions", fontsize=22, labelpad=10)

# Tick labels bigger
ax.tick_params(axis='both', labelsize=20)

# Legend in top-left corner
ax.legend(
    title_fontsize=20,
    fontsize=18,
    loc='upper left',
    frameon=True
)

# Layout & save
plt.tight_layout()
png_file = os.path.join(png_dir, "Emissions_BoxPlot_Normalized.png")
pdf_file = os.path.join(pdf_dir, "Emissions_BoxPlot_Normalized.pdf")
plt.savefig(png_file, dpi=300)
plt.savefig(pdf_file)
plt.close()
print(f"Normalized emissions box plot saved to {png_file} and {pdf_file}")
