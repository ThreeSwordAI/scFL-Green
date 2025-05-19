import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define base paths.
results_base    = r"../methodoloy/results"
analysis_data_dir = r"../Analysis/Analysis Data"
png_dir         = r"../Analysis/png"
pdf_dir         = r"../Analysis/pdf"

# Ensure output directories exist
os.makedirs(analysis_data_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# Configuration
datasets   = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
frameworks = ["scDLC", "scCDCG", "scSMD", "ACTINN"]

# Collect F1 records
records = []

# Process DL baseline F1
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_DL_results.csv"
            col_name  = "DL_Baseline_F1"
        else:
            file_name = f"{fw}_DL_results_pre_train.csv"
            col_name  = "DL_Baseline_F1"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        f1_val = df.iloc[0][col_name]
        records.append({
            "Framework": fw,
            "Learning": "Deep Learning",
            "Dataset": ds,
            "F1": f1_val
        })

# Process FL baseline F1
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_FL_results.csv"
            col_name  = "FL_Baseline_F1"
        else:
            file_name = f"{fw}_FL_results_pre_train.csv"
            col_name  = "FL_Baseline_F1"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        f1_val = df.iloc[0][col_name]
        records.append({
            "Framework": fw,
            "Learning": "Federated Learning",
            "Dataset": ds,
            "F1": f1_val
        })

# Create DataFrame and save to CSV
f1_df = pd.DataFrame(records)
f1_csv_path = os.path.join(analysis_data_dir, "F1_BoxPlot.csv")
f1_df.to_csv(f1_csv_path, index=False)
print(f"Processed F1 data saved to {f1_csv_path}")

#----------------------------------------------------------------------------
# Plotting: Boxplot of F1 by Framework and Learning type
#----------------------------------------------------------------------------

# Increase all font sizes globally
plt.rcParams.update({'font.size': 20})

# Set theme
sns.set_theme(style="whitegrid")

# Create figure + axes
fig, ax = plt.subplots(figsize=(10, 8))

# Draw boxplot with hue
sns.boxplot(
    data = f1_df,
    x = "Framework",
    y = "F1",
    hue = "Learning",
    ax = ax
)

# Title and axis labels with larger fonts
ax.set_title("F1 Score vs Framework", fontsize=24)
ax.set_xlabel("Framework",             fontsize=22)
ax.set_ylabel("F1 Score",              fontsize=22)

# Enlarge tick labels
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

# Remove top/right spines for cleanliness
sns.despine(ax=ax)

# Move legend to bottom-left corner inside plot
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels,
    title_fontsize=20,
    fontsize=18,
    loc='lower left',
    bbox_to_anchor=(0.01, 0.01),
    frameon=True
)

plt.tight_layout()

# Save outputs
png_file = os.path.join(png_dir, "F1_BoxPlot.png")
pdf_file = os.path.join(pdf_dir, "F1_BoxPlot.pdf")
plt.savefig(png_file, dpi=300)
plt.savefig(pdf_file)
plt.close()
print(f"F1 box plot saved to {png_file} and {pdf_file}")

