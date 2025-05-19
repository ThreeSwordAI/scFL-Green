import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration of Paths ---
# (Relative to the current working directory: oFed-sc\Analysis)
results_base = os.path.join("..", "methodoloy", "results", "dataset_A")
analysis_data_dir = os.path.join("..", "Analysis", "Analysis Data")
png_dir = os.path.join("..", "Analysis", "png")
pdf_dir = os.path.join("..", "Analysis", "pdf")

os.makedirs(analysis_data_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# --- Define Frameworks and file/column names ---
frameworks = ["scDLC", "scCDCG", "scSMD", "ACTINN"]
# For Deep Learning (DL):
# For ACTINN, file is "ACTINN_DL_results.csv", for others, "{framework}_DL_results_pre_train.csv"
# Column for baseline emissions is "DL_Baseline_emissions"
# For Federated Learning (FL):
# For ACTINN, file is "ACTINN_FL_results.csv", for others, "{framework}_FL_results_pre_train.csv"
# Column: "FL_Baseline_emissions"

records = []  # to hold emission records

# Process DL emissions for dataset_A
for fw in frameworks:
    if fw == "ACTINN":
        file_name = f"{fw}_DL_results.csv"
        col_name = "DL_Baseline_emissions"
    else:
        file_name = f"{fw}_DL_results_pre_train.csv"
        col_name = "DL_Baseline_emissions"
    file_path = os.path.join(results_base, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        continue
    df = pd.read_csv(file_path)
    # Assume one row per file
    emission = df.iloc[0][col_name]
    records.append({
        "Framework": fw,
        "Learning": "Deep Learning",
        "Dataset": "dataset_A",
        "Emissions": emission
    })

# Process FL emissions for dataset_A
for fw in frameworks:
    if fw == "ACTINN":
        file_name = f"{fw}_FL_results.csv"
        col_name = "FL_Baseline_emissions"
    else:
        file_name = f"{fw}_FL_results_pre_train.csv"
        col_name = "FL_Baseline_emissions"
    file_path = os.path.join(results_base, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        continue
    df = pd.read_csv(file_path)
    emission = df.iloc[0][col_name]
    records.append({
        "Framework": fw,
        "Learning": "Federated Learning",
        "Dataset": "dataset_A",
        "Emissions": emission
    })

# Create a DataFrame and save the processed data CSV.
emissions_df = pd.DataFrame(records)
csv_path = os.path.join(analysis_data_dir, "Emissions_BarPlot_dataset_A.csv")
emissions_df.to_csv(csv_path, index=False)
print(f"Processed emissions data saved to {csv_path}")

emissions_df = pd.read_csv(csv_path)
# Plot the bar plot.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=emissions_df, x="Framework", y="Emissions", hue="Learning")
ax.set_title("Baseline Emissions (Actual Values) for Dataset A")
plt.tight_layout()

png_file = os.path.join(png_dir, "Emissions_BarPlot_dataset_A.png")
pdf_file = os.path.join(pdf_dir, "Emissions_BarPlot_dataset_A.pdf")
plt.savefig(png_file, dpi=300)
plt.savefig(pdf_file)
plt.close()
print(f"Bar plot saved to {png_file} and {pdf_file}")
