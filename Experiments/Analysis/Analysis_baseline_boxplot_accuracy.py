import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define base paths.
# Assuming that "oFed-sc\Analysis" is the current working directory,
# the results are stored in "oFed-sc\methodoloy\results"
results_base = r"../methodoloy/results"  
analysis_data_dir = r"../Analysis/Analysis Data"
png_dir = r"../Analysis/png"
pdf_dir = r"../Analysis/pdf"

# Ensure output directories exist
os.makedirs(analysis_data_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)

# Define datasets and frameworks.
datasets = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
frameworks = ["scDLC", "scCDCG", "scSMD", "ACTINN"]

# For deep learning (DL) results:
# - For ACTINN, the file is named "ACTINN_DL_results.csv" and the accuracy column is "DL_Baseline_Accuracy"
# - For the other frameworks, the file is named "{framework}_DL_results_pre_train.csv"
# For federated learning (FL) results:
# - For ACTINN, file is "ACTINN_FL_results.csv"
# - For others, "{framework}_FL_results_pre_train.csv"
# And the columns for FL are "FL_Baseline_Accuracy".

records = []  # List to collect each record

# Process Deep Learning (DL) baseline accuracy for each dataset and framework.
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_DL_results.csv"
            col_name = "DL_Baseline_Accuracy"
        else:
            file_name = f"{fw}_DL_results_pre_train.csv"
            col_name = "DL_Baseline_Accuracy"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        # Assuming that the results CSV has one row and the baseline accuracy in the proper column.
        acc_val = df.iloc[0][col_name]
        records.append({
            "Framework": fw,
            "Learning": "Deep Learning",
            "Dataset": ds,
            "Accuracy": acc_val
        })

# Process Federated Learning (FL) baseline accuracy.
for ds in datasets:
    for fw in frameworks:
        if fw == "ACTINN":
            file_name = f"{fw}_FL_results.csv"
            col_name = "FL_Baseline_Accuracy"
        else:
            file_name = f"{fw}_FL_results_pre_train.csv"
            col_name = "FL_Baseline_Accuracy"
        file_path = os.path.join(results_base, ds, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
        acc_val = df.iloc[0][col_name]
        records.append({
            "Framework": fw,
            "Learning": "Federated Learning",
            "Dataset": ds,
            "Accuracy": acc_val
        })

# Create a DataFrame from records and save to CSV in the analysis data folder.
acc_df = pd.DataFrame(records)
acc_csv_path = os.path.join(analysis_data_dir, "Accuracy_BoxPlot.csv")
acc_df.to_csv(acc_csv_path, index=False)
print(f"Processed accuracy data saved to {acc_csv_path}")

# Now plot the box plot.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.boxplot(data=acc_df, x="Framework", y="Accuracy", hue="Learning")
ax.set_title("Accuracy vs Framework")
plt.tight_layout()

# Save the plot in PNG and PDF formats.
png_file = os.path.join(png_dir, "Accuracy_BoxPlot.png")
pdf_file = os.path.join(pdf_dir, "Accuracy_BoxPlot.pdf")
plt.savefig(png_file, dpi=300)
plt.savefig(pdf_file)
plt.close()
print(f"Accuracy box plot saved to {png_file} and {pdf_file}")
