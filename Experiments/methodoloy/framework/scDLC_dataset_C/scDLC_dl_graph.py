import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Update file paths for Dataset B
pre_train_results_path = "../../results/dataset_C/scDLC_DL_per_epoch_result_pre_train.csv"
fine_tune_results_A_path = "../../results/dataset_C/scDLC_DL_results_fine_tune_A.csv"
fine_tune_results_B_path = "../../results/dataset_C/scDLC_DL_results_fine_tune_B.csv"
graph_folder_path = "../../graphs/dataset_C"

os.makedirs(graph_folder_path, exist_ok=True)
os.makedirs(f"{graph_folder_path}/pdf", exist_ok=True)
os.makedirs(f"{graph_folder_path}/png", exist_ok=True)

# Load datasets
pre_train_results = pd.read_csv(pre_train_results_path)
fine_tune_results_A = pd.read_csv(fine_tune_results_A_path)
fine_tune_results_B = pd.read_csv(fine_tune_results_B_path)

sns.set_theme(style="whitegrid")

# --- First Plot: Train vs Test Accuracy per Epoch (Pre-Training) ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=pre_train_results, x="Epoch", y="Train_Accuracy", hue="Method", marker="o")
plt.title("scDLC Deep Learning Train Accuracy per Epoch (Pre-Training) - Dataset C")
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.legend(title="Method")

plt.subplot(1, 2, 2)
sns.lineplot(data=pre_train_results, x="Epoch", y="Test_Accuracy", hue="Method", marker="o")
plt.title("scDLC Deep Learning Test Accuracy per Epoch (Pre-Training) - Dataset C")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend(title="Method")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/pdf/scDLC_DL_train_vs_test_accuracy.pdf")
plt.savefig(f"{graph_folder_path}/png/scDLC_DL_train_vs_test_accuracy.png")
plt.close()

# Update fine-tuning CSV files for Clients A and B
file_paths = [
    fine_tune_results_A_path,
    fine_tune_results_B_path
]

for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    # Modify the Method column to reflect fine-tuning status
    df["Method"] = df.apply(lambda row: f"{row['Method']} with FineTune" if "FineTune" in str(row["Epoch"]) else f"{row['Method']}", axis=1)
    
    # Convert FineTune epochs to numeric values starting from 1
    df["Epoch"] = df["Epoch"].apply(lambda x: int(x.replace("FineTune_", "")) if isinstance(x, str) and "FineTune_" in x else x)
    
    # Save the modified results
    df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")

file_A = fine_tune_results_A_path
file_B = fine_tune_results_B_path
data_A = pd.read_csv(file_A)
data_B = pd.read_csv(file_B)

# Convert "FineTune_X" to integer values (if any still exist)
def convert_epoch(epoch):
    if "FineTune_" in str(epoch):
        return int(epoch.replace("FineTune_", ""))
    return int(epoch)

data_A["Epoch"] = data_A["Epoch"].apply(convert_epoch)
data_B["Epoch"] = data_B["Epoch"].apply(convert_epoch)

# Set up the figure and axes for fine-tuning results (Clients A and B)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.set(style="whitegrid")

def plot_metric(ax, data, metric, title, ylabel):
    sns.lineplot(
        data=data,
        x="Epoch",
        y=metric,
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(title="Method", loc="lower right")

# Plot Client A Accuracy and F1 Score
plot_metric(axes[0, 0], data_A, "Accuracy", "scDLC Deep Learning Client A - Accuracy per Epoch (Dataset C)", "Accuracy")
plot_metric(axes[0, 1], data_A, "F1 Score", "scDLC Deep Learning Client A - F1 Score per Epoch (Dataset C)", "F1 Score")
# Plot Client B Accuracy and F1 Score
plot_metric(axes[1, 0], data_B, "Accuracy", "scDLC Deep Learning Client B - Accuracy per Epoch (Dataset C)", "Accuracy")
plot_metric(axes[1, 1], data_B, "F1 Score", "scDLC Deep Learning Client B - F1 Score per Epoch (Dataset C)", "F1 Score")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scDLC_DL_FineTune_Results.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scDLC_DL_FineTune_Results.pdf")
#plt.show()

# Emission plots
data_path = "../../results/dataset_C/scDLC_DL_Client_A_B_emission.csv"
graph_folder_path = "../../graphs/dataset_C"

os.makedirs(f"{graph_folder_path}/png", exist_ok=True)
os.makedirs(f"{graph_folder_path}/pdf", exist_ok=True)

# Load the emissions data
df = pd.read_csv(data_path)

# Convert 'FineTune' column values for better readability
df['FineTune'] = df['FineTune'].map({'without': 'Without Fine-Tune', 'with': 'With Fine-Tune'})

# Create the figure and subplots for emissions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Plot for Client A
sns.barplot(ax=axes[0], data=df[df['Client'] == 'A'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[0].set_title("scDLC Deep Learning Carbon Emission for Client A (Dataset C)")
axes[0].set_ylabel("Emission (kWh)")
axes[0].set_xlabel("Method")
axes[0].legend(title="Fine-Tune")

# Plot for Client B
sns.barplot(ax=axes[1], data=df[df['Client'] == 'B'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[1].set_title("scDLC Deep Learning Carbon Emission for Client B (Dataset C)")
axes[1].set_ylabel("Emission (kWh)")
axes[1].set_xlabel("Method")
axes[1].legend(title="Fine-Tune")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scDLC_Client_A_B_Emission.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scDLC_Client_A_B_Emission.pdf")

plt.close()

print("Emission plots saved successfully.")

