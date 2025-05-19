import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Update result file paths for scCDCG FL experiments on dataset_C
pre_train_results_path = "../../results/dataset_C/scCDCG_FL_per_round_result_pre_train.csv"
fine_tune_results_A_path = "../../results/dataset_C/scCDCG_FL_results_fine_tune_A.csv"
fine_tune_results_B_path = "../../results/dataset_C/scCDCG_FL_results_fine_tune_B.csv"
graph_folder_path = "../../graphs/dataset_C"

os.makedirs(graph_folder_path, exist_ok=True)
os.makedirs(f"{graph_folder_path}/pdf", exist_ok=True)
os.makedirs(f"{graph_folder_path}/png", exist_ok=True)

# Load datasets
pre_train_results = pd.read_csv(pre_train_results_path)
fine_tune_results_A = pd.read_csv(fine_tune_results_A_path)
fine_tune_results_B = pd.read_csv(fine_tune_results_B_path)

sns.set_theme(style="whitegrid")

# --- First Plot: Test Accuracy and Test Loss per Global Round (Pre-Training) ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=pre_train_results, x="Global_Round", y="Test_Accuracy", hue="Method", marker="o")
plt.title("scCDCG Federated Learning Test Accuracy per Global Round (Pre-Training)")
plt.xlabel("Global Round")
plt.ylabel("Test Accuracy")
plt.legend(title="Method")

plt.subplot(1, 2, 2)
sns.lineplot(data=pre_train_results, x="Global_Round", y="Test_Loss", hue="Method", marker="o")
plt.title("scCDCG Federated Learning Test Loss per Global Round (Pre-Training)")
plt.xlabel("Global Round")
plt.ylabel("Test Loss")
plt.legend(title="Method")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/pdf/scCDCG_FL_test_accuracy_loss.pdf")
plt.savefig(f"{graph_folder_path}/png/scCDCG_FL_test_accuracy_loss.png")
plt.close()

# Update fine-tuning CSV file names for FL results (for Clients A and B)
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
    
    # Save the modified results back to the file
    df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")

file_A = fine_tune_results_A_path
file_B = fine_tune_results_B_path
data_A = pd.read_csv(file_A)
data_B = pd.read_csv(file_B)

# Ensure that Epoch column values are integers
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
plot_metric(axes[0, 0], data_A, "Accuracy", "scCDCG Federated Learning Client A - Accuracy per Epoch", "Accuracy")
plot_metric(axes[0, 1], data_A, "F1 Score", "scCDCG Federated Learning Client A - F1 Score per Epoch", "F1 Score")
# Plot Client B Accuracy and F1 Score
plot_metric(axes[1, 0], data_B, "Accuracy", "scCDCG Federated Learning Client B - Accuracy per Epoch", "Accuracy")
plot_metric(axes[1, 1], data_B, "F1 Score", "scCDCG Federated Learning Client B - F1 Score per Epoch", "F1 Score")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scCDCG_FL_FineTune_Results.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scCDCG_FL_FineTune_Results.pdf")
#plt.show()

# Emission plots
data_path = "../../results/dataset_A/scDLC_FL_Client_A_B_emission.csv"  # Update this if needed
graph_folder_path = "../../graphs/dataset_C"

os.makedirs(f"{graph_folder_path}/png", exist_ok=True)
os.makedirs(f"{graph_folder_path}/pdf", exist_ok=True)

# Load the emissions data
df = pd.read_csv(data_path)

# Map the 'FineTune' column for readability
df['FineTune'] = df['FineTune'].map({'without': 'Without Fine-Tune', 'with': 'With Fine-Tune'})

# Create the figure and subplots for emissions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Plot for Client A
sns.barplot(ax=axes[0], data=df[df['Client'] == 'A'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[0].set_title("scCDCG Federated Learning Carbon Emission for Client A")
axes[0].set_ylabel("Emission (kWh)")
axes[0].set_xlabel("Method")
axes[0].legend(title="Fine-Tune")

# Plot for Client B
sns.barplot(ax=axes[1], data=df[df['Client'] == 'B'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[1].set_title("scCDCG Federated Learning Carbon Emission for Client B")
axes[1].set_ylabel("Emission (kWh)")
axes[1].set_xlabel("Method")
axes[1].legend(title="Fine-Tune")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scCDCG_FL_Client_A_B_Emission.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scCDCG_FL_Client_A_B_Emission.pdf")

plt.close()

print("Emission plots saved successfully.")
