import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Updated result file paths for scSMD FL experiments on Dataset C:
pre_train_results_path = "../../results/dataset_C/scSMD_FL_per_round_result_pre_train.csv"
fine_tune_results_A_path = "../../results/dataset_C/scSMD_FL_results_fine_tune_A.csv"
fine_tune_results_B_path = "../../results/dataset_C/scSMD_FL_results_fine_tune_B.csv"
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
plt.title("scSMD FL Test Accuracy per Global Round (Pre-Training) - Dataset B")
plt.xlabel("Global Round")
plt.ylabel("Test Accuracy")
plt.legend(title="Method")

plt.subplot(1, 2, 2)
sns.lineplot(data=pre_train_results, x="Global_Round", y="Test_Loss", hue="Method", marker="o")
plt.title("scSMD FL Test Loss per Global Round (Pre-Training) - Dataset B")
plt.xlabel("Global Round")
plt.ylabel("Test Loss")
plt.legend(title="Method")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/pdf/scSMD_FL_test_accuracy_loss.pdf")
plt.savefig(f"{graph_folder_path}/png/scSMD_FL_test_accuracy_loss.png")
plt.close()

# Update fine-tuning CSV file names for FL results (for Clients A and B)
file_paths = [
    fine_tune_results_A_path,
    fine_tune_results_B_path
]

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df["Method"] = df.apply(lambda row: f"{row['Method']} with FineTune" if "FineTune" in str(row["Epoch"]) else f"{row['Method']}", axis=1)
    df["Epoch"] = df["Epoch"].apply(lambda x: int(x.replace("FineTune_", "")) if isinstance(x, str) and "FineTune_" in x else x)
    df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")

file_A = fine_tune_results_A_path
file_B = fine_tune_results_B_path
data_A = pd.read_csv(file_A)
data_B = pd.read_csv(file_B)

def convert_epoch(epoch):
    if "FineTune_" in str(epoch):
        return int(epoch.replace("FineTune_", ""))
    return int(epoch)

data_A["Epoch"] = data_A["Epoch"].apply(convert_epoch)
data_B["Epoch"] = data_B["Epoch"].apply(convert_epoch)

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

plot_metric(axes[0, 0], data_A, "Accuracy", "scSMD FL Client A - Accuracy per Epoch (Dataset C)", "Accuracy")
plot_metric(axes[0, 1], data_A, "F1 Score", "scSMD FL Client A - F1 Score per Epoch (Dataset C)", "F1 Score")
plot_metric(axes[1, 0], data_B, "Accuracy", "scSMD FL Client B - Accuracy per Epoch (Dataset C)", "Accuracy")
plot_metric(axes[1, 1], data_B, "F1 Score", "scSMD FL Client B - F1 Score per Epoch (Dataset C)", "F1 Score")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scSMD_FL_FineTune_Results.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scSMD_FL_FineTune_Results.pdf")
# plt.show()

# Emission plots
data_path = "../../results/dataset_C/scDLC_FL_Client_A_B_emission.csv"  # Update if needed
graph_folder_path = "../../graphs/dataset_C"

os.makedirs(f"{graph_folder_path}/png", exist_ok=True)
os.makedirs(f"{graph_folder_path}/pdf", exist_ok=True)

df = pd.read_csv(data_path)
df['FineTune'] = df['FineTune'].map({'without': 'Without Fine-Tune', 'with': 'With Fine-Tune'})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_style("whitegrid")
sns.set_palette("Set2")

sns.barplot(ax=axes[0], data=df[df['Client'] == 'A'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[0].set_title("scSMD FL Carbon Emission for Client A (Dataset B)")
axes[0].set_ylabel("Emission (kWh)")
axes[0].set_xlabel("Method")
axes[0].legend(title="Fine-Tune")

sns.barplot(ax=axes[1], data=df[df['Client'] == 'B'], x='Method', y='Emission (kWh)', hue='FineTune')
axes[1].set_title("scSMD FL Carbon Emission for Client B (Dataset B)")
axes[1].set_ylabel("Emission (kWh)")
axes[1].set_xlabel("Method")
axes[1].legend(title="Fine-Tune")

plt.tight_layout()
plt.savefig(f"{graph_folder_path}/png/scSMD_Client_A_B_Emission.png", dpi=300)
plt.savefig(f"{graph_folder_path}/pdf/scSMD_Client_A_B_Emission.pdf")
plt.close()

print("Emission plots saved successfully.")
