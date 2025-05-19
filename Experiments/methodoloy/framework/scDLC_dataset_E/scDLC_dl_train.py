import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from codecarbon import EmissionsTracker
import pandas as pd
from sklearn.metrics import f1_score

from scDLC_model import scDLC_scRNAseqClassifier

# Folder containing processed CSV files for dataset_E
folder_path = "../../processed_data/dataset_E"
# Use chunks 1 to 8 for training (CSV files)
training_files = [f"{folder_path}/dataset_E_chunk_{i + 1}.csv" for i in range(8)]
# Use chunk 11 for testing
test_file = f"{folder_path}/dataset_E_chunk_11.csv"

# Load the global label mapping (assumes mapping was saved as "key: value" per line)
mapping = {}
mapping_file = "../../processed_data/dataset_E/dataset_E_label_mapping.txt"
with open(mapping_file, "r") as f:
    for line in f:
        key, val = line.strip().split(": ")
        mapping[int(key)] = val
# Force global categories as a sorted list of expected class codes.
# If you expect 10 targets, adjust accordingly (e.g. 0 to 9):
global_categories = sorted(mapping.keys())
if len(global_categories) != 11:
    print(f"Warning: Expected 11 target classes but found {len(global_categories)} in the mapping.")

# Define a fixed categorical type using the global categories.
global_cat_dtype = pd.api.types.CategoricalDtype(categories=global_categories, ordered=True)

# Load training data from CSV files
train_dfs = [pd.read_csv(file) for file in training_files]

# Each CSV has 2001 columns: first 2000 are features, last column is the label.
X_train = torch.tensor(pd.concat([df.iloc[:, :2000] for df in train_dfs]).values, dtype=torch.float32)

# Force the label column to use the global categories before taking codes.
train_labels_series = pd.concat([df.iloc[:, 2000] for df in train_dfs])
train_labels_cat = pd.Categorical(train_labels_series, dtype=global_cat_dtype)
y_train = torch.tensor(train_labels_cat.codes, dtype=torch.long)

# Load test data similarly
test_df = pd.read_csv(test_file)
X_test = torch.tensor(test_df.iloc[:, :2000].values, dtype=torch.float32)
test_labels_series = test_df.iloc[:, 2000]
test_labels_cat = pd.Categorical(test_labels_series, dtype=global_cat_dtype)
y_test = torch.tensor(test_labels_cat.codes, dtype=torch.long)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Methods and hyperparameters
methods = {
    'Baseline': {'lstm_size': 64, 'num_layers': 2},
    'SmallBatch': {'batch_size': 16, 'lstm_size': 64, 'num_layers': 2},
    'MixedPrecision': {'use_mixed_precision': True, 'lstm_size': 64, 'num_layers': 2},
    'ReduceComplexity': {'lstm_size': 48, 'num_layers': 1},
}

results = {
    'accuracy': {},
    'loss': {},
    'f1_score': {},
    'emissions': {}
}

# Update output directories to use dataset_E
os.makedirs("../../models/dataset_E", exist_ok=True)
os.makedirs("../../results/dataset_E", exist_ok=True)

per_epoch_results = []

for method, params in methods.items():
    tracker = EmissionsTracker()
    tracker.start()

    # Save the architecture configuration
    architecture_config = {
        "lstm_size": params.get('lstm_size', 64),
        "num_layers": params.get('num_layers', 2),
    }
    config_path = f"../../models/dataset_E/scDLC_dataset_E_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)

    # Initialize the model; input_size is 2000, and number of classes is fixed using global_categories
    model = scDLC_scRNAseqClassifier(
        input_size=X_train.shape[1],
        num_classes=len(global_categories),  # force to use the global mapping (should be 10)
        lstm_size=architecture_config["lstm_size"],
        num_layers=architecture_config["num_layers"]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(15):  
        epoch_loss = 0
        correct, total = 0, 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/15", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({"Loss": loss.item(), "Accuracy": correct / total})
                pbar.update(1)
        
        train_accuracy = correct / total

        model.eval()
        correct, total, test_loss = 0, 0, 0
        with torch.no_grad():
            all_preds, all_labels = [], []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = correct / total
        per_epoch_results.append([method, epoch + 1, train_accuracy, test_accuracy])
        print(f"[Epoch {epoch + 1}] Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy >= 0.99:
            print(f"Stopping early for {method} at epoch {epoch + 1} with {test_accuracy:.4f} accuracy")
            break
    
    emissions = tracker.stop()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    results['accuracy'][method] = test_accuracy
    results['loss'][method] = test_loss / len(test_loader)
    results['f1_score'][method] = f1
    results['emissions'][method] = emissions

    # Save the pre-training model with updated naming
    torch.save(model.state_dict(), f"../../models/dataset_E/scDLC_DL_dataset_E_{method}_pre_train.pth")

per_epoch_results_df = pd.DataFrame(per_epoch_results, columns=['Method', 'Epoch', 'Train_Accuracy', 'Test_Accuracy'])
per_epoch_results_df.to_csv("../../results/dataset_E/scDLC_DL_per_epoch_result_pre_train.csv", index=False)

results_df = pd.DataFrame({
    'DL_Baseline_Accuracy': [results['accuracy']['Baseline']],
    'DL_SmallBatch_Accuracy': [results['accuracy']['SmallBatch']],
    'DL_MixedPrecision_Accuracy': [results['accuracy']['MixedPrecision']],
    'DL_ReduceComplexity_Accuracy': [results['accuracy']['ReduceComplexity']],
    'DL_Baseline_Loss': [results['loss']['Baseline']],
    'DL_SmallBatch_Loss': [results['loss']['SmallBatch']],
    'DL_MixedPrecision_Loss': [results['loss']['MixedPrecision']],
    'DL_ReduceComplexity_Loss': [results['loss']['ReduceComplexity']],
    'DL_Baseline_F1': [results['f1_score']['Baseline']],
    'DL_SmallBatch_F1': [results['f1_score']['SmallBatch']],
    'DL_MixedPrecision_F1': [results['f1_score']['MixedPrecision']],
    'DL_ReduceComplexity_F1': [results['f1_score']['ReduceComplexity']],
    'DL_Baseline_emissions': [results['emissions']['Baseline']],
    'DL_SmallBatch_emissions': [results['emissions']['SmallBatch']],
    'DL_MixedPrecision_emissions': [results['emissions']['MixedPrecision']],
    'DL_ReduceComplexity_emissions': [results['emissions']['ReduceComplexity']],
})
results_df.to_csv("../../results/dataset_E/scDLC_DL_results_pre_train.csv", index=False)

print("Pre-training complete. Results saved to CSV.")

