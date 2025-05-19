import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from codecarbon import EmissionsTracker
import pandas as pd
from sklearn.metrics import f1_score
from scDLC_model import scDLC_scRNAseqClassifier

# Define the average_weights function for federated averaging
def average_weights(state_dicts):
    """
    Averages a list of model state dictionaries.
    """
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state

def load_global_categories(mapping_file="../../processed_data/dataset_E/dataset_E_label_mapping.txt"):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            key, val = line.strip().split(": ")
            mapping[int(key)] = val
    global_categories = sorted(mapping.keys())
    return global_categories

def process_client_data(file_path):
    # Read CSV file: first 2000 columns are features and 2001st is the label.
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :2000].values, dtype=torch.float32)
    global_categories = load_global_categories()
    cat_type = pd.api.types.CategoricalDtype(categories=global_categories, ordered=True)
    y_cat = pd.Categorical(df.iloc[:, 2000], dtype=cat_type)
    y = torch.tensor(y_cat.codes, dtype=torch.long)
    return TensorDataset(X, y)

# Define folder paths for Dataset E
folder_path = "../../processed_data/dataset_E"
# Use chunks 1 to 8 for training (CSV files)
client_files = [f"{folder_path}/dataset_E_chunk_{i+1}.csv" for i in range(8)]
# Use chunk 11 for testing
test_file = f"{folder_path}/dataset_E_chunk_11.csv"

# Load test data from CSV
test_df = pd.read_csv(test_file)
X_test = torch.tensor(test_df.iloc[:, :2000].values, dtype=torch.float32)
global_categories = load_global_categories()
cat_type = pd.api.types.CategoricalDtype(categories=global_categories, ordered=True)
test_labels_cat = pd.Categorical(test_df.iloc[:, 2000], dtype=cat_type)
y_test = torch.tensor(test_labels_cat.codes, dtype=torch.long)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set the number of classes using the global mapping (should be 10)
num_classes_global = len(global_categories)
input_size = pd.read_csv(client_files[0]).iloc[:, :2000].shape[1]

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

os.makedirs("../../models/dataset_E", exist_ok=True)
os.makedirs("../../results/dataset_E", exist_ok=True)

emission_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Federated training parameters
global_rounds = 5    # Global interactive rounds
local_epochs = 3     # Local client epochs

for method, params in methods.items():
    tracker = EmissionsTracker()
    tracker.start()

    # Save the architecture configuration
    architecture_config = {
        "lstm_size": params.get('lstm_size', 64),
        "num_layers": params.get('num_layers', 2)
    }
    config_path = f"../../models/dataset_E/scDLC_FL_dataset_E_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)

    # Initialize the global model with fixed number of classes (should be 10)
    global_model = scDLC_scRNAseqClassifier(
        input_size=input_size,
        num_classes=num_classes_global,
        lstm_size=architecture_config["lstm_size"],
        num_layers=architecture_config["num_layers"]
    )
    global_model.to(device)

    # Create DataLoaders for each client using CSV files
    client_loaders = []
    for file in client_files:
        dataset = process_client_data(file)
        batch_size = params.get('batch_size', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    criterion = nn.CrossEntropyLoss()

    # Federated Averaging across global rounds
    for rnd in range(global_rounds):
        local_weights = []
        for loader in client_loaders:
            local_model = scDLC_scRNAseqClassifier(
                input_size=input_size,
                num_classes=num_classes_global,
                lstm_size=architecture_config["lstm_size"],
                num_layers=architecture_config["num_layers"]
            )
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)

            optimizer = optim.Adam(local_model.parameters(), lr=0.001)
            local_model.train()
            use_mixed_precision = params.get('use_mixed_precision', False)
            if use_mixed_precision and device.type == 'cuda':
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(local_epochs):
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    if use_mixed_precision and device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = local_model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = local_model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
            local_weights.append(local_model.state_dict())

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        global_model.eval()
        correct, total, test_loss = 0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = global_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_accuracy = correct / total
        avg_test_loss = test_loss / len(test_loader)
        print("="*30)
        print(f"[Global Round {rnd + 1}] Method: {method}, Test Accuracy: {test_accuracy:.4f}, Loss: {avg_test_loss:.4f}")
        print("="*30)
        if test_accuracy >= 0.98:
            print(f"Early stopping for {method} at round {rnd + 1} with accuracy {test_accuracy:.4f}")
            break

    emissions = tracker.stop()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    results['accuracy'][method] = test_accuracy
    results['loss'][method] = avg_test_loss
    results['f1_score'][method] = f1
    results['emissions'][method] = emissions

    torch.save(global_model.state_dict(), f"../../models/dataset_E/scDLC_FL_dataset_E_{method}_pre_train.pth")

per_round_results_df = pd.DataFrame(results, columns=['Method', 'Global_Round', 'Test_Accuracy', 'Test_Loss'])
per_round_results_df.to_csv("../../results/dataset_E/scDLC_FL_per_round_result_pre_train.csv", index=False)

results_df = pd.DataFrame({
    'FL_Baseline_Accuracy': [results['accuracy']['Baseline']],
    'FL_SmallBatch_Accuracy': [results['accuracy']['SmallBatch']],
    'FL_MixedPrecision_Accuracy': [results['accuracy']['MixedPrecision']],
    'FL_ReduceComplexity_Accuracy': [results['accuracy']['ReduceComplexity']],
    'FL_Baseline_Loss': [results['loss']['Baseline']],
    'FL_SmallBatch_Loss': [results['loss']['SmallBatch']],
    'FL_MixedPrecision_Loss': [results['loss']['MixedPrecision']],
    'FL_ReduceComplexity_Loss': [results['loss']['ReduceComplexity']],
    'FL_Baseline_F1': [results['f1_score']['Baseline']],
    'FL_SmallBatch_F1': [results['f1_score']['SmallBatch']],
    'FL_MixedPrecision_F1': [results['f1_score']['MixedPrecision']],
    'FL_ReduceComplexity_F1': [results['f1_score']['ReduceComplexity']],
    'FL_Baseline_emissions': [results['emissions']['Baseline']],
    'FL_SmallBatch_emissions': [results['emissions']['SmallBatch']],
    'FL_MixedPrecision_emissions': [results['emissions']['MixedPrecision']],
    'FL_ReduceComplexity_emissions': [results['emissions']['ReduceComplexity']],
})
results_df.to_csv("../../results/dataset_E/scDLC_FL_results_pre_train.csv", index=False)

print("Federated training complete. Results saved to CSV.")

