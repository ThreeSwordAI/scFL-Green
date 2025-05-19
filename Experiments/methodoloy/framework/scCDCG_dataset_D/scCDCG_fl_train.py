# scCDCG_fl_train.py
import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from codecarbon import EmissionsTracker
import scanpy as sc
import pandas as pd
from sklearn.metrics import f1_score

from scCDCG_model import scCDCG_scRNAseqClassifier

def average_weights(state_dicts):
    """
    Averages a list of model state dictionaries.
    """
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state

# Define file paths
folder_path = "../../processed_data/dataset_D"
client_files = [f"{folder_path}/dataset_D_chunk_{i+1}.h5ad" for i in range(8)]
test_file = f"{folder_path}/dataset_D_chunk_11.h5ad"

# Load test data
test_data = sc.read_h5ad(test_file)
X_test = torch.tensor(test_data.X.toarray(), dtype=torch.float32)
y_test = torch.tensor(pd.Series(test_data.obs['cell_type'].astype('category').cat.codes).values, dtype=torch.long)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Determine input size and number of classes from the first client file
first_client = sc.read_h5ad(client_files[0])
input_size = torch.tensor(pd.DataFrame(first_client.X.toarray()).values, dtype=torch.float32).shape[1]
num_classes = len(pd.Series(first_client.obs['cell_type'].astype('category').cat.codes).unique())

# Define different methods and their parameters
methods = {
    'Baseline': {'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'SmallBatch': {'batch_size': 16, 'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'MixedPrecision': {'use_mixed_precision': True, 'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'ReduceComplexity': {'dims_encoder': [128, 48], 'dims_decoder': [48, 128]},
}

results = {
    'accuracy': {},
    'loss': {},
    'f1_score': {},
    'emissions': {}
}

os.makedirs("../../models/dataset_D", exist_ok=True)
os.makedirs("../../results/dataset_D", exist_ok=True)

all_round_results = []  # To store per-global-round results
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Federated learning parameters
global_rounds = 5    # Global rounds
local_epochs = 3     # Local epochs per client

for method, params in methods.items():
    tracker = EmissionsTracker()
    tracker.start()

    architecture_config = {
        "dims_encoder": params.get('dims_encoder', [256, 64]),
        "dims_decoder": params.get('dims_decoder', [64, 256])
    }
    config_path = f"../../models/dataset_D/scCDCG_FL_dataset_D_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)

    # Initialize the global model
    global_model = scCDCG_scRNAseqClassifier(
        input_size=input_size,
        num_classes=num_classes,
        dims_encoder=architecture_config["dims_encoder"],
        dims_decoder=architecture_config["dims_decoder"]
    )
    global_model.to(device)

    # Create DataLoaders for each client
    client_loaders = []
    for file in client_files:
        client_data = sc.read_h5ad(file)
        X_client = torch.tensor(pd.DataFrame(client_data.X.toarray()).values, dtype=torch.float32)
        y_client = torch.tensor(pd.Series(client_data.obs['cell_type'].astype('category').cat.codes).values, dtype=torch.long)
        dataset = TensorDataset(X_client, y_client)
        batch_size = params.get('batch_size', 32)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    criterion = nn.CrossEntropyLoss()

    for rnd in range(global_rounds):
        local_weights = []

        # Each client trains locally
        for loader in client_loaders:
            local_model = scCDCG_scRNAseqClassifier(
                input_size=input_size,
                num_classes=num_classes,
                dims_encoder=architecture_config["dims_encoder"],
                dims_decoder=architecture_config["dims_decoder"]
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

        # Federated averaging
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Evaluate on the test set
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
        all_round_results.append([method, rnd + 1, test_accuracy, avg_test_loss])
        print("=============================")
        print(f"[Global Round {rnd + 1}] Method: {method}, Test Accuracy: {test_accuracy:.4f}, Loss: {avg_test_loss:.4f}")
        print("=============================")
        
        if test_accuracy >= 0.90:
            print(f"Early stopping global rounds for {method} at round {rnd + 1} with accuracy {test_accuracy:.4f}")
            break

    emissions = tracker.stop()
    f1 = f1_score(all_labels, all_preds, average='weighted')

    results['accuracy'][method] = test_accuracy
    results['loss'][method] = avg_test_loss
    results['f1_score'][method] = f1
    results['emissions'][method] = emissions

    model_save_path = f"../../models/dataset_D/scCDCG_FL_dataset_D_{method}_pre_train.pth"
    torch.save(global_model.state_dict(), model_save_path)
    
per_round_results_df = pd.DataFrame(all_round_results, columns=['Method', 'Global_Round', 'Test_Accuracy', 'Test_Loss'])
per_round_results_df.to_csv("../../results/dataset_D/scCDCG_FL_per_round_result_pre_train.csv", index=False)

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
results_df.to_csv("../../results/dataset_D/scCDCG_FL_results_pre_train.csv", index=False)

print("Federated training complete. Results saved to CSV.")
