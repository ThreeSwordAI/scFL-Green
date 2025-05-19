import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from codecarbon import EmissionsTracker
import scanpy as sc
import pandas as pd
from sklearn.metrics import f1_score

from ACTINN_model import ACTINN_scRNAseqClassifier

# Define file paths for dataset_A (AnnData format)
folder_path = "../../processed_data/dataset_A"
# Use chunk 9 and 10 as clients; chunk 11 as test
clients = {
    "A": f"{folder_path}/dataset_A_chunk_9.h5ad",
    "B": f"{folder_path}/dataset_A_chunk_10.h5ad"
}
test_file = f"{folder_path}/dataset_A_chunk_11.h5ad"

# Methods dictionary for ACTINN (for DL training) – note the use of hidden_sizes
methods = {
    'Baseline': {},
    'SmallBatch': {'batch_size': 16},
    'MixedPrecision': {'use_mixed_precision': True},
    'ReduceComplexity': {'hidden_sizes': [50, 25, 12]}  # Reduced architecture
}

os.makedirs("../../results/dataset_A", exist_ok=True)

def process_client_data(file_path):
    adata = sc.read_h5ad(file_path)
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    y = torch.tensor(pd.Series(adata.obs['cell_type'].astype('category').cat.codes).values, dtype=torch.long)
    return TensorDataset(X, y)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

emission_results = []
results = []

# For each client, first train "without fine-tuning" then fine-tune
for client, client_file in clients.items():
    train_data = process_client_data(client_file)
    test_data = process_client_data(test_file)
    train_loader = DataLoader(train_data, batch_size=methods.get('SmallBatch', {}).get('batch_size', 128), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    for method, params in methods.items():
        print(f"\n=== ACTINN DL Fine-Tuning: Method {method}, Client {client} - Without Fine-Tuning ===")
        tracker = EmissionsTracker()
        tracker.start()
        
        # Instantiate ACTINN model; if hidden_sizes provided, pass them
        model = ACTINN_scRNAseqClassifier(
            input_size=train_data[0][0].shape[0],
            num_classes=len(torch.unique(torch.tensor([label for _, label in train_data]))),
            hidden_sizes=params.get('hidden_sizes', None)
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
        criterion = nn.NLLLoss()
        
        # Training loop for a fixed number of epochs (e.g., 10)
        best_accuracy = 0.0
        for epoch in range(10):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            with tqdm(total=len(train_loader), desc=f"Method {method} | Epoch {epoch+1}/10", unit="batch") as pbar:
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
                    pbar.set_postfix({"Loss": loss.item(), "Acc": correct/total})
                    pbar.update(1)
            train_accuracy, _ = evaluate_model(model, train_loader, device)
            test_accuracy, f1 = evaluate_model(model, test_loader, device)
            best_accuracy = max(best_accuracy, test_accuracy)
            results.append([method, client, epoch+1, test_accuracy, f1])
            print(f"[Epoch {epoch+1}] Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, F1: {f1:.4f}")
        emissions = tracker.stop()
        emission_results.append([client, method, "without", emissions])
        # Save model weights for without fine-tuning
        torch.save(model.state_dict(), f"../../models/dataset_A/ACTINN_DL_dataset_A_{method}_without_fine_tune_{client}.pth")
        
        print(f"\n=== ACTINN DL Fine-Tuning: Method {method}, Client {client} - With Fine-Tuning ===")
        tracker = EmissionsTracker()
        tracker.start()
        # Load pre-trained DL weights (assumed to be saved previously from ACTINN_dl_train.py)
        pre_trained_path = f"../../models/dataset_A/ACTINN_DL_dataset_A_{method}_pre_train.pth"
        model.load_state_dict(torch.load(pre_trained_path))
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
        # Fine-tuning loop (10 epochs)
        for epoch in range(10):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            with tqdm(total=len(train_loader), desc=f"FineTune {method} | Epoch {epoch+1}/10", unit="batch") as pbar:
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
                    pbar.set_postfix({"Loss": loss.item(), "Acc": correct/total})
                    pbar.update(1)
            train_accuracy, _ = evaluate_model(model, train_loader, device)
            test_accuracy, f1 = evaluate_model(model, test_loader, device)
            results.append([method, client, f"FineTune_{epoch+1}", test_accuracy, f1])
            print(f"[FineTune Epoch {epoch+1}] Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, F1: {f1:.4f}")
            if (train_accuracy >= 0.9) and (test_accuracy > best_accuracy):
                print(f"Early stopping at FineTune Epoch {epoch+1} for method {method} with Test Acc: {test_accuracy:.4f}")
                break
        emissions = tracker.stop()
        emission_results.append([client, method, "with", emissions])
        torch.save(model.state_dict(), f"../../models/dataset_A/ACTINN_DL_dataset_A_{method}_with_fine_tune_{client}.pth")
        print(f"Fine-tuning complete for method {method}, client {client}.")
        
df_results = pd.DataFrame(results, columns=["Method", "Client", "Epoch", "Test_Accuracy", "F1_Score"])
df_results.to_csv("../../results/dataset_A/ACTINN_DL_results_fine_tune.csv", index=False)

df_emissions = pd.DataFrame(emission_results, columns=["Client", "Method", "FineTune", "Emission (kWh)"])
df_emissions.to_csv("../../results/dataset_A/ACTINN_DL_Client_emission.csv", index=False)

print("ACTINN DL Fine-tuning complete for dataset_A. Results saved to CSV.")

