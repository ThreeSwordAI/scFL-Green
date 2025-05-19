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

def load_global_categories(mapping_file="../../processed_data/dataset_E/dataset_E_label_mapping.txt"):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            key, val = line.strip().split(": ")
            mapping[int(key)] = val
    # Return sorted global categories (list of class codes)
    global_categories = sorted(mapping.keys())
    return global_categories

def process_client_data(file_path):
    # Read CSV file: first 2000 columns are features, 2001st is the label.
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :2000].values, dtype=torch.float32)
    # Force label column to use the global categories from the mapping
    global_categories = load_global_categories()
    cat_type = pd.api.types.CategoricalDtype(categories=global_categories, ordered=True)
    y_cat = pd.Categorical(df.iloc[:, 2000], dtype=cat_type)
    y = torch.tensor(y_cat.codes, dtype=torch.long)
    return TensorDataset(X, y)

def evaluate_model(model, test_loader, device):
    model.eval()
    test_correct, test_total = 0, 0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    accuracy = test_correct / test_total
    f1 = f1_score(test_labels, test_preds, average='weighted')
    return accuracy, f1

# Define folder paths for Dataset E
data_folder_path = "../../processed_data/dataset_E"
test_file = f"{data_folder_path}/dataset_E_chunk_11.csv"
clients = {
    "A": f"{data_folder_path}/dataset_E_chunk_9.csv",
    "B": f"{data_folder_path}/dataset_E_chunk_10.csv"
}

methods = {
    'Baseline': {'lstm_size': 64, 'num_layers': 2},
    'SmallBatch': {'batch_size': 16, 'lstm_size': 64, 'num_layers': 2},
    'MixedPrecision': {'use_mixed_precision': True, 'lstm_size': 64, 'num_layers': 2},
    'ReduceComplexity': {'lstm_size': 48, 'num_layers': 1},
}

os.makedirs("../../results/dataset_E", exist_ok=True)

emission_results = []

# Load global categories once (expected to be 10 classes)
global_categories = load_global_categories()
num_classes_global = len(global_categories)

for client, file in clients.items():
    train_data = process_client_data(file)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_data = process_client_data(test_file)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    results = []
    for method, params in methods.items():
        print(f"Training Method: {method}, Client: {client}, Without Fine-Tuning")
        tracker = EmissionsTracker()
        tracker.start()
        
        # Initialize model using the global number of classes (should be 10)
        model = scDLC_scRNAseqClassifier(
            input_size=train_data[0][0].shape[0],
            num_classes=num_classes_global,
            lstm_size=params.get('lstm_size', 64),
            num_layers=params.get('num_layers', 2)
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_accuracy = 0.0
        for epoch in range(10):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/10", unit="batch") as pbar:
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
            
            accuracy, f1 = evaluate_model(model, test_loader, device)
            best_accuracy = max(best_accuracy, accuracy)
            results.append([method, client, epoch + 1, accuracy, f1])
            print(f"[Epoch {epoch + 1}] Train Accuracy: {correct / total:.4f}, Test Accuracy: {accuracy:.4f}")
        
        emissions = tracker.stop()
        emission_results.append([client, method, "without", emissions])
        torch.save(model.state_dict(), f"../../models/dataset_E/scDLC_DL_dataset_E_{method}_without_fine_tune_{client}.pth")
        
        print(f"Training Method: {method}, Client: {client}, With Fine-Tuning")
        tracker = EmissionsTracker()
        tracker.start()
        
        # Load pre-trained model weights; adjust final layer if shape mismatches.
        pre_trained_path = f"../../models/dataset_E/scDLC_DL_dataset_E_{method}_pre_train.pth"
        state_dict = torch.load(pre_trained_path, map_location=device)
        model_state = model.state_dict()
        for key in list(state_dict.keys()):
            if key.startswith("fc_out"):
                if state_dict[key].shape != model_state[key].shape:
                    print(f"Mismatch in {key} for method {method}. Reinitializing final layer parameters.")
                    del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(10):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            with tqdm(total=len(train_loader), desc=f"FineTune Epoch {epoch + 1}/10", unit="batch") as pbar:
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
            accuracy, f1 = evaluate_model(model, test_loader, device)
            results.append([method, client, f"FineTune_{epoch + 1}", accuracy, f1])
            print(f"[FineTune Epoch {epoch + 1}] Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            print(f"Best Accuracy without fine-tuning: {best_accuracy:.4f} and Current Accuracy: {accuracy:.4f}")

            if (train_accuracy >= 0.90) and (accuracy > best_accuracy):
                print(f"Early stopping at FineTune Epoch {epoch + 1}, best accuracy: {accuracy:.4f}")
                break
        
        emissions = tracker.stop()
        emission_results.append([client, method, "with", emissions])
        torch.save(model.state_dict(), f"../../models/dataset_E/scDLC_DL_dataset_E_{method}_with_fine_tune_{client}.pth")
        print(f"Fine-Tuning for {method}, Client {client} complete.")
    
    df_results = pd.DataFrame(results, columns=["Method", "Client", "Epoch", "Accuracy", "F1 Score"])
    df_results.to_csv(f"../../results/dataset_E/scDLC_DL_results_fine_tune_{client}.csv", index=False)
    
    
df_emissions = pd.DataFrame(emission_results, columns=["Client", "Method", "FineTune", "Emission (kWh)"])
df_emissions.to_csv(f"../../results/dataset_E/scDLC_DL_Client_A_B_emission.csv", index=False)

print("Fine-tuning complete. Results saved.")

