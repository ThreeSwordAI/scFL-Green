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

from scSMD_model import scSMD_scRNAseqClassifier

# Define file paths and client mapping for Dataset C
data_folder_path = "../../processed_data/dataset_C"
test_file = f"{data_folder_path}/dataset_C_chunk_7.csv"
clients = {
    "A": f"{data_folder_path}/dataset_C_chunk_5.csv",
    "B": f"{data_folder_path}/dataset_C_chunk_6.csv"
}

# Methods dictionary for scSMD (using latent_dim parameter)
methods = {
    'Baseline': {'latent_dim': 64},
    'SmallBatch': {'batch_size': 16, 'latent_dim': 64},
    'MixedPrecision': {'use_mixed_precision': True, 'latent_dim': 64},
    'ReduceComplexity': {'latent_dim': 48},
}

os.makedirs("../../results/dataset_C", exist_ok=True)
os.makedirs("../../models/dataset_C", exist_ok=True)

def process_client_data(file_path):
    # Read CSV file: first 2000 columns are features, 2001st column is the label.
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :2000].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 2000].astype('category').cat.codes.values, dtype=torch.long)
    return TensorDataset(X, y)

def evaluate_model(model, test_loader, device):
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = test_correct / test_total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1

emission_results = []

# Loop over each client (Client A and Client B)
for client, client_file in clients.items():
    # Prepare client training data from CSV
    train_data = process_client_data(client_file)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    # Test data is always taken from chunk 7
    test_data = process_client_data(test_file)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    results = []
    for method, params in methods.items():
        print(f"Processing {method} model for Client {client} (FL Fine-Tuning)")
        tracker = EmissionsTracker()
        tracker.start()
        
        # Initialize scSMD model and load pre-trained FL weights
        model = scSMD_scRNAseqClassifier(
            input_size=train_data[0][0].shape[0],
            num_classes=len(torch.unique(torch.tensor([label for _, label in train_data]))),
            latent_dim=params.get('latent_dim', 64)
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        pre_trained_path = f"../../models/dataset_C/scSMD_FL_dataset_C_{method}_pre_train.pth"
        model.load_state_dict(torch.load(pre_trained_path))
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            model.train()
            epoch_loss, correct, total = 0, 0, 0
            with tqdm(total=len(train_loader), desc=f"FineTune Epoch {epoch+1}/10", unit="batch") as pbar:
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
                    pbar.set_postfix({"Loss": loss.item(), "Accuracy": correct/total})
                    pbar.update(1)
            train_accuracy = correct / total  
            accuracy, f1 = evaluate_model(model, test_loader, device)
            results.append([method, client, f"FineTune_{epoch+1}", accuracy, f1])
            print(f"[FineTune Epoch {epoch+1}] Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            if (train_accuracy >= 0.9) and (accuracy > 0.95):
                print(f"Early stopping at FineTune Epoch {epoch+1}, best accuracy: {accuracy:.4f}")
                break
        
        emissions = tracker.stop()
        emission_results.append([client, method, "with", emissions])
        torch.save(model.state_dict(), f"../../models/dataset_C/scSMD_FL_dataset_C_{method}_with_fine_tune_{client}.pth")
        print(f"Fine-Tuning for {method}, Client {client} complete.")
    
    df_results = pd.DataFrame(results, columns=["Method", "Client", "Epoch", "Accuracy", "F1 Score"])
    df_results.to_csv(f"../../results/dataset_C/scSMD_FL_results_fine_tune_{client}.csv", index=False)
    
df_emissions = pd.DataFrame(emission_results, columns=["Client", "Method", "FineTune", "Emission (kWh)"])
df_emissions.to_csv(f"../../results/dataset_C/scSMD_FL_Client_A_B_emission.csv", index=False)

print("Fine-tuning complete for scSMD (FL) on dataset_C. Results saved to CSV.")
