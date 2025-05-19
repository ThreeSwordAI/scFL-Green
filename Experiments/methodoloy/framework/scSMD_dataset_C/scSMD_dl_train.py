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

# Folder containing processed CSV files for dataset B
folder_path = "../../processed_data/dataset_C"
# Use chunks 1, 2, 3, and 4 for training (CSV files)
training_files = [f"{folder_path}/dataset_C_chunk_{i + 1}.csv" for i in range(4)]
# Use chunk 7 for testing
test_file = f"{folder_path}/dataset_C_chunk_7.csv"

# Load training data from CSV files
train_dfs = [pd.read_csv(file) for file in training_files]
# Each CSV has 2001 columns: first 2000 are features and last is the label.
X_train = torch.tensor(pd.concat([df.iloc[:, :2000] for df in train_dfs]).values, dtype=torch.float32)
y_train = torch.tensor(pd.concat([df.iloc[:, 2000] for df in train_dfs]).astype('category').cat.codes.values, dtype=torch.long)

# Load test data
test_df = pd.read_csv(test_file)
X_test = torch.tensor(test_df.iloc[:, :2000].values, dtype=torch.float32)
y_test = torch.tensor(test_df.iloc[:, 2000].astype('category').cat.codes.values, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Methods dictionary for scSMD using latent_dim parameter
methods = {
    'Baseline': {'latent_dim': 64},
    'SmallBatch': {'batch_size': 16, 'latent_dim': 64},
    'MixedPrecision': {'use_mixed_precision': True, 'latent_dim': 64},
    'ReduceComplexity': {'latent_dim': 48},
}

results = {
    'accuracy': {},
    'loss': {},
    'f1_score': {},
    'emissions': {}
}

os.makedirs("../../models/dataset_C", exist_ok=True)
os.makedirs("../../results/dataset_C", exist_ok=True)

per_epoch_results = []

for method, params in methods.items():
    tracker = EmissionsTracker()
    tracker.start()

    # Save architecture configuration for reproducibility
    architecture_config = {
        "latent_dim": params.get('latent_dim', 64)
    }
    config_path = f"../../models/dataset_C/scSMD_dataset_C_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)

    # Initialize the scSMD model
    model = scSMD_scRNAseqClassifier(
        input_size=X_train.shape[1],
        num_classes=len(torch.unique(y_train)),
        latent_dim=architecture_config["latent_dim"]
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
        all_preds, all_labels = [], []
        with torch.no_grad():
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

    # Save the pre-training model with updated naming for dataset_C
    torch.save(model.state_dict(), f"../../models/dataset_C/scSMD_DL_dataset_C_{method}_pre_train.pth")

per_epoch_results_df = pd.DataFrame(per_epoch_results, columns=['Method', 'Epoch', 'Train_Accuracy', 'Test_Accuracy'])
per_epoch_results_df.to_csv("../../results/dataset_C/scSMD_DL_per_epoch_result_pre_train.csv", index=False)

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
results_df.to_csv("../../results/dataset_C/scSMD_DL_results_pre_train.csv", index=False)

print("Pre-training complete for scSMD on dataset_C. Results saved to CSV.")
