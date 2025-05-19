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

from scSMD_model import scSMD_scRNAseqClassifier

# Define paths for dataset_D (h5ad files)
folder_path = "../../processed_data/dataset_D"
chunks_to_train = [f"{folder_path}/dataset_D_chunk_{i + 1}.h5ad" for i in range(8)]
test_file = f"{folder_path}/dataset_D_chunk_11.h5ad"

# Load data using Scanpy
train_data = [sc.read_h5ad(file) for file in chunks_to_train]
test_data = sc.read_h5ad(test_file)

# Concatenate training data
X_train = torch.tensor(pd.concat([pd.DataFrame(d.X.toarray()) for d in train_data]).values, dtype=torch.float32)
y_train = torch.tensor(pd.concat([pd.Series(d.obs['cell_type'].astype('category').cat.codes) for d in train_data]).values, dtype=torch.long)

X_test = torch.tensor(test_data.X.toarray(), dtype=torch.float32)
y_test = torch.tensor(pd.Series(test_data.obs['cell_type'].astype('category').cat.codes).values, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Methods dictionary for scSMD (using latent_dim parameter)
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

os.makedirs("../../models/dataset_D", exist_ok=True)
os.makedirs("../../results/dataset_D", exist_ok=True)

per_epoch_results = []

for method, params in methods.items():
    tracker = EmissionsTracker()
    tracker.start()

    # Save configuration for reproducibility
    architecture_config = {
        "latent_dim": params.get('latent_dim', 64)
    }
    config_path = f"../../models/dataset_D/scSMD_dataset_D_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)

    # Instantiate the model; note: input_size is X_train.shape[1] (e.g., 21946)
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
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/15", unit="batch") as pbar:
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
        per_epoch_results.append([method, epoch+1, train_accuracy, test_accuracy])
        print(f"[Epoch {epoch+1}] Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        if test_accuracy >= 0.90:
            print(f"Early stopping for {method} at epoch {epoch+1} with accuracy {test_accuracy:.4f}")
            break

    emissions = tracker.stop()
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    results['accuracy'][method] = test_accuracy
    results['loss'][method] = test_loss / len(test_loader)
    results['f1_score'][method] = f1
    results['emissions'][method] = emissions

    # Save the pre-trained model
    torch.save(model.state_dict(), f"../../models/dataset_D/scSMD_DL_dataset_D_{method}_pre_train.pth")

per_epoch_results_df = pd.DataFrame(per_epoch_results, columns=['Method', 'Epoch', 'Train_Accuracy', 'Test_Accuracy'])
per_epoch_results_df.to_csv("../../results/dataset_D/scSMD_DL_per_epoch_result_pre_train.csv", index=False)

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
results_df.to_csv("../../results/dataset_D/scSMD_DL_results_pre_train.csv", index=False)

print("Pre-training complete for scSMD on dataset_D. Results saved to CSV.")
