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

from ACTINN_model import ACTINN_scRNAseqClassifier

# Set folder path for dataset_C (CSV format)
folder_path = "../../processed_data/dataset_C"
# Use chunks 1-4 for training
training_files = [f"{folder_path}/dataset_C_chunk_{i + 1}.csv" for i in range(4)]
# Use chunk 7 for testing
test_file = f"{folder_path}/dataset_C_chunk_7.csv"

# Load training data from CSV files
train_dfs = [pd.read_csv(file) for file in training_files]
# Each CSV is assumed to have 2001 columns: first 2000 are features and the 2001st is the label.
X_train = torch.tensor(pd.concat([df.iloc[:, :2000] for df in train_dfs]).values, dtype=torch.float32)
y_train = torch.tensor(pd.concat([df.iloc[:, 2000] for df in train_dfs])
                     .astype('category').cat.codes.values, dtype=torch.long)

# Load test data
test_df = pd.read_csv(test_file)
X_test = torch.tensor(test_df.iloc[:, :2000].values, dtype=torch.float32)
y_test = torch.tensor(test_df.iloc[:, 2000]
                    .astype('category').cat.codes.values, dtype=torch.long)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define method-specific parameters for ACTINN
methods = {
    'Baseline': {},
    'SmallBatch': {'batch_size': 16},
    'MixedPrecision': {'use_mixed_precision': True},
    'ReduceComplexity': {'hidden_sizes': [50, 25, 12]}  # Reduced architecture
}

results = {
    'accuracy': {},
    'loss': {},
    'f1_score': {},
    'emissions': {}
}
per_epoch_results = []

# Loop over each method
for method, params in methods.items():
    print(f"\n=== Training ACTINN method: {method} ===")
    tracker = EmissionsTracker()
    tracker.start()
    
    # Use method-specific batch size if provided; default to 128 otherwise.
    batch_size = params.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Save architecture configuration for reproducibility
    architecture_config = {
        "model": "FeedForward Neural Network",
        "layers": params.get('hidden_sizes', [100, 50, 25]),
        "activation": "ReLU",
        "output_activation": "Softmax",
        "loss_function": "NLLLoss",
        "optimizer": "Adam",
        "initial_learning_rate": 0.0001,
        "lr_decay": {"type": "staircase_exponential_decay", "decay_rate": 0.95, "decay_step": 1000},
        "epochs": 15,
        "batch_size": batch_size,
        "L2_regularization_rate": 0.005,
        "initializer": "Xavier",
        "use_mixed_precision": params.get('use_mixed_precision', False)
    }
    os.makedirs("../../models/dataset_C", exist_ok=True)
    config_path = f"../../models/dataset_C/ACTINN_DL_dataset_C_{method}_config.json"
    with open(config_path, "w") as f:
        json.dump(architecture_config, f)
    
    # Instantiate the ACTINN model
    model = ACTINN_scRNAseqClassifier(
        input_size=X_train.shape[1],
        num_classes=len(torch.unique(y_train)),
        hidden_sizes=params.get('hidden_sizes', None)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    criterion = nn.NLLLoss()
    
    use_mixed_precision = params.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    
    # Training loop
    for epoch in range(15):
        model.train()
        epoch_loss = 0
        correct, total = 0, 0
        with tqdm(total=len(train_loader), desc=f"Method {method} | Epoch {epoch+1}/15", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({"Loss": loss.item(), "Acc": correct/total})
                pbar.update(1)
        # Evaluation on training and test sets
        def evaluate(loader):
            model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    corr += (pred == labels).sum().item()
                    tot += labels.size(0)
            return corr / tot
        train_accuracy = evaluate(train_loader)
        test_accuracy = evaluate(test_loader)
        per_epoch_results.append([method, epoch+1, train_accuracy, test_accuracy])
        print(f"[Epoch {epoch+1}] Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        if test_accuracy >= 0.90:
            print(f"Early stopping for {method} at epoch {epoch+1} with Test Acc: {test_accuracy:.4f}")
            break
    
    # Compute final test metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    final_accuracy = sum([1 for p, t in zip(all_preds, all_labels) if p == t]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    emissions = tracker.stop()
    
    results['accuracy'][method] = final_accuracy
    results['loss'][method] = epoch_loss / len(test_loader)
    results['f1_score'][method] = f1
    results['emissions'][method] = emissions
    
    torch.save(model.state_dict(), f"../../models/dataset_C/ACTINN_DL_dataset_C_{method}_pre_train.pth")

# Save per-epoch results
per_epoch_results_df = pd.DataFrame(per_epoch_results, columns=['Method', 'Epoch', 'Train_Accuracy', 'Test_Accuracy'])
per_epoch_results_df.to_csv("../../results/dataset_C/ACTINN_DL_per_epoch_result.csv", index=False)

# Save overall results summary (using same format as scDLC)
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
results_df.to_csv("../../results/dataset_C/ACTINN_DL_results.csv", index=False)

print("ACTINN DL pre-training complete for dataset_C. Results saved to CSV.")
