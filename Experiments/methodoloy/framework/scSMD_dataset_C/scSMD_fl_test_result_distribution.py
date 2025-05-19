import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from scSMD_model import scSMD_scRNAseqClassifier  # Use scSMD model

# Set file paths for Dataset C
data_folder_path = "../../processed_data/dataset_C"
test_file = f"{data_folder_path}/dataset_C_chunk_7.csv"

# Methods dictionary for scSMD (using latent_dim)
methods = {
    'Baseline': {'latent_dim': 64},
    'SmallBatch': {'batch_size': 16, 'latent_dim': 64},
    'MixedPrecision': {'use_mixed_precision': True, 'latent_dim': 64},
    'ReduceComplexity': {'latent_dim': 48},
}

os.makedirs("../../results/dataset_C", exist_ok=True)

def process_test_data(file_path):
    # Read CSV file: first 2000 columns are features and 2001st column is the label.
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :2000].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 2000].astype('category').cat.codes.values, dtype=torch.long)
    return TensorDataset(X, y), df.iloc[:, 2000].astype('category')

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
    return all_preds

# Process test data from CSV
test_data, test_labels = process_test_data(test_file)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

cell_types = list(test_labels.cat.categories)
distribution_results = []

def compute_class_distribution(labels, label_map):
    counter = Counter(labels)
    total = sum(counter.values())
    return {label_map[i]: (counter[i] / total) * 100 for i in range(len(label_map))}

# Compute distribution on the test set
test_distribution = compute_class_distribution(test_labels.cat.codes.values, cell_types)
distribution_results.append({"Method": "Test Set", **test_distribution})

for method in methods.keys():
    print(f"Processing {method} model...")
    # Use FL naming convention for the pre-trained model for scSMD FL (dataset_C)
    model_path = f"../../models/dataset_C/scSMD_FL_dataset_C_{method}_pre_train.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}, skipping...")
        continue

    params = methods[method]
    model = scSMD_scRNAseqClassifier(
        input_size=test_data[0][0].shape[0],
        num_classes=len(cell_types),
        latent_dim=params.get('latent_dim', 64)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Model loading failed for {method}. Error: {e}")
        continue

    preds = evaluate_model(model, test_loader, device)
    model_distribution = compute_class_distribution(preds, cell_types)
    distribution_results.append({"Method": method, **model_distribution})

df_results = pd.DataFrame(distribution_results)
df_results.to_csv("../../results/dataset_C/scSMD_FL_test_result_distribution.csv", index=False)

print("Test set distribution analysis complete. Results saved.")
