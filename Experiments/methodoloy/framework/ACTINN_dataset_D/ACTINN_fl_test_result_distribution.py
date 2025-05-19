import os
import torch
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from ACTINN_model import ACTINN_scRNAseqClassifier

data_folder_path = "../../processed_data/dataset_D"
test_file = f"{data_folder_path}/dataset_D_chunk_11.h5ad"

# Methods parameters for ACTINN FL (using same keys as before)
methods = {
    'Baseline': {},
    'SmallBatch': {'batch_size': 16},
    'MixedPrecision': {'use_mixed_precision': True},
    'ReduceComplexity': {'hidden_sizes': [50, 25, 12]}
}

os.makedirs("../../results/dataset_D", exist_ok=True)

def process_test_data(file_path):
    adata = sc.read_h5ad(file_path)
    X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    y = torch.tensor(pd.Series(adata.obs['cell_type'].astype('category').cat.codes).values, dtype=torch.long)
    return TensorDataset(X, y), adata.obs['cell_type'].astype('category')

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

test_data, test_labels = process_test_data(test_file)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

cell_types = list(test_labels.cat.categories)

distribution_results = []

def compute_class_distribution(labels, label_map):
    counter = Counter(labels)
    total = sum(counter.values())
    return {label_map[i]: (counter[i] / total) * 100 for i in range(len(label_map))}

test_distribution = compute_class_distribution(test_labels.cat.codes.values, cell_types)
distribution_results.append({"Method": "Test Set", **test_distribution})

for method in methods.keys():
    print(f"Processing {method} model...")
    model_path = f"../../models/dataset_D/ACTINN_FL_dataset_D_{method}_pre_train.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}, skipping...")
        continue

    params = methods[method]
    model = ACTINN_scRNAseqClassifier(
        input_size=test_data[0][0].shape[0],
        num_classes=len(cell_types),
        hidden_sizes=params.get('hidden_sizes', None)
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
df_results.to_csv("../../results/dataset_D/ACTINN_FL_test_result_distribution.csv", index=False)

print("Test set distribution analysis complete. Results saved.")

