import os
import torch
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from scCDCG_model import scCDCG_scRNAseqClassifier  # Changed import

data_folder_path = "../../processed_data/dataset_A"
test_file = f"{data_folder_path}/dataset_A_chunk_11.h5ad"

# Updated methods dictionary for scCDCG
methods = {
    'Baseline': {'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'SmallBatch': {'batch_size': 16, 'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'MixedPrecision': {'use_mixed_precision': True, 'dims_encoder': [256, 64], 'dims_decoder': [64, 256]},
    'ReduceComplexity': {'dims_encoder': [128, 48], 'dims_decoder': [48, 128]},
}

os.makedirs("../../results/dataset_A", exist_ok=True)

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
    # Updated model file path for scCDCG DL pre-trained model
    model_path = f"../../models/dataset_A/scCDCG_DL_dataset_A_{method}_pre_train.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}, skipping...")
        continue
    
    params = methods[method]
    model = scCDCG_scRNAseqClassifier(
        input_size=test_data[0][0].shape[0],
        num_classes=len(cell_types),
        dims_encoder=params['dims_encoder'],
        dims_decoder=params['dims_decoder']
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
df_results.to_csv("../../results/dataset_A/scCDCG_DL_test_result_distribution.csv", index=False)

print("Test set distribution analysis complete. Results saved.")
