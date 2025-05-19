import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from scDLC_model import scDLC_scRNAseqClassifier

# Update folder paths for Dataset E
data_folder_path = "../../processed_data/dataset_E"
test_file = f"{data_folder_path}/dataset_E_chunk_11.csv"
methods = {
    'Baseline': {'lstm_size': 64, 'num_layers': 2},
    'SmallBatch': {'batch_size': 16, 'lstm_size': 64, 'num_layers': 2},
    'MixedPrecision': {'use_mixed_precision': True, 'lstm_size': 64, 'num_layers': 2},
    'ReduceComplexity': {'lstm_size': 48, 'num_layers': 1},
}

os.makedirs("../../results/dataset_E", exist_ok=True)

def load_global_categories(mapping_file="../../processed_data/dataset_E/dataset_E_label_mapping.txt"):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            key, val = line.strip().split(": ")
            mapping[int(key)] = val
    # We expect 10 targets; global_categories are the sorted keys
    global_categories = sorted(mapping.keys())
    return global_categories

def process_test_data(file_path):
    # For Dataset E, CSV file: first 2000 columns are features, 2001st is label.
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :2000].values, dtype=torch.float32)
    # Load the global categories
    global_categories = load_global_categories()
    # Create a fixed categorical dtype using the global categories:
    cat_type = pd.api.types.CategoricalDtype(categories=global_categories, ordered=True)
    # Convert the label column to a categorical with fixed categories
    y_cat = pd.Categorical(df.iloc[:, 2000], dtype=cat_type)
    y = torch.tensor(y_cat.codes, dtype=torch.long)
    return TensorDataset(X, y), y_cat

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

# Load test data from CSV using the updated process_test_data
test_data, test_labels = process_test_data(test_file)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Get the list of unique label categories from the categorical, which is now fixed to the global mapping
cell_types = list(test_labels.categories)
print("Cell types (global):", cell_types)  # Should print 10 classes

distribution_results = []

def compute_class_distribution(labels, label_map):
    counter = Counter(labels)
    total = sum(counter.values())
    return {label_map[i]: (counter[i] / total) * 100 for i in range(len(label_map))}

# Compute distribution on the test set (ground truth)
test_distribution = compute_class_distribution(test_labels.codes, cell_types)
distribution_results.append({"Method": "Test Set", **test_distribution})

# Evaluate each pre-trained model
for method in methods.keys():
    print(f"Processing {method} model...")
    model_path = f"../../models/dataset_E/scDLC_DL_dataset_E_{method}_pre_train.pth"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}, skipping...")
        continue

    params = methods[method]
    model = scDLC_scRNAseqClassifier(
        input_size=test_data[0][0].shape[0],
        num_classes=len(cell_types),  # now using the fixed global categories (should be 10)
        lstm_size=params.get('lstm_size', 64),
        num_layers=params.get('num_layers', 2)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_state = model.state_dict()
        for key in list(checkpoint.keys()):
            if key.startswith("fc_out"):
                if checkpoint[key].shape != model_state[key].shape:
                    print(f"Skipping parameter {key} due to shape mismatch: checkpoint {checkpoint[key].shape} vs model {model_state[key].shape}.")
                    del checkpoint[key]
        model.load_state_dict(checkpoint, strict=False)
    except RuntimeError as e:
        print(f"Model loading failed for {method}. Error: {e}")
        continue
    
    preds = evaluate_model(model, test_loader, device)
    model_distribution = compute_class_distribution(preds, cell_types)
    distribution_results.append({"Method": method, **model_distribution})

df_results = pd.DataFrame(distribution_results)
df_results.to_csv("../../results/dataset_E/scDLC_DL_test_result_distribution.csv", index=False)

print("Test set distribution analysis complete. Results saved.")

