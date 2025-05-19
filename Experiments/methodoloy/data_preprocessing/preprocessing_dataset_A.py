import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
adata = sc.read_h5ad("../raw_data/dataset_A/dataset_A.h5ad")

# Retain only the transcriptomic data (gene expression matrix) as features
X = adata.X  # This contains RNA expression values

# Retain only the cell type as the label
target_column = "cell_type"
if target_column not in adata.obs.columns:
    raise ValueError("Target column 'cell_type' not found in the dataset.")

y = adata.obs[target_column]

# Encode the cell type labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

data_shuffled = np.random.permutation(len(y))  # Shuffle indices
X = X[data_shuffled]
y_encoded = y_encoded[data_shuffled]

# Save label encoding mapping
label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
with open("../processed_data/dataset_A/dataset_A_label_mapping.txt", "w") as f:
    for key, value in label_mapping.items():
        f.write(f"{key}: {value}\n")

# Split dataset into chunks
chunk_size = 10000
num_chunks = int(np.ceil(X.shape[0] / chunk_size))
print(f"Splitting dataset into {num_chunks} chunks...")

for i in range(num_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, X.shape[0])

    # Create AnnData object for the chunk
    chunk_adata = sc.AnnData(X=X[start:end])
    chunk_adata.obs[target_column] = y_encoded[start:end]
    chunk_adata.var_names = adata.var_names  # Keep gene names
    
    # Save as .h5ad file
    chunk_adata.write(f"../processed_data/dataset_A/dataset_A_chunk_{i + 1}.h5ad")
    print(f"Saved dataset_A_chunk_{i + 1}.h5ad with shape {chunk_adata.shape}, ensuring transcriptomic data is included.")

print("Data processing complete. .h5ad files saved.")
