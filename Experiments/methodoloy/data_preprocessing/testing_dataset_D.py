import scanpy as sc

# Load the dataset
adata = sc.read_h5ad(r"..\\raw_data\\dataset_D\\dataset_D.h5ad")  

# Check dataset dimensions
print(f"Dataset Shape: {adata.shape}")  # (cells, genes)

# Check metadata columns
print(f"Metadata columns: {adata.obs.columns}")

# Check first few genes
print(f"First 10 genes: {adata.var_names[:10]}")

# Check if the transcriptomic matrix (X) is sparse
is_sparse = hasattr(adata.X, "toarray")
print(f"Is X matrix sparse? {is_sparse}")

# Print the first row (RNA expression values)
if is_sparse:
    print("First row of RNA data (dense format):", adata.X[0].toarray()[0][:10])  # First 10 values
else:
    print("First row of RNA data:", adata.X[0][:10])  # First 10 values

# Check if the target labels exist
print("Unique Cell Types:", adata.obs["cell_type"].unique())
