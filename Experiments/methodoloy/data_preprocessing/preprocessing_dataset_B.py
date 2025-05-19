import pandas as pd
import os


raw_file_path = "../raw_data/dataset_B/selectgene_dataset_B.csv"
processed_folder = "../processed_data/dataset_B"


os.makedirs(processed_folder, exist_ok=True)


df_raw = pd.read_csv(raw_file_path, header=None)
print(f"Raw data shape (rows x columns): {df_raw.shape}")


df_samples = df_raw.transpose()
print(f"Transposed data shape (samples x columns): {df_samples.shape}")


feature_columns = [f"feature_{i+1}" for i in range(2000)]
df_samples.columns = feature_columns + ["label"]


df_shuffled = df_samples.sample(frac=1, random_state=42).reset_index(drop=True)


n_samples = df_shuffled.shape[0]  
n_chunks = 7
chunk_size = n_samples // n_chunks  

for i in range(n_chunks):
    start_idx = i * chunk_size
    end_idx = (i + 1) * chunk_size
    chunk = df_shuffled.iloc[start_idx:end_idx, :]
    chunk_file = os.path.join(processed_folder, f"dataset_B_chunk_{i+1}.csv")
    chunk.to_csv(chunk_file, index=False)
    print(f"Saved chunk {i+1} to {chunk_file}")

print("Dataset B processing complete.")