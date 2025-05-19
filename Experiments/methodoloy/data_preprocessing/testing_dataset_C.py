import os
import pandas as pd

# Directory containing the processed dataset_B chunks
processed_folder = "../processed_data/dataset_C"

# List all CSV files that follow the naming convention for chunks
csv_files = sorted([os.path.join(processed_folder, f)
                    for f in os.listdir(processed_folder)
                    if f.startswith("dataset_C_chunk_") and f.endswith(".csv")])

# Process each file and print label counts
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    print(f"Label counts in {os.path.basename(csv_file)}:")
    print(df['label'].value_counts())
    print("-" * 50)