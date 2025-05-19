import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DATASET = "dataset_D"  
FRAMEWORKS = ["scDLC", "scCDCG", "scSMD"]
RESULTS_DIR = f"../methodoloy/results/{DATASET}"
OUTPUT_DIR = "../Analysis"

# Emission CSV file pattern, e.g. "scDLC_FL_Client_A_B_emission.csv"
EMISSION_FILE_PATTERN = "{fw}_FL_Client_A_B_emission.csv"

# We'll only keep these 4 methods as data points:
VALID_METHODS = ["Baseline","SmallBatch","MixedPrecision","ReduceComplexity"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_emission_data(dataset):

    dfs = []
    for fw in FRAMEWORKS:
        fname = EMISSION_FILE_PATTERN.format(fw=fw)
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
        df_tmp = pd.read_csv(path)
        df_tmp["Framework"] = fw
        dfs.append(df_tmp)
    if not dfs:
        return pd.DataFrame()  # empty
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def plot_emission_finetune_side_by_side(df, dataset):
    if df.empty:
        print(f"No data to plot for {dataset} - DataFrame empty.")
        return

    # Keep only the 4 standard methods
    df = df[df["Method"].isin(VALID_METHODS)].copy()

    # Convert columns to categories for nicer ordering
    df["Client"] = df["Client"].astype("category")
    df["FineTune"] = df["FineTune"].astype("category")
    df["Framework"] = df["Framework"].astype("category")

    # Sort frameworks in the desired order
    df["Framework"] = df["Framework"].cat.set_categories(FRAMEWORKS, ordered=True)

    # We do a catplot with col="Client" (2 columns side by side)
    g = sns.catplot(
        data=df, 
        x="Framework", y="Emission (kWh)",
        hue="FineTune",
        col="Client",
        kind="box",
        sharey=True,
        height=4, aspect=1.2,
        palette="Set2"
    )

    # Label the columns: "Client A" and "Client B"
    g.set_titles(col_template="Client {col_name}")

    # Axis labels, main figure title, etc.
    g.set_axis_labels("Framework", "Emission (kWh)")
    g.fig.suptitle(f"Emission with/without Fine-Tuning - Dataset D")

    # Adjust layout
    plt.tight_layout()

    # Save
    png_path = os.path.join(OUTPUT_DIR, f"Emission_FineTune_dataset_D_SideBySide.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"Emission_FineTune_dataset_D_SideBySide.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved figure:\n  {png_path}\n  {pdf_path}")

def main():
    df = load_emission_data(DATASET)
    if df.empty:
        print(f"No data loaded for {DATASET}. Check file paths.")
        return
    plot_emission_finetune_side_by_side(df, DATASET)

if __name__ == "__main__":
    main()
