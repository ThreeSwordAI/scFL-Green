import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############################################
# Configuration
############################################

DATASET = "dataset_A"
FRAMEWORKS = ["scDLC", "scCDCG", "scSMD"]  # We only do these two
CLIENTS = ["A", "B"]
RESULTS_FOLDER = f"../methodoloy/results/{DATASET}"
OUTPUT_FOLDER = f"../Analysis/Analysis Data/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Filenames we expect:
#   scCDCG_FL_results_fine_tune_A.csv
#   scCDCG_FL_results_fine_tune_B.csv
#   scSMD_FL_results_fine_tune_A.csv
#   scSMD_FL_results_fine_tune_B.csv
#
# We will read them and combine into a single DataFrame.

############################################
# Helper function to load and transform data
############################################
def load_fine_tune_data(framework, client):
    """
    Loads scXXX_FL_results_fine_tune_{client}.csv
    from the RESULTS_FOLDER. Returns a DataFrame with columns:
      - 'Epoch'
      - 'Accuracy'
      - 'Method'
      - 'Framework'
      - 'Client'
    We also unify the naming for the fine-tune vs. not, if present.
    """
    filename = f"{framework}_FL_results_fine_tune_{client}.csv"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)

    # We assume columns: ["Method","Client","Epoch","Accuracy","F1 Score"]
    # Some rows have "FineTune_{epoch}" in the "Epoch" column. We want numeric.
    # Already done in many scripts, but let's be safe:
    def parse_epoch(x):
        if isinstance(x, str) and x.startswith("FineTune_"):
            return int(x.replace("FineTune_", ""))
        else:
            return int(x)

    df["Epoch"] = df["Epoch"].apply(parse_epoch)

    # Add columns for easy plotting
    df["Framework"] = framework
    df["Client"] = f"Client {client}"

    return df[["Epoch","Accuracy","Method","Framework","Client"]]

############################################
# Main logic
############################################
def main():
    # 1) Read scCDCG & scSMD data for Client A & B
    all_data = []
    for fw in FRAMEWORKS:
        for cl in CLIENTS:
            df_part = load_fine_tune_data(fw, cl)
            if not df_part.empty:
                all_data.append(df_part)
    if not all_data:
        print("No data loaded. Check file paths or filenames.")
        return

    df_combined = pd.concat(all_data, ignore_index=True)

    # 2) We'll produce a single figure with 2 subplots side by side:
    #    Left = Client A, Right = Client B
    #    We'll show lines for each Method (including with/without FineTune in name).
    #    We only plot Accuracy vs. Epoch.

    # Filter for dataset_A only, but we already are using dataset_A. So no extra filter needed.

    # Prepare subsets for A and B
    df_A = df_combined[df_combined["Client"] == "Client A"].copy()
    df_B = df_combined[df_combined["Client"] == "Client B"].copy()

    # 3) Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Left subplot: Client A
    sns.lineplot(
        data=df_A,
        x="Epoch",
        y="Accuracy",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=axes[0]
    )
    axes[0].set_title("Dataset A - Client A (scDLC, scCDCG & scSMD)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Method", loc="lower right")

    # Right subplot: Client B
    sns.lineplot(
        data=df_B,
        x="Epoch",
        y="Accuracy",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        ax=axes[1]
    )
    axes[1].set_title("Dataset D - Client B (scDLC, scCDCG & scSMD)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(title="Method", loc="lower right")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_FOLDER, f"scCDCG_scSMD_FL_Accuracy_{DATASET}.png")
    out_pdf = os.path.join(OUTPUT_FOLDER, f"scCDCG_scSMD_FL_Accuracy_{DATASET}.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()

    print(f"Saved 2-subplot accuracy figure for scCDCG & scSMD (Client A & B) to:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    main()
