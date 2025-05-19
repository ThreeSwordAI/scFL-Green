import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

############################################
# Configuration
############################################

DATASETS      = ["dataset_A", "dataset_D"]
FRAMEWORKS    = ["scDLC", "scCDCG", "scSMD"]
CLIENTS       = ["A", "B"]
BASE_RESULTS  = "../methodoloy/results"
OUTPUT_FOLDER = "../Analysis/Analysis Data/IndividualPlots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

############################################
# Data loading
############################################

def load_fine_tune_data(dataset, framework, client):
    folder   = os.path.join(BASE_RESULTS, dataset)
    filename = f"{framework}_FL_results_fine_tune_{client}.csv"
    path     = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # convert "FineTune_<n>" → int
    df["Epoch"] = df["Epoch"].apply(
        lambda x: int(x.replace("FineTune_", "")) if isinstance(x, str) and x.startswith("FineTune_") else int(x)
    )
    df["Framework"] = framework
    df["Client"]    = f"Client {client}"
    df["Dataset"]   = dataset
    return df[["Epoch", "Accuracy", "Method", "Framework", "Client", "Dataset"]]

############################################
# Plotting
############################################

def plot_accuracy(df, dataset, client):
    # 1) Double all font sizes
    plt.rcParams.update({'font.size': 22})
    sns.set_theme(style="whitegrid")

    # 2) Create figure + axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # 3) Draw lines (let seaborn make the legend internally first)
    sns.lineplot(
        data=df,
        x="Epoch", y="Accuracy",
        hue="Method", style="Method",
        markers=True, dashes=False,
        legend="brief",  # allow seaborn to build the legend
        ax=ax
    )

    # 4) Grab handles/labels and remove the axes legend
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # 5) Build & set centered suptitle
    ds_name = dataset.replace("_", " ").title()    
    fw_list = ", ".join(FRAMEWORKS[:-1]) + " & " + FRAMEWORKS[-1]
    title   = f"{ds_name} - Client {client} ({fw_list})"
    fig.suptitle(title, fontsize=26)

    # 6) Axis labels
    ax.set_xlabel("Epoch",    fontsize=26)
    ax.set_ylabel("Accuracy", fontsize=26)

    # 7) Big tick labels
    ax.tick_params(axis='both', labelsize=22)

    # 8) Force x‐axis from 0 to 10 with step 2
    ax.set_xlim(1, 10)
    ax.set_xticks(range(2, 11, 2))

    # 9) Carve out space on the right & top for legend + title
    fig.subplots_adjust(right=0.70, top=0.90)

    # 10) Draw a single legend on the right, slightly separated
    fig.legend(
        handles, labels,
        title="Method",
        title_fontsize=26,
        fontsize=24,
        loc='center left',
        bbox_to_anchor=(0.75, 0.5),  # moves legend further right
        frameon=False
    )

    # 11) Save as PNG & PDF
    base = os.path.join(OUTPUT_FOLDER, f"Accuracy_{dataset}_Client_{client}")
    plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{base}.pdf",               bbox_inches='tight')
    plt.close()
    print(f"Saved: {base}.png  &  {base}.pdf")

############################################
# Main
############################################

def main():
    for dataset in DATASETS:
        parts = []
        for fw in FRAMEWORKS:
            for cl in CLIENTS:
                df_part = load_fine_tune_data(dataset, fw, cl)
                if not df_part.empty:
                    parts.append(df_part)

        if not parts:
            print(f"No data for {dataset}, skipping.")
            continue

        # concatenate & wrap "with FineTune" onto two lines
        df_all = pd.concat(parts, ignore_index=True)
        df_all["Method"] = df_all["Method"].str.replace(
            r" with FineTune",
            "\nwith FineTune",
            regex=True
        )

        for cl in CLIENTS:
            df_cl = df_all[df_all["Client"] == f"Client {cl}"]
            if df_cl.empty:
                print(f"  • No data for Client {cl} in {dataset}")
                continue
            plot_accuracy(df_cl, dataset, cl)

if __name__ == "__main__":
    main()

    print("Done.")