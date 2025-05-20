# scFL-Green

**Optimizing Federated Learning for Single‑Cell Classification with Carbon‑Aware Fine‑Tuning**

scFL‑Green benchmarks and optimizes federated learning (FL) workflows for single‑cell RNA‑seq classification under sustainable, carbon‑aware settings. We integrate four state‑of‑the‑art classifiers (scCDCG, scDLC, scSMD, ACTINN) with three environmental optimizations (SmallBatch, MixedPrecision, ReduceComplexity) and a fine‑tuning stage to reduce training rounds and emissions while preserving or improving accuracy.

---

## Part I: Pipeline Usage

This section helps you quickly apply scFL‑Green to your own data.

### 1. Environment Setup

#### Windows (CMD/PowerShell)

```batch
python -m venv venv
venv\\Scripts\\activate
```

#### macOS/Linux (bash)

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

* **GPU Training:** Visit [PyTorch’s Get Started page](https://pytorch.org/get-started/locally), select your operating system, package, language, and CUDA version, then copy and run the provided installation command.

### 2. Folder Structure

```
scFL‑Green/
├── Pipeline/
│   ├── FL_main.py          # Federated training driver
│   ├── fine_tune.py        # Fine‑tuning script
│   ├── <x>_model.py        # scCDCG, scDLC, scSMD, ACTINN definitions
│   ├── Data/
│   │   ├── Training/       # CSVs for each client (e.g. Data/Training/ClientA/...)
│   │   ├── Testing/        # Optional held‑out test CSVs
│   │   └── Fine_Tune/      # CSV(s) for fine‑tuning
│   └── Results/
│       ├── Federated/      # FL outputs (global weights, metrics)
│       └── Fine_Tune/      # fine‑tune metrics, plots, checkpoints
├── Experiments/            # Part II: Experiments Reproduction
└── requirements.txt        # Dependency list
```

### 3. Running Federated Training

1. Drop client‑specific CSVs into `Data/Training/<ClientName>/`.
2. (Optional) Add test CSV(s) into `Data/Testing/`.
3. Execute:

   ```bash
   cd Pipeline
   python FL_main.py 
   ```
4. Find global model weights in `Models/`.
4. Find global model per‑round emissions, accuracy and F1 in `Results/`.

### 4. Running Fine‑Tuning

1. Place your new data CSV(s) in `Data/Fine_Tune/`.
2. Run:

   ```bash
   cd Pipeline #if you are not in the folder
   python fine_tune.py
   ```
3. Review scratch vs. fine‑tune accuracies in `Results/Fine_Tune/fine_tune_results.csv` and PNG plots in `Results/Fine_Tune/plots/`.

> **Note:** Data included here are pseudo‑simulated. Replace with your own scRNA‑seq CSVs to get valid results.

---

## Part II: Experiments Reproduction

Reproduce the benchmarks, tables, and figures from our study.

### 1. Repository Layout

```
scFL‑Green/
├── Experiments/
│    ├── Analysis/                # Scripts for generating report graphs
│    └── methodology/             # Experimental code and raw/processed data
│        ├── raw_data/            # Place original scRNA‑seq datasets A–D
│        ├── data_preprocessing/  # Data processing part
│        ├── frameworks/          # per‑model training scripts per dataset
│        ├── processed_data/      # outputs of preprocessing
│        └── graphs/              # saved experiment figures
├── Pipeline/                     # Part I: Pipeline Usage
└── requirements.txt              # Dependency list
```

### 2. Prerequisites

* Python ≥ 3.8
* `pip install -r requirements.txt`

* **GPU Training:** Visit [PyTorch’s Get Started page](https://pytorch.org/get-started/locally), select your operating system, package, language, and CUDA version, then copy and run the provided installation command.
* (GPU recommended for mixed‑precision)

### 3. Data Preparation


1. Place raw CSVs into `Experiments/methodology/raw_data/dataset_<A|B|C|D>/`.
2. From `Experiments/methodology/`:

   ```bash
   cd data_preprocessing
   python preprocessing_dataset_A.py
   # repeat for B, C, D
   ```

### 4. Running Experiments

For each framework and dataset:

```bash
cd Experiments/methodology/frameworks/scDLC_dataset_A
python scDLC_fl_train.py     # FL baseline & optimizations
python scDLC_dl_fine_tune.py # Fine‑tuning
```

Repeat for scCDCG, scSMD, ACTINN and datasets A–D.

### 5. Generating Figures

From `Experiments/Analysis/`:

```bash
python Analysis_FL_OnlyMethods_3Plots.py
python Analysis_scCDCG_scSMD_FL_Accuracy_2Subplots.py
# etc.
```

Outputs saved under `Experiments/Analysis/outputs/`.

---

## Tables

### Table 1: Frameworks

| Framework  | Paper                                                                                                                                                                          | Code                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| **scDLC**  | [scDLC: a deep learning framework to classify large sample single-cell RNA-seq data](https://pubmed.ncbi.nlm.nih.gov/35831808/)                                                | [Link](https://github.com/scDLC-code/scDLC) |
| **scCDCG** | [scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding](https://arxiv.org/abs/2404.06167)                                 | [Link](https://github.com/XPgogogo/scCDCG)  |
| **scSMD**  | [scSMD: a deep learning method for accurate clustering of single cells based on auto-encoder](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06047-x) | [Link](https://github.com/xiaoxuc/scSMD)    |
| **ACTINN** | [ACTINN: automated identification of cell types in single cell RNA sequencing](https://academic.oup.com/bioinformatics/article/36/2/533/5540320)                               | [Link](https://github.com/mafeiyang/ACTINN) |


### Table 2: Datasets

| Dataset | Tissue      | Source                                                                                               |
| ------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| **A**   | Blood       | [Link](https://cellxgene.cziscience.com/collections/0aab20b3-c30c-4606-bd2e-d20dae739c45)            |
| **B**   | Hippocampus | [Link](https://github.com/scDLC-code/scDLC/releases/tag/Data)                                        |
| **C**   | Pancreas    | [Link](https://github.com/scDLC-code/scDLC/releases/tag/Data)                                        |
| **D**   | Blood       | [Link](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b?utm_source) |


---



## License

This project is licensed under the [Apache License](LICENSE).

---

## Contact

For questions or further information, please contact:

**Mahfuzur Rahman Chowdhury**
[mahfuzur.rahman.chowdhury@fau.de](mailto:mahfuzur.rahman.chowdhury@fau.de)


