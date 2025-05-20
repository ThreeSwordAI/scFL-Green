# scFL-Green
**Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning**

This repository contains the code, data preprocessing scripts, and analysis pipelines used in the scFL-Green project, which benchmarks and optimizes federated learning (FL) for single-cell RNA-seq classification under carbon-aware settings (SmallBatch, MixedPrecision, ReduceComplexity, and Fine-Tuning). The detailed project report is included as a PDF in the repository root.

---

## Repository Structure

```

scFL-Green/
├── Experiments/
│   ├── Analysis/           # Scripts to generate graphs for the project report
│   └── methodoloy/         # All experimental code (note spelling intentional)
│       ├── data\_preprocessing/
│       │   ├── preprocessing\_dataset\_A.py
│       │   └── preprocessing\_dataset\_B.py
│       ├── testing\_dataset\_A.py
│       ├── frameworks/     # Framework implementations for each dataset
│       │   ├── ACTINN\_dataset\_A/
│       │   ├── ACTINN\_dataset\_B/
│       │   ├── ACTINN\_dataset\_C/
│       │   ├── ACTINN\_dataset\_D/
│       │   ├── scCDCG\_dataset\_A/
│       │   ├── scCDCG\_dataset\_B/
│       │   ├── scCDCG\_dataset\_C/
│       │   ├── scCDCG\_dataset\_D/
│       │   ├── scDLC\_dataset\_A/
│       │   ├── scDLC\_dataset\_B/
│       │   ├── scDLC\_dataset\_C/
│       │   ├── scDLC\_dataset\_D/
│       │   ├── scSMD\_dataset\_A/
│       │   ├── scSMD\_dataset\_B/
│       │   ├── scSMD\_dataset\_C/
│       │   └── scSMD\_dataset\_D/
│       ├── graphs/         # Folder to save generated graphs
│       ├── models/         # Folder for trained model weights (excluded due to size)
│       ├── processed\_data/ # Preprocessed datasets after running scripts
│       │   ├── dataset\_A/
│       │   ├── dataset\_B/
│       │   ├── dataset\_C/
│       │   └── dataset\_D/
│       └── raw\_data/       # Raw datasets to be placed here
│           ├── dataset\_A/
│           ├── dataset\_B/
│           ├── dataset\_C/
│           └── dataset\_D/

````

---

## Prerequisites

- Python ≥ 3.8
- Git
- Install Python dependencies:

```bash
pip install pandas numpy seaborn matplotlib
````

* PyTorch or TensorFlow (depending on framework)

---

## Step-by-Step Guide to Replicate Results

### 1. Clone the repository

```bash
git clone https://github.com/ThreeSwordAI/scFL-Green.git
cd scFL-Green/Experiments/methodoloy/raw_data
```

### 2. Download and organize datasets

Place each dataset into the corresponding `raw_data/dataset_<X>` folder:

* **Dataset A:** [CellXGene Collection](https://cellxgene.cziscience.com/collections/0aab20b3-c30c-4606-bd2e-d20dae739c45)
* **Dataset B:** [scDLC Data release](https://github.com/scDLC-code/scDLC/releases/tag/Data)
* **Dataset C:** [scDLC Data release](https://github.com/scDLC-code/scDLC/releases/tag/Data)
* **Dataset D:** [CellXGene Collection](https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b?utm_source)

### 3. Preprocess datasets

```bash
cd ../data_preprocessing
python preprocessing_dataset_A.py
python preprocessing_dataset_B.py
# similarly for other datasets
```

Preprocessed outputs go to `processed_data/dataset_<X>/`

### 4. Verify dataset preprocessing

```bash
python testing_dataset_A.py
```

### 5. Run framework training and evaluation

Navigate to desired framework folder:

```bash
cd ../frameworks/scDLC_dataset_A
python scDLC_model.py           # define model
python scDLC_dl_train.py        # deep learning training
python scDLC_fl_train.py        # federated learning training
python scDLC_dl_fine_tune.py    # fine-tuning
```

> **Note:**
>
> * `ACTINN` framework does not have fine-tuning (`dl_fine_tune`) scripts.

Repeat the above for each dataset (`A`, `B`, `C`, `D`) and each framework (`scDLC`, `scCDCG`, `scSMD`, `ACTINN`).

### 6. Generate analysis graphs

```bash
cd ../../Analysis
python Analysis_emissions_boxplot_normalized.py
python Analysis_baseline_boxplot_f1.py
python Analysis_FL_OnlyMethods_3Plots.py
python Analysis_scCDCG_scSMD_FL_Accuracy_2Subplots.py
```

Graphs will be saved in the designated `outputs` folder.

---

## Reference Tables

### Frameworks

| Framework  | Paper                                                                                                                                                                          | Code                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| **scDLC**  | [scDLC: a deep learning framework to classify large sample single-cell RNA-seq data](https://pubmed.ncbi.nlm.nih.gov/35831808/)                                                | [Link](https://github.com/scDLC-code/scDLC) |
| **scCDCG** | [scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding](https://arxiv.org/abs/2404.06167)                                 | [Link](https://github.com/XPgogogo/scCDCG)  |
| **scSMD**  | [scSMD: a deep learning method for accurate clustering of single cells based on auto-encoder](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06047-x) | [Link](https://github.com/xiaoxuc/scSMD)    |
| **ACTINN** | [ACTINN: automated identification of cell types in single cell RNA sequencing](https://academic.oup.com/bioinformatics/article/36/2/533/5540320)                               | [Link](https://github.com/mafeiyang/ACTINN) |

### Datasets

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


