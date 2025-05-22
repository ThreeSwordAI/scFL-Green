import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score

# Import model classes from Pipeline/Models
from ACTINN_model import ACTINN_scRNAseqClassifier
from scCDCG_model import scCDCG_scRNAseqClassifier
from scDLC_model import scDLC_scRNAseqClassifier
from scSMD_model import scSMD_scRNAseqClassifier


def load_h5ad_dataset(file_path: Path, model_name: str):
    """
    Load an .h5ad file and extract X and y tensors based on model_name.
    - ACTINN, scSMD use adata.layers['log1p']
    - scDLC, scCDCG use adata.layers['counts']
    Labels from adata.obs['cell_type']
    """
    adata = sc.read_h5ad(file_path)
    # Select layer
    if model_name in ('ACTINN', 'scSMD'):
        X_layer = adata.layers['log1p']
    else:
        X_layer = adata.layers['counts']
    # Dense conversion if sparse
    X_arr = X_layer.toarray() if hasattr(X_layer, 'toarray') else X_layer
    X = torch.tensor(X_arr, dtype=torch.float32)
    # Labels
    y = torch.tensor(
        pd.Series(adata.obs['cell_type'].astype('category').cat.codes).values,
        dtype=torch.long
    )
    return X, y


def average_weights(state_dicts: list):
    """
    Federated averaging of model state dicts.
    """
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
    return avg_state


def train_federated_model(
    model_name: str,
    model_class,
    train_dir: Path,
    test_dir: Path,
    results_dir: Path,
    models_dir: Path,
    device: torch.device,
    global_rounds: int = 5,
    local_epochs: int = 3
) -> pd.DataFrame:
    """
    Train a model with federated averaging across clients using .h5ad data.
    """
    # Define methods
    if model_name == 'ACTINN':
        methods = {
            'Baseline': {},
            'SmallBatch': {'batch_size': 16},
            'MixedPrecision': {'mixed_precision': True},
            'ReduceComplexity': {'hidden_sizes': [50, 25, 12]}
        }
    elif model_name == 'scCDCG':
        methods = {
            'Baseline': {'dims_encoder': [256, 64]},
            'SmallBatch': {'batch_size': 16, 'dims_encoder': [256, 64]},
            'MixedPrecision': {'mixed_precision': True, 'dims_encoder': [256, 64]},
            'ReduceComplexity': {'dims_encoder': [128, 48], 'dims_decoder': [48, 128]}
        }
    elif model_name == 'scDLC':
        methods = {
            'Baseline': {'lstm_size': 64, 'num_layers': 2},
            'SmallBatch': {'batch_size': 16, 'lstm_size': 64, 'num_layers': 2},
            'MixedPrecision': {'mixed_precision': True, 'lstm_size': 64, 'num_layers': 2},
            'ReduceComplexity': {'lstm_size': 48, 'num_layers': 1}
        }
    elif model_name == 'scSMD':
        methods = {
            'Baseline': {'latent_dim': 64},
            'SmallBatch': {'batch_size': 16, 'latent_dim': 64},
            'MixedPrecision': {'mixed_precision': True, 'latent_dim': 64},
            'ReduceComplexity': {'latent_dim': 48}
        }
    else:
        raise ValueError(f"Unknown framework: {model_name}")

    # Prepare client loaders from .h5ad
    base_bs = methods['Baseline'].get('batch_size', 32)
    client_loaders = []
    for h5_file in sorted((train_dir).glob('*.h5ad')):
        X, y = load_h5ad_dataset(h5_file, model_name)
        ds = TensorDataset(X, y)
        client_loaders.append(DataLoader(ds, batch_size=base_bs, shuffle=True))

    # Prepare test loader from .h5ad
    X_list, y_list = [], []
    for h5_file in sorted((test_dir).glob('*.h5ad')):
        Xt, yt = load_h5ad_dataset(h5_file, model_name)
        X_list.append(Xt); y_list.append(yt)
    X_test = torch.cat(X_list, dim=0)
    y_test = torch.cat(y_list, dim=0)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=base_bs, shuffle=False)

    # Infer dimensions
    sample_X, sample_y = client_loaders[0].dataset.tensors
    input_size = sample_X.size(1)
    num_classes = int(sample_y.max().item() + 1)

    # Federated training and evaluation
    summary = []
    for method, params in methods.items():
        print(f"\n>> {model_name} | Method: {method}")
        tracker = EmissionsTracker(project_name=f"{model_name}_{method}")
        tracker.start()

        # Initialize global model
        model_kwargs = {k: v for k, v in params.items() if k not in ('batch_size', 'mixed_precision')}
        global_model = model_class(input_size, num_classes, **model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if params.get('mixed_precision') and device.type=='cuda' else None

        for rnd in range(global_rounds):
            local_states = []
            for loader in client_loaders:
                local_model = model_class(input_size, num_classes, **model_kwargs).to(device)
                local_model.load_state_dict(global_model.state_dict())
                opt = optim.Adam(local_model.parameters(), lr=params.get('lr', 1e-3))
                local_model.train()
                for _ in range(local_epochs):
                    for Xb, yb in loader:
                        Xb, yb = Xb.to(device), yb.to(device)
                        opt.zero_grad()
                        if scaler:
                            with torch.cuda.amp.autocast():
                                loss = criterion(local_model(Xb), yb)
                            scaler.scale(loss).backward()
                            scaler.step(opt)
                            scaler.update()
                        else:
                            loss = criterion(local_model(Xb), yb)
                            loss.backward(); opt.step()
                local_states.append(local_model.state_dict())
            global_model.load_state_dict(average_weights(local_states))

        # Evaluate on test set
        global_model.eval()
        correct = total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = global_model(Xb).argmax(dim=1)
                correct += (preds==yb).sum().item()
                total += yb.size(0)
                all_preds += preds.cpu().tolist()
                all_labels += yb.cpu().tolist()
        accuracy = correct/total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        emissions = tracker.stop()

        # Save model
        out_model_dir = models_dir / model_name
        out_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), out_model_dir / f"{method}.pth")

        summary.append({'Method': method, 'Accuracy': accuracy, 'F1': f1, 'Emissions': emissions})

    # Write summary CSV and plots
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(results_dir / f"{model_name}_federated_summary.csv", index=False)
    sns.set_theme(style='whitegrid')
    for metric in ['Accuracy', 'F1', 'Emissions']:
        plt.figure(figsize=(8,4))
        sns.barplot(data=df_sum, x='Method', y=metric)
        plt.title(f"{model_name}: {metric} by Method")
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name}_{metric.lower()}_comparison.png", dpi=300)
        plt.close()

    return df_sum


def main():
    base = Path(__file__).parent
    train_dir = base / 'Data_layer' / 'Federated_Learning' / 'Train'
    test_dir = base / 'Data_layer' / 'Federated_Learning' / 'Test'
    results_dir = base / 'Results'
    models_dir = base / 'Models'
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    frameworks = {
        'ACTINN': ACTINN_scRNAseqClassifier,
        'scCDCG': scCDCG_scRNAseqClassifier,
        'scDLC': scDLC_scRNAseqClassifier,
        'scSMD': scSMD_scRNAseqClassifier
    }

    all_dfs = []
    for name, cls in frameworks.items():
        df = train_federated_model(
            model_name=name,
            model_class=cls,
            train_dir=train_dir,
            test_dir=test_dir,
            results_dir=results_dir,
            models_dir=models_dir,
            device=device
        )
        df['Framework'] = name
        all_dfs.append(df)

    overall = pd.concat(all_dfs, ignore_index=True)
    overall.to_csv(results_dir / 'federated_results_overall.csv', index=False)

if __name__ == '__main__':
    main()