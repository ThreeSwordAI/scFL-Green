#!/usr/bin/env python3
"""
Federated Learning Pipeline for scFL-Green

This script orchestrates federated training across multiple frameworks (ACTINN, scCDCG, scDLC, scSMD) using FedAvg,
applies optimization methods (Baseline, SmallBatch, MixedPrecision, ReduceComplexity),
and saves results and models for analysis.
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker
from sklearn.metrics import f1_score

# Import model classes from Pipeline/Models
from Models.ACTINN_model import ACTINN_scRNAseqClassifier
from Models.scCDCG_model import scCDCG_scRNAseqClassifier
from Models.scDLC_model import scDLC_scRNAseqClassifier
from Models.scSMD_model import scSMD_scRNAseqClassifier


def load_csv_data(file_path: Path):
    """
    Loads a CSV with features and a label column into tensors.

    Args:
        file_path: Path to the CSV file. Last column is assumed to be the label.

    Returns:
        Tuple of (features_tensor, labels_tensor).
    """
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].astype('category').cat.codes.values, dtype=torch.long)
    return X, y


def get_data_loaders(train_dir: Path, test_dir: Path, batch_size: int):
    """
    Prepares DataLoader objects for each client (train) and a combined test loader.

    Args:
        train_dir: Directory containing one CSV per client.
        test_dir: Directory containing one or more CSVs for testing.
        batch_size: Batch size for training and testing.

    Returns:
        Tuple (list of client DataLoaders, test DataLoader).
    """
    # Client loaders
    client_loaders = []
    for csv_file in sorted(train_dir.glob("*.csv")):
        X, y = load_csv_data(csv_file)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    # Test loader (concatenate all test CSVs)
    X_list, y_list = [], []
    for csv_file in sorted(test_dir.glob("*.csv")):
        Xf, yf = load_csv_data(csv_file)
        X_list.append(Xf)
        y_list.append(yf)
    X_test = torch.cat(X_list, dim=0)
    y_test = torch.cat(y_list, dim=0)
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader


def average_weights(state_dicts: list):
    """
    Performs federated averaging of model state dictionaries.

    Args:
        state_dicts: List of `state_dict()` outputs from client models.

    Returns:
        A new state_dict with averaged weights.
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
    methods: dict,
    device: torch.device,
    global_rounds: int = 5,
    local_epochs: int = 3
) -> pd.DataFrame:
    """
    Runs federated training for a single framework across all methods.

    Args:
        model_name: Name of the framework (e.g., 'ACTINN').
        model_class: The classifier class to instantiate.
        train_dir: Directory of client CSVs.
        test_dir: Directory of test CSVs.
        results_dir: Where to save CSV and PNG results.
        models_dir: Where to save trained model weights.
        methods: Dict mapping method names to training params.
        device: Torch device (cpu or cuda).
        global_rounds: Number of federated aggregation rounds.
        local_epochs: Local epochs per client.

    Returns:
        DataFrame summarizing final metrics per method.
    """
    # Prepare data loaders
    # Use Baseline batch size for dimension inference
    baseline_bs = methods['Baseline'].get('batch_size', 128)
    client_loaders, test_loader = get_data_loaders(train_dir, test_dir, batch_size=baseline_bs)

    # Infer input size & number of classes from first client
    sample_X, sample_y = client_loaders[0].dataset.tensors
    input_size = sample_X.size(1)
    num_classes = int(sample_y.max().item() + 1)

    # Record final metrics
    summary = []

    for method_name, params in methods.items():
        print(f"\n=== Training {model_name} with method: {method_name} ===")
        # Emissions tracking
        tracker = EmissionsTracker(project_name=f"{model_name}_{method_name}")
        tracker.start()

        # Instantiate global model
        model_kwargs = params.get('model_kwargs', {})
        global_model = model_class(input_size=input_size, num_classes=num_classes, **model_kwargs)
        global_model.to(device)

        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if params.get('mixed_precision', False) and device.type=='cuda' else None

        # Federated training loop
        for rnd in range(global_rounds):
            local_states = []
            for loader in client_loaders:
                local_model = model_class(input_size=input_size, num_classes=num_classes, **model_kwargs)
                local_model.load_state_dict(global_model.state_dict())
                local_model.to(device)

                optimizer = optim.Adam(
                    local_model.parameters(),
                    lr=params.get('lr', 1e-3),
                    weight_decay=params.get('weight_decay', 0)
                )

                local_model.train()
                for epoch in range(local_epochs):
                    for X_batch, y_batch in loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        if scaler:
                            with torch.cuda.amp.autocast():
                                logits = local_model(X_batch)
                                loss = criterion(logits, y_batch)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            logits = local_model(X_batch)
                            loss = criterion(logits, y_batch)
                            loss.backward()
                            optimizer.step()
                local_states.append(local_model.state_dict())

            # FedAvg: average client weights
            avg_state = average_weights(local_states)
            global_model.load_state_dict(avg_state)

            # Evaluate on test set
            global_model.eval()
            correct = total = 0
            all_preds, all_labels = [], []
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with torch.no_grad():
                    logits = global_model(X_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.cpu().tolist())
            accuracy = correct / total
            print(f"Round {rnd+1}/{global_rounds} | Acc: {accuracy:.4f}")

        # Stop emissions tracker
        emissions = tracker.stop()
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Save global model weights
        model_path = models_dir / f"{model_name}_{method_name}.pth"
        torch.save(global_model.state_dict(), model_path)

        # Record summary metrics
        summary.append({
            'Method': method_name,
            'Accuracy': accuracy,
            'F1': f1,
            'Emissions': emissions
        })

    # Create DataFrame and save
    df_summary = pd.DataFrame(summary)
    csv_path = results_dir / f"{model_name}_federated_summary.csv"
    df_summary.to_csv(csv_path, index=False)

    # Plot metrics
    sns.set_theme(style='whitegrid')
    for metric in ['Accuracy', 'F1', 'Emissions']:
        plt.figure(figsize=(8, 4))
        ax = sns.barplot(data=df_summary, x='Method', y=metric)
        ax.set_title(f"{model_name}: {metric} by Method", fontsize=14)
        ax.set_xlabel('')
        ax.set_ylabel(metric)
        plt.tight_layout()
        png_path = results_dir / f"{model_name}_{metric.lower()}_comparison.png"
        plt.savefig(png_path, dpi=300)
        plt.close()

    return df_summary


def main():
    """
    Main pipeline entrypoint.

    Sets up directories, device, method configs, and iterates frameworks.
    """
    base_dir = Path(__file__).parent
    train_dir = base_dir / 'Data' / 'Federated_Learning' / 'Train'
    test_dir = base_dir / 'Data' / 'Federated_Learning' / 'Test'
    results_dir = base_dir / 'Results'
    models_dir = base_dir / 'Models'
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define federated methods and parameters
    methods = {
        'Baseline': {},
        'SmallBatch': {'batch_size': 16},
        'MixedPrecision': {'mixed_precision': True},
        'ReduceComplexity': {'model_kwargs': {'hidden_sizes': [50, 25, 12]}}
    }

    # Map framework names to their classes
    frameworks = {
        'ACTINN': ACTINN_scRNAseqClassifier,
        'scCDCG': scCDCG_scRNAseqClassifier,
        'scDLC': scDLC_scRNAseqClassifier,
        'scSMD': scSMD_scRNAseqClassifier
    }

    # Run federated training for each framework
    all_dfs = []
    for name, cls in frameworks.items():
        df = train_federated_model(
            model_name=name,
            model_class=cls,
            train_dir=train_dir,
            test_dir=test_dir,
            results_dir=results_dir,
            models_dir=models_dir,
            methods=methods,
            device=device
        )
        df['Framework'] = name
        all_dfs.append(df)

    # Save overall summary
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(results_dir / 'federated_results_overall.csv', index=False)


if __name__ == '__main__':
    main()
