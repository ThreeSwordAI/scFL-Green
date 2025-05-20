"""
Fine-tuning script for scFL-Green models.

This script loads pre-trained models for scCDCG, scDLC, and scSMD,
runs scratch training for 10 epochs, then fine-tunes each model stopping
once its accuracy meets or exceeds the best scratch accuracy (after epoch 1).
It saves:
  - A CSV of accuracy per epoch for both runs
  - State dicts for each model
  - Per-method plots
  - A combined accuracy plot matching the provided example

Reuses load_csv_data from FL_main.py citeturn2file7
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

# Import model classes
from scCDCG_model import scCDCG_scRNAseqClassifier
from scDLC_model import scDLC_scRNAseqClassifier
from scSMD_model import scSMD_scRNAseqClassifier

# Configuration
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / 'Models'
DATA_DIR      = BASE_DIR / 'Data' / 'Fine_Tune'
RESULTS_DIR   = BASE_DIR / 'Results' / 'Fine_Tune'
PLOTS_DIR     = RESULTS_DIR / 'plots'
MAX_EPOCHS    = 10
DEFAULT_BATCH = 32
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
def load_csv_data(file_path: Path):
    df = pd.read_csv(file_path)
    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].astype('category').cat.codes.values, dtype=torch.long)
    return X, y

# -----------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    correct = total = 0
    for Xb, yb in dataloader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(Xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    return correct / total

# -----------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    Xs, ys = [], []
    for f in sorted(DATA_DIR.glob('*.csv')):
        Xf, yf = load_csv_data(f)
        Xs.append(Xf); ys.append(yf)
    X_all = torch.cat(Xs,0); y_all = torch.cat(ys,0)
    dataset = TensorDataset(X_all, y_all)
    input_size = X_all.size(1)
    num_classes = int(y_all.max().item())+1

    # Frameworks
    frameworks = {
        'scCDCG': (scCDCG_scRNAseqClassifier, {
            'Baseline':      {'dims_encoder':[256,64]},
            'SmallBatch':    {'batch_size':16,'dims_encoder':[256,64]},
            'MixedPrecision':{'mixed_precision':True,'dims_encoder':[256,64]},
            'ReduceComplexity':{'dims_encoder':[128,48]}
        }),
        'scDLC': (scDLC_scRNAseqClassifier, {
            'Baseline':      {'lstm_size':64,'num_layers':2},
            'SmallBatch':    {'batch_size':16,'lstm_size':64,'num_layers':2},
            'MixedPrecision':{'mixed_precision':True,'lstm_size':64,'num_layers':2},
            'ReduceComplexity':{'lstm_size':48,'num_layers':1}
        }),
        'scSMD': (scSMD_scRNAseqClassifier, {
            'Baseline':      {'latent_dim':64},
            'SmallBatch':    {'batch_size':16,'latent_dim':64},
            'MixedPrecision':{'mixed_precision':True,'latent_dim':64},
            'ReduceComplexity':{'latent_dim':48}
        })
    }

    records = []

    for fw_name, (ModelCls, methods) in frameworks.items():
        for method, params in methods.items():
            print(f"[{fw_name}|{method}] scratch→fine-tune")
            # DataLoader
            bs = params.get('batch_size', DEFAULT_BATCH)
            dl = DataLoader(dataset, batch_size=bs, shuffle=True)
            # kwargs
            kwargs = {}
            for k,v in params.items():
                if k=='dims_encoder':
                    kwargs['dims_encoder']=v; kwargs['dims_decoder']=v[::-1]
                elif k not in('batch_size','mixed_precision'):
                    kwargs[k]=v
            # models
            scratch = ModelCls(input_size,num_classes,**kwargs).to(device)
            ft = ModelCls(input_size,num_classes,**kwargs).to(device)
            ft.load_state_dict(torch.load(MODELS_DIR/fw_name/f"{method}.pth",map_location=device))
            # opts
            opt_s = optim.Adam(scratch.parameters(),lr=params.get('lr',1e-3))
            opt_ft= optim.Adam(ft.parameters(),     lr=params.get('lr',1e-3))
            crit = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler() if params.get('mixed_precision') and device.type=='cuda' else None

            # 1) scratch full run
            scratch_best = 0.0
            for ep in range(1,MAX_EPOCHS+1):
                acc_s = train_one_epoch(scratch,dl,crit,opt_s,device)
                records.append({'Framework':fw_name,'Method':method,'Run':'scratch','Epoch':ep,'Accuracy':acc_s})
                scratch_best = max(scratch_best,acc_s)

            # 2) fine-tune run
            for ep in range(1,MAX_EPOCHS+1):
                acc_ft = train_one_epoch(ft,dl,crit,opt_ft,device,scaler)
                records.append({'Framework':fw_name,'Method':method,'Run':'fine_tune','Epoch':ep,'Accuracy':acc_ft})
                if ep>1 and acc_ft>=scratch_best:
                    print(f"→ stopped at ep{ep} ft {acc_ft:.3f} ≥ scratch_best {scratch_best:.3f}")
                    break

            # save models
            od = RESULTS_DIR/fw_name; od.mkdir(parents=True,exist_ok=True)
            torch.save(scratch.state_dict(),od/f"{method}_scratch.pth")
            torch.save(ft.state_dict(),     od/f"{method}_finetuned.pth")
            # per-method plot
            dfm = pd.DataFrame([r for r in records if r['Framework']==fw_name and r['Method']==method])
            plt.figure(figsize=(8,5))
            sns.lineplot(data=dfm,x='Epoch',y='Accuracy',hue='Run',style='Run',markers=True)
            plt.title(f"{fw_name} | {method}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR/f"{fw_name}_{method}.png",dpi=300)
            plt.close()

    # CSV
    df_all = pd.DataFrame(records)
    df_all.to_csv(RESULTS_DIR/'fine_tune_results.csv',index=False)
    # combined plot
    df_all['Label']=df_all.apply(lambda r: r['Method']+(' with FineTune' if r['Run']=='fine_tune' else ''),axis=1)
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df_all,x='Epoch',y='Accuracy',hue='Label',marker='o')
    plt.title('Dataset A - Client A (scDLC, scCDCG & scSMD)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'combined_accuracy.png',dpi=300)
    plt.close()
    print("Done. Plots & CSV in",RESULTS_DIR)

if __name__=='__main__':
    main()