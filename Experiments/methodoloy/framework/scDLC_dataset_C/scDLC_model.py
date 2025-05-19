import torch
import torch.nn as nn
import torch.optim as optim

class scDLC_scRNAseqClassifier(nn.Module):
    def __init__(self, input_size, num_classes, lstm_size=64, num_layers=2, dropout=0.3):
        super(scDLC_scRNAseqClassifier, self).__init__()
        print("Input Size is ", input_size)
        print("Num Classes is ", num_classes)
        self.input_size = input_size
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(input_size, 128)
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 if i == 0 else lstm_size, lstm_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(lstm_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.fc_layers:
            x = layer(x)
        x = self.fc_out(x)
        return x