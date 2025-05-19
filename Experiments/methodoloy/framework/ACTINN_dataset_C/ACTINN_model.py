import torch
import torch.nn as nn

class ACTINN_scRNAseqClassifier(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3, hidden_sizes=None):
        super(ACTINN_scRNAseqClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        # Use default hidden sizes if none provided
        if hidden_sizes is None:
            hidden_sizes = [100, 50, 25]
        self.hidden_sizes = hidden_sizes
        
        # Define layers based on the hidden_sizes list
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # LogSoftmax output for compatibility with NLLLoss
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Initialize weights using Xavier initializer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.logsoftmax(x)
        return x
