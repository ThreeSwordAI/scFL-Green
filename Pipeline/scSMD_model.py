# scSMD: a deep learning method for accurate clustering of single cells based on auto-encoder
# https://bmcbioinformatics.biomedcentral.com/

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(FCEncoder, self).__init__()
        # A simple fully connected encoder: input -> 1024 -> latent
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        latent = self.fc2(x)
        return latent

class scSMD_scRNAseqClassifier(nn.Module):
    def __init__(self, input_size, num_classes, latent_dim=64, dropout=0.3):
        """
        input_size: Number of input features (e.g., 21946 for dataset_A)
        num_classes: Number of target clusters/classes.
        latent_dim: Dimension of the latent embedding.
        dropout: Dropout rate applied before classification.
        """
        super(scSMD_scRNAseqClassifier, self).__init__()
        print("scSMD Model - Input Size:", input_size)
        print("scSMD Model - Number of Classes:", num_classes)
        self.encoder = FCEncoder(input_size, latent_dim=latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dim, num_classes)
    
    def forward(self, x, adj=None):
        # Ignore adj since we use an FC encoder
        latent = self.encoder(x)
        latent = self.dropout(latent)
        logits = self.classifier(latent)
        return logits
