# scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding
# https://arxiv.org/abs/2404.

import torch
import torch.nn as nn

# Define the AE_NN autoencoder architecture directly.
class AE_NN(nn.Module):
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        """
        dim_input: Dimension of the input features.
        dims_encoder: List of hidden dimensions for the encoder.
        dims_decoder: List of hidden dimensions for the decoder.
        """
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]
        self.num_layer = len(self.dims_en) - 1
        
        self.Encoder = nn.ModuleList()
        self.Decoder = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        for index in range(self.num_layer):
            self.Encoder.append(nn.Linear(self.dims_en[index], self.dims_en[index+1]))
            self.Decoder.append(nn.Linear(self.dims_de[index], self.dims_de[index+1]))
    
    def forward(self, x, adj=None):
        # Note: The 'adj' is ignored in this simple NN variant.
        for layer in self.Encoder:
            x = layer(x)
            # Optionally, you could add non-linearity here:
            x = self.leakyrelu(x)
        h = x  # latent embedding
        
        for layer in self.Decoder:
            x = layer(x)
            # Optionally add non-linearity in decoder too
            x = self.leakyrelu(x)
        x_hat = x      
        return h, x_hat

# Now, define the scCDCG classifier that uses the AE_NN autoencoder as its feature extractor.
class scCDCG_scRNAseqClassifier(nn.Module):
    def __init__(self, input_size, num_classes, embedding_num=64, dims_encoder=None, dims_decoder=None, dropout=0.3):
        """
        input_size: Number of input features.
        num_classes: Number of target classes.
        embedding_num: Dimension of the latent embedding.
        dims_encoder: List of hidden dimensions for the encoder; default is [256, embedding_num].
        dims_decoder: List of hidden dimensions for the decoder; default is [embedding_num, 256].
        dropout: Dropout rate for the classifier head.
        """
        super(scCDCG_scRNAseqClassifier, self).__init__()
        if dims_encoder is None:
            dims_encoder = [256, embedding_num]
        if dims_decoder is None:
            dims_decoder = [embedding_num, 256]
        
        # Create the autoencoder (using our AE_NN defined above)
        self.ae = AE_NN(dim_input=input_size, dims_encoder=dims_encoder, dims_decoder=dims_decoder)
        
        # Classification head: maps latent embedding to class logits.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims_encoder[-1], num_classes)
        )
        
        # Logging the configuration
        print("scCDCG Model - Input Size:", input_size)
        print("scCDCG Model - Number of Classes:", num_classes)
        print("scCDCG Model - Encoder Dims:", dims_encoder)

    def forward(self, x, adj=None):
        """
        x: Input tensor of shape (batch_size, input_size).
        adj: Optional adjacency matrix; not used in AE_NN.
        """
        # If no adjacency is provided, we simply ignore it.
        h, x_hat = self.ae(x, adj)
        # Compute the logits from the latent embedding.
        logits = self.classifier(h)
        return logits

