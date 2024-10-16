import torch
import torch.nn as nn
import math


class TransformerSequenceModel(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, sequence_length: int):
        """
        Initializes the Transformer sequence model.

        Parameters:
            input_dim (int): Size of the input vocabulary.
            embed_dim (int): Dimensionality of the embedding layer.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimensionality of the feedforward layer.
            num_layers (int): Number of layers in the Transformer encoder.
            sequence_length (int): Length of input sequences.
        """
        super(TransformerSequenceModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = self.get_sinusoidal_positional_encoding(sequence_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    @staticmethod
    def get_sinusoidal_positional_encoding(sequence_length: int, embed_dim: int) -> torch.Tensor:
        """
        Generates sinusoidal positional encodings.

        Parameters:
            sequence_length (int): Length of sequences.
            embed_dim (int): Dimensionality of the embedding.

        Returns:
            torch.Tensor: Positional encoding tensor.
        """
        position = torch.arange(0, sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(sequence_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.embedding(x) + pos_enc
        x = self.transformer_encoder(x)
        return self.fc_out(x)

