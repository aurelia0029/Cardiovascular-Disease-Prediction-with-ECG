"""
Transformer model for ECG arrhythmia prediction.

This model uses self-attention mechanisms to capture temporal patterns
in ECG beat sequences for predicting future abnormalities.
"""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """
    Transformer-based model for ECG beat sequence classification.

    Architecture:
    1. Linear projection of each beat into embedding space
    2. Positional encoding to capture sequence order
    3. Transformer encoder layers with self-attention
    4. Classification head for binary prediction

    Args:
        seq_len (int): Number of beats in input sequence (default: 7)
        beat_len (int): Length of each beat window (default: 256)
        d_model (int): Embedding dimension (default: 128)
        nhead (int): Number of attention heads (default: 4)
        num_layers (int): Number of transformer layers (default: 2)
        dim_feedforward (int): Feedforward network dimension (default: 256)
    """

    def __init__(self, seq_len=7, beat_len=256, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256):
        super(TransformerClassifier, self).__init__()

        self.seq_len = seq_len
        self.beat_len = beat_len
        self.d_model = d_model

        # Project each beat (1D signal) into an embedding vector
        self.input_proj = nn.Sequential(
            nn.Flatten(start_dim=2),  # (batch, seq_len, beat_len)
            nn.Linear(beat_len, d_model)  # (batch, seq_len, d_model)
        )

        # Learnable positional encoding
        # This helps the model understand the order of beats in the sequence
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, 1, beat_len)

        Returns:
            torch.Tensor: Output predictions of shape (batch, 1)
        """
        # x: (batch, seq_len, 1, beat_len)
        x = x.squeeze(2)  # → (batch, seq_len, beat_len)
        x = self.input_proj(x)  # → (batch, seq_len, d_model)
        x = x + self.pos_embedding  # Add positional encoding

        # Apply transformer encoder
        encoded = self.transformer(x)  # (batch, seq_len, d_model)

        # Take the representation of the last time step
        # This contains information about the entire sequence
        out = encoded[:, -1, :]  # (batch, d_model)

        # Classify
        return self.classifier(out)  # (batch, 1)
