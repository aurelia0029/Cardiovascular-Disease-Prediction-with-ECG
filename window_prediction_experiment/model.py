"""
CNN-LSTM model for ECG arrhythmia classification.

This model combines:
- 1D CNN for extracting spatial features from individual beats
- LSTM for capturing temporal patterns across beat sequences
- Fully connected classifier for binary classification
"""

import torch
import torch.nn as nn


class CNNLSTMClassifier(nn.Module):
    """
    CNN-LSTM hybrid model for ECG beat sequence classification.

    Architecture:
    1. 1D CNN processes each beat individually to extract spatial features
    2. LSTM processes the sequence of CNN features to capture temporal patterns
    3. Fully connected layers for final binary classification

    Args:
        beat_len (int): Length of each beat window (default: 200)
        cnn_out_channels (int): Number of CNN output channels (default: 16)
        lstm_hidden_size (int): LSTM hidden state size (default: 64)
    """

    def __init__(self, beat_len=200, cnn_out_channels=16, lstm_hidden_size=64):
        super(CNNLSTMClassifier, self).__init__()

        # CNN block to process each beat (1D Conv over individual beat)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # downsample to beat_len/2
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output shape: (batch, channels, 1)
        )

        # LSTM to process sequence of CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
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
        batch_size, seq_len, _, beat_len = x.size()
        x = x.view(batch_size * seq_len, 1, beat_len)

        # Apply CNN to each beat
        cnn_feat = self.cnn(x)  # (batch * seq_len, channels, 1)
        cnn_feat = cnn_feat.view(batch_size, seq_len, -1)  # (batch, seq_len, cnn_out_channels)

        # Apply LSTM to sequence of CNN features
        lstm_out, _ = self.lstm(cnn_feat)  # (batch, seq_len, hidden)
        last_output = lstm_out[:, -1, :]  # get last time step

        # Classify
        out = self.classifier(last_output)
        return out
