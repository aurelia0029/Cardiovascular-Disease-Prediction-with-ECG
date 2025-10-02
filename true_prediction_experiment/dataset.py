"""
PyTorch Dataset class for ECG beat sequences.
"""

import torch
from torch.utils.data import Dataset


class BeatSequenceDataset(Dataset):
    """
    PyTorch Dataset for ECG beat sequences.

    Args:
        X (numpy.ndarray): Beat sequences, shape (num_samples, seq_len, beat_len)
        y (numpy.ndarray): Labels, shape (num_samples,)
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dimension for CNN: (seq_len, 1, beat_len)
        return self.X[idx].unsqueeze(1), self.y[idx]
