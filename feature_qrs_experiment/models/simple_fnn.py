from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.split_strategies import DatasetSplits


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32)


class SimpleFNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainedFNN:
    model: SimpleFNN
    device: torch.device

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor = _to_tensor(X).to(self.device)
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1.0 - probs, probs]).T

    def state_dict(self):  # convenience for saving
        return self.model.state_dict()


def train(splits: DatasetSplits, params: Dict, seed: int):
    torch.manual_seed(seed)

    input_dim = splits.X_train.shape[1]
    hidden_dim = params.get("hidden_dim", 16)
    dropout = params.get("dropout", 0.2)
    lr = params.get("lr", 1e-3)
    batch_size = params.get("batch_size", 128)
    epochs = params.get("epochs", 100)
    patience = params.get("patience", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFNN(input_dim, hidden_dim, dropout).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(_to_tensor(splits.X_train), _to_tensor(splits.y_train))
    val_dataset = TensorDataset(_to_tensor(splits.X_val), _to_tensor(splits.y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)

        epoch_loss /= max(1, len(train_dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_X)

        val_loss /= max(1, len(val_dataset))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedFNN(model, device)


__all__ = ["train", "TrainedFNN"]
