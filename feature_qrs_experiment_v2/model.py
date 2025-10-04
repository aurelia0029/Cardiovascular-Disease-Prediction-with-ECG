"""
Simple Feedforward Neural Network (SimpleFNN) for Binary Classification

A straightforward fully-connected network for classifying QRS features
as normal (0) or abnormal (1).
"""

import torch
import torch.nn as nn


class SimpleFNN(nn.Module):
    """
    Simple Feedforward Neural Network.

    Architecture:
    - Input: 4 QRS features
    - Hidden layers: Configurable (default: [16, 8])
    - Output: 1 (binary classification with sigmoid)
    """

    def __init__(self, input_size=4, hidden_sizes=None, dropout=0.3):
        """
        Initialize SimpleFNN.

        Parameters:
        -----------
        input_size : int
            Number of input features (default: 4 for QRS features)
        hidden_sizes : list
            List of hidden layer sizes (default: [16, 8])
        dropout : float
            Dropout rate (default: 0.3)
        """
        super(SimpleFNN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [16, 8]

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size)

        Returns:
        --------
        output : torch.Tensor
            Predicted probabilities of shape (batch_size, 1)
        """
        return self.network(x)


class SimpleFNNTrainer:
    """
    Trainer for SimpleFNN model.
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.

        Parameters:
        -----------
        model : SimpleFNN
            Model to train
        device : str
            Device to use ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.

        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            Training data loader
        optimizer : torch.optim.Optimizer
            Optimizer

        Returns:
        --------
        loss : float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device).float()
            targets = targets.to(self.device).float().unsqueeze(1)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, data_loader):
        """
        Evaluate model on data.

        Parameters:
        -----------
        data_loader : torch.utils.data.DataLoader
            Data loader

        Returns:
        --------
        metrics : dict
            Dictionary with loss, accuracy, predictions, targets
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float().unsqueeze(1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Collect predictions and targets
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        avg_loss = total_loss / len(data_loader)
        all_preds = torch.tensor(all_preds)
        all_targets = torch.tensor(all_targets)

        accuracy = (all_preds == all_targets).float().mean().item()

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds.numpy(),
            'targets': all_targets.numpy()
        }
