"""
Training script for CNN-LSTM ECG arrhythmia classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

from data_loader import load_record_list, extract_beat_windows, create_sequences
from dataset import BeatSequenceDataset
from model import CNNLSTMClassifier


def balance_dataset(X, y, random_seed=42):
    """
    Balance dataset by undersampling majority class.

    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (X_balanced, y_balanced)
    """
    X = np.array(X)
    y = np.array(y)

    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    min_count = min(len(normal_idx), len(abnormal_idx))

    # Undersample both to the same count
    np.random.seed(random_seed)
    selected_normal = np.random.choice(normal_idx, min_count, replace=False)
    selected_abnormal = np.random.choice(abnormal_idx, min_count, replace=False)

    # Combine and shuffle
    balanced_idx = np.concatenate([selected_normal, selected_abnormal])
    np.random.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx]


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001, patience=5):
    """
    Train the CNN-LSTM model with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch.device
        num_epochs (int): Maximum number of epochs
        lr (float): Learning rate
        patience (int): Early stopping patience

    Returns:
        tuple: (model with best weights, train_losses, val_losses)
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()


def main():
    """Main training pipeline."""

    # Configuration
    DATA_DIR = '../dataset/mitdb'  # MIT-BIH dataset
    SELECTED_SYMBOLS = ['N', 'L', 'R', 'V', 'A', 'F']
    NORMAL_SYMBOLS = ['N']
    SEQUENCE_LENGTH = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 5
    RANDOM_SEED = 42

    print("Loading MIT-BIH dataset...")
    record_list = load_record_list(DATA_DIR)
    print(f"Found {len(record_list)} records")

    # Extract beat windows
    print("\nExtracting beat windows...")
    beat_windows, beat_labels = extract_beat_windows(
        DATA_DIR, record_list, SELECTED_SYMBOLS, NORMAL_SYMBOLS
    )
    print(f"Extracted beats: {beat_windows.shape}")
    print(f"Label distribution: {Counter(beat_labels)}")

    # Create sequences
    print(f"\nCreating sequences of length {SEQUENCE_LENGTH}...")
    X_seq, y_seq = create_sequences(beat_windows, beat_labels, SEQUENCE_LENGTH)
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")

    # Split into train+val and test
    print("\nSplitting data...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=RANDOM_SEED
    )

    # Balance training data
    print("\nBalancing training data...")
    X_balanced, y_balanced = balance_dataset(X_trainval, y_trainval, RANDOM_SEED)
    print(f"Balanced data shape: {X_balanced.shape}")
    print(f"Balanced distribution: {Counter(y_balanced)}")

    # Split balanced data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=RANDOM_SEED
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train distribution: {Counter(y_train)}")
    print(f"Val distribution: {Counter(y_val)}")
    print(f"Test distribution: {Counter(y_test)}")

    # Create datasets and dataloaders
    print("\nCreating data loaders...")
    train_dataset = BeatSequenceDataset(X_train, y_train)
    val_dataset = BeatSequenceDataset(X_val, y_val)
    test_dataset = BeatSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = CNNLSTMClassifier(beat_len=X_train.shape[2]).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    print("\nStarting training...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=PATIENCE
    )

    # Plot training curves
    plot_training_curves(train_losses, val_losses, save_path='training_curve.png')

    # Save model
    torch.save(model.state_dict(), "cnn_lstm_best_model.pth")
    print("\n✅ Model saved as cnn_lstm_best_model.pth")

    # Save test data for evaluation
    np.save('test_data.npy', {'X_test': X_test, 'y_test': y_test})
    print("✅ Test data saved as test_data.npy")


if __name__ == "__main__":
    main()
