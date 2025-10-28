"""
LSTM Model Module
=================
PyTorch LSTM model for price prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from loguru import logger


class LSTMModel(nn.Module):
    """LSTM-based neural network for price prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Number of output predictions
        """
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        logger.info(
            f"Initialized LSTM model: input={input_size}, "
            f"hidden={hidden_size}, layers={num_layers}"
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Optional hidden state tuple (h0, c0)

        Returns:
            Output tensor and hidden state tuple
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last time step
        last_time_step = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc(last_time_step)

        return out, hidden

    def init_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device to place tensors on

        Returns:
            Tuple of (h0, c0) hidden state tensors
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


class LSTMTrainer:
    """Trainer for LSTM model."""

    def __init__(
        self,
        model: LSTMModel,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: LSTM model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized LSTM trainer on {device}")

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output, _ = self.model(data)

            # Calculate loss
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}")
        return avg_loss

    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output, _ = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        logger.info(f"Validation Loss = {avg_loss:.6f}")
        return avg_loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            x: Input data as numpy array

        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            
            output, _ = self.model(x_tensor)
            predictions = output.cpu().numpy()

        return predictions

    def save_model(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")
