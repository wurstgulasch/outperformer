"""
Test LSTM Model
===============
Unit tests for LSTM model.
"""

import pytest
import torch
import numpy as np
from src.models import LSTMModel, LSTMTrainer


class TestLSTMModel:
    """Test cases for LSTMModel."""

    def test_init(self):
        """Test model initialization."""
        model = LSTMModel(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_forward_pass(self):
        """Test forward pass."""
        model = LSTMModel(input_size=10, hidden_size=64)
        
        # Create dummy input
        batch_size = 8
        seq_length = 20
        x = torch.randn(batch_size, seq_length, 10)
        
        # Forward pass
        output, hidden = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 1)
        assert len(hidden) == 2  # h and c

    def test_init_hidden(self):
        """Test hidden state initialization."""
        model = LSTMModel(input_size=10, hidden_size=64, num_layers=2)
        
        batch_size = 8
        h0, c0 = model.init_hidden(batch_size)
        
        assert h0.shape == (2, batch_size, 64)
        assert c0.shape == (2, batch_size, 64)


class TestLSTMTrainer:
    """Test cases for LSTMTrainer."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return LSTMModel(input_size=5, hidden_size=32)

    def test_init(self, model):
        """Test trainer initialization."""
        trainer = LSTMTrainer(model, learning_rate=0.001)
        
        assert trainer.model is not None
        assert trainer.device == 'cpu'

    def test_predict(self, model):
        """Test prediction."""
        trainer = LSTMTrainer(model)
        
        # Create dummy input
        x = np.random.randn(10, 5)
        predictions = trainer.predict(x)
        
        assert predictions.shape == (1, 1)
