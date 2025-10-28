"""
Test RL Agent
=============
Unit tests for RL agent.
"""

import pytest
import pandas as pd
import numpy as np
from src.models import RLAgent, TradingEnv


class TestTradingEnv:
    """Test cases for TradingEnv."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for environment."""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': 30000 + np.cumsum(np.random.randn(100) * 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        }, index=dates)
        
        return df

    def test_init(self, sample_data):
        """Test environment initialization."""
        env = TradingEnv(sample_data, initial_balance=10000.0)
        
        assert env.initial_balance == 10000.0
        assert env.balance == 10000.0
        assert env.position == 0.0

    def test_reset(self, sample_data):
        """Test environment reset."""
        env = TradingEnv(sample_data, initial_balance=10000.0)
        
        obs = env.reset()
        
        assert obs.shape[0] == env.window_size
        assert env.balance == 10000.0
        assert env.position == 0.0

    def test_step_buy(self, sample_data):
        """Test buy action."""
        env = TradingEnv(sample_data, initial_balance=10000.0)
        env.reset()
        
        obs, reward, done, info = env.step(1)  # Buy
        
        assert env.position > 0  # Should have BTC
        assert env.balance == 0  # Used all balance

    def test_step_sell(self, sample_data):
        """Test sell action."""
        env = TradingEnv(sample_data, initial_balance=10000.0)
        env.reset()
        
        # First buy
        env.step(1)
        
        # Then sell
        obs, reward, done, info = env.step(2)
        
        assert env.position == 0  # No BTC
        assert env.balance > 0  # Got USD back


class TestRLAgent:
    """Test cases for RLAgent."""

    def test_init(self):
        """Test agent initialization."""
        agent = RLAgent(algorithm='PPO', learning_rate=0.001)
        
        assert agent.algorithm == 'PPO'
        assert agent.learning_rate == 0.001
        assert agent.model is None  # Not trained yet

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        np.random.seed(42)
        
        df = pd.DataFrame({
            'close': 30000 + np.cumsum(np.random.randn(100) * 100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        }, index=dates)
        
        return df

    def test_create_env(self, sample_data):
        """Test environment creation."""
        agent = RLAgent()
        env = agent.create_env(sample_data, initial_balance=10000.0)
        
        assert env is not None
        assert isinstance(env, TradingEnv)
