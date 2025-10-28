"""
Test Backtester
===============
Unit tests for backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting import Backtester, MLStrategy


class TestBacktester:
    """Test cases for Backtester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1D')
        np.random.seed(42)
        
        close_prices = 30000 + np.cumsum(np.random.randn(200) * 100)
        
        df = pd.DataFrame({
            'open': close_prices - np.random.rand(200) * 50,
            'high': close_prices + np.random.rand(200) * 100,
            'low': close_prices - np.random.rand(200) * 100,
            'close': close_prices,
            'volume': np.random.rand(200) * 1000
        }, index=dates)
        
        return df

    def test_init(self):
        """Test backtester initialization."""
        backtester = Backtester(initial_cash=10000.0, commission=0.001)
        
        assert backtester.initial_cash == 10000.0
        assert backtester.commission == 0.001

    def test_prepare_data(self, sample_data):
        """Test data preparation."""
        backtester = Backtester()
        data = backtester.prepare_data(sample_data)
        
        assert data is not None

    def test_run_backtest(self, sample_data):
        """Test running backtest."""
        backtester = Backtester(initial_cash=10000.0, commission=0.001)
        
        results = backtester.run_backtest(sample_data)
        
        assert 'start_value' in results
        assert 'end_value' in results
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        
        assert results['start_value'] == 10000.0
