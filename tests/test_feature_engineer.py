"""
Test Feature Engineer
======================
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=300, freq='h')
        np.random.seed(42)
        
        close_prices = 30000 + np.cumsum(np.random.randn(300) * 100)
        
        df = pd.DataFrame({
            'open': close_prices - np.random.rand(300) * 50,
            'high': close_prices + np.random.rand(300) * 100,
            'low': close_prices - np.random.rand(300) * 100,
            'close': close_prices,
            'volume': np.random.rand(300) * 1000
        }, index=dates)
        
        return df

    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(use_talib=False)
        assert engineer.use_talib == False

    def test_add_technical_indicators(self, sample_data):
        """Test adding technical indicators."""
        engineer = FeatureEngineer(use_talib=False)
        df = engineer.add_technical_indicators(sample_data)
        
        # Check that indicators were added
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns
        assert 'ema_12' in df.columns
        assert 'rsi_14' in df.columns
        assert 'bb_upper' in df.columns
        assert 'bb_lower' in df.columns

    def test_add_volatility_features(self, sample_data):
        """Test adding volatility features."""
        engineer = FeatureEngineer()
        df = engineer.add_volatility_features(sample_data)
        
        assert 'volatility_7' in df.columns
        assert 'volatility_30' in df.columns
        assert 'hl_spread' in df.columns

    def test_add_price_features(self, sample_data):
        """Test adding price features."""
        engineer = FeatureEngineer()
        df = engineer.add_price_features(sample_data)
        
        assert 'returns' in df.columns
        assert 'momentum_1' in df.columns
        assert 'momentum_7' in df.columns

    def test_add_volume_features(self, sample_data):
        """Test adding volume features."""
        engineer = FeatureEngineer()
        df = engineer.add_volume_features(sample_data)
        
        assert 'volume_sma_7' in df.columns
        assert 'volume_ratio' in df.columns

    def test_engineer_features(self, sample_data):
        """Test full feature engineering pipeline."""
        engineer = FeatureEngineer(use_talib=False)
        df = engineer.engineer_features(sample_data)
        
        # Should have all original columns plus features
        assert len(df.columns) > len(sample_data.columns)
        
        # Check for key features
        assert 'sma_20' in df.columns
        assert 'rsi_14' in df.columns
        assert 'volatility_7' in df.columns
        assert 'returns' in df.columns
        assert 'volume_ratio' in df.columns
        
        # No NaN values after filling
        assert df.isna().sum().sum() == 0
