"""
Integration Tests
=================
End-to-end integration tests for the trading bot.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


class TestIntegration:
    """Integration tests for complete trading workflows."""

    @pytest.fixture
    def sample_market_data(self):
        """Create realistic sample market data."""
        dates = pd.date_range('2024-01-01', periods=500, freq='h')
        np.random.seed(42)
        
        # Simulate BTC price movement
        close_prices = 30000 + np.cumsum(np.random.randn(500) * 100)
        
        df = pd.DataFrame({
            'open': close_prices - np.random.rand(500) * 50,
            'high': close_prices + np.random.rand(500) * 100,
            'low': close_prices - np.random.rand(500) * 100,
            'close': close_prices,
            'volume': np.random.rand(500) * 1000
        }, index=dates)
        
        return df

    def test_complete_backtest_workflow(self, sample_market_data):
        """Test complete backtest workflow from data to results."""
        from src.features import FeatureEngineer
        from src.backtesting import Backtester
        
        # Step 1: Engineer features
        engineer = FeatureEngineer(use_talib=False)
        df_features = engineer.engineer_features(sample_market_data)
        
        # Verify features were added
        assert len(df_features.columns) > len(sample_market_data.columns)
        assert 'sma_20' in df_features.columns
        assert 'rsi_14' in df_features.columns
        
        # Step 2: Run backtest
        backtester = Backtester(initial_cash=10000.0, commission=0.001)
        results = backtester.run_backtest(sample_market_data)
        
        # Verify results
        assert 'start_value' in results
        assert 'end_value' in results
        assert 'sharpe_ratio' in results
        assert 'calmar_ratio' in results
        assert 'sortino_ratio' in results
        assert results['start_value'] == 10000.0

    def test_feature_engineering_with_sentiment(self, sample_market_data):
        """Test feature engineering with sentiment integration."""
        from src.features import FeatureEngineer
        
        engineer = FeatureEngineer(use_talib=False)
        
        # Create mock sentiment scores
        sentiment_scores = pd.Series(
            np.random.uniform(-1, 1, len(sample_market_data)),
            index=sample_market_data.index
        )
        
        # Engineer features with sentiment
        df_features = engineer.engineer_features(
            sample_market_data,
            sentiment_scores=sentiment_scores
        )
        
        # Verify sentiment features were added
        assert 'sentiment_score' in df_features.columns
        assert 'sentiment_ma_7' in df_features.columns
        assert 'sentiment_momentum' in df_features.columns
        assert 'sentiment_volatility' in df_features.columns

    @patch('ccxt.binance')
    def test_data_fetcher_with_retry(self, mock_exchange_class):
        """Test data fetcher retry logic."""
        from src.data import DataFetcher
        import ccxt
        
        # Setup mock exchange
        mock_exchange = Mock()
        mock_exchange_class.return_value = mock_exchange
        
        # Simulate rate limit on first call, success on second
        mock_exchange.fetch_ohlcv.side_effect = [
            ccxt.RateLimitExceeded("Rate limit exceeded"),
            [[1609459200000, 29000, 29500, 28800, 29200, 1000]]
        ]
        
        fetcher = DataFetcher(exchange_id='binance', max_retries=3, retry_delay=0.1)
        fetcher.exchange = mock_exchange
        
        # Should succeed after retry
        df = fetcher.fetch_ohlcv(symbol='BTC/USDT', limit=1)
        
        assert len(df) == 1
        assert mock_exchange.fetch_ohlcv.call_count == 2

    def test_rl_environment_trading_cycle(self, sample_market_data):
        """Test complete RL trading environment cycle."""
        from src.features import FeatureEngineer
        from src.models import TradingEnv
        
        # Prepare features
        engineer = FeatureEngineer(use_talib=False)
        df_features = engineer.engineer_features(sample_market_data)
        
        # Create environment
        env = TradingEnv(df_features, initial_balance=10000.0)
        
        # Test trading cycle
        obs = env.reset()
        assert obs is not None
        
        total_reward = 0
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        # Verify environment state
        assert 'total_value' in info
        assert 'balance' in info
        assert 'position' in info

    def test_risk_manager_dynamic_stop_loss(self):
        """Test dynamic stop loss adjustment based on volatility."""
        from src.execution import RiskManager
        
        risk_mgr = RiskManager(use_dynamic_sl=True, stop_loss_pct=0.02)
        
        # Test with low volatility (small ATR)
        current_price = 30000
        atr_low = 150  # 0.5% volatility
        sl_low = risk_mgr.adjust_stop_loss_for_volatility(atr_low, current_price)
        
        # Should use base stop loss (2%)
        assert sl_low == 0.02
        
        # Test with high volatility (large ATR)
        atr_high = 900  # 3% volatility
        sl_high = risk_mgr.adjust_stop_loss_for_volatility(atr_high, current_price)
        
        # Should adjust to 2x ATR (6%)
        assert sl_high == 0.06
        
        # Test with extreme volatility
        atr_extreme = 2000  # 6.67% volatility
        sl_extreme = risk_mgr.adjust_stop_loss_for_volatility(atr_extreme, current_price)
        
        # Should cap at 10%
        assert sl_extreme == 0.10

    def test_backtester_performance_metrics(self, sample_market_data):
        """Test that backtester calculates all performance metrics."""
        from src.backtesting import Backtester
        
        backtester = Backtester(initial_cash=10000.0)
        results = backtester.run_backtest(sample_market_data)
        
        # Verify all expected metrics are present
        expected_metrics = [
            'start_value', 'end_value', 'total_return', 'profit',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'annualized_return',
            'total_trades', 'won_trades', 'lost_trades', 'win_rate'
        ]
        
        for metric in expected_metrics:
            assert metric in results, f"Missing metric: {metric}"

    def test_feature_columns_extraction(self, sample_market_data):
        """Test feature column extraction after engineering."""
        from src.features import FeatureEngineer
        
        engineer = FeatureEngineer(use_talib=False)
        df_features = engineer.engineer_features(sample_market_data)
        
        # Test getting all columns
        all_cols = engineer.get_feature_columns(df_features, exclude_ohlcv=False)
        assert len(all_cols) == len(df_features.columns)
        
        # Test excluding OHLCV
        feature_cols = engineer.get_feature_columns(df_features, exclude_ohlcv=True)
        assert 'close' not in feature_cols
        assert 'volume' not in feature_cols
        assert 'sma_20' in feature_cols
