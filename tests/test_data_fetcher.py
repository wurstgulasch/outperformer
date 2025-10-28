"""
Test Data Fetcher
=================
Unit tests for data fetching module.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data import DataFetcher


class TestDataFetcher:
    """Test cases for DataFetcher."""

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        mock = Mock()
        mock.fetch_ohlcv.return_value = [
            [1609459200000, 29000, 29500, 28800, 29200, 1000],
            [1609462800000, 29200, 29800, 29100, 29500, 1100],
            [1609466400000, 29500, 30000, 29400, 29800, 1200],
        ]
        mock.fetch_ticker.return_value = {
            'symbol': 'BTC/USDT',
            'last': 29500,
            'bid': 29490,
            'ask': 29510
        }
        return mock

    @patch('src.data.data_fetcher.ccxt')
    def test_init(self, mock_ccxt):
        """Test DataFetcher initialization."""
        mock_ccxt.binance = Mock
        fetcher = DataFetcher(exchange_id='binance')
        
        assert fetcher.exchange_id == 'binance'
        assert fetcher.exchange is not None

    @patch('src.data.data_fetcher.ccxt')
    def test_fetch_ohlcv(self, mock_ccxt, mock_exchange):
        """Test fetching OHLCV data."""
        mock_ccxt.binance = Mock(return_value=mock_exchange)
        
        fetcher = DataFetcher(exchange_id='binance')
        fetcher.exchange = mock_exchange
        
        df = fetcher.fetch_ohlcv(symbol='BTC/USDT', timeframe='1h')
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df['close'].iloc[-1] == 29800

    @patch('src.data.data_fetcher.ccxt')
    def test_get_ticker(self, mock_ccxt, mock_exchange):
        """Test fetching ticker data."""
        mock_ccxt.binance = Mock(return_value=mock_exchange)
        
        fetcher = DataFetcher(exchange_id='binance')
        fetcher.exchange = mock_exchange
        
        ticker = fetcher.get_ticker(symbol='BTC/USDT')
        
        assert ticker['symbol'] == 'BTC/USDT'
        assert ticker['last'] == 29500
