"""
Data Fetcher Module
===================
Handles data ingestion from cryptocurrency exchanges using CCXT.
"""

import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from loguru import logger


class DataFetcher:
    """Fetches OHLCV data from exchanges using CCXT."""

    def __init__(
        self,
        exchange_id: str = 'binance',
        config: Optional[Dict] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize DataFetcher.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            config: Exchange configuration dictionary
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.exchange_id = exchange_id
        self.config = config or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(self.config)
        
        logger.info(f"Initialized DataFetcher with {exchange_id}")

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limit exceeded, retrying in {delay}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    raise
            except ccxt.NetworkError as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Network error: {e}, retrying in {delay}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Network error after {self.max_retries} attempts: {e}")
                    raise
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange with retry logic.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        ohlcv = self._retry_with_backoff(
            self.exchange.fetch_ohlcv,
            symbol=symbol,
            timeframe=timeframe,
            since=since,
            limit=limit
        )
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Fetched {len(df)} candles for {symbol}")
        return df

    def fetch_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical data for backtesting.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            days: Number of days of historical data

        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        since = self.exchange.parse8601(
            (datetime.now() - timedelta(days=days)).isoformat()
        )
        
        while True:
            data = self._retry_with_backoff(
                self.exchange.fetch_ohlcv,
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=1000
            )
            
            if not data:
                break
                
            all_data.extend(data)
            since = data[-1][0] + 1
            
            # Check if we've reached the present
            if since >= self.exchange.milliseconds():
                break
        
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Fetched {len(df)} historical candles for {symbol}")
        return df

    def get_ticker(self, symbol: str = 'BTC/USDT') -> Dict:
        """
        Get current ticker information.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with ticker data
        """
        ticker = self._retry_with_backoff(self.exchange.fetch_ticker, symbol)
        logger.debug(f"Fetched ticker for {symbol}: {ticker['last']}")
        return ticker

    def get_order_book(self, symbol: str = 'BTC/USDT', limit: int = 20) -> Dict:
        """
        Get order book data.

        Args:
            symbol: Trading pair symbol
            limit: Number of order book levels

        Returns:
            Dictionary with order book data
        """
        order_book = self._retry_with_backoff(self.exchange.fetch_order_book, symbol, limit)
        return order_book
