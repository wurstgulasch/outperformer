"""
Data Fetcher Module
===================
Handles data ingestion from cryptocurrency exchanges using CCXT.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from loguru import logger


class DataFetcher:
    """Fetches OHLCV data from exchanges using CCXT."""

    def __init__(self, exchange_id: str = 'binance', config: Optional[Dict] = None):
        """
        Initialize DataFetcher.

        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase')
            config: Exchange configuration dictionary
        """
        self.exchange_id = exchange_id
        self.config = config or {}
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(self.config)
        
        logger.info(f"Initialized DataFetcher with {exchange_id}")

    def fetch_ohlcv(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.

        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
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
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise

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
            try:
                data = self.exchange.fetch_ohlcv(
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
                    
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
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
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"Fetched ticker for {symbol}: {ticker['last']}")
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            raise

    def get_order_book(self, symbol: str = 'BTC/USDT', limit: int = 20) -> Dict:
        """
        Get order book data.

        Args:
            symbol: Trading pair symbol
            limit: Number of order book levels

        Returns:
            Dictionary with order book data
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise
