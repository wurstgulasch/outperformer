"""
Example: Fetch Data
====================
Example script showing how to fetch cryptocurrency data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DataFetcher
from loguru import logger


def main():
    """Main example function."""
    logger.info("Starting data fetch example")
    
    # Initialize data fetcher
    fetcher = DataFetcher(exchange_id='binance')
    
    # Fetch recent OHLCV data
    logger.info("Fetching recent BTC/USDT data...")
    df = fetcher.fetch_ohlcv(
        symbol='BTC/USDT',
        timeframe='1h',
        limit=100
    )
    
    print(f"\nFetched {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nLatest prices:")
    print(df.tail())
    
    # Fetch ticker
    logger.info("Fetching ticker...")
    ticker = fetcher.get_ticker('BTC/USDT')
    print(f"\nCurrent price: ${ticker['last']:,.2f}")
    
    # Fetch historical data
    logger.info("Fetching historical data...")
    df_hist = fetcher.fetch_historical_data(
        symbol='BTC/USDT',
        timeframe='1h',
        days=30
    )
    
    print(f"\nHistorical data: {len(df_hist)} candles")
    print(f"Date range: {df_hist.index[0]} to {df_hist.index[-1]}")
    
    logger.success("Example completed")


if __name__ == '__main__':
    main()
