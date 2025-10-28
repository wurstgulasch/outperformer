"""
Feature Engineer Module
========================
Handles feature engineering using technical indicators (TA-Lib) and custom features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using simplified indicators.")


class FeatureEngineer:
    """Engineers features from raw OHLCV data for ML models."""

    def __init__(self, use_talib: bool = True):
        """
        Initialize FeatureEngineer.

        Args:
            use_talib: Whether to use TA-Lib for indicators
        """
        self.use_talib = use_talib and TALIB_AVAILABLE
        logger.info(f"FeatureEngineer initialized (TA-Lib: {self.use_talib})")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()

        if self.use_talib:
            df = self._add_talib_indicators(df)
        else:
            df = self._add_basic_indicators(df)

        return df

    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using TA-Lib."""
        # Moving Averages
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        # RSI
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )

        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )

        # ATR (Average True Range) - Volatility
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # ADX (Average Directional Index) - Trend Strength
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # OBV (On Balance Volume)
        df['obv'] = talib.OBV(df['close'], df['volume'])

        logger.debug("Added TA-Lib indicators")
        return df

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic indicators without TA-Lib."""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        logger.debug("Added basic indicators")
        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-related features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volatility features
        """
        df = df.copy()

        # Price volatility
        df['volatility_7'] = df['close'].pct_change().rolling(7).std()
        df['volatility_30'] = df['close'].pct_change().rolling(30).std()

        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']

        # Volume volatility
        df['volume_volatility'] = df['volume'].pct_change().rolling(7).std()

        logger.debug("Added volatility features")
        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price features
        """
        df = df.copy()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price momentum
        df['momentum_1'] = df['close'].pct_change(1)
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_7'] = df['close'].pct_change(7)

        # Price position in range
        df['price_range_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        logger.debug("Added price features")
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with volume features
        """
        df = df.copy()

        # Volume moving averages
        df['volume_sma_7'] = df['volume'].rolling(7).mean()
        df['volume_sma_30'] = df['volume'].rolling(30).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_30']

        # Price-Volume trend
        # Note: This method can be called independently, so we check for 'returns'
        # If called after add_price_features(), 'returns' will already exist
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        df['pv_trend'] = df['returns'] * df['volume']

        logger.debug("Added volume features")
        return df

    def add_sentiment_features(self, df: pd.DataFrame, sentiment_scores: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Add sentiment analysis features.

        Args:
            df: DataFrame with OHLCV data
            sentiment_scores: Optional series of sentiment scores (-1 to 1)

        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()

        if sentiment_scores is not None:
            # Align sentiment scores with dataframe index
            df['sentiment_score'] = sentiment_scores.reindex(df.index, method='ffill')
            
            # Rolling sentiment averages
            df['sentiment_ma_7'] = df['sentiment_score'].rolling(7).mean()
            df['sentiment_ma_30'] = df['sentiment_score'].rolling(30).mean()
            
            # Sentiment momentum
            df['sentiment_momentum'] = df['sentiment_score'].diff(7)
            
            # Sentiment volatility
            df['sentiment_volatility'] = df['sentiment_score'].rolling(7).std()
            
            logger.debug("Added sentiment features")
        else:
            # Add placeholder sentiment features if no scores provided
            df['sentiment_score'] = 0.0
            df['sentiment_ma_7'] = 0.0
            df['sentiment_ma_30'] = 0.0
            df['sentiment_momentum'] = 0.0
            df['sentiment_volatility'] = 0.0
            logger.debug("Added placeholder sentiment features")

        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        sentiment_scores: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Args:
            df: DataFrame with OHLCV data
            sentiment_scores: Optional sentiment scores to integrate

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering")
        
        df = self.add_technical_indicators(df)
        df = self.add_volatility_features(df)
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.add_sentiment_features(df, sentiment_scores)

        # Fill NaN values
        df = df.bfill().ffill()

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def get_feature_columns(self, df: pd.DataFrame, exclude_ohlcv: bool = True) -> List[str]:
        """
        Get list of feature columns from a dataframe.

        Args:
            df: DataFrame with engineered features
            exclude_ohlcv: Whether to exclude basic OHLCV columns

        Returns:
            List of feature column names
        """
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Returns all columns except the basic OHLCV if requested
        if exclude_ohlcv:
            return [col for col in df.columns if col not in base_cols]
        return list(df.columns)
