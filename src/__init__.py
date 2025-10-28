"""
Outperformer AI Trading Bot
============================
Fully autonomous AI trading bot for BTC/USD outperformance using ML for predictions and trades.
"""

__version__ = "0.1.0"
__author__ = "Outperformer Team"

from src.data.data_fetcher import DataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMModel
from src.models.rl_agent import RLAgent
from src.execution.trade_executor import TradeExecutor
from src.backtesting.backtester import Backtester

__all__ = [
    "DataFetcher",
    "FeatureEngineer",
    "LSTMModel",
    "RLAgent",
    "TradeExecutor",
    "Backtester",
]
