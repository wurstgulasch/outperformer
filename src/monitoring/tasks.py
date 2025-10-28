"""
Monitoring Module
=================
Celery-based monitoring and task scheduling.
"""

from celery import Celery
from celery.schedules import crontab
from typing import Dict
from loguru import logger
from datetime import datetime
import os


# Initialize Celery
celery_app = Celery(
    'outperformer',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
)


@celery_app.task(name='outperformer.fetch_data')
def fetch_data_task(symbol: str = 'BTC/USDT', timeframe: str = '1h') -> Dict:
    """
    Celery task to fetch market data.

    Args:
        symbol: Trading pair symbol
        timeframe: Candlestick timeframe

    Returns:
        Dictionary with task results
    """
    try:
        from src.data import DataFetcher
        
        logger.info(f"Fetching data for {symbol}")
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe)
        
        return {
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'records': len(df),
            'latest_price': float(df['close'].iloc[-1])
        }
    except Exception as e:
        logger.error(f"Error in fetch_data_task: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task(name='outperformer.train_model')
def train_model_task(algorithm: str = 'PPO', timesteps: int = 100000) -> Dict:
    """
    Celery task to train RL model.

    Args:
        algorithm: RL algorithm to use
        timesteps: Training timesteps

    Returns:
        Dictionary with task results
    """
    try:
        from src.data import DataFetcher
        from src.features import FeatureEngineer
        from src.models import RLAgent
        
        logger.info(f"Training {algorithm} model")
        
        # Fetch data
        fetcher = DataFetcher()
        df = fetcher.fetch_historical_data(days=365)
        
        # Engineer features
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)
        
        # Train model
        agent = RLAgent(algorithm=algorithm)
        agent.train(df, total_timesteps=timesteps)
        
        # Save model
        model_path = f'/tmp/model_{algorithm}_{timesteps}.zip'
        agent.save(model_path)
        
        return {
            'status': 'success',
            'algorithm': algorithm,
            'timesteps': timesteps,
            'model_path': model_path
        }
    except Exception as e:
        logger.error(f"Error in train_model_task: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task(name='outperformer.execute_trade')
def execute_trade_task(
    symbol: str,
    action: str,
    amount: float
) -> Dict:
    """
    Celery task to execute trade.

    Args:
        symbol: Trading pair symbol
        action: Trade action ('buy' or 'sell')
        amount: Trade amount

    Returns:
        Dictionary with task results
    """
    try:
        from src.execution import TradeExecutor
        
        logger.info(f"Executing {action} for {amount} {symbol}")
        
        executor = TradeExecutor(testnet=True)
        
        if action == 'buy':
            order = executor.create_market_order(symbol, 'buy', amount)
        elif action == 'sell':
            order = executor.create_market_order(symbol, 'sell', amount)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        if order:
            return {
                'status': 'success',
                'order_id': order.get('id'),
                'symbol': symbol,
                'action': action,
                'amount': amount
            }
        else:
            return {
                'status': 'failed',
                'error': 'Order creation failed'
            }
            
    except Exception as e:
        logger.error(f"Error in execute_trade_task: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


@celery_app.task(name='outperformer.monitor_performance')
def monitor_performance_task() -> Dict:
    """
    Celery task to monitor bot performance.

    Returns:
        Dictionary with performance metrics
    """
    try:
        from src.execution import TradeExecutor
        
        logger.info("Monitoring performance")
        
        executor = TradeExecutor(testnet=True)
        balance = executor.get_balance()
        position = executor.get_position('BTC/USDT')
        
        metrics = {
            'status': 'success',
            'balance': balance,
            'position': position,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error in monitor_performance_task: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


# Periodic task scheduling
celery_app.conf.beat_schedule = {
    'fetch-data-every-hour': {
        'task': 'outperformer.fetch_data',
        'schedule': crontab(minute=0),  # Every hour
    },
    'monitor-performance-every-15min': {
        'task': 'outperformer.monitor_performance',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
}


class MonitoringService:
    """Service for monitoring bot operations."""

    def __init__(self):
        """Initialize monitoring service."""
        self.metrics_history = []
        logger.info("MonitoringService initialized")

    def record_trade(self, trade_info: Dict):
        """
        Record trade information.

        Args:
            trade_info: Dictionary with trade details
        """
        self.metrics_history.append({
            'type': 'trade',
            'timestamp': datetime.now().isoformat(),
            'info': trade_info
        })
        logger.info(f"Trade recorded: {trade_info}")

    def record_prediction(self, prediction_info: Dict):
        """
        Record prediction information.

        Args:
            prediction_info: Dictionary with prediction details
        """
        self.metrics_history.append({
            'type': 'prediction',
            'timestamp': datetime.now().isoformat(),
            'info': prediction_info
        })
        logger.debug(f"Prediction recorded: {prediction_info}")

    def get_metrics(self) -> Dict:
        """
        Get aggregated metrics.

        Returns:
            Dictionary with aggregated metrics
        """
        trades = [m for m in self.metrics_history if m['type'] == 'trade']
        predictions = [m for m in self.metrics_history if m['type'] == 'prediction']
        
        return {
            'total_trades': len(trades),
            'total_predictions': len(predictions),
            'history_size': len(self.metrics_history)
        }

    def clear_old_metrics(self, keep_last: int = 1000):
        """
        Clear old metrics to manage memory.

        Args:
            keep_last: Number of recent metrics to keep
        """
        if len(self.metrics_history) > keep_last:
            self.metrics_history = self.metrics_history[-keep_last:]
            logger.info(f"Cleared old metrics, kept last {keep_last}")
