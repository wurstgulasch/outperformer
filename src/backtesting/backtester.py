"""
Backtester Module
=================
Backtesting framework using Backtrader.
"""

import backtrader as bt
import pandas as pd
from typing import Optional, Dict, List
from loguru import logger


class MLStrategy(bt.Strategy):
    """Backtrader strategy using ML predictions."""

    params = (
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
        ('position_size', 0.95),
    )

    def __init__(self):
        """Initialize strategy."""
        self.order = None
        self.buy_price = None
        self.predictions = []
        
        logger.info("MLStrategy initialized")

    def set_predictions(self, predictions: List[float]):
        """
        Set ML model predictions.

        Args:
            predictions: List of price predictions
        """
        self.predictions = predictions

    def next(self):
        """Execute on each bar."""
        # Skip if we don't have predictions
        if len(self.predictions) <= len(self):
            return

        # Check for open orders
        if self.order:
            return

        # Get current prediction
        current_idx = len(self) - 1
        if current_idx >= len(self.predictions):
            return
            
        prediction = self.predictions[current_idx]
        current_price = self.data.close[0]

        # Trading logic based on prediction
        if not self.position:
            # Buy signal: prediction indicates price will go up
            if prediction > current_price * 1.01:  # 1% threshold
                size = (self.broker.get_cash() * self.params.position_size) / current_price
                self.order = self.buy(size=size)
                self.buy_price = current_price
                logger.debug(f"BUY signal at {current_price:.2f}")
        else:
            # Sell signal: prediction indicates price will go down or take profit
            if prediction < current_price * 0.99:  # 1% threshold
                self.order = self.sell(size=self.position.size)
                logger.debug(f"SELL signal at {current_price:.2f}")
            
            # Stop loss check
            elif self.buy_price and current_price < self.buy_price * (1 - self.params.stop_loss):
                self.order = self.sell(size=self.position.size)
                logger.debug(f"STOP LOSS at {current_price:.2f}")
            
            # Take profit check
            elif self.buy_price and current_price > self.buy_price * (1 + self.params.take_profit):
                self.order = self.sell(size=self.position.size)
                logger.debug(f"TAKE PROFIT at {current_price:.2f}")

    def notify_order(self, order):
        """Notification of order status."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED: {order.executed.price:.2f}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED: {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        """Notification of trade completion."""
        if trade.isclosed:
            logger.info(f"Trade PNL: {trade.pnl:.2f}")


class Backtester:
    """Backtesting framework for trading strategies."""

    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001
    ):
        """
        Initialize backtester.

        Args:
            initial_cash: Initial portfolio cash
            commission: Trading commission
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = None
        self.results = None
        
        logger.info(f"Backtester initialized with ${initial_cash}")

    def prepare_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """
        Prepare data feed for Backtrader.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Backtrader data feed
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create Backtrader data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        return data

    def run_backtest(
        self,
        df: pd.DataFrame,
        strategy=MLStrategy,
        predictions: Optional[List[float]] = None,
        strategy_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run backtest.

        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy class to use
            predictions: Optional ML predictions
            strategy_params: Optional strategy parameters

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest")

        # Initialize Cerebro
        self.cerebro = bt.Cerebro()
        
        # Add data
        data = self.prepare_data(df[['open', 'high', 'low', 'close', 'volume']])
        self.cerebro.adddata(data)
        
        # Add strategy
        if strategy_params:
            self.cerebro.addstrategy(strategy, **strategy_params)
        else:
            self.cerebro.addstrategy(strategy)
        
        # Set initial cash
        self.cerebro.broker.setcash(self.initial_cash)
        
        # Set commission
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Record starting value
        start_value = self.cerebro.broker.getvalue()
        logger.info(f"Starting Portfolio Value: ${start_value:.2f}")
        
        # Run backtest
        self.results = self.cerebro.run()
        
        # Record ending value
        end_value = self.cerebro.broker.getvalue()
        logger.info(f"Ending Portfolio Value: ${end_value:.2f}")
        
        # Extract results
        strategy_result = self.results[0]
        
        # Calculate metrics
        metrics = self._calculate_metrics(strategy_result, start_value, end_value)
        
        logger.info(f"Backtest complete: Total Return={metrics['total_return']:.2%}, "
                   f"Sharpe={metrics['sharpe_ratio']:.2f}")
        
        return metrics

    def _calculate_metrics(
        self,
        strategy_result,
        start_value: float,
        end_value: float
    ) -> Dict:
        """Calculate backtest metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['start_value'] = start_value
        metrics['end_value'] = end_value
        metrics['total_return'] = (end_value - start_value) / start_value
        metrics['profit'] = end_value - start_value
        
        # Sharpe Ratio
        sharpe = strategy_result.analyzers.sharpe.get_analysis()
        metrics['sharpe_ratio'] = sharpe.get('sharperatio', 0.0) or 0.0
        
        # Drawdown
        drawdown = strategy_result.analyzers.drawdown.get_analysis()
        metrics['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0.0) or 0.0
        
        # Returns
        returns = strategy_result.analyzers.returns.get_analysis()
        metrics['annualized_return'] = returns.get('rnorm100', 0.0) or 0.0
        
        # Trade analysis
        trades = strategy_result.analyzers.trades.get_analysis()
        metrics['total_trades'] = trades.get('total', {}).get('total', 0) or 0
        metrics['won_trades'] = trades.get('won', {}).get('total', 0) or 0
        metrics['lost_trades'] = trades.get('lost', {}).get('total', 0) or 0
        
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['won_trades'] / metrics['total_trades']
        else:
            metrics['win_rate'] = 0.0
        
        return metrics

    def plot_results(self, **kwargs):
        """
        Plot backtest results.

        Args:
            **kwargs: Additional plotting arguments
        """
        if self.cerebro:
            self.cerebro.plot(**kwargs)
        else:
            logger.warning("No backtest results to plot")

    def get_results(self) -> Optional[Dict]:
        """
        Get last backtest results.

        Returns:
            Dictionary with results or None
        """
        return self.results


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy optimization."""

    def __init__(
        self,
        train_periods: int = 252,  # ~1 year of trading days
        test_periods: int = 63,     # ~3 months
        step_size: int = 21         # ~1 month
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_size: Step size for rolling window
        """
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_size = step_size
        
        logger.info(f"WalkForwardAnalyzer initialized: train={train_periods}, "
                   f"test={test_periods}, step={step_size}")

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        train_func,
        test_func
    ) -> List[Dict]:
        """
        Run walk-forward analysis.

        Args:
            df: Full dataset
            train_func: Function to train model (takes train_df, returns model)
            test_func: Function to test model (takes model, test_df, returns metrics)

        Returns:
            List of test results for each window
        """
        results = []
        total_length = len(df)
        
        start_idx = 0
        while start_idx + self.train_periods + self.test_periods <= total_length:
            # Split data
            train_end = start_idx + self.train_periods
            test_end = train_end + self.test_periods
            
            train_df = df.iloc[start_idx:train_end]
            test_df = df.iloc[train_end:test_end]
            
            logger.info(f"Walk-forward: Training on {len(train_df)} samples, "
                       f"testing on {len(test_df)} samples")
            
            # Train and test
            model = train_func(train_df)
            metrics = test_func(model, test_df)
            
            results.append({
                'train_start': train_df.index[0],
                'train_end': train_df.index[-1],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'metrics': metrics
            })
            
            # Move window
            start_idx += self.step_size
        
        logger.info(f"Walk-forward analysis complete: {len(results)} windows")
        return results
