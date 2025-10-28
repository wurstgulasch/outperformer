"""
Main Trading Bot
================
Main entry point for the AI trading bot.
"""

import sys
import os
import yaml
import argparse
from loguru import logger
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DataFetcher
from src.features import FeatureEngineer
from src.models import RLAgent
from src.execution import TradeExecutor, RiskManager
from src.backtesting import Backtester
from src.utils import SentimentAnalyzer, MarketSentimentTracker


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize trading bot.

        Args:
            config_path: Path to configuration file
        """
        # Load environment variables
        load_dotenv('config/.env')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        log_level = os.getenv('LOG_LEVEL', self.config['monitoring']['log_level'])
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        logger.add(
            self.config['paths']['logs'] + '/bot_{time}.log',
            rotation='1 day',
            retention='30 days',
            level=log_level
        )
        
        # Initialize components
        self.data_fetcher = DataFetcher(
            exchange_id=self.config['exchange']['id'],
            config={
                'apiKey': os.getenv('EXCHANGE_API_KEY'),
                'secret': os.getenv('EXCHANGE_API_SECRET'),
            }
        )
        
        self.feature_engineer = FeatureEngineer(
            use_talib=self.config['features']['use_talib']
        )
        
        self.risk_manager = RiskManager(
            max_position_size=self.config['risk']['max_position_size'],
            stop_loss_pct=self.config['risk']['stop_loss_pct'],
            take_profit_pct=self.config['risk']['take_profit_pct'],
            max_drawdown=self.config['risk']['max_drawdown']
        )
        
        self.trade_executor = TradeExecutor(
            exchange_id=self.config['exchange']['id'],
            api_key=os.getenv('EXCHANGE_API_KEY'),
            api_secret=os.getenv('EXCHANGE_API_SECRET'),
            testnet=self.config['exchange']['testnet'],
            risk_manager=self.risk_manager
        )
        
        if self.config['sentiment']['enabled']:
            self.sentiment_analyzer = SentimentAnalyzer(
                model_name=self.config['sentiment']['model']
            )
            self.sentiment_tracker = MarketSentimentTracker(self.sentiment_analyzer)
        else:
            self.sentiment_analyzer = None
            self.sentiment_tracker = None
        
        self.rl_agent = None
        
        logger.info("Trading bot initialized")

    def fetch_data(self) -> 'pd.DataFrame':
        """
        Fetch market data.

        Returns:
            DataFrame with OHLCV data
        """
        logger.info("Fetching market data")
        
        df = self.data_fetcher.fetch_historical_data(
            symbol=self.config['trading']['symbol'],
            timeframe=self.config['trading']['timeframe'],
            days=self.config['data']['historical_days']
        )
        
        return df

    def prepare_features(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Prepare features from raw data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features")
        
        df = self.feature_engineer.engineer_features(df)
        
        return df

    def train_rl_agent(self, df: 'pd.DataFrame'):
        """
        Train RL agent.

        Args:
            df: DataFrame with features
        """
        logger.info("Training RL agent")
        
        self.rl_agent = RLAgent(
            algorithm=self.config['rl']['algorithm'],
            policy=self.config['rl']['policy'],
            learning_rate=self.config['rl']['learning_rate']
        )
        
        self.rl_agent.train(
            df,
            total_timesteps=self.config['rl']['total_timesteps'],
            window_size=self.config['rl']['window_size']
        )
        
        # Save model
        model_path = f"{self.config['paths']['models']}/rl_agent.zip"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.rl_agent.save(model_path)
        
        logger.info(f"RL agent trained and saved to {model_path}")

    def run_backtest(self, df: 'pd.DataFrame') -> dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with backtest results
        """
        logger.info("Running backtest")
        
        backtester = Backtester(
            initial_cash=self.config['backtest']['initial_cash'],
            commission=self.config['backtest']['commission']
        )
        
        results = backtester.run_backtest(df)
        
        logger.info(f"Backtest results: {results}")
        
        # Check if Sharpe ratio meets target
        if results['sharpe_ratio'] >= self.config['backtest']['target_sharpe']:
            logger.success(f"Sharpe ratio {results['sharpe_ratio']:.2f} exceeds target {self.config['backtest']['target_sharpe']}")
        else:
            logger.warning(f"Sharpe ratio {results['sharpe_ratio']:.2f} below target {self.config['backtest']['target_sharpe']}")
        
        return results

    def trade(self):
        """Execute trading logic."""
        logger.info("Starting trading")
        
        # Fetch latest data
        df = self.data_fetcher.fetch_ohlcv(
            symbol=self.config['trading']['symbol'],
            timeframe=self.config['trading']['timeframe']
        )
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Get prediction from RL agent
        if self.rl_agent:
            # Get current observation
            from src.models.rl_agent import TradingEnv
            env = TradingEnv(df, window_size=self.config['rl']['window_size'])
            obs = env.reset()
            
            # Predict action
            action = self.rl_agent.predict(obs)
            
            logger.info(f"RL agent action: {action}")
            
            # Execute trade based on action
            if action == 1:  # Buy
                logger.info("Buy signal from RL agent")
                # Implement buy logic
            elif action == 2:  # Sell
                logger.info("Sell signal from RL agent")
                # Implement sell logic
            else:
                logger.info("Hold signal from RL agent")
        else:
            logger.warning("RL agent not trained")

    def run(self, mode: str = 'backtest'):
        """
        Run trading bot.

        Args:
            mode: Operating mode ('backtest', 'train', 'trade')
        """
        logger.info(f"Running bot in {mode} mode")
        
        try:
            if mode == 'backtest':
                # Fetch data
                df = self.fetch_data()
                df = self.prepare_features(df)
                
                # Run backtest
                self.run_backtest(df)
                
                logger.success("Backtest completed")
                
            elif mode == 'train':
                # Fetch data
                df = self.fetch_data()
                df = self.prepare_features(df)
                
                # Train RL agent
                self.train_rl_agent(df)
                
                # Run backtest to evaluate
                self.run_backtest(df)
                
                logger.success("Training completed")
                
            elif mode == 'trade':
                # Load trained model
                model_path = f"{self.config['paths']['models']}/rl_agent.zip"
                if os.path.exists(model_path):
                    self.rl_agent = RLAgent(algorithm=self.config['rl']['algorithm'])
                    self.rl_agent.load(model_path)
                    logger.info("Loaded trained RL agent")
                else:
                    logger.error("No trained model found. Run in 'train' mode first.")
                    return
                
                # Execute trading
                self.trade()
                
            else:
                logger.error(f"Unknown mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument(
        '--mode',
        choices=['backtest', 'train', 'trade'],
        default='backtest',
        help='Operating mode'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize and run bot
    bot = TradingBot(config_path=args.config)
    bot.run(mode=args.mode)


if __name__ == '__main__':
    main()
