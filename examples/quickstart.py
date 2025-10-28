#!/usr/bin/env python3
"""
Quick Start Script
==================
Run this script to quickly test the outperformer bot.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data...")
    
    dates = pd.date_range('2024-01-01', periods=365, freq='h')
    np.random.seed(42)
    
    close_prices = 30000 + np.cumsum(np.random.randn(365) * 100)
    
    df = pd.DataFrame({
        'open': close_prices - np.random.rand(365) * 50,
        'high': close_prices + np.random.rand(365) * 100,
        'low': close_prices - np.random.rand(365) * 100,
        'close': close_prices,
        'volume': np.random.rand(365) * 1000
    }, index=dates)
    
    return df


def test_feature_engineering():
    """Test feature engineering."""
    logger.info("Testing feature engineering...")
    
    from src.features import FeatureEngineer
    
    df = create_sample_data()
    engineer = FeatureEngineer(use_talib=False)
    df_features = engineer.engineer_features(df)
    
    logger.success(f"Features engineered: {df_features.shape[1]} columns")
    return df_features


def test_backtesting():
    """Test backtesting."""
    logger.info("Testing backtesting...")
    
    from src.backtesting import Backtester
    
    df = create_sample_data()
    
    backtester = Backtester(initial_cash=10000.0, commission=0.001)
    results = backtester.run_backtest(df)
    
    logger.success(f"Backtest complete:")
    logger.info(f"  Total Return: {results['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Win Rate: {results['win_rate']:.2%}")
    
    return results


def test_rl_environment():
    """Test RL environment."""
    logger.info("Testing RL environment...")
    
    from src.models import TradingEnv
    
    df = create_sample_data()
    from src.features import FeatureEngineer
    engineer = FeatureEngineer(use_talib=False)
    df = engineer.engineer_features(df)
    
    env = TradingEnv(df, initial_balance=10000.0)
    env.reset()
    
    # Take some random actions
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    logger.success(f"RL Environment test complete:")
    logger.info(f"  Total Reward: {total_reward:.4f}")
    logger.info(f"  Final Value: ${info['total_value']:.2f}")


def main():
    """Run quick start tests."""
    logger.info("Starting Outperformer Quick Test")
    logger.info("=" * 50)
    
    try:
        # Test 1: Feature Engineering
        test_feature_engineering()
        print()
        
        # Test 2: Backtesting
        test_backtesting()
        print()
        
        # Test 3: RL Environment
        test_rl_environment()
        print()
        
        logger.success("All quick tests passed!")
        logger.info("You can now:")
        logger.info("  1. Configure API keys in config/.env")
        logger.info("  2. Run: python src/bot.py --mode backtest")
        logger.info("  3. Run: python src/bot.py --mode train")
        logger.info("  4. Run: python src/bot.py --mode trade")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == '__main__':
    main()
