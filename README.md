# Outperformer - AI Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fully autonomous AI trading bot for BTC/USD outperformance using machine learning for predictions and trade execution.

## Features

- **Data Ingestion**: Real-time and historical data fetching using CCXT
- **Feature Engineering**: Technical indicators using TA-Lib and Pandas
- **Machine Learning Models**:
  - PyTorch LSTM for price prediction
  - Reinforcement Learning (PPO/A2C/SAC) via Stable Baselines3 for adaptive strategies
- **Trade Execution**: Automated trading with CCXT and comprehensive risk management
- **Backtesting**: Historical data backtesting using Backtrader
- **Sentiment Analysis**: Market sentiment tracking using Hugging Face transformers
- **Monitoring**: Task scheduling and monitoring with Celery
- **Target**: Sharpe Ratio > 1.5

## Architecture

```
outperformer/
├── src/
│   ├── data/              # Data ingestion (CCXT)
│   ├── features/          # Feature engineering (Pandas, TA-Lib)
│   ├── models/            # ML models (LSTM, RL)
│   ├── execution/         # Trade execution with risk management
│   ├── backtesting/       # Backtesting framework (Backtrader)
│   ├── monitoring/        # Monitoring and task scheduling (Celery)
│   ├── utils/             # Utilities (sentiment analysis)
│   └── bot.py             # Main bot orchestrator
├── config/                # Configuration files
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Redis (for Celery)
- TA-Lib (optional, for advanced indicators)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/wurstgulasch/outperformer.git
cd outperformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install TA-Lib (optional but recommended):
```bash
# On Ubuntu/Debian
sudo apt-get install ta-lib

# On macOS
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

4. Configure environment:
```bash
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

5. Configure bot settings:
```bash
# Edit config/config.yaml as needed
```

## Usage

### Backtesting

Run backtest on historical data:
```bash
python src/bot.py --mode backtest
```

### Training

Train the RL agent:
```bash
python src/bot.py --mode train
```

### Live Trading

Execute live trading (testnet by default):
```bash
python src/bot.py --mode trade
```

### Running Tests

```bash
pytest tests/ -v
```

### Starting Celery Workers

For monitoring and scheduled tasks:
```bash
# Start Redis (if not running)
redis-server

# Start Celery worker
celery -A src.monitoring.tasks worker --loglevel=info

# Start Celery beat for scheduled tasks
celery -A src.monitoring.tasks beat --loglevel=info
```

## Configuration

Key configuration options in `config/config.yaml`:

- **Exchange**: API credentials, testnet mode
- **Trading**: Symbol, timeframe, initial balance
- **Risk Management**: Position sizing, stop loss, take profit, max drawdown
- **Models**: LSTM and RL hyperparameters
- **Backtesting**: Initial cash, commission, target Sharpe ratio
- **Sentiment**: Enable/disable sentiment analysis

## Risk Management

The bot includes comprehensive risk management:

- **Position Sizing**: Configurable maximum position size
- **Stop Loss**: Automatic stop loss orders
- **Take Profit**: Automatic take profit orders
- **Drawdown Protection**: Maximum drawdown limits
- **Volatility Handling**: Dynamic position sizing based on market volatility

## Models

### LSTM Price Prediction

- Multi-layer LSTM network for time series forecasting
- Technical indicators as input features
- Price movement prediction

### Reinforcement Learning

- PPO (Proximal Policy Optimization) as default algorithm
- Custom trading environment with realistic transaction costs
- Continuous learning and adaptation
- Target: Sharpe Ratio > 1.5

### Sentiment Analysis

- Financial sentiment analysis using FinBERT
- Market sentiment tracking and aggregation
- Integration with trading decisions

## Monitoring

- Real-time performance tracking
- Automated task scheduling via Celery
- Metrics collection and logging
- Trade history and analytics

## Testing

Comprehensive test suite covering:

- Data fetching and processing
- Feature engineering
- Model training and prediction
- Backtesting
- Trade execution

Run tests with:
```bash
pytest tests/ -v --cov=src
```

## Development

### Project Structure

- `src/data/`: Data fetching and management
- `src/features/`: Feature engineering pipeline
- `src/models/`: ML models (LSTM, RL)
- `src/execution/`: Trade execution and risk management
- `src/backtesting/`: Backtesting framework
- `src/monitoring/`: Monitoring and task scheduling
- `src/utils/`: Utility functions (sentiment, etc.)
- `config/`: Configuration files
- `tests/`: Unit and integration tests

### Adding New Features

1. Implement feature in appropriate module
2. Add tests in `tests/`
3. Update configuration if needed
4. Update documentation

## Performance Targets

- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 10%
- **Win Rate**: Optimized through RL
- **Risk-Adjusted Returns**: Primary optimization goal

## Disclaimer

**This software is for educational and research purposes only.**

- Cryptocurrency trading carries significant risk
- Past performance does not guarantee future results
- Test thoroughly on testnet before live trading
- Never invest more than you can afford to lose
- The authors assume no liability for financial losses

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for usage examples

## Acknowledgments

- **CCXT**: Cryptocurrency exchange integration
- **Freqtrade**: Trading framework inspiration
- **Stable Baselines3**: RL algorithms
- **PyTorch**: Deep learning framework
- **Backtrader**: Backtesting engine
- **Hugging Face**: NLP models
