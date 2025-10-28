# Outperformer - Getting Started

Welcome to Outperformer, your AI-powered trading bot for BTC/USD!

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: TA-Lib is optional but recommended. If you skip it, the bot will use simplified indicators.

### 2. Run the Quick Test

```bash
python examples/quickstart.py
```

This will:
- Test feature engineering ✓
- Run a backtest ✓
- Test the RL environment ✓

### 3. Configure Your Bot

Copy the example environment file:

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your API keys (for live trading only):

```env
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
TRADING_MODE=testnet  # or 'live' for real trading
```

### 4. Choose Your Mode

#### Backtesting (Safe - No Real Money)

Test the bot on historical data:

```bash
python src/bot.py --mode backtest
```

#### Training (Computational)

Train the RL agent on historical data:

```bash
python src/bot.py --mode train
```

This will:
- Fetch 1 year of historical data
- Engineer features
- Train the RL agent
- Save the model to `./models/`
- Run a backtest to evaluate performance

**Time**: 10-30 minutes depending on your hardware

#### Trading (Testnet/Live)

Execute live trades using the trained model:

```bash
python src/bot.py --mode trade
```

**⚠️ IMPORTANT**: 
- Always test on testnet first (default)
- Change `testnet: true` to `testnet: false` in `config/config.yaml` for live trading
- Start with small amounts
- Monitor closely

## Understanding the Bot

### What It Does

1. **Fetches Data**: Gets BTC/USD price data from exchanges
2. **Analyzes**: Creates 29+ technical features from price data
3. **Predicts**: Uses LSTM neural network and RL agent to predict price movements
4. **Trades**: Executes buy/sell orders based on predictions
5. **Manages Risk**: Applies stop-loss, take-profit, and position sizing rules
6. **Learns**: Adapts trading strategy through reinforcement learning

### Key Features

✅ **Fully Modular**: Each component works independently
✅ **Risk Management**: Built-in stop-loss and position limits
✅ **Backtesting**: Test strategies before risking real money
✅ **Machine Learning**: LSTM + Reinforcement Learning
✅ **Sentiment Analysis**: Analyze market sentiment (optional)
✅ **Monitoring**: Track performance in real-time
✅ **Well-Tested**: 23 unit tests covering all components

### Configuration

Edit `config/config.yaml` to customize:

```yaml
# Trading pair and timeframe
trading:
  symbol: BTC/USDT
  timeframe: 1h
  initial_balance: 10000.0

# Risk limits
risk:
  max_position_size: 0.95  # Use 95% of balance
  stop_loss_pct: 0.02      # 2% stop loss
  take_profit_pct: 0.05    # 5% take profit
  max_drawdown: 0.10       # 10% max drawdown

# Target performance
backtest:
  target_sharpe: 1.5       # Sharpe ratio target
```

## Project Structure

```
outperformer/
├── src/                    # Source code
│   ├── data/              # Data fetching
│   ├── features/          # Feature engineering
│   ├── models/            # ML models (LSTM, RL)
│   ├── execution/         # Trade execution
│   ├── backtesting/       # Backtesting engine
│   ├── monitoring/        # Task scheduling
│   ├── utils/             # Utilities (sentiment)
│   └── bot.py             # Main entry point
├── config/                # Configuration
├── tests/                 # Unit tests
├── examples/              # Example scripts
├── docs/                  # Documentation
├── models/                # Saved models
├── logs/                  # Log files
└── data/                  # Downloaded data
```

## Running Tests

Ensure everything works:

```bash
pytest tests/ -v
```

Expected: All 23 tests should pass ✓

## Examples

### Example 1: Fetch Data

```bash
python examples/fetch_data_example.py
```

Demonstrates how to fetch market data from exchanges.

### Example 2: Quick Test

```bash
python examples/quickstart.py
```

Runs a complete test of all components.

## Monitoring

### With Celery (Advanced)

For automated scheduling:

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery worker
celery -A src.monitoring.tasks worker --loglevel=info

# Terminal 3: Start Celery beat (scheduler)
celery -A src.monitoring.tasks beat --loglevel=info
```

This enables:
- Automatic hourly data updates
- Performance monitoring every 15 minutes
- Scheduled retraining

## Performance Targets

The bot aims for:

- **Sharpe Ratio**: > 1.5 (risk-adjusted returns)
- **Max Drawdown**: < 10% (capital preservation)
- **Win Rate**: Optimized through RL
- **Volatility Handling**: Adaptive position sizing

## Safety Tips

### Before Live Trading

1. ✅ Run backtests on different time periods
2. ✅ Test on testnet for at least a week
3. ✅ Verify risk management works (test stop-loss)
4. ✅ Start with small amounts
5. ✅ Monitor closely for the first few days
6. ✅ Have a plan to stop if things go wrong

### Risk Management

- **Never invest more than you can afford to lose**
- Use stop-losses (configured in risk settings)
- Monitor drawdown closely
- Have exit strategies
- Keep API keys secure
- Use testnet first, always

## Common Issues

### Issue: TA-Lib not installed

**Solution**: The bot will work without TA-Lib using simplified indicators. To install TA-Lib:

```bash
# Ubuntu/Debian
sudo apt-get install ta-lib
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib
```

### Issue: Exchange API errors

**Solution**: 
- Check API keys are correct
- Verify API permissions (read + trade)
- Check rate limits
- Try testnet first

### Issue: Model training is slow

**Solution**:
- Reduce `total_timesteps` in config
- Use GPU if available
- Reduce historical data days
- Use smaller model architecture

## Getting Help

1. **Read the docs**: Check `docs/ARCHITECTURE.md`
2. **Check examples**: Look at `examples/`
3. **Run tests**: `pytest tests/ -v`
4. **Check logs**: Look in `logs/` directory

## Next Steps

1. **Learn More**: Read `docs/ARCHITECTURE.md`
2. **Customize**: Edit `config/config.yaml`
3. **Experiment**: Try different parameters
4. **Backtest**: Test various strategies
5. **Train**: Optimize the RL agent
6. **Monitor**: Watch performance metrics

## Disclaimer

**This software is for educational purposes only.**

- Cryptocurrency trading is risky
- Past performance ≠ future results
- Test thoroughly before live trading
- Start with small amounts
- Never invest more than you can afford to lose
- The authors assume no liability for losses

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for usage examples

---

**Ready to start?** Run: `python examples/quickstart.py`

**Need help?** Read: `docs/ARCHITECTURE.md`

**Want to contribute?** Check: `CONTRIBUTING.md` (coming soon)

Happy trading! 🚀📈
