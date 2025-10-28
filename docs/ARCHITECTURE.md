# Outperformer Architecture Documentation

## Overview

Outperformer is a fully autonomous AI trading bot designed for BTC/USD outperformance using machine learning for both price prediction and adaptive trading strategies.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Outperformer Bot                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │ Data Layer   │───>│Feature Layer │───>│  ML Layer    │    │
│  │   (CCXT)     │    │ (TA-Lib)     │    │ (LSTM + RL)  │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                                         │            │
│         v                                         v            │
│  ┌──────────────┐                       ┌──────────────┐      │
│  │ Monitoring   │<─────────────────────>│  Execution   │      │
│  │  (Celery)    │                       │    (CCXT)    │      │
│  └──────────────┘                       └──────────────┘      │
│         │                                         │            │
│         v                                         v            │
│  ┌──────────────┐                       ┌──────────────┐      │
│  │ Backtesting  │                       │Risk Manager  │      │
│  │(Backtrader)  │                       │              │      │
│  └──────────────┘                       └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Layer (`src/data/`)

**Purpose**: Fetch and manage cryptocurrency market data

**Technologies**: CCXT

**Key Classes**:
- `DataFetcher`: Main data ingestion class
  - Fetches OHLCV (Open, High, Low, Close, Volume) data
  - Supports multiple exchanges (default: Binance)
  - Historical data retrieval for backtesting
  - Real-time ticker and orderbook data

**Features**:
- Exchange abstraction via CCXT
- Rate limiting to avoid API bans
- Error handling and retry logic
- Configurable timeframes (1m, 5m, 1h, 1d, etc.)

### 2. Feature Engineering Layer (`src/features/`)

**Purpose**: Transform raw price data into ML-ready features

**Technologies**: Pandas, TA-Lib (optional)

**Key Classes**:
- `FeatureEngineer`: Feature transformation pipeline

**Features Generated**:

**Technical Indicators**:
- Moving Averages (SMA 20/50/200, EMA 12/26)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
- ADX (Average Directional Index)
- OBV (On Balance Volume)

**Volatility Features**:
- Rolling volatility (7/30 periods)
- High-Low spread
- Volume volatility

**Price Features**:
- Returns (simple and log)
- Momentum (1/3/7 periods)
- Price position in range

**Volume Features**:
- Volume moving averages
- Volume ratio
- Price-Volume trend

### 3. Machine Learning Layer (`src/models/`)

**Purpose**: Predict price movements and make trading decisions

**Technologies**: PyTorch, Stable Baselines3, Gymnasium

#### 3.1 LSTM Model

**Purpose**: Time series forecasting for price prediction

**Architecture**:
- Multi-layer LSTM network (default: 2 layers, 128 hidden units)
- Dropout regularization (0.2)
- Fully connected output layers
- Sequence-to-one prediction

**Training**:
- Adam optimizer
- MSE loss function
- Gradient clipping
- Early stopping capability

#### 3.2 Reinforcement Learning Agent

**Purpose**: Adaptive trading strategy optimization

**Environment** (`TradingEnv`):
- Custom Gymnasium environment
- State: Window of technical features + account state
- Action space: [Hold, Buy, Sell]
- Reward: Portfolio return

**Algorithms**:
- PPO (Proximal Policy Optimization) - Default
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)

**Features**:
- Transaction cost modeling
- Position tracking
- Portfolio value optimization
- Target Sharpe ratio > 1.5

### 4. Execution Layer (`src/execution/`)

**Purpose**: Execute trades with risk management

**Technologies**: CCXT

**Key Classes**:

#### 4.1 RiskManager
- Position sizing limits (95% default)
- Stop loss (2% default)
- Take profit (5% default)
- Drawdown protection (10% max)
- Peak balance tracking

#### 4.2 TradeExecutor
- Market and limit orders
- Order management (create, cancel, track)
- Position management
- Automatic SL/TP placement
- Testnet support

**Safety Features**:
- Pre-trade risk checks
- Drawdown limits
- Maximum position size enforcement
- Order validation

### 5. Backtesting Layer (`src/backtesting/`)

**Purpose**: Test strategies on historical data

**Technologies**: Backtrader

**Key Classes**:

#### 5.1 Backtester
- Historical simulation
- Commission modeling
- Performance metrics calculation
- Multiple analyzers (Sharpe, Drawdown, Returns, Trades)

#### 5.2 MLStrategy
- ML prediction-based strategy
- Configurable stop loss/take profit
- Position sizing
- Signal generation

#### 5.3 WalkForwardAnalyzer
- Rolling window optimization
- Out-of-sample testing
- Strategy robustness validation

**Metrics Calculated**:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Total trades
- Profit factor

### 6. Monitoring Layer (`src/monitoring/`)

**Purpose**: Task scheduling and system monitoring

**Technologies**: Celery, Redis

**Celery Tasks**:
- `fetch_data_task`: Periodic data updates
- `train_model_task`: Model retraining
- `execute_trade_task`: Trade execution
- `monitor_performance_task`: Performance tracking

**Scheduling**:
- Hourly data fetches
- 15-minute performance checks
- Configurable via crontab

**Monitoring Service**:
- Trade recording
- Prediction logging
- Metrics aggregation
- History management

### 7. Sentiment Analysis (`src/utils/`)

**Purpose**: Analyze market sentiment from text data

**Technologies**: Hugging Face Transformers

**Key Classes**:

#### 7.1 SentimentAnalyzer
- FinBERT model for financial text
- Batch processing
- Sentiment scoring (-1 to 1)

#### 7.2 MarketSentimentTracker
- Historical sentiment tracking
- Trend analysis
- Aggregated metrics

**Features**:
- Positive/negative/neutral classification
- Confidence scores
- Temporal sentiment trends

## Configuration

### Configuration Files

**config/config.yaml**:
- Exchange settings
- Trading parameters
- Risk management rules
- Model hyperparameters
- Monitoring settings

**config/.env**:
- API credentials
- Environment variables
- Secret keys

## Data Flow

1. **Data Acquisition**:
   ```
   Exchange → DataFetcher → Raw OHLCV Data
   ```

2. **Feature Engineering**:
   ```
   Raw Data → FeatureEngineer → Engineered Features
   ```

3. **Model Training** (offline):
   ```
   Features → LSTM/RL Training → Trained Models
   ```

4. **Prediction & Trading** (online):
   ```
   Current Data → Features → Model → Prediction → Trade Signal → Execution
   ```

5. **Risk Management**:
   ```
   Trade Signal → RiskManager → Validated Trade → TradeExecutor
   ```

6. **Monitoring**:
   ```
   All Components → MonitoringService → Metrics & Logs
   ```

## Deployment Modes

### 1. Backtest Mode
Tests strategy on historical data without real trading.

```bash
python src/bot.py --mode backtest
```

### 2. Train Mode
Trains ML models on historical data.

```bash
python src/bot.py --mode train
```

### 3. Trade Mode
Executes live trading with trained models.

```bash
python src/bot.py --mode trade
```

## Performance Targets

- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 10%
- **Win Rate**: Optimized via RL
- **Transaction Costs**: Minimized through smart execution

## Risk Management

### Position Sizing
- Maximum 95% of available capital per trade
- No leverage by default
- Configurable limits

### Stop Loss & Take Profit
- Automatic SL at 2% loss
- Automatic TP at 5% gain
- Adjustable per trade

### Drawdown Protection
- Halts trading at 10% drawdown
- Peak balance tracking
- Automatic position reduction

## Testing

### Unit Tests
- 23 test cases covering all major components
- Mock exchanges for safe testing
- Isolated component testing

### Integration Testing
- End-to-end workflow validation
- Backtest validation
- Example scripts

### Security
- No known vulnerabilities in dependencies
- Input validation
- API key protection
- Testnet-first approach

## Scalability

### Horizontal Scaling
- Celery workers can run on multiple machines
- Redis as central message broker
- Stateless design

### Vertical Scaling
- GPU support for ML training
- Batch processing
- Efficient data structures

## Monitoring & Logging

### Logging
- Structured logging via Loguru
- Log levels: DEBUG, INFO, WARNING, ERROR
- Rotating log files (30-day retention)

### Metrics
- Trade history
- Performance metrics
- Model predictions
- System health

## Future Enhancements

Potential improvements:
- Multi-asset support
- Advanced RL algorithms
- Ensemble models
- Real-time news sentiment
- Portfolio optimization
- Options trading
- DeFi integration

## Security Considerations

1. **API Keys**: Never commit to version control
2. **Testnet First**: Always test on testnet before live trading
3. **Rate Limits**: Respect exchange rate limits
4. **Error Handling**: Graceful degradation
5. **Monitoring**: Active system monitoring
6. **Backups**: Model and config backups

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- ccxt (exchange integration)
- pandas (data manipulation)
- torch (deep learning)
- stable-baselines3 (RL)
- gymnasium (RL environment)
- backtrader (backtesting)
- transformers (NLP)
- celery (task scheduling)

All dependencies are kept up-to-date with latest security patches.
