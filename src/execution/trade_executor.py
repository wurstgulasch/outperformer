"""
Trade Executor Module
=====================
Handles trade execution with risk management using CCXT.
"""

import ccxt
from typing import Dict, Optional, List
from loguru import logger
from datetime import datetime


class RiskManager:
    """Risk management for trade execution."""

    def __init__(
        self,
        max_position_size: float = 0.95,
        max_leverage: float = 1.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        max_drawdown: float = 0.10
    ):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum position size as fraction of balance
            max_leverage: Maximum leverage allowed
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_drawdown: Maximum allowed drawdown
        """
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        
        self.peak_balance = 0.0
        
        logger.info("RiskManager initialized")

    def check_position_size(self, amount: float, balance: float) -> float:
        """
        Check and adjust position size according to risk rules.

        Args:
            amount: Requested position size
            balance: Current account balance

        Returns:
            Adjusted position size
        """
        max_amount = balance * self.max_position_size
        if amount > max_amount:
            logger.warning(f"Position size {amount} exceeds max {max_amount}, adjusting")
            return max_amount
        return amount

    def check_drawdown(self, current_balance: float) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.

        Args:
            current_balance: Current account balance

        Returns:
            True if drawdown is acceptable, False otherwise
        """
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if drawdown > self.max_drawdown:
                logger.error(f"Drawdown {drawdown:.2%} exceeds maximum {self.max_drawdown:.2%}")
                return False

        return True

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: Trade side ('buy' or 'sell')

        Returns:
            Stop loss price
        """
        if side == 'buy':
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: Trade side ('buy' or 'sell')

        Returns:
            Take profit price
        """
        if side == 'buy':
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)


class TradeExecutor:
    """Executes trades on exchange with risk management."""

    def __init__(
        self,
        exchange_id: str = 'binance',
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize trade executor.

        Args:
            exchange_id: Exchange identifier
            api_key: API key for exchange
            api_secret: API secret for exchange
            testnet: Whether to use testnet
            risk_manager: Risk manager instance
        """
        self.exchange_id = exchange_id
        self.testnet = testnet
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        }
        
        if testnet:
            config['options'] = {'defaultType': 'future'}
            
        self.exchange = exchange_class(config)
        
        # Set testnet if available
        if hasattr(self.exchange, 'set_sandbox_mode'):
            self.exchange.set_sandbox_mode(testnet)
        
        self.risk_manager = risk_manager or RiskManager()
        
        # Trade tracking
        self.open_orders = []
        self.position = None
        
        logger.info(f"TradeExecutor initialized with {exchange_id} (testnet={testnet})")

    def get_balance(self, currency: str = 'USDT') -> float:
        """
        Get account balance.

        Args:
            currency: Currency to check

        Returns:
            Balance amount
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance.get(currency, {}).get('free', 0.0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Create market order.

        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            params: Additional order parameters

        Returns:
            Order information dictionary
        """
        try:
            # Apply risk management
            balance = self.get_balance()
            amount = self.risk_manager.check_position_size(amount, balance)
            
            # Check drawdown
            if not self.risk_manager.check_drawdown(balance):
                logger.error("Maximum drawdown exceeded, order rejected")
                return None
            
            # Create order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params=params or {}
            )
            
            logger.info(f"Market order created: {side} {amount} {symbol}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Create limit order.

        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Limit price
            params: Additional order parameters

        Returns:
            Order information dictionary
        """
        try:
            # Apply risk management
            balance = self.get_balance()
            amount = self.risk_manager.check_position_size(amount, balance)
            
            # Create order
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
                params=params or {}
            )
            
            logger.info(f"Limit order created: {side} {amount} {symbol} @ {price}")
            self.open_orders.append(order)
            return order
            
        except Exception as e:
            logger.error(f"Error creating limit order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            
            # Remove from tracking
            self.open_orders = [o for o in self.open_orders if o['id'] != order_id]
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders.

        Args:
            symbol: Optional symbol to filter

        Returns:
            List of open orders
        """
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            self.open_orders = orders
            return orders
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position information
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                self.position = positions[0]
                return self.position
            return None
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
            return None

    def close_position(self, symbol: str) -> bool:
        """
        Close current position.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            position = self.get_position(symbol)
            if not position or position.get('contracts', 0) == 0:
                logger.info("No position to close")
                return True
            
            # Determine close side
            side = 'sell' if float(position['contracts']) > 0 else 'buy'
            amount = abs(float(position['contracts']))
            
            # Close position with market order
            order = self.create_market_order(symbol, side, amount)
            
            return order is not None
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def set_stop_loss_take_profit(
        self,
        symbol: str,
        entry_price: float,
        side: str
    ) -> bool:
        """
        Set stop loss and take profit orders.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            side: Entry side ('buy' or 'sell')

        Returns:
            True if successful, False otherwise
        """
        try:
            sl_price = self.risk_manager.calculate_stop_loss(entry_price, side)
            tp_price = self.risk_manager.calculate_take_profit(entry_price, side)
            
            position = self.get_position(symbol)
            if not position:
                return False
                
            amount = abs(float(position.get('contracts', 0)))
            close_side = 'sell' if side == 'buy' else 'buy'
            
            # Create stop loss order
            sl_params = {'stopPrice': sl_price, 'type': 'stop_market'}
            self.create_limit_order(symbol, close_side, amount, sl_price, sl_params)
            
            # Create take profit order
            tp_params = {'type': 'take_profit_market'}
            self.create_limit_order(symbol, close_side, amount, tp_price, tp_params)
            
            logger.info(f"Set SL={sl_price:.2f}, TP={tp_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting SL/TP: {e}")
            return False
