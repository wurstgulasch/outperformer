"""
Reinforcement Learning Agent Module
====================================
RL agent using Stable Baselines3 for adaptive trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from loguru import logger
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


class TradingEnv(gym.Env):
    """Custom trading environment for RL agent."""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        window_size: int = 20
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with features and price data
            initial_balance: Starting account balance
            transaction_fee: Transaction fee percentage
            window_size: Observation window size
        """
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size

        # Current step in the episode
        self.current_step = window_size
        
        # Account state
        self.balance = initial_balance
        self.position = 0.0  # BTC holdings
        self.total_value = initial_balance

        # Feature columns (exclude price columns used for trading)
        self.feature_columns = [
            col for col in df.columns
            if col not in ['open', 'high', 'low', 'close', 'volume']
        ]

        # Action space: [0: hold, 1: buy, 2: sell]
        self.action_space = spaces.Discrete(3)

        # Observation space: features from window_size periods
        n_features = len(self.feature_columns) + 3  # +3 for balance, position, price
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32
        )

        logger.info(f"Initialized TradingEnv with {len(self.df)} steps")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_value = self.initial_balance
        
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        start = self.current_step - self.window_size
        end = self.current_step

        # Get feature window
        features = self.df[self.feature_columns].iloc[start:end].values

        # Add account state to each time step
        current_price = self.df['close'].iloc[self.current_step - 1]
        account_state = np.array([
            self.balance / self.initial_balance,
            self.position,
            current_price
        ])
        
        # Broadcast account state to all time steps
        account_states = np.tile(account_state, (self.window_size, 1))
        
        # Concatenate features and account state
        observation = np.concatenate([features, account_states], axis=1)
        
        return observation.astype(np.float32)

    def step(self, action: int) -> tuple:
        """
        Execute action in environment.

        Args:
            action: Trading action (0: hold, 1: buy, 2: sell)

        Returns:
            Tuple of (observation, reward, done, info)
        """
        current_price = self.df['close'].iloc[self.current_step]
        
        # Execute action
        if action == 1:  # Buy
            # Use all balance to buy BTC
            if self.balance > 0:
                buy_amount = self.balance / current_price
                fee = buy_amount * self.transaction_fee
                self.position += (buy_amount - fee)
                self.balance = 0
                
        elif action == 2:  # Sell
            # Sell all BTC
            if self.position > 0:
                sell_value = self.position * current_price
                fee = sell_value * self.transaction_fee
                self.balance += (sell_value - fee)
                self.position = 0

        # Calculate total portfolio value
        self.total_value = self.balance + (self.position * current_price)

        # Move to next step
        self.current_step += 1
        
        # Calculate reward (portfolio return)
        reward = (self.total_value - self.initial_balance) / self.initial_balance

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        # Additional info
        info = {
            'total_value': self.total_value,
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price
        }

        observation = self._get_observation() if not done else np.zeros_like(self._get_observation())

        return observation, reward, done, info

    def render(self, mode: str = 'human'):
        """Render environment state."""
        profit = self.total_value - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position:.6f} BTC")
        print(f"Total Value: ${self.total_value:.2f}")
        print(f"Profit: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)")


class RLAgent:
    """Reinforcement Learning trading agent using Stable Baselines3."""

    def __init__(
        self,
        algorithm: str = 'PPO',
        policy: str = 'MlpPolicy',
        learning_rate: float = 0.0003,
        **kwargs
    ):
        """
        Initialize RL agent.

        Args:
            algorithm: RL algorithm ('PPO', 'A2C', 'SAC')
            policy: Policy network architecture
            learning_rate: Learning rate
            **kwargs: Additional arguments for the algorithm
        """
        self.algorithm = algorithm
        self.policy = policy
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        self.model = None
        self.env = None

        logger.info(f"Initialized RLAgent with {algorithm}")

    def create_env(self, df: pd.DataFrame, **env_kwargs) -> gym.Env:
        """
        Create trading environment.

        Args:
            df: DataFrame with features and price data
            **env_kwargs: Additional environment arguments

        Returns:
            Trading environment
        """
        self.env = TradingEnv(df, **env_kwargs)
        return self.env

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 100000,
        **env_kwargs
    ):
        """
        Train RL agent.

        Args:
            df: Training data DataFrame
            total_timesteps: Number of training timesteps
            **env_kwargs: Environment configuration
        """
        logger.info(f"Starting RL training for {total_timesteps} timesteps")

        # Create vectorized environment
        env = DummyVecEnv([lambda: self.create_env(df, **env_kwargs)])

        # Create model
        if self.algorithm == 'PPO':
            self.model = PPO(
                self.policy,
                env,
                learning_rate=self.learning_rate,
                verbose=1,
                **self.kwargs
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                self.policy,
                env,
                learning_rate=self.learning_rate,
                verbose=1,
                **self.kwargs
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                self.policy,
                env,
                learning_rate=self.learning_rate,
                verbose=1,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Train model
        self.model.learn(total_timesteps=total_timesteps)
        
        logger.info("RL training completed")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action for given observation.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def evaluate(self, df: pd.DataFrame, n_episodes: int = 10, **env_kwargs) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            df: Evaluation data DataFrame
            n_episodes: Number of episodes to evaluate
            **env_kwargs: Environment configuration

        Returns:
            Dictionary with evaluation metrics
        """
        env = self.create_env(df, **env_kwargs)
        
        total_rewards = []
        total_values = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.predict(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            total_values.append(info['total_value'])

        metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_final_value': np.mean(total_values),
            'sharpe_ratio': np.mean(total_rewards) / (np.std(total_rewards) + 1e-10)
        }

        logger.info(f"Evaluation: Mean Reward={metrics['mean_reward']:.4f}, "
                   f"Sharpe={metrics['sharpe_ratio']:.4f}")

        return metrics

    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        logger.info(f"Model loaded from {path}")
