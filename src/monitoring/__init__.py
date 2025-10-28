"""Monitoring module initialization."""

from src.monitoring.tasks import (
    celery_app,
    fetch_data_task,
    train_model_task,
    execute_trade_task,
    monitor_performance_task,
    MonitoringService
)

__all__ = [
    "celery_app",
    "fetch_data_task",
    "train_model_task",
    "execute_trade_task",
    "monitor_performance_task",
    "MonitoringService"
]
