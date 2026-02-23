import pytest
from unittest.mock import patch, MagicMock
from scheduler import StockScheduler


def test_scheduler_creates_jobs():
    scheduler = StockScheduler(autostart=False)
    job_ids = [job.id for job in scheduler.scheduler.get_jobs()]
    assert "daily_screener" in job_ids
    assert "hourly_watchlist" in job_ids
    assert "weekly_retraining" in job_ids


def test_scheduler_does_not_start_when_autostart_false():
    scheduler = StockScheduler(autostart=False)
    assert not scheduler.scheduler.running
