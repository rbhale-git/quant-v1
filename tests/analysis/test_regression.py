import pytest
import pandas as pd
import numpy as np
from analysis.regression import compute_regression


@pytest.fixture
def trending_data():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def flat_data():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.random.randn(n) * 0.1
    return pd.DataFrame({"date": dates, "close": close})


def test_regression_uptrend(trending_data):
    result = compute_regression(trending_data, window=30)
    assert "reg_slope_30" in result.columns
    assert "reg_r2_30" in result.columns
    assert "reg_predicted_30" in result.columns
    last = result.iloc[-1]
    assert last["reg_slope_30"] > 0
    assert last["reg_r2_30"] > 0.8


def test_regression_flat(flat_data):
    result = compute_regression(flat_data, window=30)
    last = result.iloc[-1]
    assert abs(last["reg_slope_30"]) < 0.1


def test_regression_multiple_windows(trending_data):
    result = compute_regression(trending_data, window=30)
    result = compute_regression(result, window=60)
    assert "reg_slope_30" in result.columns
    assert "reg_slope_60" in result.columns
