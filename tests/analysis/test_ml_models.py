import pytest
import os
import pandas as pd
import numpy as np
from analysis.ml_models import StockPredictor


@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="B"),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "sma_20": np.random.randn(n),
        "sma_50": np.random.randn(n),
        "ema_12": np.random.randn(n),
        "ema_26": np.random.randn(n),
        "bb_pct": np.random.rand(n),
        "rsi": np.random.rand(n) * 100,
        "macd_histogram": np.random.randn(n),
        "vwap": 100 + np.random.randn(n),
        "atr": np.abs(np.random.randn(n)) + 0.5,
        "reg_slope_30": np.random.randn(n),
        "reg_r2_30": np.random.rand(n),
        "volume": np.random.randint(1_000_000, 10_000_000, n),
        "sentiment_score": np.random.randn(n) * 0.5,
    })
    return df


def test_prepare_features(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    assert len(X) < len(sample_features)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({"BUY", "HOLD", "SELL"})


def test_train_and_predict(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    metrics = predictor.train(X, y)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    preds = predictor.predict(X.iloc[-10:])
    assert len(preds) == 10
    assert all(p["signal"] in ["BUY", "HOLD", "SELL"] for p in preds)
    assert all(0 <= p["confidence"] <= 1 for p in preds)


def test_save_and_load(sample_features, tmp_path):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    predictor.train(X, y)

    model_path = str(tmp_path / "model.joblib")
    predictor.save(model_path)
    assert os.path.exists(model_path)

    predictor2 = StockPredictor()
    predictor2.load(model_path)
    preds = predictor2.predict(X.iloc[-5:])
    assert len(preds) == 5


def test_feature_importances(sample_features):
    predictor = StockPredictor()
    X, y = predictor.prepare_features(sample_features, forward_days=7)
    predictor.train(X, y)
    importances = predictor.feature_importances()
    assert isinstance(importances, dict)
    assert len(importances) > 0
