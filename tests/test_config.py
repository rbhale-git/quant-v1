from config import Settings


def test_default_settings():
    s = Settings()
    assert s.db_path == "stock_analyzer.db"
    assert s.sma_windows == [20, 50, 200]
    assert s.ema_windows == [12, 26]
    assert s.rsi_window == 14
    assert s.backtest_starting_capital == 10_000.0
    assert s.ml_forward_days == 7


def test_custom_settings():
    s = Settings(db_path="custom.db", rsi_window=21)
    assert s.db_path == "custom.db"
    assert s.rsi_window == 21
