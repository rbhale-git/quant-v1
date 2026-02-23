import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import settings


FEATURE_COLUMNS = [
    "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_pct", "rsi", "macd_histogram", "atr",
    "reg_slope_30", "reg_r2_30", "sentiment_score",
    "volume_change_pct", "sma_20_ratio", "sma_50_ratio", "vwap_ratio",
]


class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            random_state=42, n_jobs=-1,
        )
        self._feature_cols = []

    def prepare_features(
        self, df: pd.DataFrame, forward_days: int = None,
        buy_threshold: float = None, sell_threshold: float = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        forward_days = forward_days or settings.ml_forward_days
        buy_threshold = buy_threshold or settings.ml_buy_threshold
        sell_threshold = sell_threshold or settings.ml_sell_threshold

        df = df.copy()

        # Derived features
        df["volume_change_pct"] = df["volume"].pct_change()
        df["sma_20_ratio"] = df["close"] / df["sma_20"] if "sma_20" in df.columns else 1.0
        df["sma_50_ratio"] = df["close"] / df["sma_50"] if "sma_50" in df.columns else 1.0
        df["vwap_ratio"] = df["close"] / df["vwap"] if "vwap" in df.columns else 1.0

        # Target: forward return
        df["forward_return"] = df["close"].shift(-forward_days) / df["close"] - 1

        def label(ret):
            if pd.isna(ret):
                return np.nan
            if ret > buy_threshold:
                return "BUY"
            elif ret < sell_threshold:
                return "SELL"
            return "HOLD"

        df["target"] = df["forward_return"].apply(label)

        # Select available feature columns
        self._feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

        # Drop rows with NaN in features or target
        subset = df[self._feature_cols + ["target"]].dropna()
        X = subset[self._feature_cols]
        y = subset["target"]
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._feature_cols = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False,
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

    def predict(self, X: pd.DataFrame) -> list[dict]:
        probas = self.model.predict_proba(X)
        preds = self.model.predict(X)
        results = []
        for pred, proba in zip(preds, probas):
            confidence = float(max(proba))
            results.append({"signal": pred, "confidence": confidence})
        return results

    def feature_importances(self) -> dict:
        importances = self.model.feature_importances_
        return dict(zip(self._feature_cols, [float(x) for x in importances]))

    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_cols": self._feature_cols}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
