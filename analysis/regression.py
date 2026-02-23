import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_regression(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.copy()
    slopes = []
    r2s = []
    predictions = []

    for i in range(len(df)):
        if i < window - 1:
            slopes.append(np.nan)
            r2s.append(np.nan)
            predictions.append(np.nan)
            continue
        segment = df["close"].iloc[i - window + 1: i + 1].values
        X = np.arange(window).reshape(-1, 1)
        y = segment
        model = LinearRegression()
        model.fit(X, y)
        slopes.append(model.coef_[0])
        r2s.append(model.score(X, y))
        next_X = np.array([[window]])
        predictions.append(model.predict(next_X)[0])

    df[f"reg_slope_{window}"] = slopes
    df[f"reg_r2_{window}"] = r2s
    df[f"reg_predicted_{window}"] = predictions
    return df
