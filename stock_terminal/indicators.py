from __future__ import annotations

import pandas as pd


def calculate_moving_average(data: pd.DataFrame, window: int):
    return data["Close"].rolling(window=window).mean()


def calculate_exponential_moving_average(data: pd.DataFrame, window: int):
    return data["Close"].ewm(span=window, adjust=False).mean()


def calculate_bollinger_bands(data: pd.DataFrame, window: int):
    sma = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower


CALCULATIONS_DICT = {
    "MAVG": calculate_moving_average,
    "EMA": calculate_exponential_moving_average,
    "BOLLINGER": calculate_bollinger_bands,
}
