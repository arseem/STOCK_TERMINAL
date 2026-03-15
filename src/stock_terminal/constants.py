from __future__ import annotations

from typing import Literal

# Period/interval options
PERIOD_UNITS = [
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]
PERIOD_UNITS_TYPE = Literal[
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
]

INTERVAL_UNITS = [
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
]
INTERVAL_UNITS_TYPE = Literal[
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
]

# Indicators
CALCULATIONS = ["MAVG", "EMA", "BOLLINGER"]
CALCULATIONS_TYPE = Literal["MAVG", "EMA", "BOLLINGER"]

# Chart modes
CHART_MODES = ["candles", "dots"]
CHART_MODE_TYPE = Literal["candles", "dots"]
