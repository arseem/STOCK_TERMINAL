from __future__ import annotations

import yfinance as yf

from . import log
from .constants import (
    INTERVAL_UNITS,
    INTERVAL_UNITS_TYPE,
    PERIOD_UNITS,
    PERIOD_UNITS_TYPE,
    CHART_MODE_TYPE,
)


# EXCEPTIONS ########################################


class PeriodNotSupportedException(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message or f"Period must be one of {PERIOD_UNITS}")


class IntervalNotSupportedException(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message or f"Interval must be one of {INTERVAL_UNITS}")


class StockNotFoundException(Exception):
    def __init__(self, symbol: str):
        super().__init__(f"Stock with symbol '{symbol}' not found.")


class StockDataFetchException(Exception):
    def __init__(self, message: str = "Failed to fetch stock data."):
        super().__init__(message)


class InvalidGraphConfigurationException(Exception):
    def __init__(self, message: str = "Invalid graph configuration."):
        super().__init__(message)


# MODELS ############################################


class Stock:
    def __init__(self, symbol: str):
        self.symbol = symbol
        try:
            self.ticker = yf.Ticker(symbol)
            if self.ticker.history(period="5d").empty:
                raise StockNotFoundException(symbol)
        except Exception:
            raise StockNotFoundException(symbol)

        self.last_refreshed = None
        self.current_value = None
        self.change_percent = None
        self.current_data = None

        self.graph_interval: INTERVAL_UNITS_TYPE = "5m"
        self.graph_period: PERIOD_UNITS_TYPE = "1d"

        # Default UX: dots inline (table view), candles in fullscreen.
        self.inline_chart_mode: CHART_MODE_TYPE = "dots"
        self.fullscreen_chart_mode: CHART_MODE_TYPE = "candles"

        self.active_graph = False
        self.active_graph_thread = None
        self.calculations: list[str] = []
        self.refresh()

    def __str__(self) -> str:
        return (
            f"{self.symbol} ({self.last_refreshed}): {self.current_value} "
            f"({round(self.change_percent, 2)}%)"
        )

    def refresh(self) -> None:
        data = self.ticker.history()
        last_quote = data["Close"].iloc[-1]
        pre_last_quote = data["Close"].iloc[-2]
        self.current_value = last_quote
        self.change_percent = (last_quote - pre_last_quote) / pre_last_quote * 100
        self.last_refreshed = data.index[-1]
        log.write(
            f"{self.symbol} ({self.last_refreshed}): {self.current_value} ({self.change_percent})"
        )

    def get_data(self, period: PERIOD_UNITS_TYPE, interval: INTERVAL_UNITS_TYPE):
        try:
            if period not in PERIOD_UNITS:
                raise PeriodNotSupportedException()
            if interval not in INTERVAL_UNITS:
                raise IntervalNotSupportedException()

            try:
                self.current_data = self.ticker.history(period=period, interval=interval)
            except Exception as e:
                raise StockDataFetchException(str(e))

        except (PeriodNotSupportedException, IntervalNotSupportedException, StockDataFetchException) as e:
            log.write(f"Error fetching data: {e}")

        return self.current_data


class Dashboard:
    def __init__(self):
        self.stocks: list[Stock] = []

    def add_stock(self, stock: Stock) -> None:
        self.stocks.append(stock)

    def remove_stock(self, stock: Stock) -> None:
        self.stocks.remove(stock)

    def refresh(self) -> None:
        for stock in self.stocks:
            stock.refresh()

    def get_data(
        self,
        period: PERIOD_UNITS_TYPE,
        interval: INTERVAL_UNITS_TYPE,
        symbol: str | None = None,
    ):
        data = {}
        for stock in self.stocks:
            if not symbol:
                data[stock.symbol] = stock.get_data(period, interval)
            elif stock.symbol == symbol:
                data[stock.symbol] = stock.get_data(period, interval)
                break
        return data
