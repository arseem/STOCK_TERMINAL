from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd
from textual import events
from textual_plotext import PlotextPlot

from .constants import CALCULATIONS_TYPE, CHART_MODE_TYPE, INTERVAL_UNITS_TYPE
from .indicators import CALCULATIONS_DICT
from .models import InvalidGraphConfigurationException


def plot_time_series(
    data: pd.DataFrame,
    title: str,
    interval: INTERVAL_UNITS_TYPE,
    calculations_TYPE: Tuple[CALCULATIONS_TYPE, ...] = (),
    *,
    plot: PlotextPlot,
    cursor_index: int | None = None,
    mode: CHART_MODE_TYPE = "candles",
) -> None:
    """Render a chart inside the terminal (Textual + plotext)."""

    df = data.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        raise InvalidGraphConfigurationException(
            "No data to plot. Please check the period and interval."
        )

    # Keep charts readable in typical terminal sizes.
    plot_width = plot.size.width or 80
    plot_height = plot.size.height or 24
    if mode == "dots":
        # Dots can be denser (braille markers), scale with width.
        max_points = plot_width * 2
    else:
        max_points = 200
    if len(df) > max_points:
        df = df.iloc[-max_points:]

    # Use 0..N-1 for both modes so cursor index maps consistently.
    x = list(range(len(df)))

    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    tick_label_format = "%m-%d %H:%M" if interval in intraday_intervals else "%Y-%m-%d"
    full_label_format = "%Y-%m-%d %H:%M" if interval in intraday_intervals else "%Y-%m-%d"
    date_labels_ticks = [
        d.strftime(tick_label_format) if hasattr(d, "strftime") else str(d)
        for d in df.index
    ]
    date_labels_full = [
        d.strftime(full_label_format) if hasattr(d, "strftime") else str(d)
        for d in df.index
    ]
    ohlc = {
        "Open": df["Open"].tolist(),
        "Close": df["Close"].tolist(),
        "High": df["High"].tolist(),
        "Low": df["Low"].tolist(),
    }

    plt = plot.plt
    plt.clear_figure()
    plt.title(title)
    plt.grid(True, True)

    # X ticks (shared).
    target_labels = max(6, min(18, plot_width // 8))
    label_step = max(1, len(x) // target_labels) if x else 1
    xticks = x[::label_step] if x else []
    xlabels = date_labels_ticks[::label_step] if x else []
    if xticks:
        plt.xticks(xticks, xlabels)
    plt.xlabel("Time")
    plt.ylabel("Price")

    # More y-ticks makes the scale feel more precise, within terminal limits.
    plt.yfrequency(max(7, min(14, plot_height // 2)))

    if mode == "dots":
        # Single series with a contrasting color.
        # Using plot() (not scatter) keeps the braille rendering dense.
        y = df["Close"].tolist()
        plt.plot(
            x,
            y,
            label="Price",
            marker="braille",
            color="red" if df["Close"].iloc[0] > df["Close"].iloc[-1] else "green",
        )
    else:
        plt.candlestick(x, ohlc, colors=["green", "red"], label="Price")

    # Optional cursor (used in fullscreen mode with arrow navigation).
    if cursor_index is not None and x:
        cursor_index = max(0, min(cursor_index, len(x) - 1))
        plt.vline(x[cursor_index], color="cyan")

    # Optional overlays.
    for calculation in calculations_TYPE:
        if calculation not in CALCULATIONS_DICT:
            continue

        result = CALCULATIONS_DICT[calculation](df, 14)
        if isinstance(result, tuple) and len(result) == 2:
            upper, lower = result
            if mode == "dots":
                plt.plot(
                    x,
                    upper.tolist(),
                    label=f"{calculation} Upper",
                    marker="dot",
                    color="red",
                )
                plt.plot(
                    x,
                    lower.tolist(),
                    label=f"{calculation} Lower",
                    marker="dot",
                    color="red",
                )
            else:
                plt.plot(x, upper.tolist(), label=f"{calculation} Upper")
                plt.plot(x, lower.tolist(), label=f"{calculation} Lower")
        else:
            if mode == "dots":
                plt.plot(x, result.tolist(), label=calculation, marker="dot")
            else:
                plt.plot(x, result.tolist(), label=calculation)

    # If the plot widget supports hover, provide it with the series.
    if hasattr(plot, "set_hover_series"):
        try:
            plot.set_hover_series(
                x=x,
                labels=date_labels_full,
                open=df["Open"].tolist(),
                high=df["High"].tolist(),
                low=df["Low"].tolist(),
                close=df["Close"].tolist(),
                volume=(df["Volume"].tolist() if "Volume" in df.columns else []),
            )
        except Exception:
            pass

    plot.refresh()


class HoverPlotextPlot(PlotextPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._x: list[int] = []
        self._labels: list[str] = []
        self._open: list[float] = []
        self._high: list[float] = []
        self._low: list[float] = []
        self._close: list[float] = []
        self._volume: list[float] = []
        self._last_index: int | None = None
        self._on_hover_index: Callable[[int], None] | None = None

    def set_hover_series(
        self,
        *,
        x: list[int],
        labels: list[str],
        open: list[float],
        high: list[float],
        low: list[float],
        close: list[float],
        volume: list[float],
    ) -> None:
        self._x = x
        self._labels = labels
        self._open = open
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume

    @property
    def series_len(self) -> int:
        return len(self._x)

    def format_tooltip(self, index: int) -> str:
        index = max(0, min(index, len(self._x) - 1))
        label = self._labels[index] if index < len(self._labels) else str(index)
        o = self._open[index] if index < len(self._open) else float("nan")
        h = self._high[index] if index < len(self._high) else float("nan")
        l = self._low[index] if index < len(self._low) else float("nan")
        c = self._close[index] if index < len(self._close) else float("nan")
        v = self._volume[index] if index < len(self._volume) else None
        vol = f"\nV: {v:,.0f}" if v is not None else ""
        return f"{label}\nO: {o:.2f}  H: {h:.2f}\nL: {l:.2f}  C: {c:.2f}{vol}"

    def set_hover_callback(self, callback: Callable[[int], None] | None) -> None:
        self._on_hover_index = callback

    def _emit_hover(self, index: int) -> None:
        if self._on_hover_index is None:
            return
        try:
            self._on_hover_index(index)
        except Exception:
            pass

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._x or not self.size.width:
            return

        # Map mouse x-coordinate to the nearest data index.
        left_margin = 8
        width = max(1, self.size.width - left_margin)
        x_pos = max(0, min(event.x - left_margin, width - 1))
        index = int(round((x_pos / max(1, width - 1)) * (len(self._x) - 1)))
        index = max(0, min(index, len(self._x) - 1))

        self._last_index = index
        self._emit_hover(index)
