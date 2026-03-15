from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, DataTable, OptionList, RichLog, LoadingIndicator, Static
from textual.widget import Widget
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual import events
import re
from typing import Callable, Tuple, Literal
import yfinance as yf
import pandas as pd
import json
from textual_plotext import PlotextPlot


# CONSTANTS ########################################

PERIOD_UNITS = ['1d', '5d', '1mo', '3mo', '6mo',
                '1y', '2y', '5y', '10y', 'ytd', 'max']
PERIOD_UNITS_TYPE = Literal['1d', '5d', '1mo', '3mo', '6mo',
                            '1y', '2y', '5y', '10y', 'ytd', 'max']

INTERVAL_UNITS = ['1m', '2m', '5m', '15m', '30m',
                  '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']
INTERVAL_UNITS_TYPE = Literal['1m', '2m', '5m', '15m', '30m',
                              '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']

CALCULATIONS = ['MAVG', 'EMA', 'BOLLINGER']
CALCULATIONS_TYPE = Literal['MAVG', 'EMA',
                            'BOLLINGER']

CHART_MODES = ["candles", "dots"]
CHART_MODE_TYPE = Literal["candles", "dots"]


# FUNCTIONS ########################################

def calculate_moving_average(data, window: int):
    return data['Close'].rolling(window=window).mean()


def calculate_exponential_moving_average(data, window: int):
    return data['Close'].ewm(span=window, adjust=False).mean()


def calculate_bollinger_bands(data, window: int):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower


CALCULATIONS_DICT = {
    'MAVG': calculate_moving_average,
    'EMA': calculate_exponential_moving_average,
    'BOLLINGER': calculate_bollinger_bands,
}


def plot_time_series(
    data: pd.DataFrame,
    title: str,
    interval: INTERVAL_UNITS_TYPE,
    calculations_TYPE: Tuple[CALCULATIONS_TYPE, ...] = (),
    plot: PlotextPlot | None = None,
    cursor_index: int | None = None,
    mode: CHART_MODE_TYPE = "candles",
):
    """Render a candlestick chart inside the terminal (Textual + plotext).

    Note: The `interval` argument is kept for API compatibility with the
    rest of the app, but sizing is handled automatically by `PlotextPlot`.
    """

    if plot is None:
        raise ValueError("plot must be provided (PlotextPlot)")

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

    # More y-ticks makes the scale feel more precise, within the limits of terminal resolution.
    plt.yfrequency(max(7, min(14, plot_height // 2)))

    if mode == "dots":
        # Single series with a contrasting color.
        # Using plot() (not scatter) keeps the braille rendering dense and avoids visual gaps.
        y = df["Close"].tolist()
        plt.plot(x, y, label="Price", marker="braille", color="red" if df["Close"].iloc[0] > df["Close"].iloc[-1] else "green")
    else:
        # Candlestick chart (built-in to plotext).
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
                plt.plot(x, upper.tolist(), label=f"{calculation} Upper", marker="dot", color="red")
                plt.plot(x, lower.tolist(), label=f"{calculation} Lower", marker="dot", color="red")
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
    """PlotextPlot with a dynamic tooltip showing nearest candle value."""

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
        """Set a callback called with the nearest hovered index."""
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
        # Plotext has a left margin for y-axis labels; approximate it.
        left_margin = 8
        width = max(1, self.size.width - left_margin)
        x_pos = max(0, min(event.x - left_margin, width - 1))
        index = int(round((x_pos / max(1, width - 1)) * (len(self._x) - 1)))
        index = max(0, min(index, len(self._x) - 1))

        self._last_index = index
        self._emit_hover(index)


# EXCEPTIONS ########################################

class PeriodNotSupportedException(Exception):
    def __init__(self, message=f"Period must be one of {PERIOD_UNITS}"):
        self.message = message
        super().__init__(self.message)


class IntervalNotSupportedException(Exception):
    def __init__(self, message=f"Interval must be one of {INTERVAL_UNITS}"):
        self.message = message
        super().__init__(self.message)


class StockNotFoundException(Exception):
    def __init__(self, symbol: str):
        super().__init__(f"Stock with symbol '{symbol}' not found.")


class StockDataFetchException(Exception):
    def __init__(self, message="Failed to fetch stock data."):
        super().__init__(message)


class InvalidGraphConfigurationException(Exception):
    def __init__(self, message="Invalid graph configuration."):
        super().__init__(message)


# CLASSES ########################################

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
        self.graph_interval = '5m'
        self.graph_period = '1d'
        # Default UX: dots inline (table view), candles in fullscreen.
        self.inline_chart_mode: CHART_MODE_TYPE = "dots"
        self.fullscreen_chart_mode: CHART_MODE_TYPE = "candles"
        self.active_graph = False
        self.active_graph_thread = None
        self.calculations = []
        self.refresh()

    def __str__(self):
        return f'{self.symbol} ({self.last_refreshed}): {self.current_value} ({round(self.change_percent, 2)}%)'

    def refresh(self):
        data = self.ticker.history()
        last_quote = data['Close'].iloc[-1]
        pre_last_quote = data['Close'].iloc[-2]
        self.current_value = last_quote
        self.change_percent = (
            last_quote - pre_last_quote) / pre_last_quote * 100
        self.last_refreshed = data.index[-1]
        LOGGER.write(
            f'{self.symbol} ({self.last_refreshed}): {self.current_value} ({self.change_percent})')

    def get_data(self, period: PERIOD_UNITS_TYPE, interval: INTERVAL_UNITS_TYPE):
        try:
            if period not in PERIOD_UNITS:
                raise PeriodNotSupportedException()
            if interval not in INTERVAL_UNITS:
                raise IntervalNotSupportedException()

            try:
                self.current_data = self.ticker.history(
                    period=period, interval=interval)
            except Exception as e:
                raise StockDataFetchException(str(e))

        except PeriodNotSupportedException as e:
            LOGGER.write(F"Error fetching data: {e}")
        except IntervalNotSupportedException as e:
            LOGGER.write(F"Error fetching data: {e}")
        except StockDataFetchException as e:
            LOGGER.write(F"Error fetching data: {e}")

        return self.current_data


class Dashboard:
    def __init__(self):
        self.stocks = []

    def add_stock(self, stock: Stock):
        self.stocks.append(stock)

    def remove_stock(self, stock: Stock):
        self.stocks.remove(stock)

    def refresh(self):
        for stock in self.stocks:
            stock.refresh()

    def get_data(self, period: PERIOD_UNITS_TYPE, interval: INTERVAL_UNITS_TYPE, symbol: str | None = None):
        data = {}
        for stock in self.stocks:
            if not symbol:
                data[stock.symbol] = stock.get_data(
                    period, interval)
            elif stock.symbol == symbol:
                data[stock.symbol] = stock.get_data(
                    period, interval)
                break
        return data


###############################################
###############################################
#####                 #########################
#####   TEXTUAL GUI   #########################
#####                 #########################
###############################################
###############################################


LOGGER = RichLog()


class SelectorWidget(OptionList):
    def __init__(self, *args, change_value, **kwargs):
        self.change_value = change_value
        super().__init__(*args, **kwargs)

    def on_option_list_option_selected(self, event):
        LOGGER.write(event)
        self.change_value(event.option.prompt)
        self.refresh()


class StockInputBar(Input):

    BINDINGS = [
        ("enter", "input_submitted", "Add"),
        ("escape", "escape", "Cancel"),
    ]

    def __init__(self, add_stock: callable):
        super().__init__(placeholder="Enter stock symbol")
        self.add_stock = add_stock

    def action_input_submitted(self):
        try:
            self.add_stock(self.value)
            LOGGER.write(f'Adding stock {self.value}')
        except Exception as e:
            LOGGER.write(f'Adding stock {self.value} FAILED: {e}')
        self.clear()
        self.remove()

    def action_escape(self):
        self.clear()
        self.remove()


class DashboardWidget(Widget):

    def __init__(self, dashboard: Dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.data_table = DataTable(cursor_type='cell', zebra_stripes=True)
        self.graph_plot: PlotextPlot | None = None
        self.selected_stock = None
        self.selected_option = None
        self.selected_stock_index = None
        self._pending_inline_refresh = False

        self.period_options = SelectorWidget(
            *PERIOD_UNITS, change_value=self.__select_stock_period, classes='selector')
        self.period_options.display = 'none'
        self.interval_options = SelectorWidget(
            *INTERVAL_UNITS, change_value=self.__select_stock_interval)
        self.interval_options.display = 'none'

    def __show_options(self, options, index, value):
        options.display = 'block'
        self.screen.set_focus(options)

    def __hide_options(self, options):
        options.display = 'none'
        self.screen.set_focus(self.data_table)
        self.data_table.move_cursor(
            row=self.selected_stock_index, column=self.selected_option)

    def __select_stock_period(self, period):
        self.selected_stock.graph_period = period
        self.data_table.update_cell_at(
            (self.selected_stock_index, self.selected_option), period)
        self.__hide_options(self.period_options)
        self._schedule_inline_graph_refresh()

    def __select_stock_interval(self, interval):
        self.selected_stock.graph_interval = interval
        self.data_table.update_cell_at(
            (self.selected_stock_index, self.selected_option), interval)
        self.__hide_options(self.interval_options)
        self._schedule_inline_graph_refresh()

    def compose(self) -> ComposeResult:
        self.create_table()
        yield self.data_table
        self.graph_plot = PlotextPlot(classes="graph")
        yield self.graph_plot
        yield self.period_options
        yield self.interval_options

    def on_mount(self) -> None:
        # Render initial inline graph for the first row (if any).
        if self.dashboard.stocks:
            self.selected_stock = self.dashboard.stocks[0]
            self.selected_stock_index = 0
            self._schedule_inline_graph_refresh()

    def create_table(self):
        self.data_table.add_columns(
            "Symbol", "Price (USD)", "Change (%)", "Active Graph", "Period", "Interval", *[calc for calc in CALCULATIONS])
        for stock in self.dashboard.stocks:
            color = "green" if stock.change_percent >= 0 else "red"
            period_string_regex = re.search(r'\] (.*?) \[', stock.graph_period)
            interval_string_regex = re.search(
                r'\] (.*?) \[', stock.graph_interval)
            self.data_table.add_row(
                f"[bold]{stock.symbol}[/bold]",
                f"[bold]{stock.current_value:.2f}[/bold]",
                f"[{color}]{stock.change_percent:.2f}%[/{color}]",
                '✔' if stock.active_graph else '✖',
                period_string_regex if period_string_regex else stock.graph_period,
                interval_string_regex if interval_string_regex else stock.graph_interval,
                *['✔' if calc in stock.calculations else '✖' for calc in CALCULATIONS]
            )

    def add_stock(self, stock: Stock):
        self.dashboard.add_stock(stock)
        self.refresh_dashboard()

    def remove_stock(self):
        stock = [i for i in self.dashboard.stocks if i.symbol ==
                 self.selected_stock.symbol][0]
        self.dashboard.remove_stock(stock)
        self.refresh_dashboard()

    def refresh_dashboard(self):
        LOGGER.write('Refreshing dashboard...')
        self.dashboard.refresh()
        self.data_table.clear(columns=True)
        self.create_table()
        self.refresh()
        LOGGER.write('Dashboard refreshed.')

    def on_data_table_cell_highlighted(self, event):
        self.selected_stock = self.dashboard.stocks[event.coordinate[0]]
        self.selected_option = event.coordinate[1]
        self.selected_stock_index = event.coordinate[0]
        self._schedule_inline_graph_refresh()

    def on_data_table_cell_selected(self, event):
        self.selected_stock = self.dashboard.stocks[event.coordinate[0]]
        self.selected_stock_index = event.coordinate[0]
        self.selected_option = event.coordinate[1]
        LOGGER.write(
            f'Cell selected: {self.selected_stock.symbol}, {self.selected_option}')
        match event.coordinate[1]:
            case 3:
                self.open_fullscreen_graph()
            case 4:
                self.__show_options(self.period_options, PERIOD_UNITS.index(
                    self.selected_stock.graph_period), event.value)

            case 5:
                self.__show_options(self.interval_options, INTERVAL_UNITS.index(
                    self.selected_stock.graph_interval), event.value)

            case _:
                if event.coordinate[1] > 5:
                    calc = CALCULATIONS[event.coordinate[1] - 6]
                    if calc in self.selected_stock.calculations:
                        self.selected_stock.calculations.remove(calc)
                    else:
                        self.selected_stock.calculations.append(calc)
                    self.data_table.update_cell_at(
                        event.coordinate, '✔' if calc in self.selected_stock.calculations else '✖')
                    self._schedule_inline_graph_refresh()

    def _schedule_inline_graph_refresh(self) -> None:
        if self._pending_inline_refresh:
            return
        self._pending_inline_refresh = True

        def _do() -> None:
            self._pending_inline_refresh = False
            self.update_inline_graph()

        # Debounce quick cursor movement.
        self.set_timer(0.15, _do)

    def update_inline_graph(self) -> None:
        try:
            if self.selected_stock is None or self.graph_plot is None:
                return
            interval = self.selected_stock.graph_interval
            period = self.selected_stock.graph_period
            symbol = self.selected_stock.symbol
            data = self.dashboard.get_data(period, interval, symbol=symbol)[symbol]
            plot_time_series(
                data,
                f"{symbol} Price",
                interval,
                tuple(self.selected_stock.calculations),
                plot=self.graph_plot,
                mode=self.selected_stock.inline_chart_mode,
            )
            self.refresh(layout=True)
        except Exception as e:
            LOGGER.write(f"Graph render error: {e}")

    def toggle_inline_chart_mode(self) -> None:
        if self.selected_stock is None:
            return
        self.selected_stock.inline_chart_mode = (
            "dots" if self.selected_stock.inline_chart_mode == "candles" else "candles"
        )
        self._schedule_inline_graph_refresh()

    def open_fullscreen_graph(self) -> None:
        if self.selected_stock is None:
            return
        stock = self.selected_stock

        def _close() -> None:
            stock.active_graph = False
            try:
                self.data_table.update_cell_at((self.selected_stock_index, 3), '✖')
            except Exception:
                pass

        stock.active_graph = True
        try:
            self.data_table.update_cell_at((self.selected_stock_index, 3), '✔')
        except Exception:
            pass

        self.app.push_screen(GraphScreen(dashboard=self.dashboard, stock=stock, on_close=_close))


class GraphScreen(Screen):
    BINDINGS = [
        ("enter", "close", "Close"),
        ("escape", "close", "Close"),
        ("left", "cursor_left", "Cursor left"),
        ("right", "cursor_right", "Cursor right"),
        ("shift+left", "cursor_left_fast", "Cursor left fast"),
        ("shift+right", "cursor_right_fast", "Cursor right fast"),
        ("home", "cursor_home", "Cursor home"),
        ("end", "cursor_end", "Cursor end"),
        ("t", "toggle_chart", "Chart mode"),
    ]

    CSS = """
    GraphScreen {
        layout: vertical;
        padding: 0;
    }

    # Cursor info line (keyboard-friendly tooltip).
    Static#cursor_info {
        height: auto;
        width: 1fr;
        padding: 0 1;
    }

    PlotextPlot {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, *, dashboard: Dashboard, stock: Stock, on_close: callable):
        super().__init__()
        self.dashboard = dashboard
        self.stock = stock
        self._on_close = on_close
        self.plot: HoverPlotextPlot | None = None
        self.cursor_info = Static(id="cursor_info")
        self._cursor_index: int = 0
        self._data_len: int = 0

    def compose(self) -> ComposeResult:
        yield self.cursor_info
        self.plot = HoverPlotextPlot()
        yield self.plot

    def on_mount(self) -> None:
        if self.plot is not None:
            # Update the top ribbon when the mouse hovers over the plot.
            self.plot.set_hover_callback(self._on_plot_hover_index)
        self.render_graph()

    def _on_plot_hover_index(self, index: int) -> None:
        if self.plot is None:
            return
        try:
            tooltip_text = self.plot.format_tooltip(index)
            self.cursor_info.update(tooltip_text.replace("\n", "  "))
        except Exception:
            pass

    def render_graph(self) -> None:
        if self.plot is None:
            return
        interval = self.stock.graph_interval
        period = self.stock.graph_period
        symbol = self.stock.symbol
        data = self.dashboard.get_data(period, interval, symbol=symbol)[symbol]

        # Keep cursor in bounds.
        try:
            self._data_len = len(data.dropna(subset=["Open", "High", "Low", "Close"]))
        except Exception:
            self._data_len = 0
        if self._data_len:
            self._cursor_index = max(0, min(self._cursor_index, self._data_len - 1))

        plot_time_series(
            data,
            f"{symbol} Price",
            interval,
            tuple(self.stock.calculations),
            plot=self.plot,
            cursor_index=self._cursor_index if self._data_len else None,
            mode=self.stock.fullscreen_chart_mode,
        )

        # Align navigation bounds to what we actually rendered (after truncation).
        self._data_len = self.plot.series_len
        if self._data_len:
            self._cursor_index = max(0, min(self._cursor_index, self._data_len - 1))

        # Ensure the tooltip/cursor info updates even when navigating with keys.
        if self._data_len:
            try:
                tooltip_text = self.plot.format_tooltip(self._cursor_index)
                self.cursor_info.update(tooltip_text.replace("\n", "  "))
            except Exception:
                pass
        else:
            self.cursor_info.update("")

    def action_cursor_left(self) -> None:
        if self._data_len:
            self._cursor_index = max(0, self._cursor_index - 1)
            self.render_graph()

    def action_cursor_right(self) -> None:
        if self._data_len:
            self._cursor_index = min(self._data_len - 1, self._cursor_index + 1)
            self.render_graph()

    def action_cursor_left_fast(self) -> None:
        if self._data_len:
            self._cursor_index = max(0, self._cursor_index - 10)
            self.render_graph()

    def action_cursor_right_fast(self) -> None:
        if self._data_len:
            self._cursor_index = min(self._data_len - 1, self._cursor_index + 10)
            self.render_graph()

    def action_cursor_home(self) -> None:
        if self._data_len:
            self._cursor_index = 0
            self.render_graph()

    def action_cursor_end(self) -> None:
        if self._data_len:
            self._cursor_index = self._data_len - 1
            self.render_graph()

    def action_close(self) -> None:
        try:
            self._on_close()
        finally:
            self.app.pop_screen()

    def action_toggle_chart(self) -> None:
        self.stock.fullscreen_chart_mode = (
            "dots" if self.stock.fullscreen_chart_mode == "candles" else "candles"
        )
        self.render_graph()


class StockApp(App):
    BINDINGS = [
        ("a", "add_stock", "Add stock"),
        ("f5", "refresh", "Refresh data"),
        ("R", "remove_stock", "Remove stock"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("t", "toggle_inline_chart", "Chart mode"),
        ("q", "quit", "Save and quit"),
        ("ctrl+d", "debug_mode", "Toggle debug"),
    ]

    CSS = """
    Screen {
        overflow: auto;
        layers: below above;
    }

    DashboardWidget {
        overflow: auto;
        height: 1fr;
        layout: vertical;
        padding: 0;
    }

    DataTable {
        layer: below;
        height: auto;
        max-height: 12;
        margin: 0;
    }

    PlotextPlot.graph {
        height: 1fr;
        layer: below;
        margin: 0;
    }

    SelectorWidget {
        layer: above;
    }

    StockInputBar {
        layer: above;
    }

    RichLog {
        height: 20%;
    }
    """

    def __init__(self, dashboard: Dashboard):
        super().__init__()
        self.dashboard = DashboardWidget(dashboard=dashboard)
        self.stock_input_bar = StockInputBar(
            add_stock=self.__add_stock)
        self.debug_mode = False
        self.footer = Footer()
        LOGGER.display = 'none'

    def __add_stock(self, stock_symbol: str):
        self.dashboard.add_stock(Stock(stock_symbol))
        self.refresh()
        self.__save_data()

    def __save_data(self):
        stock_symbols = [
            stock.symbol for stock in self.dashboard.dashboard.stocks]
        with open('data.json', 'w') as f:
            json.dump({"stocks": stock_symbols}, f)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield self.dashboard
        yield Header()
        yield LOGGER
        yield self.footer

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_debug_mode(self):
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            LOGGER.display = 'block'
        else:
            LOGGER.display = 'none'

    def action_refresh(self) -> None:
        loading = LoadingIndicator()
        self.mount(loading)
        self.dashboard.refresh_dashboard()
        loading.remove()
        self.refresh()

    def action_add_stock(self) -> None:
        self.dashboard.mount(self.stock_input_bar)
        self.screen.set_focus(self.stock_input_bar)

    def action_remove_stock(self) -> None:
        self.dashboard.remove_stock()
        self.__save_data()

    def action_toggle_inline_chart(self) -> None:
        # Toggle the inline chart mode for the currently highlighted stock.
        self.dashboard.toggle_inline_chart_mode()


def main():
    dashboard = Dashboard()

    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
            for stock_symbol in data['stocks']:
                dashboard.add_stock(Stock(stock_symbol))
    except FileNotFoundError:
        print('No data file found.')
    except json.JSONDecodeError:
        print('Error decoding JSON file.')

    app = StockApp(dashboard=dashboard)
    app.run()


if __name__ == '__main__':
    main()
