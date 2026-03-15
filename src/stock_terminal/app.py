from __future__ import annotations

import json
import re
from typing import Any, Callable

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    LoadingIndicator,
    OptionList,
    RichLog,
    Static,
)
from textual_plotext import PlotextPlot

from . import log
from .constants import CALCULATIONS, INTERVAL_UNITS, PERIOD_UNITS
from .models import Dashboard, Stock
from .plotting import HoverPlotextPlot, plot_time_series


LOGGER = RichLog()


class SelectorWidget(OptionList):
    def __init__(self, *args, change_value: Callable[[str], None], **kwargs):
        self.change_value = change_value
        super().__init__(*args, **kwargs)

    def on_option_list_option_selected(self, event) -> None:
        LOGGER.write(event)
        self.change_value(event.option.prompt)
        self.refresh()


class StockInputBar(Input):
    BINDINGS = [
        ("enter", "input_submitted", "Add"),
        ("escape", "escape", "Cancel"),
    ]

    def __init__(self, add_stock: Callable[[str], None]):
        super().__init__(placeholder="Enter stock symbol")
        self.add_stock = add_stock

    def action_input_submitted(self) -> None:
        try:
            self.add_stock(self.value)
            LOGGER.write(f"Adding stock {self.value}")
        except Exception as e:
            LOGGER.write(f"Adding stock {self.value} FAILED: {e}")
        self.clear()
        self.remove()

    def action_escape(self) -> None:
        self.clear()
        self.remove()


class DashboardWidget(Widget):
    def __init__(self, dashboard: Dashboard):
        super().__init__()
        self.dashboard = dashboard
        self.data_table = DataTable(cursor_type="cell", zebra_stripes=True)
        self.graph_plot: PlotextPlot | None = None
        self.selected_stock: Stock | None = None
        self.selected_option: int | None = None
        self.selected_stock_index: int | None = None
        self._pending_inline_refresh = False

        self.period_options = SelectorWidget(
            *PERIOD_UNITS, change_value=self.__select_stock_period, classes="selector"
        )
        self.period_options.display = "none"
        self.interval_options = SelectorWidget(
            *INTERVAL_UNITS, change_value=self.__select_stock_interval
        )
        self.interval_options.display = "none"

    def __show_options(self, options: SelectorWidget, index: int, value: Any) -> None:
        options.display = "block"
        self.screen.set_focus(options)

    def __hide_options(self, options: SelectorWidget) -> None:
        options.display = "none"
        self.screen.set_focus(self.data_table)
        if self.selected_stock_index is not None and self.selected_option is not None:
            self.data_table.move_cursor(
                row=self.selected_stock_index, column=self.selected_option
            )

    def __select_stock_period(self, period: str) -> None:
        if self.selected_stock is None:
            return
        self.selected_stock.graph_period = period
        if self.selected_stock_index is not None and self.selected_option is not None:
            self.data_table.update_cell_at(
                (self.selected_stock_index, self.selected_option), period
            )
        self.__hide_options(self.period_options)
        self._schedule_inline_graph_refresh()

    def __select_stock_interval(self, interval: str) -> None:
        if self.selected_stock is None:
            return
        self.selected_stock.graph_interval = interval
        if self.selected_stock_index is not None and self.selected_option is not None:
            self.data_table.update_cell_at(
                (self.selected_stock_index, self.selected_option), interval
            )
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
        if self.dashboard.stocks:
            self.selected_stock = self.dashboard.stocks[0]
            self.selected_stock_index = 0
            self._schedule_inline_graph_refresh()

    def create_table(self) -> None:
        self.data_table.add_columns(
            "Symbol",
            "Price (USD)",
            "Change (%)",
            "Active Graph",
            "Period",
            "Interval",
            *[calc for calc in CALCULATIONS],
        )
        for stock in self.dashboard.stocks:
            color = "green" if (stock.change_percent or 0) >= 0 else "red"
            period_string_regex = re.search(r"\] (.*?) \[", str(stock.graph_period))
            interval_string_regex = re.search(r"\] (.*?) \[", str(stock.graph_interval))
            self.data_table.add_row(
                f"[bold]{stock.symbol}[/bold]",
                f"[bold]{stock.current_value:.2f}[/bold]",
                f"[{color}]{stock.change_percent:.2f}%[/{color}]",
                "✔" if stock.active_graph else "✖",
                period_string_regex if period_string_regex else stock.graph_period,
                interval_string_regex if interval_string_regex else stock.graph_interval,
                *["✔" if calc in stock.calculations else "✖" for calc in CALCULATIONS],
            )

    def add_stock(self, stock: Stock) -> None:
        self.dashboard.add_stock(stock)
        self.refresh_dashboard()

    def remove_stock(self) -> None:
        if self.selected_stock is None:
            return
        stock = [i for i in self.dashboard.stocks if i.symbol == self.selected_stock.symbol][0]
        self.dashboard.remove_stock(stock)
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        LOGGER.write("Refreshing dashboard...")
        self.dashboard.refresh()
        self.data_table.clear(columns=True)
        self.create_table()
        self.refresh()
        LOGGER.write("Dashboard refreshed.")

    def on_data_table_cell_highlighted(self, event) -> None:
        self.selected_stock = self.dashboard.stocks[event.coordinate[0]]
        self.selected_option = event.coordinate[1]
        self.selected_stock_index = event.coordinate[0]
        self._schedule_inline_graph_refresh()

    def on_data_table_cell_selected(self, event) -> None:
        self.selected_stock = self.dashboard.stocks[event.coordinate[0]]
        self.selected_stock_index = event.coordinate[0]
        self.selected_option = event.coordinate[1]
        LOGGER.write(f"Cell selected: {self.selected_stock.symbol}, {self.selected_option}")

        match event.coordinate[1]:
            case 3:
                self.open_fullscreen_graph()
            case 4:
                self.__show_options(
                    self.period_options,
                    PERIOD_UNITS.index(self.selected_stock.graph_period),
                    event.value,
                )
            case 5:
                self.__show_options(
                    self.interval_options,
                    INTERVAL_UNITS.index(self.selected_stock.graph_interval),
                    event.value,
                )
            case _:
                if event.coordinate[1] > 5:
                    calc = CALCULATIONS[event.coordinate[1] - 6]
                    if calc in self.selected_stock.calculations:
                        self.selected_stock.calculations.remove(calc)
                    else:
                        self.selected_stock.calculations.append(calc)
                    self.data_table.update_cell_at(
                        event.coordinate,
                        "✔" if calc in self.selected_stock.calculations else "✖",
                    )
                    self._schedule_inline_graph_refresh()

    def _schedule_inline_graph_refresh(self) -> None:
        if self._pending_inline_refresh:
            return
        self._pending_inline_refresh = True

        def _do() -> None:
            self._pending_inline_refresh = False
            self.update_inline_graph()

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
                if self.selected_stock_index is not None:
                    self.data_table.update_cell_at((self.selected_stock_index, 3), "✖")
            except Exception:
                pass

        stock.active_graph = True
        try:
            if self.selected_stock_index is not None:
                self.data_table.update_cell_at((self.selected_stock_index, 3), "✔")
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

    def __init__(self, *, dashboard: Dashboard, stock: Stock, on_close: Callable[[], None]):
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

        self._data_len = self.plot.series_len
        if self._data_len:
            self._cursor_index = max(0, min(self._cursor_index, self._data_len - 1))

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


class TuiTicker(App):
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

    def __init__(self, dashboard: Dashboard, *, log_buffer: list[Any] | None = None):
        super().__init__()
        self.dashboard = DashboardWidget(dashboard=dashboard)
        self.stock_input_bar = StockInputBar(add_stock=self.__add_stock)
        self.debug_mode = False
        self.footer = Footer()
        self._log_buffer = log_buffer or []
        LOGGER.display = "none"

    def on_mount(self) -> None:
        log.set_writer(LOGGER.write)
        for item in self._log_buffer:
            try:
                LOGGER.write(item)
            except Exception:
                pass
        self._log_buffer.clear()

    def __add_stock(self, stock_symbol: str) -> None:
        self.dashboard.add_stock(Stock(stock_symbol))
        self.refresh()
        self.__save_data()

    def __save_data(self) -> None:
        stock_symbols = [stock.symbol for stock in self.dashboard.dashboard.stocks]
        with open("data.json", "w") as f:
            json.dump({"stocks": stock_symbols}, f)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield self.dashboard
        yield Header()
        yield LOGGER
        yield self.footer

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_debug_mode(self) -> None:
        self.debug_mode = not self.debug_mode
        LOGGER.display = "block" if self.debug_mode else "none"

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
        self.dashboard.toggle_inline_chart_mode()


def main() -> None:
    dashboard = Dashboard()

    preload_logs: list[Any] = []
    log.set_writer(preload_logs.append)

    try:
        with open("data.json", "r") as f:
            data = json.load(f)
            for stock_symbol in data["stocks"]:
                dashboard.add_stock(Stock(stock_symbol))
    except FileNotFoundError:
        print("No data file found.")
    except json.JSONDecodeError:
        print("Error decoding JSON file.")

    app = TuiTicker(dashboard=dashboard, log_buffer=preload_logs)
    app.run()
