from tkinter import TclError
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, DataTable, OptionList, RichLog, LoadingIndicator
from textual.widget import Widget
from textual.app import App, ComposeResult
import re
from typing import Tuple, Literal
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import json
import matplotlib
import tkinter as tk
matplotlib.rcParams['toolbar'] = 'None'


# CONSTANTS ########################################

PERIOD_UNITS = ['1d', '5d', '1mo', '3mo', '6mo',
                '1y', '2y', '5y', '10y', 'ytd', 'max']
PERIOD_UNITS_TYPE = Literal['1d', '5d', '1mo', '3mo', '6mo',
                            '1y', '2y', '5y', '10y', 'ytd', 'max']

INTERVAL_UNITS = ['1m', '2m', '5m', '15m', '30m',
                  '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']
INTERVAL_UNITS_TYPE = Literal['1m', '2m', '5m', '15m', '30m',
                              '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']

CALCULATIONS = ['MAVG', 'EMA', 'RSI', 'MACD', 'BOLLINGER', 'STOCHASTIC']
CALCULATIONS_TYPE = Literal['MAVG', 'EMA',
                            'RSI', 'MACD', 'BOLLINGER', 'STOCHASTIC']


# FUNCTIONS ########################################

def calculate_moving_average(data, window: int):
    return data['Close'].rolling(window=window).mean()


def calculate_exponential_moving_average(data, window: int):
    return data['Close'].ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window: int):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(data, window: int):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower


def calculate_stochastic_oscillator(data, window: int):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d


CALCULATIONS_DICT = {
    'MAVG': calculate_moving_average,
    'EMA': calculate_exponential_moving_average,
    'RSI': calculate_rsi,
    'BOLLINGER': calculate_bollinger_bands,
    'STOCHASTIC': calculate_stochastic_oscillator
}


def plot_time_series(data, title: str, interval: INTERVAL_UNITS_TYPE, calculations_TYPE: Tuple[CALCULATIONS_TYPE] = (), fig: Figure = plt.figure(figsize=(16, 8))):
    ax = fig.gca()
    ax.clear()
    up = data['Close'] > data['Open']
    down = data['Open'] > data['Close']

    col1 = 'green'
    col2 = 'red'

    widthBox = pd.Timedelta(interval) * 0.5
    widthTails = 0.5

    LOGGER.write(data)

    ax.vlines(data.index, data['Low'], data['High'],
              color='black', linewidth=widthTails, zorder=0)
    ax.bar(data.index[up], data['Close'][up] - data['Open'][up],
           width=widthBox, bottom=data['Open'][up], color=col1, label='Up', zorder=1)
    ax.bar(data.index[down], data['Open'][down] - data['Close'][down],
           width=widthBox, bottom=data['Close'][down], color=col2, label='Down', zorder=1)

    print(calculations_TYPE)
    for calculation in calculations_TYPE:
        if calculation in CALCULATIONS_DICT:
            result = CALCULATIONS_DICT[calculation](data, 14)
            if len(result) == 2:
                ax.plot(data.index, result[0],
                        label=f'{calculation} Upper')
                ax.plot(data.index, result[1],
                        label=f'{calculation} Lower')
            else:
                ax.plot(data.index, result, label=calculation, alpha=0.5)
        else:
            print(f'Calculation {calculation} not found.')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price (USD)', fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.yticks(fontsize=10)

    fig.autofmt_xdate()

    ax.legend()

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


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
        self.active_graph = None
        self.active_graph_thread = None
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
        if period not in PERIOD_UNITS:
            raise PeriodNotSupportedException()
        if interval not in INTERVAL_UNITS:
            raise IntervalNotSupportedException()

        try:
            self.current_data = self.ticker.history(
                period=period, interval=interval)
        except Exception as e:
            raise StockDataFetchException(str(e))

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
        self.add_stock(self.value)
        LOGGER.write(f'Adding stock {self.value}')
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
        self.selected_stock = None
        self.selected_option = None
        self.selected_stock_index = None

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
        if self.selected_stock.active_graph:
            self.graph(no_close=True)

    def __select_stock_interval(self, interval):
        self.selected_stock.graph_interval = interval
        self.data_table.update_cell_at(
            (self.selected_stock_index, self.selected_option), interval)
        self.__hide_options(self.interval_options)
        if self.selected_stock.active_graph:
            self.graph(no_close=True)

    def compose(self) -> ComposeResult:
        self.create_table()
        yield self.data_table
        yield self.period_options
        yield self.interval_options

    def create_table(self):
        self.data_table.add_columns(
            "Symbol", "Price (USD)", "Change (%)", "Active Graph", "Period", "Interval")
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

    def on_data_table_cell_selected(self, event):
        self.selected_stock = self.dashboard.stocks[event.coordinate[0]]
        self.selected_stock_index = event.coordinate[0]
        self.selected_option = event.coordinate[1]
        LOGGER.write(
            f'Cell selected: {self.selected_stock.symbol}, {self.selected_option}')
        match event.coordinate[1]:
            case 3:
                self.graph()
                self.data_table.update_cell_at(
                    event.coordinate, '✔' if self.selected_stock.active_graph else '✖')
            case 4:
                self.__show_options(self.period_options, PERIOD_UNITS.index(
                    self.selected_stock.graph_period), event.value)

            case 5:
                self.__show_options(self.interval_options, INTERVAL_UNITS.index(
                    self.selected_stock.graph_interval), event.value)

    def graph(self, no_close=False):
        try:
            if self.selected_stock.active_graph and not no_close:
                try:
                    plt.close(self.selected_stock.active_graph)

                except TclError as e:
                    LOGGER.write(f'Error closing graph: {e}')

                self.selected_stock.active_graph = None

            else:
                interval = self.selected_stock.graph_interval
                period = self.selected_stock.graph_period
                symbol = self.selected_stock.symbol
                data = self.dashboard.get_data(
                    period, interval, symbol=symbol)[symbol]
                if data.empty:
                    raise InvalidGraphConfigurationException(
                        "No data to plot. Please check the period and interval.")

                if not no_close:
                    self.selected_stock.active_graph = plt.figure(
                        figsize=(16, 8))
                plot_time_series(
                    data, f'{symbol} Price', interval, ('MAVG', ), self.selected_stock.active_graph)
        except InvalidGraphConfigurationException as e:
            LOGGER.write(str(e))
        except Exception as e:
            LOGGER.write(f'Unexpected error: {e}')


class StockApp(App):
    BINDINGS = [
        ("a", "add_stock", "Add stock"),
        ("f5", "refresh", "Refresh data"),
        ("R", "remove_stock", "Remove stock"),
        ("d", "toggle_dark", "Toggle dark mode"),
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
    }

    DataTable {
        layer: below;
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
        yield Header()
        yield LOGGER
        yield self.footer
        with Horizontal():
            yield self.dashboard

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

    def action_quit(self):
        return super().action_quit()


def main():
    plt.ion()
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
