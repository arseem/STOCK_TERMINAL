# 📈 tui-ticker - Terminal Stock Tracker

A lightweight, asynchronous Text User Interface (TUI) for real-time stock monitoring and financial indicator visualization directly in the terminal. Built for developers who want to track markets without leaving their command-line workflow.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Framework:** [Textual](https://github.com/Textualize/textual) (Asynchronous TUI framework)
- **Features:** Real-time data fetching, dynamic time-series graphing, and applied financial indicators.

## 🚀 Features & Indicators

- **Real-Time Tracking:** Add, remove, and monitor stock symbols instantly.
- **Dynamic Charting:** Interactive terminal-based graphs with adjustable periods and intervals.
- **Technical Analysis:** Toggle overlays for:
  - **MAVG** (Moving Average)
  - **EMA** (Exponential Moving Average)
  - **Bollinger Bands**

## 💻 Installation & Usage

The easiest way to install `tui-ticker` globally is using [pipx](https://pipx.pypa.io/):

```bash
pipx install git+https://github.com/arseem/tui-ticker.git
```

Or, if you want to run it from source yourself:

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the application:**

```bash
textual run main.py
```

## ⌨️ Keybindings & Controls

- `a` - Add a new stock symbol
- `R` - Remove the selected symbol (row under cursor)
- `F5` - Refresh market data
- `q` - Save current portfolio and quit
- `Ctrl+D` - Toggle debug mode (log console)
- `t` - Toggle **inline** chart mode for the highlighted stock (dots ↔ candles)

**Interactive Table Controls:**

- **Inline graph (always visible):** The chart below the table always shows the stock on the currently highlighted row.
  - Default inline mode is **dots** (braille marker) for higher visual precision.
- **Fullscreen graph:** Select the **Active Graph** column and press `Enter`.
  - `Enter` or `Esc` closes fullscreen.
  - Default fullscreen mode is **candles**.

**Graph Controls (Fullscreen):**

- `←` / `→` - Move candle cursor by 1
- `Shift+←` / `Shift+→` - Move candle cursor by 10
- `Home` / `End` - Jump to first / last candle
- `t` - Toggle fullscreen chart mode (candles ↔ dots)
- Mouse hover - Updates the top info ribbon to the nearest candle

**Indicators & Data Controls:**

- **Change period/interval:** Focus **Period** or **Interval**, press `Enter`, use `Up/Down`, then `Enter` to confirm.
- **Toggle indicators:** Focus the indicator column (MAVG, EMA, BOLLINGER) and press `Enter` to apply/remove.

**Info ribbon (Fullscreen):**

- The top ribbon shows the full timestamp and OHLC (and Volume when available) for the hovered/cursor candle.
