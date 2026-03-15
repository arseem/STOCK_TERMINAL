"""Thin entrypoint.

The actual app code lives under the `stock_terminal` package.
Run with: `textual run main.py`
"""

from stock_terminal.app import main


def run_app():
    main()

if __name__ == "__main__":
    run_app()
