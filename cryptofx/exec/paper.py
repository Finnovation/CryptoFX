"""Paper executor: the backtest fill model driven by live candles.

A distinct class so logs, state and the dashboard can never confuse a paper
run with a backtest or, later, a live account."""

from cryptofx.exec.backtest import BacktestExecutor


class PaperExecutor(BacktestExecutor):
    mode = "paper"
