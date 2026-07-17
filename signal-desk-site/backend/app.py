"""
Backend API for the trading signal website.
Wraps signal_engine.py (indicators, scoring, backtesting) behind a small
Flask REST API that the frontend (or a Telegram/Discord bot, later) calls.

Endpoints:
    GET /api/health
    GET /api/signal?symbol=EURUSD=X&interval=1h&period=60d&demo=true
    GET /api/backtest?symbol=EURUSD=X&interval=1h&period=60d&demo=true
                      &min_confidence=25&risk_reward=1.5&atr_mult=1.5

Run locally:
    pip install -r requirements.txt --break-system-packages
    python app.py
    # -> http://localhost:5000/api/signal?demo=true

Production: served by gunicorn (see Dockerfile / README), not app.run().
"""

import os
from flask import Flask, jsonify, request

from signal_engine import (
    generate_demo_data,
    load_from_yfinance,
    latest_signal,
    backtest_forex_style,
    add_all_indicators,
)

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    """Manual CORS so the frontend (a different origin in dev, and possibly
    a different domain in production) can call this API without extra deps."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _load_data(args):
    """Shared data-loading logic for both endpoints, driven by query params."""
    demo = args.get("demo", "false").lower() == "true"
    symbol = args.get("symbol")
    interval = args.get("interval", "1h")
    period = args.get("period", "60d")

    if demo or not symbol:
        return generate_demo_data(), "demo"
    return load_from_yfinance(symbol, interval, period), symbol


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/signal")
def api_signal():
    try:
        df, label = _load_data(request.args)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"failed to load data: {e}"}), 500

    if len(df) < 60:
        return jsonify({"error": "not enough bars to compute indicators (need 60+)"}), 400

    sig = latest_signal(df)
    last_row = add_all_indicators(df).iloc[-1]

    return jsonify(
        {
            "symbol": label,
            "direction": sig.direction,
            "confidence": sig.confidence,
            "binary_call": sig.binary_call,
            "entry": sig.entry,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "reasons": sig.reasons,
            "last_close": float(df["close"].iloc[-1]),
            "indicators": {
                "rsi_14": round(float(last_row["rsi_14"]), 2),
                "adx_14": round(float(last_row["adx_14"]), 2),
                "macd_hist": round(float(last_row["macd_hist"]), 5),
                "sma_20": round(float(last_row["sma_20"]), 5),
                "sma_50": round(float(last_row["sma_50"]), 5),
            },
        }
    )


@app.route("/api/backtest")
def api_backtest():
    try:
        df, label = _load_data(request.args)
    except SystemExit as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"failed to load data: {e}"}), 500

    if len(df) < 60:
        return jsonify({"error": "not enough bars to backtest (need 60+)"}), 400

    min_confidence = float(request.args.get("min_confidence", 25.0))
    risk_reward = float(request.args.get("risk_reward", 1.5))
    atr_mult = float(request.args.get("atr_mult", 1.5))
    risk_pct = float(request.args.get("risk_pct", 1.0))

    res = backtest_forex_style(
        df,
        atr_mult_sl=atr_mult,
        risk_reward=risk_reward,
        min_confidence=min_confidence,
        risk_per_trade_pct=risk_pct,
    )

    return jsonify(
        {
            "symbol": label,
            "trades": res.trades,
            "wins": res.wins,
            "losses": res.losses,
            "win_rate": res.win_rate,
            "total_return_pct": res.total_return_pct,
            "max_drawdown_pct": res.max_drawdown_pct,
            "avg_r_multiple": res.avg_r_multiple,
            "equity_curve": [round(float(x), 2) for x in res.equity_curve.tolist()],
        }
    )

from flask import Flask, jsonify, request, render_template
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    # Use Render's dynamic port environment variable, or fallback to 5000 locally
    server_port = int(os.environ.get("PORT", 5000))
app.run (host="0.0.0.0", port=server_port, debug=True, use_reloader=False)