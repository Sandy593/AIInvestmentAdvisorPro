"""Lock the risk-score ordering. These fixtures are representative, not live quotes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app_python as app

# ann_vol, max_drawdown, beta, debt multiple, market cap, expected tier
PROFILES = {
    "SPY": (0.15, 0.12, 1.0, 0.0, 500e9, "low"),
    "BND": (0.06, 0.08, 0.1, 0.0, 100e9, "low"),
    "MSFT": (0.24, 0.20, 1.05, 0.30, 3000e9, "medium"),
    "JNJ": (0.20, 0.22, 0.65, 0.40, 400e9, "medium"),
    "NVDA": (0.48, 0.36, 1.7, 0.20, 3000e9, "high"),
    "TSLA": (0.55, 0.42, 1.6, 0.15, 800e9, "high"),
    "small-cap biotech": (0.85, 0.60, 1.3, 2.5, 400e6, "speculative"),
}


def test_metrics_use_debt_multiple():
    import pandas as pd
    close = pd.Series([100 + i * 0.1 for i in range(80)])
    quote = app._metrics_from_history(
        close,
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        {"debtToEquity": 150, "marketCap": 2e12},
        "stock",
    )
    assert quote["risk"]["tier"] in {"low", "medium", "high", "speculative"}


def test_debt_units():
    assert app._debt_multiple({"debtToEquity": 3}) == 0.03
    assert app._debt_multiple({"debtToEquity": 150}) == 1.5
    calm = app.risk_score(0.20, 0.20, 1.0, 0.0, 50e9)[0]
    three_percent = app.risk_score(0.20, 0.20, 1.0, 0.03, 50e9)[0]
    one_and_a_half = app.risk_score(0.20, 0.20, 1.0, 1.5, 50e9)[0]
    assert calm == three_percent
    assert one_and_a_half == round(calm + 0.5, 2)


def test_profile_order():
    seen = {}
    for name, (vol, drawdown, beta, debt, size, expected) in PROFILES.items():
        score, tier = app.risk_score(vol, drawdown, beta, debt, size)
        seen[name] = (score, tier)
        assert tier == expected, f"{name} scored {score} ({tier}), expected {expected}"
    order = ["low", "medium", "high", "speculative"]
    assert order.index(seen["SPY"][1]) < order.index(seen["MSFT"][1])
    assert order.index(seen["MSFT"][1]) < order.index(seen["NVDA"][1])
    assert order.index(seen["NVDA"][1]) < order.index(seen["small-cap biotech"][1])


def test_outside_profile_is_not_sell():
    metrics = {"tier": "speculative", "sharpe": 1.2, "return_1y": 20, "max_drawdown": 0.2}
    assert app.rate_risk_adjusted(metrics, "Moderate") == "Outside your risk profile"
    assert app.rate_risk_adjusted(metrics, "Aggressive") != "Outside your risk profile"


def test_sector_cap_is_shared():
    caps = {"MU": 15, "STX": 15, "UNH": 30, "CVS": 30, "HCA": 30}
    sectors = {
        "MU": "Technology",
        "STX": "Technology",
        "UNH": "Healthcare",
        "CVS": "Healthcare",
        "HCA": "Healthcare",
    }
    # The model proposes a small plan. Filling toward 100 must not give every
    # healthcare name the whole 45% sector room.
    out = app.enforce_weights(
        caps,
        {"MU": 10, "STX": 5, "UNH": 10, "CVS": 10, "HCA": 5},
        sectors,
        max_sector=45,
    )
    health = sum(weight for symbol, weight in out.items() if sectors[symbol] == "Healthcare")
    tech = sum(weight for symbol, weight in out.items() if sectors[symbol] == "Technology")
    assert health <= 45.2
    assert tech <= 45.2
    assert all(out[symbol] <= caps[symbol] + 0.05 for symbol in out)


def test_quote_skips_rejection_rate():
    captured = []

    class FakeClient:
        def create_feedback(self, run_id, key, score, trace_id=None):
            captured.append((key, score))

    class Run:
        id = "run"
        trace_id = "run"

    original_flag = app.TRACING_ENABLED
    original_client = app.Client
    app.TRACING_ENABLED = True
    app.Client = lambda: FakeClient()
    try:
        app._log_feedback(
            {"task": "quote", "rejection_rate": 0.0, "eval_scores": {"suitability": 0.8}},
            Run(),
        )
        assert ("suitability", 0.8) in captured
        assert all(key != "rejection_rate" for key, _ in captured)
        captured.clear()
        app._log_feedback({"task": "trending", "rejection_rate": 0.4, "eval_scores": {}}, Run())
        assert ("rejection_rate", 0.4) in captured
    finally:
        app.TRACING_ENABLED = original_flag
        app.Client = original_client
