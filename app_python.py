"""
AI Investment Advisor — LangGraph multi-agent app with LangSmith tracing.

One router reads the tab the UI selected and sends the run into that tab's
subgraph. Trending research fans out to one researcher per sector plus an
emerging scout. Deep research runs the company and sentiment analysts together.
Get Quote does not enter the graph. An evaluator can send a plan back for at
most two revisions.

LangSmith tracing is enabled only when LANGSMITH_API_KEY is set.
"""

from __future__ import annotations

import logging
import operator
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.text import MIMEText
from typing import Annotated, Any, Dict, List, Literal, Optional

import gradio as gr
import markdown
import pandas as pd
import requests
import smtplib
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-investment-advisor")
TRACING_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_TRACING"] = "true" if TRACING_ENABLED else "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true" if TRACING_ENABLED else "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advisor.langgraph")

# This Space is hosted on ZeroGPU, which requires at least one decorated function.
# The decorator is a no-op when the spaces package is not installed.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=60)
    def _zero_gpu_noop():
        return None

MODEL_NAME = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
MAX_TOOL_CALLS = 4
SECTOR_TOOL_CALLS = 2
CACHE_SECONDS = 300
MAX_REVISIONS = 2
MIN_HISTORY_DAYS = 60
MIN_DOLLAR_VOLUME = 1_000_000
SEC_USER_AGENT = "AIInvestmentAdvisorPro/1.0 (educational research)"

RATING_LABELS = {
    "Strong Buy": "🟢 Strong Buy",
    "Moderate Buy": "🟡 Moderate Buy",
    "Hold": "🔵 Hold",
    "Sell": "🔴 Sell",
    "Outside your risk profile": "⚪ Outside your risk profile",
}
TRENDING_HEADERS = [
    "Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating",
    "Market Cap", "P/E Ratio", "PEG Ratio", "ROE", "P/B Ratio", "D/E Ratio",
]
STABLE_HEADERS = [
    "Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating",
    "P/E Ratio", "PEG Ratio", "ROE", "P/B Ratio", "D/E Ratio",
]
TIER_CAPS = {
    "Conservative": {"low": 45.0, "medium": 20.0, "high": 0.0, "speculative": 0.0},
    "Moderate": {"low": 40.0, "medium": 30.0, "high": 15.0, "speculative": 0.0},
    "Aggressive": {"low": 45.0, "medium": 35.0, "high": 25.0, "speculative": 10.0},
}
SHARE_CLASS_GROUPS = (
    {"GOOG", "GOOGL"},
    {"BRK-A", "BRK-B", "BRK.A", "BRK.B"},
    {"FOX", "FOXA"},
    {"NWS", "NWSA"},
)
SLEEVES = (
    {"key": "bonds", "symbol": "BND", "label": "Bonds", "name": "Vanguard Total Bond Market ETF"},
    {"key": "dividend_stocks", "symbol": "VYM", "label": "Dividend Stocks", "name": "Vanguard High Dividend Yield ETF"},
    {"key": "reits", "symbol": "VNQ", "label": "REITs", "name": "Vanguard Real Estate ETF"},
    {"key": "stocks", "symbol": "VTI", "label": "Stocks", "name": "Vanguard Total Stock Market ETF"},
    {"key": "growth_stocks", "symbol": "VUG", "label": "Growth Stocks", "name": "Vanguard Growth ETF"},
    {"key": "cash", "symbol": "BIL", "label": "Cash", "name": "SPDR Bloomberg 1-3 Month T-Bill ETF"},
)
SLEEVE_BY_SYMBOL = {item["symbol"]: item for item in SLEEVES}
MIN_CASH = {"Short-term": 10.0, "Medium-term": 5.0, "Long-term": 2.0}
PROMISES = (
    "guaranteed return",
    "guaranteed profit",
    "can't lose",
    "cannot lose",
    "risk-free",
    "risk free",
    "sure thing",
    "will definitely",
    "certain profit",
)
_CIK_BY_TICKER: Optional[Dict[str, str]] = None


def _add_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AdvisorState(TypedDict, total=False):
    task: str
    research_type: str
    sectors: List[str]
    symbol: str
    budget: float
    time_horizon: str
    risk_tolerance: str
    goals: str
    discovered: List[Dict[str, Any]]
    emerging: List[Dict[str, Any]]
    market_data: Dict[str, Dict[str, Any]]
    company: Dict[str, Any]
    sentiment: Dict[str, Any]
    ratings: List[Dict[str, Any]]
    weights: Dict[str, float]
    portfolio: Dict[str, Any]
    report: str
    summary_rows: List[Dict[str, Any]]
    stable_rows: List[Dict[str, Any]]
    rejections: List[Dict[str, Any]]
    rejection_rate: float
    agent_log: List[str]
    errors: List[str]
    eval_passed: bool
    eval_forced: bool
    eval_attempts: int
    eval_feedback: str
    eval_scores: Dict[str, float]
    headlines_found: bool
    trace_url: str


class FlowState(TypedDict, total=False):
    """Subgraph state. List fields use reducers so parallel branches append."""

    task: str
    research_type: str
    sectors: List[str]
    sector: str
    kind: str
    symbol: str
    budget: float
    time_horizon: str
    risk_tolerance: str
    goals: str
    discovered: Annotated[List[Dict[str, Any]], operator.add]
    emerging: Annotated[List[Dict[str, Any]], operator.add]
    rejections: Annotated[List[Dict[str, Any]], operator.add]
    agent_log: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    market_data: Annotated[Dict[str, Dict[str, Any]], _add_dicts]
    company: Dict[str, Any]
    sentiment: Dict[str, Any]
    ratings: List[Dict[str, Any]]
    weights: Dict[str, float]
    portfolio: Dict[str, Any]
    report: str
    summary_rows: List[Dict[str, Any]]
    stable_rows: List[Dict[str, Any]]
    rejection_rate: float
    eval_passed: bool
    eval_forced: bool
    eval_attempts: int
    eval_feedback: str
    eval_scores: Dict[str, float]
    headlines_found: bool
    trace_url: str


class Candidate(BaseModel):
    symbol: str = Field(description="Ticker only, for example AAPL or BTC")
    name: str
    sector: str = ""
    catalyst: str
    thesis: str
    source_url: str = ""


class CandidateBatch(BaseModel):
    assets: List[Candidate]


class CompanyProfile(BaseModel):
    company_name: str
    founded_year: str
    business_description: str
    sector: str
    market_cap_category: str
    key_products: List[str]
    competitive_advantages: str
    sourced_developments: str


class SentimentProfile(BaseModel):
    overall_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    sentiment_score: float = Field(description="Score from -5 to 5")
    developments: List[str]
    risk_factors: List[str]
    growth_catalysts: List[str]
    analyst_outlook: str


class WeightChoice(BaseModel):
    symbol: str
    weight_pct: float


class WeightPlan(BaseModel):
    holdings: List[WeightChoice]
    rationale: str
    risk_notes: List[str]
    goal_alignment: str
    rebalancing: str


class JudgeScore(BaseModel):
    suitability: float = Field(description="0 to 1")
    grounding: float = Field(description="0 to 1")
    compliance: float = Field(description="0 to 1")
    feedback: str


def _note(message: str) -> List[str]:
    logger.info(message)
    return [message]


def _fail(message: str) -> List[str]:
    logger.warning(message)
    return [message]


def _listed_size(info: Dict[str, Any]) -> float:
    """Stocks report marketCap. ETFs usually report total assets instead."""
    return _num(info.get("marketCap") or info.get("totalAssets") or info.get("netAssets"))


def _debt_multiple(info: Dict[str, Any]) -> float:
    """Yahoo reports debtToEquity as a percent (150 means 1.5x)."""
    raw = _num(info.get("debtToEquity"))
    return raw / 100.0 if raw > 0 else 0.0


def _dividend_percent(info: Dict[str, Any], price: float) -> float:
    rate = _num(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))
    if rate > 0 and price > 0:
        return rate / price * 100
    raw = _num(info.get("trailingAnnualDividendYield"))
    if raw <= 0:
        raw = _num(info.get("dividendYield"))
    if 0 < raw < 0.25:
        return raw * 100
    return raw


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the environment")
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature)


def asset_kind(research_type: str) -> str:
    if "Crypto" in (research_type or ""):
        return "crypto"
    if "ETF" in (research_type or ""):
        return "etf"
    return "stock"


def clean_symbol(symbol: str, kind: str) -> str:
    token = (symbol or "").upper().strip().replace("$", "")
    token = token.split()[0] if token else ""
    token = token.replace("-USD", "")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
    token = "".join(ch for ch in token if ch in allowed)
    max_len = 10 if kind == "crypto" else 5
    if not token or len(token) > max_len:
        return ""
    return token


def yf_symbol(symbol: str, kind: str) -> str:
    return f"{symbol}-USD" if kind == "crypto" else symbol


def format_market_cap(market_cap: float) -> str:
    if not market_cap:
        return "N/A"
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.1f}T"
    if market_cap >= 1e9:
        return f"${market_cap / 1e9:.1f}B"
    if market_cap >= 1e6:
        return f"${market_cap / 1e6:.0f}M"
    return f"${market_cap:,.0f}"


def names_match(claimed: str, long_name: str) -> bool:
    """True when the model's company name and the exchange name share a meaningful token."""
    stop = {
        "inc", "incorporated", "corp", "corporation", "ltd", "limited", "plc", "co",
        "company", "the", "class", "common", "stock", "shares", "share", "etf",
        "trust", "fund", "holdings", "holding", "group", "sa", "nv", "ag",
    }

    def tokens(text: str) -> set:
        found = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        return {token for token in found if token not in stop and len(token) > 1}

    claimed_tokens = tokens(claimed)
    official_tokens = tokens(long_name)
    if not claimed_tokens or not official_tokens:
        return False
    return bool(claimed_tokens & official_tokens)


def position_cap(tier: str, tolerance: str, horizon: str = "", sleeve: str = "") -> float:
    caps = TIER_CAPS.get(tolerance or "Moderate", TIER_CAPS["Moderate"])
    cap = float(caps.get(tier or "high", 0.0))
    if horizon == "Short-term" and tier in {"high", "speculative"}:
        cap = min(cap, 10.0)
    if horizon == "Short-term" and sleeve == "growth_stocks":
        cap = min(cap, 10.0)
    return cap


def rate_risk_adjusted(metrics: Dict[str, Any], tolerance: str, sentiment: str = "NEUTRAL") -> str:
    """Rate from risk-adjusted history. Strong Buy does not require a bullish label."""
    tier = metrics.get("tier") or "high"
    cap = position_cap(tier, tolerance or "Moderate")
    sharpe = _num(metrics.get("sharpe"))
    ret_1y = _num(metrics.get("return_1y"))
    drawdown = _num(metrics.get("max_drawdown"))
    if cap <= 0:
        return "Outside your risk profile"
    if sentiment == "BEARISH" and sharpe < 1.2:
        return "Sell" if sharpe < 0 else "Hold"
    if sentiment != "BEARISH" and sharpe >= 1.0 and ret_1y >= 8 and drawdown <= 0.30 and tier in {"low", "medium"}:
        return "Strong Buy"
    if sentiment != "BEARISH" and sharpe >= 0.5 and ret_1y > 0 and tier != "speculative":
        return "Moderate Buy"
    if sentiment != "BEARISH" and sharpe >= 0.0 and drawdown <= 0.45:
        return "Hold"
    return "Sell"


def risk_score(ann_vol: float, max_drawdown: float, beta: Optional[float], debt_ratio: float, market_cap: float) -> tuple:
    """Score 0-10. debt_ratio is a multiple (1.5 means 1.5x), not Yahoo's percent."""
    vol_pts = min(4.0, max(0.0, ann_vol / 0.12))
    dd_pts = min(3.0, max(0.0, max_drawdown / 0.15))
    beta_pts = 0.0 if beta is None else min(1.5, max(0.0, (beta - 0.8) / 0.4))
    lev_pts = min(1.5, max(0.0, debt_ratio - 1.0))
    if market_cap >= 10e9:
        size_pts = 0.0
    elif market_cap >= 2e9:
        size_pts = 0.5
    elif market_cap > 0:
        size_pts = 1.5
    else:
        size_pts = 1.0
    score = round(min(10.0, vol_pts + dd_pts + beta_pts + lev_pts + size_pts), 2)
    if score < 3:
        tier = "low"
    elif score < 5.5:
        tier = "medium"
    elif score < 8.5:
        tier = "high"
    else:
        tier = "speculative"
    return score, tier


def enforce_weights(
    caps: Dict[str, float],
    proposed: Dict[str, float],
    sectors: Dict[str, str],
    max_sector: Optional[float] = None,
    cash_symbol: Optional[str] = None,
    min_cash: float = 0.0,
) -> Dict[str, float]:
    """Clip proposed weights to code-side caps and try to make them add to 100."""
    limits = {symbol: float(cap) for symbol, cap in caps.items() if _num(cap) > 0}
    if not limits:
        return {}
    weights = {
        symbol: min(max(_num(proposed.get(symbol)), 0.0), limit)
        for symbol, limit in limits.items()
    }
    if cash_symbol in limits and weights.get(cash_symbol, 0.0) < min_cash:
        weights[cash_symbol] = min(limits[cash_symbol], float(min_cash))
    if sum(weights.values()) <= 0:
        share = 100.0 / len(limits)
        weights = {symbol: min(limit, share) for symbol, limit in limits.items()}
    if max_sector:
        for sector in set(sectors.values()):
            members = [symbol for symbol in weights if symbol != cash_symbol and sectors.get(symbol, symbol) == sector]
            used = sum(weights[symbol] for symbol in members)
            if used > max_sector > 0:
                scale = max_sector / used
                for symbol in members:
                    weights[symbol] *= scale

    def sector_used(trial: Dict[str, float], sector: str) -> float:
        return sum(
            weight for symbol, weight in trial.items()
            if symbol != cash_symbol and sectors.get(symbol, symbol) == sector
        )

    def heads_for(trial: Dict[str, float]) -> Dict[str, float]:
        """Room left under each position cap. A sector cap is shared by its members."""
        position = {symbol: max(0.0, limits[symbol] - trial.get(symbol, 0.0)) for symbol in limits}
        if not max_sector:
            return position
        sector_room = {
            sector: max(0.0, float(max_sector) - sector_used(trial, sector))
            for sector in set(sectors.values())
        }
        heads: Dict[str, float] = {}
        for symbol, room in position.items():
            if symbol == cash_symbol:
                heads[symbol] = room
                continue
            sector = sectors.get(symbol, symbol)
            peers = [
                name for name, peer_room in position.items()
                if name != cash_symbol and sectors.get(name, name) == sector and peer_room > 0
            ]
            pool = sum(position[name] for name in peers)
            share = sector_room.get(sector, 0.0) * (room / pool) if pool else 0.0
            heads[symbol] = min(room, share)
        return heads

    for _ in range(12):
        total = sum(weights.values())
        if total > 100.05:
            overflow = total - 100.0
            spare = {
                symbol: max(0.0, weight - (min_cash if symbol == cash_symbol else 0.0))
                for symbol, weight in weights.items()
            }
            pool = sum(spare.values())
            if pool <= 0:
                break
            for symbol, extra in spare.items():
                floor = min_cash if symbol == cash_symbol else 0.0
                weights[symbol] = max(floor, weights[symbol] - overflow * (extra / pool))
            continue
        if abs(total - 100.0) <= 0.05:
            break
        need = 100.0 - total
        rooms = heads_for(weights)
        free = sum(rooms.values())
        if free <= 1e-6:
            break
        give = min(need, free)
        for symbol, head in rooms.items():
            weights[symbol] += give * (head / free)

    rounded = {symbol: round(weight, 2) for symbol, weight in weights.items() if weight >= 0.25}
    if not rounded:
        return {}
    drift = round(100.0 - sum(rounded.values()), 2)

    def _can_absorb(symbol: str, extra: float) -> bool:
        updated = rounded[symbol] + extra
        if updated < 0 or updated > limits.get(symbol, 0.0) + 0.05:
            return False
        if max_sector and symbol != cash_symbol and extra > 0:
            sector = sectors.get(symbol, symbol)
            used = sum(
                weight for name, weight in rounded.items()
                if name != cash_symbol and sectors.get(name, name) == sector
            )
            if used + extra > max_sector + 0.05:
                return False
        return True

    order = sorted(rounded, key=lambda symbol: (symbol != cash_symbol, -rounded[symbol]))
    for symbol in order:
        if _can_absorb(symbol, drift):
            rounded[symbol] = round(rounded[symbol] + drift, 2)
            break
    if max_sector:
        for sector in set(sectors.values()):
            members = [symbol for symbol in rounded if symbol != cash_symbol and sectors.get(symbol, symbol) == sector]
            used = sum(rounded[symbol] for symbol in members)
            if used > max_sector + 0.05:
                scale = max_sector / used
                for symbol in members:
                    rounded[symbol] = round(rounded[symbol] * scale, 2)
    return {symbol: weight for symbol, weight in rounded.items() if weight > 0}


def _ordered_row(data: Dict[str, Any], headers: List[str]) -> Dict[str, Any]:
    return {key: data.get(key, "N/A") for key in headers}


def init_database() -> None:
    conn = sqlite3.connect("investment_advisor.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_date DATE,
            budget REAL,
            risk_tolerance TEXT,
            time_horizon TEXT,
            strategy_content TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE,
            report_content TEXT,
            report_type TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_research(report: str, report_type: str) -> None:
    conn = sqlite3.connect("investment_advisor.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stock_research (report_date, report_content, report_type) VALUES (?, ?, ?)",
        (datetime.now().date(), report, report_type),
    )
    conn.commit()
    conn.close()


def save_portfolio(report: str, budget: float, risk: str, horizon: str) -> None:
    conn = sqlite3.connect("investment_advisor.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO portfolio_strategies
        (strategy_date, budget, risk_tolerance, time_horizon, strategy_content)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now().date(), budget, risk, horizon, report),
    )
    conn.commit()
    conn.close()


def _history_frame(data: pd.DataFrame, symbol: str, field: str) -> pd.Series:
    if data is None or data.empty or field is None:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        if (symbol, field) in data.columns:
            series = data[(symbol, field)]
        elif (field, symbol) in data.columns:
            series = data[(field, symbol)]
        else:
            return pd.Series(dtype=float)
    else:
        if symbol not in (None, "") and len(getattr(data, "columns", [])) and field not in data.columns:
            return pd.Series(dtype=float)
        if field not in data.columns:
            return pd.Series(dtype=float)
        series = data[field]
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series


_INFO_CACHE: Dict[str, tuple] = {}
_HEADLINE_CACHE: Dict[str, tuple] = {}


def _cache_get(cache: Dict[str, tuple], key: str) -> Any:
    hit = cache.get(key)
    if not hit:
        return None
    saved_at, value = hit
    if time.time() - saved_at > CACHE_SECONDS:
        cache.pop(key, None)
        return None
    return value


def _load_info(ticker: str) -> tuple:
    cached = _cache_get(_INFO_CACHE, ticker)
    if cached is not None:
        return ticker, cached
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Info failed for %s: %s", ticker, exc)
        info = {}
    if info:
        _INFO_CACHE[ticker] = (time.time(), info)
    return ticker, info


def _business_days_old(last_ts: Any) -> int:
    last = pd.Timestamp(last_ts)
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    last = last.normalize()
    today = pd.Timestamp.now().normalize()
    if last >= today:
        return 0
    return max(0, len(pd.bdate_range(last, today)) - 1)


def _calendar_days_old(last_ts: Any) -> int:
    last = pd.Timestamp(last_ts)
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    return max(0, (pd.Timestamp.now().normalize() - last.normalize()).days)


def _metrics_from_history(close: pd.Series, volume: pd.Series, benchmark: pd.Series, info: Dict[str, Any], kind: str) -> Dict[str, Any]:
    returns = close.pct_change().dropna()
    periods = 365 if kind == "crypto" else 252
    ann_vol = _num(returns.std() * (periods ** 0.5))
    peak = close.cummax()
    drawdown = _num(((close / peak) - 1).min())
    max_drawdown = abs(drawdown)
    beta = None
    if benchmark is not None and not benchmark.empty:
        bench_returns = benchmark.pct_change().dropna()
        aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
        if len(aligned) >= 40 and _num(aligned.iloc[:, 1].var()) > 0:
            beta = _num(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
    ann_return = _num(returns.mean() * periods)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol else 0.0
    start_1y = _num(close.iloc[0])
    current = _num(close.iloc[-1])
    return_1y = ((current - start_1y) / start_1y) * 100 if start_1y else 0.0
    lookback = 30 if kind == "crypto" else 22
    month_ago = close.iloc[-lookback] if len(close) > lookback else close.iloc[0]
    return_1m = ((current - _num(month_ago)) / _num(month_ago)) * 100 if _num(month_ago) else 0.0
    debt_ratio = _debt_multiple(info)
    market_cap = _listed_size(info)
    dollar_volume = 0.0
    if volume is not None and not volume.empty:
        paired = pd.concat([close, volume], axis=1, join="inner").dropna().tail(20)
        if not paired.empty:
            dollar_volume = _num((paired.iloc[:, 0] * paired.iloc[:, 1]).mean())
    score, tier = risk_score(ann_vol, max_drawdown, beta, debt_ratio if kind != "crypto" else 0.0, market_cap)
    high = _num(info.get("_high"))
    low = _num(info.get("_low"))
    dividend = _dividend_percent(info, current)
    return {
        "current_price": round(current, 4 if kind == "crypto" else 2),
        "1m_performance": round(return_1m, 2),
        "1y_performance": round(return_1y, 2),
        "52w_high": round(high or _num(close.max()), 2),
        "52w_low": round(low or _num(close.min()), 2),
        "dividend_yield": round(float(dividend), 2) if dividend else 0,
        "sector": info.get("sector") or info.get("category") or ("Cryptocurrency" if kind == "crypto" else "Unknown"),
        "long_name": info.get("longName") or info.get("shortName") or "",
        "quote_type": info.get("quoteType") or "",
        "market_cap": market_cap,
        "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
        "peg_ratio": round(info.get("pegRatio"), 2) if info.get("pegRatio") else "N/A",
        "roe": round(info.get("returnOnEquity") * 100, 2) if info.get("returnOnEquity") else "N/A",
        "pb_ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else "N/A",
        "de_ratio": round(info.get("debtToEquity"), 2) if info.get("debtToEquity") else "N/A",
        "risk": {
            "ann_vol": round(ann_vol, 4),
            "max_drawdown": round(max_drawdown, 4),
            "beta": None if beta is None else round(beta, 2),
            "sharpe": round(sharpe, 2),
            "score": score,
            "tier": tier,
            "dollar_volume": round(dollar_volume, 2),
            "return_1y": round(return_1y, 2),
        },
    }


def _reject_reason(close: pd.Series, volume: pd.Series, info: Dict[str, Any], claimed_name: str, kind: str, symbol: str) -> str:
    expected = {"stock": "EQUITY", "etf": "ETF", "crypto": "CRYPTOCURRENCY"}.get(kind, "EQUITY")
    if close.empty or len(close) < MIN_HISTORY_DAYS:
        return f"fewer than {MIN_HISTORY_DAYS} trading days"
    if (close <= 0).any():
        return "zero or negative prices"
    daily = close.pct_change().dropna().abs()
    if not daily.empty and bool((daily >= 0.50).any()):
        return "a daily move of 50% or more"
    last = close.index[-1]
    if kind == "crypto":
        if _calendar_days_old(last) > 2:
            return "stale price"
    elif _business_days_old(last) > 5:
        return "stale price"
    quote_type = (info.get("quoteType") or "").upper()
    if quote_type != expected:
        return f"quote type {quote_type or 'unknown'} is not {expected}"
    long_name = info.get("longName") or info.get("shortName") or ""
    claimed = (claimed_name or "").strip()
    ticker_only = claimed.upper() == symbol.upper()
    if not ticker_only and not names_match(claimed, long_name):
        return f"name does not match {long_name or 'the listed security'}"
    market_cap = _listed_size(info)
    if market_cap <= 0:
        return "market cap is missing or zero"
    paired = pd.concat([close, volume], axis=1, join="inner").dropna().tail(20) if volume is not None else pd.DataFrame()
    dollar_volume = _num((paired.iloc[:, 0] * paired.iloc[:, 1]).mean()) if not paired.empty else 0.0
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return "20-day average dollar volume is under $1M"
    return ""


def verify_candidates(candidates: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    """Batch-download prices, check each name, and drop duplicates."""
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for item in candidates:
        symbol = clean_symbol(item.get("symbol") or "", kind)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        ordered.append({**item, "symbol": symbol})
    if not ordered:
        return {"quotes": {}, "rejections": [], "rejection_rate": 0.0}

    ticker_for = {item["symbol"]: yf_symbol(item["symbol"], kind) for item in ordered}
    benchmark = "BTC-USD" if kind == "crypto" else "SPY"
    download_list = list(dict.fromkeys([*ticker_for.values(), benchmark]))
    try:
        history = yf.download(
            download_list,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        logger.warning("Batch download failed: %s", exc)
        history = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = dict(pool.map(_load_info, download_list))

    bench_close = _history_frame(history, benchmark, "Close")
    quotes: Dict[str, Dict[str, Any]] = {}
    rejections: List[Dict[str, Any]] = []
    for item in ordered:
        symbol = item["symbol"]
        ticker = ticker_for[symbol]
        close = _history_frame(history, ticker, "Close")
        high = _history_frame(history, ticker, "High")
        low = _history_frame(history, ticker, "Low")
        volume = _history_frame(history, ticker, "Volume")
        info = dict(infos.get(ticker) or {})
        if not high.empty:
            info["_high"] = _num(high.max())
        if not low.empty:
            info["_low"] = _num(low.min())
        reason = _reject_reason(close, volume, info, item.get("name") or symbol, kind, symbol)
        if reason:
            rejections.append({"symbol": symbol, "name": item.get("name") or symbol, "reason": reason})
            continue
        quote = _metrics_from_history(close, volume, bench_close, info, kind)
        quote["claimed_name"] = item.get("name") or ""
        quotes[symbol] = quote

    rejections.extend(_drop_share_classes(quotes))
    total = len(ordered)
    rate = round(len(rejections) / total, 4) if total else 0.0
    return {"quotes": quotes, "rejections": rejections, "rejection_rate": rate}


def _drop_share_classes(quotes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    dropped: List[Dict[str, Any]] = []
    for group in SHARE_CLASS_GROUPS:
        present = [symbol for symbol in group if symbol in quotes]
        if len(present) < 2:
            continue
        present.sort(key=lambda symbol: _num((quotes[symbol].get("risk") or {}).get("dollar_volume")), reverse=True)
        for symbol in present[1:]:
            dropped.append({
                "symbol": symbol,
                "name": quotes[symbol].get("long_name") or symbol,
                "reason": f"duplicate share class of {present[0]}",
            })
            quotes.pop(symbol, None)
    return dropped


def _headline_lines(payload: Any, limit: int = 5) -> List[str]:
    lines: List[str] = []
    for article in payload or []:
        if isinstance(article, dict) and "content" in article and isinstance(article["content"], dict):
            content = article["content"]
            title = content.get("title") or ""
            url = ((content.get("canonicalUrl") or {}) or {}).get("url") or content.get("clickThroughUrl") or ""
            if isinstance(url, dict):
                url = url.get("url") or ""
            published = content.get("pubDate") or ""
        elif isinstance(article, dict):
            title = article.get("title") or ""
            url = article.get("link") or article.get("url") or ""
            published = article.get("providerPublishTime") or ""
        else:
            continue
        title = re.sub(r"\s+", " ", str(title)).strip()
        url = str(url or "").strip()
        if not title:
            continue
        when = f" ({published})" if published else ""
        lines.append(f"- {title}{when} {url}".strip())
        if len(lines) >= limit:
            break
    return lines


def search_headlines(query: str) -> str:
    """Search Yahoo Finance headlines and return titles with URLs."""
    query = (query or "").strip()[:120]
    if not query:
        return "No query."
    cached = _cache_get(_HEADLINE_CACHE, query.lower())
    if cached is not None:
        return cached
    articles: List[Any] = []
    try:
        articles.extend(yf.Search(query, news_count=5).news or [])
    except Exception as exc:
        logger.warning("Headline search failed: %s", exc)
    token = clean_symbol(query, "stock") or clean_symbol(query, "crypto")
    if token:
        try:
            articles.extend(yf.Ticker(token).news or [])
        except Exception:
            pass
    lines = _headline_lines(articles, limit=5)
    text = "\n".join(lines) if lines else "No headlines returned."
    _HEADLINE_CACHE[query.lower()] = (time.time(), text)
    return text


def _cik_map() -> Dict[str, str]:
    global _CIK_BY_TICKER
    if _CIK_BY_TICKER is not None:
        return _CIK_BY_TICKER
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=12,
    )
    response.raise_for_status()
    mapping = {}
    for row in (response.json() or {}).values():
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            mapping[ticker] = str(row.get("cik_str")).zfill(10)
    _CIK_BY_TICKER = mapping
    return mapping


def company_filings(symbol: str) -> str:
    """Return the latest SEC filing titles and URLs for a US ticker."""
    token = clean_symbol(symbol, "stock")
    if not token:
        return "No ticker to look up."
    try:
        cik = _cik_map().get(token)
    except Exception as exc:
        return f"SEC ticker list unavailable: {exc}"
    if not cik:
        return f"No SEC company match for {token}."
    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers={"User-Agent": SEC_USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
        recent = (response.json().get("filings") or {}).get("recent") or {}
    except Exception as exc:
        return f"SEC filings unavailable: {exc}"
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    lines = []
    for index, form in enumerate(forms[:3]):
        accession = str(accessions[index]).replace("-", "") if index < len(accessions) else ""
        document = documents[index] if index < len(documents) else ""
        filed = dates[index] if index < len(dates) else ""
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{document}" if accession and document else ""
        lines.append(f"- {form} filed {filed} {url}".strip())
    return "\n".join(lines) if lines else "No SEC filings returned."


search_headlines_tool = tool(search_headlines)
company_filings_tool = tool(company_filings)
RESEARCH_TOOLS = [search_headlines_tool, company_filings_tool]


def _tool_notes(task: str, prefetched: str, max_calls: int = MAX_TOOL_CALLS) -> str:
    """Let the model call news or filings, then stop. Never more than max_calls."""
    system = (
        "You research public markets with tools. Headlines may already be attached. "
        "Call search_headlines or company_filings only if you still need a source. "
        f"Stop after {max_calls} tool calls. Do not invent headlines, filings, or URLs."
    )
    messages: List[Any] = [
        SystemMessage(content=system),
        HumanMessage(content=f"Prefetched notes:\n{prefetched or 'None.'}\n\nTask:\n{task}"),
    ]
    try:
        llm = get_llm(LLM_TEMPERATURE).bind_tools(RESEARCH_TOOLS)
    except Exception as exc:
        logger.warning("Tool setup failed: %s", exc)
        return prefetched
    used = 0
    notes = [prefetched] if prefetched else []
    tool_map = {item.name: item for item in RESEARCH_TOOLS}
    for _ in range(max_calls):
        if used >= max_calls:
            break
        try:
            ai_message = llm.invoke(messages)
        except Exception as exc:
            logger.warning("Tool-calling model failed: %s", exc)
            break
        calls = getattr(ai_message, "tool_calls", None) or []
        if not calls:
            break
        messages.append(ai_message)
        for call in calls:
            if used >= max_calls:
                break
            used += 1
            name = call.get("name")
            args = call.get("args") or {}
            try:
                payload = tool_map[name].invoke(args)
            except Exception as exc:
                payload = f"{name} failed: {exc}"
            text = str(payload)[:2500]
            notes.append(text)
            messages.append(ToolMessage(content=text, tool_call_id=call.get("id") or f"call-{used}"))
    return "\n\n".join(note for note in notes if note)


def _urls_in(text: str) -> set:
    return {url.rstrip(").,]>\"'") for url in re.findall(r"https?://[^\s)>\"]+", text or "")}


def _accepted_url(url: str, notes: str) -> str:
    cleaned = (url or "").strip().rstrip(").,]>\"'")
    if not cleaned:
        return ""
    return cleaned if cleaned in _urls_in(notes) else ""


def _source_rule(notes: str) -> str:
    if "http" in (notes or "").lower():
        return (
            "Use only URLs that appear in the notes. "
            "If a catalyst is not supported by a note, call it unverified and do not describe it as recent."
        )
    return (
        "No headlines or filings were retrieved. Do not use the word recent. "
        "Mark every catalyst as unverified."
    )


def _propose_candidates(sector: str, research_type: str, kind: str, bucket: str) -> Dict[str, Any]:
    focus = "cryptocurrencies" if kind == "crypto" else sector
    count = 3
    task = (
        f"Find {count} real, currently tradable {kind}s in {focus} for {datetime.now():%B %Y}. "
        "Use real ticker symbols only. For crypto, use symbols such as BTC, without a -USD suffix. "
        "Give each asset its own sector, not a combined list. Keep each thesis to one sentence."
    )
    if bucket == "emerging":
        task += (
            f" Stay inside these sectors: {focus}. "
            "Prefer names that are less obvious than the largest mega-caps."
        )
    label = "Emerging scout" if bucket == "emerging" else f"Sector researcher ({sector})"
    try:
        prefetched = search_headlines(focus)
        notes = _tool_notes(task, prefetched if "No headlines" not in prefetched else "", max_calls=SECTOR_TOOL_CALLS)
        prompt = f"{_source_rule(notes)}\n\nNotes:\n{notes or 'None.'}\n\n{task}"
        result = get_llm(LLM_TEMPERATURE).with_structured_output(CandidateBatch).invoke(prompt)
    except Exception as exc:
        key = "emerging" if bucket == "emerging" else "discovered"
        return {
            key: [],
            "agent_log": _note(f"{label}: failed"),
            "errors": _fail(f"{label}: {exc}"),
        }
    assets = []
    seen = set()
    for item in result.assets:
        symbol = clean_symbol(item.symbol, kind)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        if kind == "crypto":
            own_sector = "Cryptocurrency"
        elif bucket == "emerging":
            own_sector = (item.sector or "").strip() or "Unspecified"
        else:
            own_sector = sector
        assets.append({
            "symbol": symbol,
            "name": item.name,
            "sector": own_sector,
            "catalyst": item.catalyst,
            "thesis": item.thesis,
            "source_url": _accepted_url(item.source_url, notes),
            "bucket": bucket,
            "kind": kind,
        })
        if len(assets) >= count:
            break
    key = "emerging" if bucket == "emerging" else "discovered"
    return {
        key: assets,
        "agent_log": _note(f"{label}: selected {len(assets)} {kind} symbols"),
    }


def sector_researcher(state: FlowState) -> Dict[str, Any]:
    return _propose_candidates(
        state.get("sector") or "Technology",
        state.get("research_type") or "Trending Stocks",
        state.get("kind") or asset_kind(state.get("research_type") or ""),
        "trending",
    )


def emerging_scout(state: FlowState) -> Dict[str, Any]:
    kind = state.get("kind") or asset_kind(state.get("research_type") or "")
    if kind == "crypto":
        focus = "smaller cryptocurrencies"
    else:
        focus = ", ".join(state.get("sectors") or []) or "the selected sectors"
    return _propose_candidates(
        focus,
        state.get("research_type") or "Trending Stocks",
        kind,
        "emerging",
    )


def fan_trending(state: FlowState):
    research_type = state.get("research_type") or "Trending Stocks"
    kind = asset_kind(research_type)
    if kind == "crypto":
        sectors = ["Cryptocurrency"]
    else:
        sectors = [sector for sector in (state.get("sectors") or []) if sector] or ["Technology"]
    base = {
        "research_type": research_type,
        "kind": kind,
        "risk_tolerance": state.get("risk_tolerance") or "Moderate",
        "sectors": sectors,
    }
    sends = [Send("sector_researcher", {**base, "sector": sector}) for sector in sectors]
    sends.append(Send("emerging_scout", base))
    return sends


def join_research(state: FlowState) -> Dict[str, Any]:
    discovered = len(state.get("discovered") or [])
    emerging = len(state.get("emerging") or [])
    return {"agent_log": _note(f"Research joined: {discovered} sector names, {emerging} emerging names")}


def _unique_candidates(state: FlowState) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in list(state.get("discovered") or []) + list(state.get("emerging") or []):
        symbol = item.get("symbol")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        merged.append(item)
    return merged


def market_data_trending(state: FlowState) -> Dict[str, Any]:
    kind = asset_kind(state.get("research_type") or "Trending Stocks")
    verified = verify_candidates(_unique_candidates(state), kind)
    accepted = len(verified["quotes"])
    rejected = len(verified["rejections"])
    return {
        "market_data": verified["quotes"],
        "rejections": verified["rejections"],
        "rejection_rate": verified["rejection_rate"],
        "agent_log": _note(f"Market data: accepted {accepted}, rejected {rejected}"),
    }


def _rated_items(candidates: List[Dict[str, Any]], quotes: Dict[str, Dict[str, Any]], tolerance: str, sentiment: str = "NEUTRAL") -> List[Dict[str, Any]]:
    ratings = []
    for item in candidates:
        quote = quotes.get(item.get("symbol") or "")
        if not quote:
            continue
        metrics = dict(quote.get("risk") or {})
        rating = rate_risk_adjusted(metrics, tolerance, sentiment)
        tier = metrics.get("tier") or "high"
        ratings.append({
            **item,
            "name": quote.get("long_name") or item.get("name"),
            "sector": item.get("sector") or quote.get("sector") or "Unknown",
            "rating": rating,
            "tier": tier,
            "cap": position_cap(tier, tolerance),
            "quote": quote,
        })
    return ratings


def risk_trending(state: FlowState) -> Dict[str, Any]:
    tolerance = state.get("risk_tolerance") or "Moderate"
    ratings = _rated_items(_unique_candidates(state), state.get("market_data") or {}, tolerance)
    return {
        "ratings": ratings,
        "agent_log": _note(f"Risk analyst: scored {len(ratings)} verified symbols"),
    }


def _weight_prompt(rows: List[str], feedback: str, extra: str) -> str:
    revision = f"\n\nPrevious plan was rejected:\n{feedback}\nRevise it." if feedback else ""
    return (
        "Choose weights for the verified instruments below. Weights are percentages. "
        "Stay inside each position cap. Do not use a symbol that is not listed. "
        "This is educational, not a forecast or a promise of returns.\n\n"
        + "\n".join(rows)
        + f"\n\n{extra}{revision}"
    )


def strategist_trending(state: FlowState) -> Dict[str, Any]:
    tolerance = state.get("risk_tolerance") or "Moderate"
    ratings = list(state.get("ratings") or [])
    allowed = [item for item in ratings if _num(item.get("cap")) > 0]
    proposed: Dict[str, float] = {}
    narrative = {"rationale": "No verified names were available to weight.", "risk_notes": [], "goal_alignment": "", "rebalancing": ""}
    if allowed:
        rows = []
        for item in allowed:
            metrics = (item.get("quote") or {}).get("risk") or {}
            rows.append(
                f"- {item['symbol']} | sector {item.get('sector')} | tier {item.get('tier')} | "
                f"cap {item.get('cap')}% | sharpe {metrics.get('sharpe')} | "
                f"1y {metrics.get('return_1y')}% | thesis {item.get('thesis', '')}"
            )
        sectors = {item["symbol"]: item.get("sector") or item["symbol"] for item in allowed}
        try:
            plan = get_llm(LLM_TEMPERATURE).with_structured_output(WeightPlan).invoke(
                _weight_prompt(
                    rows,
                    state.get("eval_feedback") or "",
                    f"Risk tolerance: {tolerance}. Spread the list across more than one name when several names qualify.",
                )
            )
            proposed = {choice.symbol: choice.weight_pct for choice in plan.holdings}
            narrative = plan.model_dump()
        except Exception as exc:
            narrative = {"rationale": "Weights were set from the position caps because the model call failed.", "risk_notes": [str(exc)], "goal_alignment": "", "rebalancing": ""}
        sector_names = {sectors[symbol] for symbol in sectors}
        weights = enforce_weights(
            {item["symbol"]: _num(item.get("cap")) for item in allowed},
            proposed,
            sectors,
            max_sector=45.0 if len(sector_names) >= 2 else None,
        )
    else:
        weights = {}
    return {
        "weights": weights,
        "portfolio": {"narrative": narrative, "allocation": weights},
        "agent_log": _note(f"Strategist: assigned weights to {len(weights)} symbols"),
    }


def _promise_hits(text: str) -> List[str]:
    lowered = (text or "").lower()
    return [phrase for phrase in PROMISES if phrase in lowered]


def _judge(packet: str) -> JudgeScore:
    prompt = (
        "Score this educational investment draft from 0 to 1 on three axes: "
        "suitability for the stated profile, grounding in the supplied numbers and URLs, "
        "and compliance (no guaranteed returns or hype). "
        "A claim of a recent catalyst without a URL in the packet must lower grounding. "
        "Reply with scores and one sentence of feedback.\n\n"
        + packet[:6000]
    )
    score = get_llm(0.0).with_structured_output(JudgeScore).invoke(prompt)
    score.suitability = min(1.0, max(0.0, _num(score.suitability)))
    score.grounding = min(1.0, max(0.0, _num(score.grounding)))
    score.compliance = min(1.0, max(0.0, _num(score.compliance)))
    return score


def _finish_eval(
    passed: bool,
    attempts: int,
    reasons: List[str],
    scores: Optional[JudgeScore],
    judge_unavailable: bool = False,
) -> Dict[str, Any]:
    forced = (not passed) and attempts >= MAX_REVISIONS
    feedback = " ".join(reason for reason in reasons if reason)
    if not feedback:
        feedback = "Checks passed." if passed or forced else "Revise the plan."
    logged: Dict[str, float] = {}
    if scores is not None:
        logged = {
            "suitability": scores.suitability,
            "grounding": scores.grounding,
            "compliance": scores.compliance,
            "eval_forced": 1.0 if forced else 0.0,
        }
        detail = (
            f"suitability {scores.suitability:.2f}, grounding {scores.grounding:.2f}, "
            f"compliance {scores.compliance:.2f}"
        )
    else:
        if judge_unavailable:
            logged["judge_unavailable"] = 1.0
        if forced:
            logged["eval_forced"] = 1.0
        detail = "judge unavailable" if judge_unavailable else "judge not called"
    label = "forced pass" if forced else ("pass" if passed else "revise")
    return {
        "eval_passed": passed or forced,
        "eval_forced": forced,
        "eval_attempts": attempts + 1,
        "eval_feedback": feedback,
        "eval_scores": logged,
        "agent_log": _note(f"Evaluator: {label} ({detail})"),
    }


def _apply_judge(reasons: List[str], packet: str) -> Optional[JudgeScore]:
    try:
        scores = _judge(packet)
    except Exception as exc:
        logger.warning("Judge unavailable: %s", exc)
        return None
    if min(scores.suitability, scores.grounding, scores.compliance) < 0.5:
        reasons.append(scores.feedback or "The judge rejected suitability, grounding, or compliance.")
    return scores


def evaluator_trending(state: FlowState) -> Dict[str, Any]:
    attempts = int(state.get("eval_attempts") or 0)
    ratings = {item["symbol"]: item for item in (state.get("ratings") or [])}
    quotes = state.get("market_data") or {}
    weights = state.get("weights") or {}
    reasons: List[str] = []
    if not ratings:
        reasons.append("No verified symbols were left to weight.")
    for symbol, weight in weights.items():
        item = ratings.get(symbol)
        if symbol not in quotes or item is None:
            reasons.append(f"{symbol} was not in the verified market data.")
            continue
        if weight - _num(item.get("cap")) > 0.2:
            reasons.append(f"{symbol} weight {weight}% is above its {item.get('cap')}% cap.")
    sector_totals: Dict[str, float] = {}
    for symbol, weight in weights.items():
        sector_totals[ratings.get(symbol, {}).get("sector") or symbol] = sector_totals.get(ratings.get(symbol, {}).get("sector") or symbol, 0.0) + _num(weight)
    if len(sector_totals) >= 2:
        for sector, total in sector_totals.items():
            if total > 45.2:
                reasons.append(f"{sector} is {total:.1f}% of the list, above the 45% concentration limit.")
    allowed = [item for item in ratings.values() if _num(item.get("cap")) > 0]
    if not allowed:
        return _finish_eval(True, attempts, ["No verified name is inside this risk profile."], None)
    if len(allowed) >= 4 and len(weights) < 3:
        reasons.append("Use at least three names when four or more pass the risk caps.")
    weight_sum = sum(_num(weight) for weight in weights.values())
    if weight_sum > 100.5:
        reasons.append(f"Weights add up to {weight_sum:.1f}%, above 100%.")
    narrative = ((state.get("portfolio") or {}).get("narrative") or {})
    blob = " ".join(str(narrative.get(key, "")) for key in ("rationale", "goal_alignment", "rebalancing"))
    hits = _promise_hits(blob)
    if hits:
        reasons.append("Remove promise language: " + ", ".join(hits))
    metric_lines = []
    for symbol, item in ratings.items():
        metrics = (item.get("quote") or {}).get("risk") or {}
        metric_lines.append(
            f"- {symbol} | profile tier {item.get('tier')} | cap {item.get('cap')}% | "
            f"weight {weights.get(symbol, 0)}% | sharpe {metrics.get('sharpe')} | "
            f"drawdown {metrics.get('max_drawdown')} | vol {metrics.get('ann_vol')} | "
            f"source {item.get('source_url') or 'none'}"
        )
    packet = (
        f"Risk tolerance: {state.get('risk_tolerance') or 'Moderate'}. "
        f"Sectors: {', '.join(state.get('sectors') or []) or 'n/a'}.\n"
        + "\n".join(metric_lines)
        + f"\nWeights: {weights}\nReasons so far: {reasons}\nNarrative: {blob}"
    )
    scores = _apply_judge(reasons, packet)
    return _finish_eval(not reasons, attempts, reasons, scores, judge_unavailable=scores is None)


def route_trending_eval(state: FlowState) -> str:
    if state.get("eval_passed", False):
        return "report_writer"
    return "strategist"


def _evaluator_section(state: FlowState) -> List[str]:
    if not state.get("eval_feedback") and not state.get("eval_forced"):
        return []
    lines = ["", "## Evaluator", ""]
    if state.get("eval_forced"):
        lines.append("Did not fully pass review.")
        lines.append("")
    if state.get("eval_feedback"):
        lines.append(state["eval_feedback"])
    return lines


def _rejection_lines(rejections: List[Dict[str, Any]]) -> List[str]:
    if not rejections:
        return ["None."]
    return [f"- **{item.get('symbol')}** ({item.get('name', '')}): {item.get('reason')}" for item in rejections]


def _quote_fields(item: Dict[str, Any], kind: str) -> Dict[str, Any]:
    quote = item.get("quote") or {}
    price = _num(quote.get("current_price"))
    price_display = f"${price:.4f}" if kind == "crypto" and price < 1 else f"${price:.2f}"
    return {
        "Symbol": item.get("symbol"),
        "Name": item.get("name"),
        "Price": price_display,
        "Type": {"stock": "Stock", "crypto": "Crypto", "etf": "ETF"}.get(kind, "Stock"),
        "Sector": item.get("sector") or quote.get("sector"),
        "1M Return": f"{_num(quote.get('1m_performance')):+.1f}%",
        "Rating": RATING_LABELS.get(item.get("rating"), item.get("rating")),
        "Market Cap": format_market_cap(_num(quote.get("market_cap"))),
        "P/E Ratio": quote.get("pe_ratio", "N/A"),
        "PEG Ratio": quote.get("peg_ratio", "N/A"),
        "ROE": quote.get("roe", "N/A"),
        "P/B Ratio": quote.get("pb_ratio", "N/A"),
        "D/E Ratio": quote.get("de_ratio", "N/A"),
    }


def report_trending(state: FlowState) -> Dict[str, Any]:
    research_type = state.get("research_type") or "Trending Stocks"
    kind = asset_kind(research_type)
    ratings = list(state.get("ratings") or [])
    weights = state.get("weights") or {}
    lines = [
        f"# {research_type}",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Sectors**: {', '.join(state.get('sectors') or []) or 'N/A'}  ",
        f"**Risk tolerance**: {state.get('risk_tolerance') or 'Moderate'}  ",
        f"**Symbols verified**: {len(ratings)}",
        "",
        "Ratings use volatility, drawdown, beta, Sharpe, leverage, and size. They are not a 1-month momentum rank, and they are not a forecast.",
        "",
        "## Trending names",
        "",
    ]
    summary_rows: List[Dict[str, Any]] = []
    stable_rows: List[Dict[str, Any]] = []

    def write_item(item: Dict[str, Any]) -> None:
        quote = item.get("quote") or {}
        metrics = quote.get("risk") or {}
        beta = metrics.get("beta")
        beta_text = "n/a" if beta is None else str(beta)
        lines.extend([
            f"### {item['symbol']} — {item.get('name', item['symbol'])}",
            (
                f"**Rating**: {item['rating']} · **Risk**: {item.get('tier')} "
                f"({metrics.get('score')}/10) · **Cap**: {item.get('cap')}% · "
                f"**Weight**: {weights.get(item['symbol'], 0)}%"
            ),
            (
                f"**Price**: ${quote.get('current_price', 'N/A')} · "
                f"**1M**: {_num(quote.get('1m_performance')):+.1f}% · "
                f"**1Y**: {_num(quote.get('1y_performance')):+.1f}%"
            ),
            (
                f"Volatility {metrics.get('ann_vol')} · max drawdown {metrics.get('max_drawdown')} · "
                f"Sharpe {metrics.get('sharpe')} · beta {beta_text}"
            ),
            "",
            item.get("thesis") or "",
            "",
            f"Catalyst: {item.get('catalyst') or 'Unverified.'}",
            f"Source: {item.get('source_url') or 'No source URL.'}",
            "",
        ])
        row = _quote_fields(item, item.get("kind") or kind)
        if item.get("rating") in {"Strong Buy", "Moderate Buy"}:
            summary_rows.append(_ordered_row(row, TRENDING_HEADERS))
        else:
            stable_rows.append(_ordered_row(row, STABLE_HEADERS))

    trending = [item for item in ratings if item.get("bucket") != "emerging"]
    emerging = [item for item in ratings if item.get("bucket") == "emerging"]
    if not trending:
        lines.append("No verified trending names were returned.")
        lines.append("")
    for item in trending:
        write_item(item)
    lines.extend(["## Emerging ideas", ""])
    if not emerging:
        lines.append("No verified emerging names were returned.")
        lines.append("")
    for item in emerging:
        write_item(item)
    lines.extend(["## Weights", ""])
    if not weights:
        lines.append("No weights were assigned.")
    for symbol, weight in weights.items():
        lines.append(f"- **{symbol}**: {weight}%")
    allocated = sum(_num(weight) for weight in weights.values())
    if weights and allocated < 99:
        lines.append(f"- **Unallocated**: {100 - allocated:.1f}% because a position cap or the 45% sector limit left no room.")
    lines.extend(["", "## Rejected by data checks", ""])
    lines.extend(_rejection_lines(list(state.get("rejections") or [])))
    lines.extend(_evaluator_section(state))
    return {
        "report": "\n".join(lines),
        "summary_rows": summary_rows,
        "stable_rows": stable_rows,
        "agent_log": _note("Report writer: assembled the trending report"),
    }


def build_trending_graph():
    graph = StateGraph(FlowState)
    graph.add_node("sector_researcher", sector_researcher)
    graph.add_node("emerging_scout", emerging_scout)
    graph.add_node("join_research", join_research)
    graph.add_node("market_data", market_data_trending)
    graph.add_node("risk_agent", risk_trending)
    graph.add_node("strategist", strategist_trending)
    graph.add_node("evaluator", evaluator_trending)
    graph.add_node("report_writer", report_trending)
    graph.add_conditional_edges(START, fan_trending, ["sector_researcher", "emerging_scout"])
    graph.add_edge("sector_researcher", "join_research")
    graph.add_edge("emerging_scout", "join_research")
    graph.add_edge("join_research", "market_data")
    graph.add_edge("market_data", "risk_agent")
    graph.add_edge("risk_agent", "strategist")
    graph.add_edge("strategist", "evaluator")
    graph.add_conditional_edges("evaluator", route_trending_eval, {"strategist": "strategist", "report_writer": "report_writer"})
    graph.add_edge("report_writer", END)
    return graph.compile()


def sleeves_for(tolerance: str) -> List[Dict[str, str]]:
    if tolerance == "Conservative":
        keys = {"bonds", "dividend_stocks", "reits", "cash"}
    elif tolerance == "Aggressive":
        keys = {"stocks", "growth_stocks", "reits", "bonds", "cash"}
    else:
        keys = {"bonds", "stocks", "dividend_stocks", "reits", "cash"}
    return [item for item in SLEEVES if item["key"] in keys]


def market_data_portfolio(state: FlowState) -> Dict[str, Any]:
    sleeves = sleeves_for(state.get("risk_tolerance") or "Moderate")
    candidates = [
        {"symbol": item["symbol"], "name": item["name"], "sector": item["label"], "bucket": "sleeve", "kind": "etf", "sleeve": item["key"]}
        for item in sleeves
    ]
    verified = verify_candidates(candidates, "etf")
    return {
        "market_data": verified["quotes"],
        "rejections": verified["rejections"],
        "rejection_rate": verified["rejection_rate"],
        "discovered": candidates,
        "agent_log": _note(f"Market data: priced {len(verified['quotes'])} portfolio ETFs"),
    }


def risk_portfolio(state: FlowState) -> Dict[str, Any]:
    tolerance = state.get("risk_tolerance") or "Moderate"
    horizon = state.get("time_horizon") or "Medium-term"
    quotes = state.get("market_data") or {}
    ratings = []
    for item in sleeves_for(tolerance):
        quote = quotes.get(item["symbol"])
        if not quote:
            continue
        metrics = dict(quote.get("risk") or {})
        tier = metrics.get("tier") or "high"
        cap = position_cap(tier, tolerance, horizon, item["key"])
        if item["key"] == "cash":
            cap = 100.0
        ratings.append({
            "symbol": item["symbol"],
            "name": quote.get("long_name") or item["name"],
            "sector": item["label"],
            "sleeve": item["key"],
            "label": item["label"],
            "bucket": "sleeve",
            "kind": "etf",
            "rating": rate_risk_adjusted(metrics, tolerance),
            "tier": tier,
            "cap": cap,
            "quote": quote,
            "thesis": f"Liquid {item['label'].lower()} sleeve.",
            "catalyst": "Verified market data.",
        })
    return {"ratings": ratings, "agent_log": _note(f"Risk analyst: scored {len(ratings)} sleeves")}


def strategist_portfolio(state: FlowState) -> Dict[str, Any]:
    tolerance = state.get("risk_tolerance") or "Moderate"
    horizon = state.get("time_horizon") or "Medium-term"
    budget = _num(state.get("budget"))
    goals = state.get("goals") or "General long-term growth"
    ratings = [item for item in (state.get("ratings") or []) if _num(item.get("cap")) > 0]
    rows = []
    for item in ratings:
        metrics = (item.get("quote") or {}).get("risk") or {}
        rows.append(
            f"- {item['symbol']} ({item['label']}) | tier {item['tier']} | cap {item['cap']}% | "
            f"trailing 1y {_num((item.get('quote') or {}).get('1y_performance'))}% | sharpe {metrics.get('sharpe')}"
        )
    proposed: Dict[str, float] = {}
    narrative: Dict[str, Any] = {}
    try:
        plan = get_llm(LLM_TEMPERATURE).with_structured_output(WeightPlan).invoke(
            _weight_prompt(
                rows,
                state.get("eval_feedback") or "",
                (
                    f"Profile: {tolerance}, {horizon}. Budget: ${budget:,.0f}. Goals: {goals}. "
                    f"Keep at least {MIN_CASH.get(horizon, 5)}% in BIL. "
                    "A short horizon should not be heavy in higher-risk sleeves. "
                    "Describe the trailing return as history, not as an expected future return."
                ),
            )
        )
        proposed = {choice.symbol: choice.weight_pct for choice in plan.holdings}
        narrative = plan.model_dump()
    except Exception as exc:
        narrative = {
            "rationale": "Weights were set from the sleeve caps because the model call failed.",
            "risk_notes": [str(exc)],
            "goal_alignment": goals,
            "rebalancing": "Review the mix about once a quarter.",
        }
    cash_floor = MIN_CASH.get(horizon, 5.0)
    weights = enforce_weights(
        {item["symbol"]: _num(item.get("cap")) for item in ratings},
        proposed,
        {item["symbol"]: item["label"] for item in ratings},
        max_sector=None,
        cash_symbol="BIL" if any(item["symbol"] == "BIL" for item in ratings) else None,
        min_cash=cash_floor if any(item["symbol"] == "BIL" for item in ratings) else 0.0,
    )
    trailing_values = []
    trailing = 0.0
    for symbol, weight in weights.items():
        perf = _num(((state.get("market_data") or {}).get(symbol) or {}).get("1y_performance"))
        trailing += (weight / 100.0) * perf
        trailing_values.append(perf)
    trailing = round(trailing, 2)
    portfolio = {
        "budget": budget,
        "allocation": weights,
        "expected_annual_return": trailing,
        "return_basis": "trailing 1-year weighted return",
        "return_range": [round(min(trailing_values), 2), round(max(trailing_values), 2)] if trailing_values else [0, 0],
        "projected_value_5y": round(budget * ((1 + trailing / 100.0) ** 5), 2) if budget else 0,
        "narrative": narrative,
    }
    return {
        "weights": weights,
        "portfolio": portfolio,
        "agent_log": _note(f"Strategist: trailing 1-year mix return {trailing}%"),
    }


def evaluator_portfolio(state: FlowState) -> Dict[str, Any]:
    attempts = int(state.get("eval_attempts") or 0)
    horizon = state.get("time_horizon") or "Medium-term"
    ratings = {item["symbol"]: item for item in (state.get("ratings") or [])}
    quotes = state.get("market_data") or {}
    weights = dict(state.get("weights") or {})
    reasons: List[str] = []
    if not weights:
        reasons.append("The plan has no verified holdings.")
    total = round(sum(_num(weight) for weight in weights.values()), 2)
    if weights and abs(total - 100.0) > 1.0:
        reasons.append(f"Weights add up to {total}%, not 100%.")
    cash_floor = MIN_CASH.get(horizon, 5.0)
    if "BIL" in ratings and _num(weights.get("BIL")) + 0.2 < cash_floor:
        reasons.append(f"Cash (BIL) is below the {cash_floor}% minimum for a {horizon.lower()} horizon.")
    for symbol, weight in weights.items():
        item = ratings.get(symbol)
        if symbol not in quotes or item is None:
            reasons.append(f"{symbol} was not verified.")
            continue
        if _num(weight) - _num(item.get("cap")) > 0.2:
            reasons.append(f"{symbol} is above its {item.get('cap')}% cap.")
        if horizon == "Short-term" and item.get("tier") in {"high", "speculative"} and _num(weight) > 10.2:
            reasons.append(f"{symbol} is a {item.get('tier')} sleeve above 10% on a short horizon.")
    if len([item for item in ratings.values() if _num(item.get("cap")) > 0]) >= 3 and len(weights) < 2:
        reasons.append("Use at least two sleeves.")
    narrative = ((state.get("portfolio") or {}).get("narrative") or {})
    blob = " ".join(str(narrative.get(key, "")) for key in ("rationale", "goal_alignment", "rebalancing"))
    hits = _promise_hits(blob)
    if hits:
        reasons.append("Remove promise language: " + ", ".join(hits))
    metric_lines = []
    for symbol, item in ratings.items():
        metrics = (item.get("quote") or {}).get("risk") or {}
        metric_lines.append(
            f"- {symbol} ({item.get('label')}) sharpe {metrics.get('sharpe')} "
            f"drawdown {metrics.get('max_drawdown')} vol {metrics.get('ann_vol')} cap {item.get('cap')}%"
        )
    packet = (
        f"Profile {state.get('risk_tolerance')} {horizon}. Budget context is educational.\n"
        + "\n".join(metric_lines)
        + f"\nWeights {weights}\nNarrative {blob}\nIssues {reasons}"
    )
    scores = _apply_judge(reasons, packet)
    return _finish_eval(not reasons, attempts, reasons, scores, judge_unavailable=scores is None)


def route_portfolio_eval(state: FlowState) -> str:
    if state.get("eval_passed", False):
        return "report_writer"
    return "strategist"


def report_portfolio(state: FlowState) -> Dict[str, Any]:
    portfolio = state.get("portfolio") or {}
    narrative = portfolio.get("narrative") or {}
    weights = portfolio.get("allocation") or {}
    budget = _num(portfolio.get("budget"))
    ratings = {item["symbol"]: item for item in (state.get("ratings") or [])}
    low, high = (portfolio.get("return_range") or [0, 0])[:2]
    lines = [
        "# Portfolio strategy",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Profile**: {state.get('risk_tolerance')} risk, {state.get('time_horizon')} horizon  ",
        f"**Budget**: ${budget:,.2f}  ",
        f"**Trailing 1-year return of this mix**: {portfolio.get('expected_annual_return', 0)}%  ",
        f"**Trailing sleeve range**: {low}% to {high}%  ",
        f"**Illustrative 5-year value if that trailing return repeated**: ${_num(portfolio.get('projected_value_5y')):,.2f}",
        "",
        "The trailing figure is history. It is not a forecast, and the five-year number only shows compounding of that history.",
        "",
        narrative.get("rationale") or narrative.get("summary") or "",
        "",
        "## Allocation",
        "",
    ]
    summary_rows = []
    for symbol, weight in weights.items():
        item = ratings.get(symbol) or SLEEVE_BY_SYMBOL.get(symbol) or {}
        label = item.get("label") or item.get("name") or symbol
        tier = str(item.get("tier") or "medium").title()
        amount = budget * _num(weight) / 100.0
        lines.append(f"- **{label} ({symbol})**: {weight}% (${amount:,.0f})")
        summary_rows.append({
            "Asset Class": f"{label} ({symbol})",
            "Allocation": f"{weight}%",
            "Amount": f"${amount:,.0f}",
            "Risk Level": tier,
        })
    lines.extend(["", "## Goals", "", narrative.get("goal_alignment") or state.get("goals") or "", "", "## Risk notes"])
    notes = narrative.get("risk_notes") or ["Review this mix against your own constraints."]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Rebalancing", "", narrative.get("rebalancing") or "Review about once a quarter."])
    lines.extend(["", "## Rejected by data checks", ""])
    lines.extend(_rejection_lines(list(state.get("rejections") or [])))
    lines.extend(_evaluator_section(state))
    return {
        "report": "\n".join(lines),
        "summary_rows": summary_rows,
        "stable_rows": [],
        "agent_log": _note("Report writer: assembled the portfolio strategy"),
    }


def build_portfolio_graph():
    graph = StateGraph(FlowState)
    graph.add_node("market_data", market_data_portfolio)
    graph.add_node("risk_agent", risk_portfolio)
    graph.add_node("strategist", strategist_portfolio)
    graph.add_node("evaluator", evaluator_portfolio)
    graph.add_node("report_writer", report_portfolio)
    graph.add_edge(START, "market_data")
    graph.add_edge("market_data", "risk_agent")
    graph.add_edge("risk_agent", "strategist")
    graph.add_edge("strategist", "evaluator")
    graph.add_conditional_edges("evaluator", route_portfolio_eval, {"strategist": "strategist", "report_writer": "report_writer"})
    graph.add_edge("report_writer", END)
    return graph.compile()


def market_data_individual(state: FlowState) -> Dict[str, Any]:
    symbol = clean_symbol(state.get("symbol") or "", "stock")
    if not symbol:
        return {
            "symbol": "",
            "market_data": {},
            "rejections": [{"symbol": state.get("symbol") or "", "name": "", "reason": "symbol could not be parsed"}],
            "rejection_rate": 1.0,
            "agent_log": _note("Market data: no usable symbol"),
        }
    verified = verify_candidates([{"symbol": symbol, "name": symbol, "sector": "Equity"}], "stock")
    return {
        "symbol": symbol,
        "market_data": verified["quotes"],
        "rejections": verified["rejections"],
        "rejection_rate": verified["rejection_rate"],
        "agent_log": _note(f"Market data: {'accepted' if symbol in verified['quotes'] else 'rejected'} {symbol}"),
    }


def route_after_individual_market(state: FlowState):
    symbol = state.get("symbol") or ""
    if (state.get("market_data") or {}).get(symbol):
        payload = {
            "symbol": symbol,
            "market_data": state.get("market_data") or {},
            "risk_tolerance": state.get("risk_tolerance") or "Moderate",
            "eval_feedback": state.get("eval_feedback") or "",
        }
        return [Send("company_analyst", payload), Send("sentiment_analyst", payload)]
    return "report_writer"


def _profile_prompt(symbol: str, quote: Dict[str, Any], notes: str, feedback: str) -> str:
    revision = f"\n\nFix this review note:\n{feedback}" if feedback else ""
    return (
        f"{_source_rule(notes)}\n\n"
        f"Profile {symbol}. Live price: {quote.get('current_price', 'unknown')}. "
        f"Listed name: {quote.get('long_name') or symbol}. Sector: {quote.get('sector', 'unknown')}. "
        f"Market cap: {format_market_cap(_num(quote.get('market_cap')))}.\n\n"
        f"Notes:\n{notes or 'None.'}{revision}"
    )


def company_analyst(state: FlowState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    quote = (state.get("market_data") or {}).get(symbol, {})
    try:
        filings = company_filings(symbol)
        headlines = search_headlines(symbol)
        notes = _tool_notes(f"Company facts and filings for {symbol}.", "\n\n".join(part for part in (headlines, filings) if part))
        profile = get_llm(LLM_TEMPERATURE).with_structured_output(CompanyProfile).invoke(
            _profile_prompt(symbol, quote, notes, state.get("eval_feedback") or "")
        )
        data = profile.model_dump()
        if "http" not in notes.lower():
            data["sourced_developments"] = "No sourced developments. Treat nothing here as current news."
        return {"company": data, "agent_log": _note(f"Company analyst: profiled {symbol}")}
    except Exception as exc:
        return {
            "company": {"company_name": quote.get("long_name") or symbol, "sector": quote.get("sector", "Unknown"), "sourced_developments": "No sourced developments."},
            "agent_log": _note("Company analyst: failed"),
            "errors": _fail(f"Company analyst: {exc}"),
        }


def sentiment_analyst(state: FlowState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    quote = (state.get("market_data") or {}).get(symbol, {})
    metrics = quote.get("risk") or {}
    try:
        headlines = search_headlines(f"{symbol} stock")
        notes = _tool_notes(f"Recent traded news for {symbol}, only if the tools return it.", headlines if "No headlines" not in headlines else "")
        found = "http" in notes.lower()
        prompt = (
            f"{_source_rule(notes)}\n\n"
            f"Assess {symbol}. Price {quote.get('current_price')}. "
            f"1-month {quote.get('1m_performance')}%. 1-year {quote.get('1y_performance')}%. "
            f"Sharpe {metrics.get('sharpe')}. Max drawdown {metrics.get('max_drawdown')}. "
            "Score sentiment from -5 to 5. Each development must include a URL from the notes when one exists.\n\n"
            f"Notes:\n{notes or 'None.'}\n\n{state.get('eval_feedback') or ''}"
        )
        sentiment = get_llm(LLM_TEMPERATURE).with_structured_output(SentimentProfile).invoke(prompt)
        data = sentiment.model_dump()
        data["sentiment_score"] = max(-5.0, min(5.0, _num(data.get("sentiment_score"))))
        if not found:
            data["developments"] = ["No sourced headlines. Catalysts are unverified."]
        else:
            cleaned = []
            for item in data.get("developments") or []:
                if "http" in item.lower() or "recent" not in item.lower():
                    cleaned.append(item)
            data["developments"] = cleaned or ["No development in the notes included a usable URL."]
        return {
            "sentiment": data,
            "headlines_found": found,
            "agent_log": _note(f"Sentiment analyst: {data['overall_sentiment']} ({data['sentiment_score']:+.1f})"),
        }
    except Exception as exc:
        return {
            "sentiment": {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 0,
                "developments": ["No sourced headlines. Catalysts are unverified."],
                "risk_factors": ["Market volatility"],
                "growth_catalysts": [],
                "analyst_outlook": "Sentiment analysis was unavailable.",
            },
            "headlines_found": False,
            "agent_log": _note("Sentiment analyst: failed"),
            "errors": _fail(f"Sentiment analyst: {exc}"),
        }


def revise_research(state: FlowState) -> Dict[str, Any]:
    """One revision node so the parallel join does not have to run again."""
    company_update = company_analyst(state)
    sentiment_update = sentiment_analyst(state)
    merged = {**company_update, **sentiment_update}
    logs = list(company_update.get("agent_log") or []) + list(sentiment_update.get("agent_log") or [])
    errors = list(company_update.get("errors") or []) + list(sentiment_update.get("errors") or [])
    merged["agent_log"] = logs
    if errors:
        merged["errors"] = errors
    return merged


def risk_individual(state: FlowState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    quote = (state.get("market_data") or {}).get(symbol) or {}
    sentiment = (state.get("sentiment") or {}).get("overall_sentiment", "NEUTRAL")
    metrics = dict(quote.get("risk") or {})
    tolerance = state.get("risk_tolerance") or "Moderate"
    rating = rate_risk_adjusted(metrics, tolerance, sentiment)
    tier = metrics.get("tier") or "high"
    ratings = [{
        "symbol": symbol,
        "name": quote.get("long_name") or (state.get("company") or {}).get("company_name") or symbol,
        "sector": quote.get("sector") or "Unknown",
        "rating": rating,
        "tier": tier,
        "cap": position_cap(tier, tolerance),
        "bucket": "individual",
        "kind": "stock",
        "quote": quote,
    }]
    return {"ratings": ratings, "agent_log": _note(f"Risk analyst: rated {symbol} {rating} ({tier})")}


def evaluator_individual(state: FlowState) -> Dict[str, Any]:
    attempts = int(state.get("eval_attempts") or 0)
    symbol = state.get("symbol") or ""
    sentiment = state.get("sentiment") or {}
    company = state.get("company") or {}
    reasons: List[str] = []
    if symbol and symbol not in (state.get("market_data") or {}):
        reasons.append(f"{symbol} did not pass market-data checks.")
    developments = [str(item) for item in (sentiment.get("developments") or [])]
    if not state.get("headlines_found"):
        if any("recent" in item.lower() for item in developments):
            reasons.append("A sentiment note calls something recent, but no headline tool returned an article.")
    else:
        for item in developments:
            if "recent" in item.lower() and "http" not in item.lower():
                reasons.append("A development is described as recent without a source URL.")
                break
    text = " ".join(
        [str(company.get("sourced_developments") or "")]
        + developments
        + [str(sentiment.get("analyst_outlook") or "")]
    )
    hits = _promise_hits(text)
    if hits:
        reasons.append("Remove promise language: " + ", ".join(hits))
    scores = None
    judge_unavailable = False
    if symbol in (state.get("market_data") or {}):
        scores = _apply_judge(reasons, f"Symbol {symbol}. Company {company}. Sentiment {sentiment}. Issues {reasons}")
        judge_unavailable = scores is None
    no_quote = symbol not in (state.get("market_data") or {})
    return _finish_eval(no_quote or not reasons, attempts, reasons, scores, judge_unavailable=judge_unavailable)


def route_individual_eval(state: FlowState) -> str:
    if state.get("eval_passed", False):
        return "report_writer"
    return "revise_research"


def report_individual(state: FlowState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    company = state.get("company") or {}
    sentiment = state.get("sentiment") or {}
    quote = (state.get("market_data") or {}).get(symbol, {})
    rating_row = (state.get("ratings") or [{}])[0]
    rating = rating_row.get("rating", "Hold")
    metrics = quote.get("risk") or {}
    score = _num(sentiment.get("sentiment_score"))
    developments = sentiment.get("developments") or ["No sourced developments."]
    catalysts = sentiment.get("growth_catalysts") or ["No catalysts returned."]
    risks = sentiment.get("risk_factors") or ["Standard market risk."]
    if not quote:
        lines = [
            f"# {symbol or 'Symbol'} research",
            "",
            "This symbol did not pass the market-data checks, so it is not rated.",
            "",
            "## Rejected by data checks",
            "",
            *_rejection_lines(list(state.get("rejections") or [])),
        ]
        return {"report": "\n".join(lines), "summary_rows": [], "stable_rows": [], "agent_log": _note("Report writer: rejection only")}
    beta = metrics.get("beta")
    lines = [
        f"# {symbol} research",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Company**: {company.get('company_name', quote.get('long_name') or symbol)}  ",
        f"**Sector**: {company.get('sector') or quote.get('sector', 'Unknown')}  ",
        f"**Price**: ${quote.get('current_price', 0)} · **1M**: {_num(quote.get('1m_performance')):+.1f}% · **1Y**: {_num(quote.get('1y_performance')):+.1f}%",
        f"**Market cap**: {format_market_cap(_num(quote.get('market_cap')))}  ",
        f"**Sentiment**: {sentiment.get('overall_sentiment', 'NEUTRAL')} ({score:+.1f}/5)  ",
        f"**Rating**: {rating} · **Risk**: {rating_row.get('tier', metrics.get('tier'))} ({metrics.get('score')}/10)",
        f"**Volatility**: {metrics.get('ann_vol')} · **Max drawdown**: {metrics.get('max_drawdown')} · **Sharpe**: {metrics.get('sharpe')} · **Beta**: {'n/a' if beta is None else beta}",
        "",
        "## Company",
        "",
        company.get("business_description") or "Company profile was unavailable.",
        "",
        f"Founded: {company.get('founded_year', 'N/A')}. {company.get('competitive_advantages', '')}",
        "",
        f"Sourced developments: {company.get('sourced_developments', 'No sourced developments.')}",
        "",
        "## Sentiment",
        "",
        "### Developments",
    ]
    lines.extend(f"- {item}" for item in developments[:4])
    lines.extend(["", "### Catalysts"])
    lines.extend(f"- {item}" for item in catalysts[:4])
    lines.extend(["", "### Risks"])
    lines.extend(f"- {item}" for item in risks[:4])
    lines.extend(["", "## Outlook", "", sentiment.get("analyst_outlook") or "Outlook was unavailable.", "", "## Rejected by data checks", ""])
    lines.extend(_rejection_lines(list(state.get("rejections") or [])))
    lines.extend(_evaluator_section(state))
    return {
        "report": "\n".join(lines),
        "summary_rows": [],
        "stable_rows": [],
        "agent_log": _note("Report writer: assembled the company report"),
    }


def build_individual_graph():
    graph = StateGraph(FlowState)
    graph.add_node("market_data", market_data_individual)
    graph.add_node("company_analyst", company_analyst)
    graph.add_node("sentiment_analyst", sentiment_analyst)
    graph.add_node("revise_research", revise_research)
    graph.add_node("risk_agent", risk_individual)
    graph.add_node("evaluator", evaluator_individual)
    graph.add_node("report_writer", report_individual)
    graph.add_edge(START, "market_data")
    graph.add_conditional_edges(
        "market_data",
        route_after_individual_market,
        ["company_analyst", "sentiment_analyst", "report_writer"],
    )
    graph.add_edge("company_analyst", "risk_agent")
    graph.add_edge("sentiment_analyst", "risk_agent")
    graph.add_edge("revise_research", "risk_agent")
    graph.add_edge("risk_agent", "evaluator")
    graph.add_conditional_edges("evaluator", route_individual_eval, {"revise_research": "revise_research", "report_writer": "report_writer"})
    graph.add_edge("report_writer", END)
    return graph.compile()


def router_node(state: AdvisorState) -> Dict[str, Any]:
    task = state.get("task", "trending")
    return {"agent_log": [f"Router: dispatched the {task} workflow"]}


def route_start(state: AdvisorState) -> str:
    task = state.get("task")
    if task == "portfolio":
        return "portfolio_flow"
    if task == "individual":
        return "individual_flow"
    return "trending_flow"


def build_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("router", router_node)
    graph.add_node("trending_flow", build_trending_graph())
    graph.add_node("individual_flow", build_individual_graph())
    graph.add_node("portfolio_flow", build_portfolio_graph())
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_start,
        {
            "trending_flow": "trending_flow",
            "individual_flow": "individual_flow",
            "portfolio_flow": "portfolio_flow",
        },
    )
    graph.add_edge("trending_flow", END)
    graph.add_edge("individual_flow", END)
    graph.add_edge("portfolio_flow", END)
    return graph.compile()


ADVISOR = build_graph()


def _blank_state(**kwargs: Any) -> AdvisorState:
    state: AdvisorState = {
        "task": "trending",
        "research_type": "Trending Stocks",
        "sectors": ["Technology"],
        "symbol": "",
        "budget": 10000,
        "time_horizon": "Medium-term",
        "risk_tolerance": "Moderate",
        "goals": "",
        "discovered": [],
        "emerging": [],
        "market_data": {},
        "company": {},
        "sentiment": {},
        "ratings": [],
        "weights": {},
        "portfolio": {},
        "report": "",
        "summary_rows": [],
        "stable_rows": [],
        "rejections": [],
        "rejection_rate": 0.0,
        "agent_log": [],
        "errors": [],
        "eval_passed": False,
        "eval_forced": False,
        "eval_attempts": 0,
        "eval_feedback": "",
        "eval_scores": {},
        "headlines_found": False,
        "trace_url": "",
    }
    state.update(kwargs)
    return state


def load_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Price one symbol for the quote button. This does not run the research graph."""
    cleaned = clean_symbol(symbol, "stock")
    if not cleaned:
        return None
    verified = verify_candidates([{"symbol": cleaned, "name": cleaned, "sector": "Equity"}], "stock")
    quote = (verified.get("quotes") or {}).get(cleaned)
    if quote:
        return quote
    # A quote can still show a live price when the research checks reject the name.
    ticker = yf_symbol(cleaned, "stock")
    try:
        history = yf.download(ticker, period="1y", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return None
    close = _history_frame(history, ticker, "Close")
    if close.empty:
        close = pd.to_numeric(history.get("Close"), errors="coerce").dropna() if history is not None and not history.empty else pd.Series(dtype=float)
    if close.empty:
        return None
    _, info = _load_info(ticker)
    high = _history_frame(history, ticker, "High")
    low = _history_frame(history, ticker, "Low")
    volume = _history_frame(history, ticker, "Volume")
    if not high.empty:
        info["_high"] = _num(high.max())
    if not low.empty:
        info["_low"] = _num(low.min())
    quote = _metrics_from_history(close, volume, pd.Series(dtype=float), info, "stock")
    quote["verification"] = (verified.get("rejections") or [{}])[0].get("reason", "")
    return quote


def _log_feedback(result: AdvisorState, run) -> None:
    if not TRACING_ENABLED or run is None:
        return
    scores = dict(result.get("eval_scores") or {})
    # A quote inherits 0.0 from the blank state and would pull the average toward zero.
    if result.get("task") != "quote" and result.get("rejection_rate") is not None:
        scores["rejection_rate"] = _num(result.get("rejection_rate"))
    if not scores:
        return
    try:
        client = Client()
        run_id = getattr(run, "id", None)
        trace_id = getattr(run, "trace_id", None) or run_id
        for key, score in scores.items():
            client.create_feedback(run_id, key=key, score=_num(score), trace_id=trace_id)
    except Exception as exc:
        logger.warning("LangSmith feedback was not saved: %s", exc)


@traceable(name="investment_advisor", run_type="chain", tags=["langgraph", "investment-advisor"])
def run_advisor(state: AdvisorState) -> AdvisorState:
    if state.get("task") == "quote":
        symbol = clean_symbol(state.get("symbol") or "", "stock")
        quote = load_quote(symbol) if symbol else None
        result = dict(state)
        result["symbol"] = symbol
        result["market_data"] = {symbol: quote} if quote else {}
        result["agent_log"] = list(result.get("agent_log") or []) + ["Quote: loaded market data directly"]
        result["report"] = ""
        _attach_trace(result)
        return result

    config: Dict[str, Any] = {
        "run_name": f"advisor-{state.get('task', 'run')}",
        "tags": ["investment-advisor", state.get("task", "run")],
        "metadata": {
            "task": state.get("task"),
            "research_type": state.get("research_type"),
            "symbol": state.get("symbol"),
            "sectors": state.get("sectors"),
            "risk_tolerance": state.get("risk_tolerance"),
        },
    }
    result = ADVISOR.invoke(state, config=config)
    _attach_trace(result)
    _persist(result)
    return result


def _attach_trace(result: AdvisorState) -> None:
    run = get_current_run_tree()
    trace_url = ""
    if run is not None:
        try:
            trace_url = run.get_url() or ""
        except Exception:
            trace_url = ""
    result["trace_url"] = trace_url
    if trace_url and result.get("report"):
        result["report"] = result["report"].rstrip() + f"\n\n---\n\n[LangSmith trace]({trace_url})\n"
    _log_feedback(result, run)


def _persist(result: AdvisorState) -> None:
    task = result.get("task")
    report = result.get("report") or ""
    if not report or task == "quote":
        return
    try:
        if task == "trending":
            sectors = "+".join(result.get("sectors") or ["All"])
            save_research(report, f"LangGraph_{result.get('research_type')}_{sectors}")
        elif task == "individual":
            save_research(report, f"LangGraph_Stock_{result.get('symbol')}")
        elif task == "portfolio":
            portfolio = result.get("portfolio") or {}
            save_portfolio(
                report,
                float(portfolio.get("budget") or 0),
                result.get("risk_tolerance") or "",
                result.get("time_horizon") or "",
            )
    except Exception as exc:
        logger.warning("Could not save report: %s", exc)


# ================================
# UI
# ================================

CUSTOM_CSS = """

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
    color: #e0e6ed !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 3rem 2rem !important;
    border-radius: 20px !important;
    margin: 2rem 0 !important;
}

.main-header h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    letter-spacing: -0.02em !important;
}

.main-header h2 {
    color: #e0e6ed !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
    margin-bottom: 1rem !important;
}

.main-header p {
    color: #e0e6ed !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
    max-width: 600px !important;
    margin: 0 auto 2rem !important;
}

.tabs button {
    color: white !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.section-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}

.metric-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%) !important;
    backdrop-filter: blur(10px) !important;
    color: white !important;
    padding: 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    margin: 0.5rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.success-msg {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.9) 0%, rgba(168, 230, 207, 0.9) 100%) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
}

.warning-msg {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(245, 87, 108, 0.9) 100%) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
}

.markdown-content {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    line-height: 1.6 !important;
    height: 500px !important;
    overflow-y: auto !important;
}

/* FIX: Ensure all markdown elements have proper dark colors on light background */
.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
    color: #1e293b !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}

.markdown-content p {
    color: #374151 !important;
    margin-bottom: 1rem !important;
    line-height: 1.6 !important;
}

.markdown-content ul,
.markdown-content ol {
    color: #374151 !important;
    margin-bottom: 1rem !important;
    padding-left: 1.5rem !important;
}

.markdown-content li {
    color: #374151 !important;
    margin-bottom: 0.25rem !important;
}

.markdown-content strong {
    color: #1e293b !important;
    font-weight: 600 !important;
}

.markdown-content em {
    color: #374151 !important;
    font-style: italic !important;
}

.markdown-content a {
    color: #2563eb !important;
    text-decoration: underline !important;
}

.markdown-content code {
    background: rgba(0, 0, 0, 0.1) !important;
    color: #1e293b !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
}

.markdown-content pre {
    background: rgba(0, 0, 0, 0.05) !important;
    color: #1e293b !important;
    padding: 1rem !important;
    border-radius: 6px !important;
    overflow-x: auto !important;
    margin: 1rem 0 !important;
}

.markdown-content blockquote {
    border-left: 4px solid #667eea !important;
    padding-left: 1rem !important;
    margin: 1rem 0 !important;
    color: #4b5563 !important;
    font-style: italic !important;
}

.markdown-content hr {
    border: none !important;
    height: 1px !important;
    background: rgba(0, 0, 0, 0.1) !important;
    margin: 2rem 0 !important;
}

.primary-btn,
button.primary {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%) !important;
    color: white !important;
    padding: 0.875rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

::placeholder {
    color: #9ca3af !important;
}

/* Tables inside markdown content (reports) - dark text on light background */
.markdown-content table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    background: white !important;
}

.markdown-content table th {
    background: rgba(102, 126, 234, 0.1) !important;
    color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: none !important;
    -webkit-text-fill-color: #000000 !important;
    padding: 0.75rem !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    text-align: left !important;
}

.markdown-content table td {
    color: #1f2937 !important;
    text-shadow: none !important;
    -webkit-text-fill-color: #1f2937 !important;
    padding: 0.75rem !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
}

.markdown-content table tr:nth-child(even) {
    background: rgba(0, 0, 0, 0.02) !important;
}

/* Gradio dataframe tables - light text on dark background for dark theme */
.gradio-dataframe {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

.gradio-dataframe table {
    background: transparent !important;
    color: #e0e6ed !important;
    width: 100% !important;
}

.gradio-dataframe th {
    background: rgba(102, 126, 234, 0.2) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 0.75rem !important;
    text-align: left !important;
}

.gradio-dataframe td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: #e0e6ed !important;
    padding: 0.75rem !important;
}

.gradio-dataframe tr:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}

"""

def convert_markdown_to_html(markdown_text):
    """Convert markdown text to styled HTML"""
    if not markdown_text or markdown_text.startswith("❌"):
        return create_placeholder_html("Report", "Please generate a report to see AI results here")

    try:
        html_content = markdown.markdown(markdown_text, extensions=['nl2br', 'tables'])
        return f"<div class='markdown-content'>{html_content}</div>"
    except Exception as e:
        return create_placeholder_html("Report", f"Error rendering report: {str(e)}")


def create_placeholder_html(title, description):
    """Create placeholder HTML for empty reports"""
    return f"""
    <div class='markdown-content' style='display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 300px;'>
        <div style='font-weight: 600; font-size: 1.1rem; color: #374151; margin-bottom: 0.5rem;'>{title}</div>
        <div style='color: #6b7280;'>{description}</div>
    </div>
    """


def get_research_history():
    """Get research history from database"""
    try:
        conn = sqlite3.connect('investment_advisor.db')

        stock_query = """
        SELECT id, report_date, report_type,
               SUBSTR(report_content, 1, 100) || '...' as preview
        FROM stock_research
        ORDER BY report_date DESC
        LIMIT 20
        """
        stock_df = pd.read_sql_query(stock_query, conn)

        portfolio_query = """
        SELECT id, strategy_date, budget, risk_tolerance, time_horizon,
               SUBSTR(strategy_content, 1, 100) || '...' as preview
        FROM portfolio_strategies
        ORDER BY strategy_date DESC
        LIMIT 20
        """
        portfolio_df = pd.read_sql_query(portfolio_query, conn)

        conn.close()
        return stock_df, portfolio_df

    except Exception as e:
        print(f"Database error: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_full_report_by_id(report_id, report_type):
    """Get full report content by ID"""
    try:
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()

        if report_type == "stock":
            cursor.execute("SELECT report_content FROM stock_research WHERE id = ?", (report_id,))
        else:
            cursor.execute("SELECT strategy_content FROM portfolio_strategies WHERE id = ?", (report_id,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else "Report not found"

    except Exception as e:
        return f"Error retrieving report: {str(e)}"


def send_email_report(email, content, content_type="portfolio"):
    """Send email with strategy/research content"""
    try:
        if not email or "@" not in email:
            return "Please provide a valid email address"

        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')

        if not all([sender_email, sender_password]):
            return "Email credentials not configured. Set SENDER_EMAIL and SENDER_PASSWORD environment variables."

        subject = f"Your {content_type.title()} Report - {datetime.now().strftime('%Y-%m-%d')}"

        msg = MIMEText(content, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = email

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return f"{content_type.title()} report sent successfully to {email}!"

    except Exception as e:
        return f"Failed to send email: {str(e)}"


def _quote_html(symbol, data):
    performance_1m_color = "#22c55e" if data['1m_performance'] > 0 else "#ef4444"
    return f"""
            <div class="section-card">
                <h4 style="color: #d1d5db; margin-bottom: 1.5rem;">{symbol.upper()} - Live Quote</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div class="metric-card">
                        <h5>Current Price</h5>
                        <p style="font-size: 1.5rem; font-weight: bold;">${data['current_price']}</p>
                    </div>
                    <div class="metric-card">
                        <h5>1M Performance</h5>
                        <p style="font-size: 1.5rem; font-weight: bold; color: {performance_1m_color};">{data['1m_performance']:+.1f}%</p>
                    </div>
                    <div class="metric-card">
                        <h5>52W High</h5>
                        <p style="font-size: 1.3rem;">${data['52w_high']}</p>
                    </div>
                    <div class="metric-card">
                        <h5>52W Low</h5>
                        <p style="font-size: 1.3rem;">${data['52w_low']}</p>
                    </div>
                    <div class="metric-card">
                        <h5>Dividend Yield</h5>
                        <p style="font-size: 1.3rem;">{data['dividend_yield']}%</p>
                    </div>
                    <div class="metric-card">
                        <h5>Sector</h5>
                        <p style="font-size: 1.1rem;">{data.get('sector', 'Unknown')}</p>
                    </div>
                </div>
            </div>
            """


def create_enhanced_gradio_app():
    """Create the Gradio application"""

    init_database()

    with gr.Blocks(title="AI Investment Advisor Pro", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:

        gr.HTML("""
        <div class="main-header">
            <h1>AI Investment Advisor Pro</h1>
            <h2>AI-Powered Portfolio Strategies & Market Intelligence</h2>
            <p>Harness the power of AI to build sophisticated investment strategies and analyze market trends.</p>
            <!-- <p>Create personalized investment strategies and discover varied assets through AI agent research with asset-type specific emerging opportunities.</p> -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                <div class="metric-card">
                    <h3>Portfolio Strategy</h3>
                    <p>AI-Powered Allocation</p>
                </div>
                <div class="metric-card">
                    <h3>Trending Assets Report</h3>
                    <p>AI Agent Analysis</p>
                </div>
                <div class="metric-card">
                    <h3>Stock Lookup</h3>
                    <p>Deep Stock Research</p>
                </div>
                <div class="metric-card">
                    <h3>Research History</h3>
                    <p>Saved Reports</p>
                </div>
            </div>
        </div>
        """)

        with gr.Tabs():

            with gr.TabItem("Portfolio Strategy"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Investment Profile</h3>")

                        budget = gr.Number(
                            label="Initial Investment Amount ($)",
                            value=10000,
                            minimum=200,
                            maximum=10000000,
                            info='Minimum: $200'
                        )

                        time_horizon = gr.Radio(
                            label="Investment Time Horizon",
                            choices=["Short-term", "Medium-term", "Long-term"],
                            value="Medium-term",
                            info="How long will you invest?"
                        )

                        risk_tolerance = gr.Radio(
                            label="Risk Tolerance",
                            choices=["Conservative", "Moderate", "Aggressive"],
                            value="Moderate",
                            info="How comfortable are you with risk?"
                        )

                        goals = gr.Textbox(
                             label="Investment Goals & Interests",
                            placeholder="e.g., passive income, retirement, real estate, tech focus, ESG investing...",
                            lines=3,
                            info="Tell us about your investment objectives"
                        )

                        email = gr.Textbox(
                            label="📧 Email (Optional)",
                            placeholder="your.email@example.com",
                            info="📨 Receive your strategy via email"
                        )

                        generate_portfolio_btn = gr.Button(
                            "Generate Portfolio Strategy",
                            variant="primary",
                            elem_classes="primary-btn",
                            size="lg"
                        )

                    with gr.Column(scale=2, elem_classes="results-section"):
                        portfolio_status = gr.HTML(
                                    value="""
                                    <div style='text-align: center;'>
                                        <h3 style='color: #d1d5db; margin-bottom: 0.5rem;'>Your Portfolio Strategy</h3>
                                        <div style='color: #9ca3af;'>Fill out your profile and click 'Generate Portfolio Strategy' to begin</div>
                                    </div>
                                    """
                                )

                        portfolio_output = gr.HTML(
                            value=create_placeholder_html("Portfolio Strategy", "Generate your strategy and uncover smarter investment opportunities")
                        )
                        portfolio_table = gr.Dataframe(label="Asset Allocation Summary")

                        with gr.Row():
                            portfolio_email_btn = gr.Button("Email Report", variant="primary")
                            portfolio_email_status = gr.Textbox(label="Email Status", interactive=False)

            with gr.TabItem("Trending Assets Report"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Trending Assets</h3>")

                        research_type = gr.Radio(
                            label="Research Type",
                            choices=["Trending Stocks", "Trending Cryptos", "Trending ETFs"],
                            value="Trending Stocks",
                            info="Select your research focus"
                        )

                        market_focus = gr.CheckboxGroup(
                            label="Market Focus Areas",
                            choices=["Technology", "Healthcare", "Energy", "Finance", "Consumer", "Real Estate"],
                            value=["Technology"],
                            info="Choose sectors of interest (Ignored for Cryptos)"
                        )

                        research_risk = gr.Radio(
                            label="Risk Tolerance",
                            choices=["Conservative", "Moderate", "Aggressive"],
                            value="Moderate",
                            info="How comfortable are you with risk?"
                        )

                        research_email = gr.Textbox(
                            label="Email (Optional)",
                            placeholder="your.email@example.com",
                            info="📨 Receive research report via email"
                        )

                        generate_research_btn = gr.Button(
                            "Generate Trending Assets Report",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        research_status = gr.HTML(
                                    value="""
                                    <div style='text-align: center;'>
                                        <h3 style='color: #d1d5db; margin-bottom: 0.5rem;'>Trending Assets Report</h3>
                                        <div style='color: #9ca3af;'>Select your research preferences and click 'Generate Research Report' to begin</div>
                                    </div>
                                    """
                                )

                        stock_analysis_output = gr.HTML(
                            value=create_placeholder_html("Trending Assets Report", "Comprehensive market analysis and stock recommendations")
                        )

                        trending_table = gr.Dataframe(
                            label="Buy Ratings",
                            headers=["Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating", "Market Cap", "P/E Ratio", "PEG Ratio", "ROE", "P/B Ratio", "D/E Ratio"]
                        )
                        stable_table = gr.Dataframe(
                            label="Other Ratings",
                            headers=["Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating", "P/E Ratio", "PEG Ratio", "ROE", "P/B Ratio", "D/E Ratio"]
                        )

                        with gr.Row():
                            research_email_btn = gr.Button("Email Asset Report", variant="primary")
                            research_email_status = gr.Textbox(label="Email Status", interactive=False)

            with gr.TabItem("Stock Lookup"):
                gr.HTML("<h3 style='text-align: center; color: #d1d5db;'>Deep Stock Research</h3>")

                with gr.Row():
                    symbol_input = gr.Textbox(
                        label="Enter Stock Symbol",
                        placeholder="e.g., AAPL, TSLA, MSFT, NVDA",
                        scale=3
                    )
                    with gr.Column(scale=1):
                        lookup_btn = gr.Button("Get Quote", variant="primary")
                        news_btn = gr.Button("Deep Research", variant="primary")

                quote_output = gr.HTML()

                news_output = gr.HTML(
                    value=create_placeholder_html("Deep Research", "Enter a stock symbol and click 'Deep Research' for comprehensive analysis")
                )

            with gr.TabItem("Research History"):
                gr.HTML("<h3 style='text-align: center; color: #d1d5db;'>Analysis History & Saved Reports</h3>")

                with gr.Row():
                    refresh_history_btn = gr.Button("Refresh History", variant="secondary")

                with gr.Tabs():
                    with gr.TabItem("Stock Research History"):
                        stock_history_table = gr.Dataframe(
                            label="Recent Asset Analysis Reports",
                            headers=["ID", "Date", "Type", "Preview"]
                        )

                        with gr.Row():
                            stock_report_id = gr.Number(label="Report ID", precision=0)
                            view_stock_report_btn = gr.Button("View Report", variant="primary")

                        stock_report_display = gr.HTML(
                            value=create_placeholder_html("Full Report", "Select a report ID and click 'View Report'")
                        )

                    with gr.TabItem("Portfolio Strategy History"):
                        portfolio_history_table = gr.Dataframe(
                            label="Recent Portfolio Strategies",
                            headers=["ID", "Date", "Budget", "Risk", "Horizon", "Preview"]
                        )

                        with gr.Row():
                            portfolio_report_id = gr.Number(label="Strategy ID", precision=0)
                            view_portfolio_report_btn = gr.Button("View Strategy", variant="primary")

                        portfolio_report_display = gr.HTML(
                            value=create_placeholder_html("Full Strategy", "Select a strategy ID and click 'View Strategy'")
                        )

        gr.HTML("""
        <div style="text-align: center; margin-top: 0 rem; padding: 1rem; background: rgba(255, 255, 255, 0.02); border-radius: 16px;">
            <h4 style="color: #fbbf24;">⚠️ Investment Disclaimer</h4>
            <p style="color: #9ca3af;">
                This AI-generated investment analysis is for informational and educational purposes only. All investments carry inherent risks, and past performance does not guarantee future results. Cryptocurrency investments are particularly volatile and speculative. Always conduct your own research and consult with qualified financial professionals before making investment decisions.
            </p>
            <!--<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; margin-top: 1rem; font-weight: 600;">
                <strong>AI Agent Research • Asset-Type Specific Emerging Opportunities • Strictly AI-Only Content • Proper Rating Distribution • Professional Reports</strong>
            </div> -->
        </div>
        """)

        def portfolio_wrapper(budget_val, time_val, risk_val, goals_val):
            """Handle portfolio strategy generation"""
            if budget_val is None or budget_val < 200:
                return "<div class='warning-msg'>Minimum investment is $200</div>", create_placeholder_html("Error", "Minimum investment required"), pd.DataFrame()

            try:
                result = run_advisor(_blank_state(
                    task="portfolio",
                    budget=float(budget_val),
                    time_horizon=time_val,
                    risk_tolerance=risk_val,
                    goals=goals_val or "",
                ))
                expected = (result.get("portfolio") or {}).get("expected_annual_return", 0)
                status = f"Portfolio strategy generated! Trailing 1-year return of this mix: {expected}%"
                status_html = f"<div class='success-msg'>Success: {status}</div>"
                strategy_html = convert_markdown_to_html(result.get("report", ""))
                return status_html, strategy_html, pd.DataFrame(result.get("summary_rows") or [])
            except Exception as e:
                error_html = f"<div class='warning-msg'>Error: {str(e)}</div>"
                return error_html, create_placeholder_html("Error", str(e)), pd.DataFrame()

        def research_wrapper(research_type_val, market_focus_val, risk_val):
            """Handle trending assets research generation"""
            try:
                if not os.getenv('OPENAI_API_KEY'):
                    report = "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
                    status_html = "<div class='warning-msg'>API Key Required</div>"
                    return status_html, convert_markdown_to_html(report), pd.DataFrame(), pd.DataFrame()

                result = run_advisor(_blank_state(
                    task="trending",
                    research_type=research_type_val,
                    sectors=list(market_focus_val or ["Technology"]),
                    risk_tolerance=risk_val or "Moderate",
                ))
                if result.get("ratings"):
                    status = f"Generated {research_type_val} report with emerging opportunities!"
                    status_html = f"<div class='success-msg'>Success: {status}</div>"
                else:
                    status = f"No {research_type_val.lower()} discovered"
                    status_html = f"<div class='warning-msg'>{status}</div>"
                report_html = convert_markdown_to_html(result.get("report", ""))
                trending_df = pd.DataFrame(result.get("summary_rows") or [])
                stable_df = pd.DataFrame(result.get("stable_rows") or [])
                return status_html, report_html, trending_df, stable_df
            except Exception as e:
                error_html = f"<div class='warning-msg'>Error: {str(e)}</div>"
                placeholder = create_placeholder_html("Error", str(e))
                return error_html, placeholder, pd.DataFrame(), pd.DataFrame()

        def lookup_wrapper(symbol_val):
            """Handle individual stock quote lookup"""
            if not symbol_val:
                return "<div class='warning-msg'>Please enter a stock symbol</div>"

            try:
                symbol = symbol_val.strip().upper()
                result = run_advisor(_blank_state(task="quote", symbol=symbol))
                data = (result.get("market_data") or {}).get(symbol)
                if data:
                    return _quote_html(symbol, data)
                return "<div class='warning-msg'>Symbol not found or no data available</div>"
            except Exception as e:
                return f"<div class='warning-msg'>Error fetching data: {str(e)}</div>"

        def news_wrapper(symbol_val):
            """Handle individual stock AI research"""
            if not symbol_val:
                return "<div class='warning-msg'>Please enter a stock symbol</div>"

            try:
                if not os.getenv('OPENAI_API_KEY'):
                    return convert_markdown_to_html("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
                result = run_advisor(_blank_state(task="individual", symbol=symbol_val.strip().upper()))
                return convert_markdown_to_html(result.get("report", ""))
            except Exception as e:
                return f"<div class='warning-msg'>AI research error for {symbol_val.upper()}: {str(e)}</div>"

        def refresh_history_wrapper():
            """Handle history refresh"""
            stock_df, portfolio_df = get_research_history()
            return stock_df, portfolio_df

        def view_stock_report_wrapper(report_id):
            """Handle viewing full stock report"""
            if not report_id:
                return create_placeholder_html("Error", "Please enter a valid report ID")
            try:
                content = get_full_report_by_id(int(report_id), "stock")
                return convert_markdown_to_html(content)
            except Exception as e:
                return create_placeholder_html("Error", f"Error loading report: {str(e)}")

        def view_portfolio_report_wrapper(strategy_id):
            """Handle viewing full portfolio strategy"""
            if not strategy_id:
                return create_placeholder_html("Error", "Please enter a valid strategy ID")
            try:
                content = get_full_report_by_id(int(strategy_id), "portfolio")
                return convert_markdown_to_html(content)
            except Exception as e:
                return create_placeholder_html("Error", f"Error loading strategy: {str(e)}")

        def email_portfolio_wrapper(email_val, portfolio_content):
            """Handle portfolio email sending"""
            if not email_val:
                return "Please provide an email address"
            return send_email_report(email_val, portfolio_content, "portfolio")

        def email_research_wrapper(email_val, research_report):
            """Handle research email sending"""
            if not email_val:
                return "Please provide an email address"
            return send_email_report(email_val, research_report, "enhanced_stock_analysis")

        generate_portfolio_btn.click(
            fn=portfolio_wrapper,
            inputs=[budget, time_horizon, risk_tolerance, goals],
            outputs=[portfolio_status, portfolio_output, portfolio_table]
        )

        generate_research_btn.click(
            fn=research_wrapper,
            inputs=[research_type, market_focus, research_risk],
            outputs=[research_status, stock_analysis_output, trending_table, stable_table]
        )

        portfolio_email_btn.click(
            fn=email_portfolio_wrapper,
            inputs=[email, portfolio_output],
            outputs=[portfolio_email_status]
        )

        research_email_btn.click(
            fn=email_research_wrapper,
            inputs=[research_email, stock_analysis_output],
            outputs=[research_email_status]
        )

        lookup_btn.click(
            fn=lookup_wrapper,
            inputs=[symbol_input],
            outputs=[quote_output]
        )

        news_btn.click(
            fn=news_wrapper,
            inputs=[symbol_input],
            outputs=[news_output]
        )

        symbol_input.submit(
            fn=lookup_wrapper,
            inputs=[symbol_input],
            outputs=[quote_output]
        )

        refresh_history_btn.click(
            fn=refresh_history_wrapper,
            inputs=[],
            outputs=[stock_history_table, portfolio_history_table]
        )

        view_stock_report_btn.click(
            fn=view_stock_report_wrapper,
            inputs=[stock_report_id],
            outputs=[stock_report_display]
        )

        view_portfolio_report_btn.click(
            fn=view_portfolio_report_wrapper,
            inputs=[portfolio_report_id],
            outputs=[portfolio_report_display]
        )

        try:
            stock_df, portfolio_df = get_research_history()
            stock_history_table.value = stock_df
            portfolio_history_table.value = portfolio_df
        except:
            pass

    return app


if __name__ == "__main__":
    print("Starting AI Investment Advisor Pro...")

    if os.getenv('OPENAI_API_KEY'):
        print("OpenAI API key detected - AI agents enabled")
    else:
        print("No OPENAI_API_KEY found - Add to .env file to enable AI agents")

    app = create_enhanced_gradio_app()
    app.launch(
        share=False,
        debug=False,
        show_error=True
    )
