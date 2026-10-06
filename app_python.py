"""
AI Investment Advisor — LangGraph multi-agent app with LangSmith tracing.

Agents
------
Router                  chooses the workflow
Discovery Agent         finds trending stocks, cryptos, or ETFs
Emerging Agent          finds lower-priced, higher-upside names of the same type
Market Data Agent       pulls live quotes from Yahoo Finance
Company Analyst         company profile for a single ticker
Sentiment Analyst       bullish / bearish read and catalysts
Risk Agent              ratings from price action and sentiment
Portfolio Strategist    allocation plus a goal-aligned narrative
Report Writer           markdown report saved to SQLite

LangSmith
---------
Set LANGSMITH_API_KEY in .env. Tracing is enabled on startup and every graph
run is tagged so it shows up in the LangSmith project named below.
"""

import os
import random
import sqlite3
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import gradio as gr
import markdown
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from email.mime.text import MIMEText
import smtplib
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from typing_extensions import TypedDict

load_dotenv()

LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ai-investment-advisor")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

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


# ================================
# STATE
# ================================

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
    portfolio: Dict[str, Any]
    report: str
    summary_rows: List[Dict[str, Any]]
    stable_rows: List[Dict[str, Any]]
    agent_log: List[str]
    errors: List[str]
    trace_url: str


# ================================
# STRUCTURED LLM OUTPUTS
# ================================

class DiscoveredAsset(BaseModel):
    symbol: str = Field(description="Ticker only, for example AAPL or BTC")
    name: str
    category: str
    catalyst: str
    thesis: str
    expected_return_pct: float


class DiscoveryBatch(BaseModel):
    assets: List[DiscoveredAsset]


class EmergingAsset(BaseModel):
    symbol: str
    name: str
    catalyst: str
    upside: str


class EmergingBatch(BaseModel):
    assets: List[EmergingAsset]


class CompanyProfile(BaseModel):
    company_name: str
    founded_year: str
    business_description: str
    sector: str
    market_cap_category: str
    key_products: List[str]
    competitive_advantages: str
    recent_developments: str


class SentimentProfile(BaseModel):
    overall_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    sentiment_score: float = Field(description="Score from -5 to 5")
    recent_developments: List[str]
    risk_factors: List[str]
    growth_catalysts: List[str]
    analyst_outlook: str


class PortfolioNarrative(BaseModel):
    summary: str
    risk_notes: List[str]
    goal_alignment: str
    rebalancing: str


# ================================
# HELPERS
# ================================

def _log(state: AdvisorState, message: str) -> List[str]:
    logs = list(state.get("agent_log") or [])
    logs.append(message)
    logger.info(message)
    return logs


def _errors(state: AdvisorState, message: str) -> List[str]:
    errors = list(state.get("errors") or [])
    errors.append(message)
    logger.warning(message)
    return errors


def get_llm(temperature: float = 0.4) -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from the environment")
    return ChatOpenAI(model=MODEL_NAME, temperature=temperature)


def asset_kind(research_type: str) -> str:
    if "Crypto" in research_type:
        return "crypto"
    if "ETF" in research_type:
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


# ================================
# MARKET DATA TOOL
# ================================

@traceable(name="yahoo_finance_quotes", run_type="tool")
def fetch_quotes(symbols: List[str], kind: str) -> Dict[str, Dict[str, Any]]:
    quotes: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols[:12]:
        ticker_symbol = f"{symbol}-USD" if kind == "crypto" else symbol
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist_1m = ticker.history(period="1mo")
            hist_1y = ticker.history(period="1y")
            info = ticker.info or {}
            if hist_1m.empty:
                continue
            current = float(hist_1m["Close"].iloc[-1])
            start_1m = float(hist_1m["Close"].iloc[0])
            performance_1m = ((current - start_1m) / start_1m) * 100 if start_1m else 0.0
            performance_1y = 0.0
            high_52w = float(hist_1m["High"].max())
            low_52w = float(hist_1m["Low"].min())
            if not hist_1y.empty:
                start_1y = float(hist_1y["Close"].iloc[0])
                performance_1y = ((current - start_1y) / start_1y) * 100 if start_1y else 0.0
                high_52w = float(hist_1y["High"].max())
                low_52w = float(hist_1y["Low"].min())
            dividend = info.get("dividendYield") or 0
            if isinstance(dividend, (int, float)) and 0 < dividend < 1:
                dividend *= 100
            quotes[symbol] = {
                "current_price": round(current, 4 if kind == "crypto" else 2),
                "1m_performance": round(performance_1m, 2),
                "1y_performance": round(performance_1y, 2),
                "52w_high": round(high_52w, 2),
                "52w_low": round(low_52w, 2),
                "dividend_yield": round(float(dividend), 2) if dividend else 0,
                "sector": info.get("sector") or ("Cryptocurrency" if kind == "crypto" else "Unknown"),
                "market_cap": info.get("marketCap") or 0,
                "pe_ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
                "peg_ratio": round(info.get("pegRatio"), 2) if info.get("pegRatio") else "N/A",
                "roe": round(info.get("returnOnEquity") * 100, 2) if info.get("returnOnEquity") else "N/A",
                "pb_ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else "N/A",
                "de_ratio": round(info.get("debtToEquity"), 2) if info.get("debtToEquity") else "N/A",
            }
        except Exception as exc:
            logger.warning("Quote failed for %s: %s", symbol, exc)
    return quotes


RATING_LABELS = {
    "Strong Buy": "🟢 Strong Buy",
    "Moderate Buy": "🟡 Moderate Buy",
    "Hold": "🔵 Hold",
    "Sell": "🔴 Sell",
}


def _quote_row(item: Dict[str, Any], quote: Dict[str, Any], rating: str, kind: str, include_market_cap: bool) -> Dict[str, Any]:
    price = quote.get("current_price") or 0
    price_display = f"${price:.4f}" if kind == "crypto" and price < 1 else f"${price:.2f}"
    type_label = {"stock": "Stock", "crypto": "Crypto", "etf": "ETF"}.get(kind, "Stock")
    row = {
        "Symbol": item.get("symbol"),
        "Name": item.get("name"),
        "Price": price_display,
        "Type": type_label,
        "Sector": item.get("sector") or quote.get("sector"),
        "1M Return": f"{quote.get('1m_performance', 0):+.1f}%",
        "Rating": RATING_LABELS.get(rating, rating),
    }
    if include_market_cap:
        row["Market Cap"] = format_market_cap(quote.get("market_cap") or 0)
    row.update({
        "P/E Ratio": quote.get("pe_ratio", "N/A"),
        "PEG Ratio": quote.get("peg_ratio", "N/A"),
        "ROE": quote.get("roe", "N/A"),
        "P/B Ratio": quote.get("pb_ratio", "N/A"),
        "D/E Ratio": quote.get("de_ratio", "N/A"),
    })
    return row


def rate_performance(performance_1m: float, sentiment: str = "NEUTRAL") -> str:
    if sentiment == "BULLISH" and performance_1m > 5:
        return "Strong Buy"
    if sentiment == "BULLISH" or performance_1m > 3:
        return "Moderate Buy"
    if sentiment != "BEARISH" and performance_1m > -5:
        return "Hold"
    return "Sell"


# ================================
# AGENT NODES
# ================================

def router_node(state: AdvisorState) -> Dict[str, Any]:
    task = state.get("task", "trending")
    return {"agent_log": _log(state, f"Router: dispatched the {task} workflow")}


def route_start(state: AdvisorState) -> str:
    task = state.get("task")
    if task == "portfolio":
        return "portfolio_strategist"
    if task == "trending":
        return "discovery"
    return "market_data"


def discovery_node(state: AdvisorState) -> Dict[str, Any]:
    research_type = state.get("research_type") or "Trending Stocks"
    sectors = state.get("sectors") or ["Technology"]
    kind = asset_kind(research_type)
    context = random.choice([
        "mix established leaders with newer growth names",
        "emphasize recent earnings, product, or partnership catalysts",
        "include large-cap, mid-cap, and smaller names",
        "favor names with momentum and a clear investment thesis",
    ])
    prompt = f"""Find 6 real, currently tradable {kind}s for {datetime.now():%B %Y}.
Focus: {', '.join(sectors) if kind != 'crypto' else 'cryptocurrencies'}.
Research angle: {context}.
Use real ticker symbols only. Do not invent tickers. For crypto, use symbols such as BTC or ETH, without a -USD suffix.
Keep each thesis to one sentence."""
    try:
        result = get_llm(0.6).with_structured_output(DiscoveryBatch).invoke(prompt)
        assets = []
        for item in result.assets:
            symbol = clean_symbol(item.symbol, kind)
            if not symbol:
                continue
            assets.append({
                "symbol": symbol,
                "name": item.name,
                "category": item.category,
                "catalyst": item.catalyst,
                "thesis": item.thesis,
                "expected_return_pct": item.expected_return_pct,
                "bucket": "trending",
                "sector": "Cryptocurrency" if kind == "crypto" else ", ".join(sectors),
            })
        return {
            "discovered": assets[:6],
            "agent_log": _log(state, f"Discovery Agent: selected {len(assets[:6])} {kind} symbols"),
        }
    except Exception as exc:
        return {
            "discovered": [],
            "agent_log": _log(state, "Discovery Agent: failed"),
            "errors": _errors(state, f"Discovery Agent: {exc}"),
        }


def emerging_node(state: AdvisorState) -> Dict[str, Any]:
    research_type = state.get("research_type") or "Trending Stocks"
    sectors = state.get("sectors") or ["Technology"]
    kind = asset_kind(research_type)
    exclude = [item["symbol"] for item in state.get("discovered") or []]
    prompt = f"""Find 4 real emerging {kind}s for {datetime.now():%B %Y}.
Focus: {', '.join(sectors) if kind != 'crypto' else 'smaller cryptocurrencies'}.
Prefer names that are less obvious than the mega-caps, with a specific recent catalyst.
Exclude: {', '.join(exclude) or 'none'}.
Use real ticker symbols only."""
    try:
        result = get_llm(0.7).with_structured_output(EmergingBatch).invoke(prompt)
        assets = []
        seen = set(exclude)
        for item in result.assets:
            symbol = clean_symbol(item.symbol, kind)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            assets.append({
                "symbol": symbol,
                "name": item.name,
                "category": "emerging",
                "catalyst": item.catalyst,
                "thesis": item.upside,
                "bucket": "emerging",
                "sector": "Cryptocurrency" if kind == "crypto" else ", ".join(sectors),
            })
        return {
            "emerging": assets[:4],
            "agent_log": _log(state, f"Emerging Agent: selected {len(assets[:4])} additional symbols"),
        }
    except Exception as exc:
        return {
            "emerging": [],
            "agent_log": _log(state, "Emerging Agent: failed"),
            "errors": _errors(state, f"Emerging Agent: {exc}"),
        }


def market_data_node(state: AdvisorState) -> Dict[str, Any]:
    task = state.get("task")
    kind = "stock"
    symbols: List[str] = []
    if task == "trending":
        kind = asset_kind(state.get("research_type") or "Trending Stocks")
        symbols = [item["symbol"] for item in (state.get("discovered") or []) + (state.get("emerging") or [])]
    else:
        symbol = clean_symbol(state.get("symbol") or "", "stock")
        symbols = [symbol] if symbol else []
    quotes = fetch_quotes(symbols, kind) if symbols else {}
    missing = [symbol for symbol in symbols if symbol not in quotes]
    update: Dict[str, Any] = {
        "market_data": quotes,
        "agent_log": _log(state, f"Market Data Agent: priced {len(quotes)} of {len(symbols)} symbols"),
    }
    if missing:
        update["errors"] = _errors(state, f"Market Data Agent: no quote for {', '.join(missing)}")
    if task in {"quote", "individual"} and symbols:
        update["symbol"] = symbols[0]
    return update


def after_market_data(state: AdvisorState) -> str:
    task = state.get("task")
    if task == "individual":
        return "company_analyst"
    if task == "trending":
        return "risk_agent"
    return "report_writer"


def company_analyst_node(state: AdvisorState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    quote = (state.get("market_data") or {}).get(symbol, {})
    prompt = f"""Profile the public company with ticker {symbol}.
Live price: {quote.get('current_price', 'unknown')}. Sector from market data: {quote.get('sector', 'unknown')}.
Stick to well-known public facts. If you are unsure of the founding year, use N/A."""
    try:
        profile = get_llm(0.2).with_structured_output(CompanyProfile).invoke(prompt)
        return {
            "company": profile.model_dump(),
            "agent_log": _log(state, f"Company Analyst: profiled {symbol}"),
        }
    except Exception as exc:
        return {
            "company": {"company_name": symbol, "sector": quote.get("sector", "Unknown")},
            "agent_log": _log(state, "Company Analyst: failed"),
            "errors": _errors(state, f"Company Analyst: {exc}"),
        }


def sentiment_analyst_node(state: AdvisorState) -> Dict[str, Any]:
    symbol = state.get("symbol") or ""
    company = state.get("company") or {}
    quote = (state.get("market_data") or {}).get(symbol, {})
    prompt = f"""Assess sentiment for {symbol} ({company.get('company_name', symbol)}).
1-month return: {quote.get('1m_performance', 'unknown')}%.
1-year return: {quote.get('1y_performance', 'unknown')}%.
Give a sentiment score from -5 (very bearish) to +5 (very bullish).
Keep each bullet to one sentence."""
    try:
        sentiment = get_llm(0.3).with_structured_output(SentimentProfile).invoke(prompt)
        data = sentiment.model_dump()
        data["sentiment_score"] = max(-5.0, min(5.0, float(data["sentiment_score"])))
        return {
            "sentiment": data,
            "agent_log": _log(state, f"Sentiment Analyst: {data['overall_sentiment']} ({data['sentiment_score']:+.1f})"),
        }
    except Exception as exc:
        return {
            "sentiment": {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 0,
                "recent_developments": [],
                "risk_factors": ["Market volatility"],
                "growth_catalysts": [],
                "analyst_outlook": "Sentiment analysis was unavailable.",
            },
            "agent_log": _log(state, "Sentiment Analyst: failed"),
            "errors": _errors(state, f"Sentiment Analyst: {exc}"),
        }


def risk_agent_node(state: AdvisorState) -> Dict[str, Any]:
    task = state.get("task")
    ratings: List[Dict[str, Any]] = []
    if task == "individual":
        symbol = state.get("symbol") or ""
        quote = (state.get("market_data") or {}).get(symbol, {})
        sentiment = (state.get("sentiment") or {}).get("overall_sentiment", "NEUTRAL")
        rating = rate_performance(float(quote.get("1m_performance") or 0), sentiment)
        ratings.append({"symbol": symbol, "rating": rating, "bucket": "individual"})
        message = f"Risk Agent: rated {symbol} {rating}"
    else:
        kind = asset_kind(state.get("research_type") or "Trending Stocks")
        quotes = state.get("market_data") or {}
        for item in (state.get("discovered") or []) + (state.get("emerging") or []):
            quote = quotes.get(item["symbol"])
            if not quote:
                continue
            rating = rate_performance(float(quote.get("1m_performance") or 0))
            ratings.append({**item, "rating": rating, "quote": quote, "kind": kind})
        message = f"Risk Agent: rated {len(ratings)} symbols"
    return {"ratings": ratings, "agent_log": _log(state, message)}


ALLOCATIONS = {
    "Conservative": {
        "Short-term": {"bonds": 50, "dividend_stocks": 30, "reits": 15, "cash": 5},
        "Medium-term": {"bonds": 40, "dividend_stocks": 35, "reits": 20, "cash": 5},
        "Long-term": {"bonds": 30, "dividend_stocks": 40, "reits": 25, "cash": 5},
    },
    "Moderate": {
        "Short-term": {"bonds": 35, "stocks": 40, "reits": 15, "cash": 10},
        "Medium-term": {"stocks": 50, "bonds": 25, "reits": 20, "cash": 5},
        "Long-term": {"stocks": 60, "bonds": 20, "reits": 15, "cash": 5},
    },
    "Aggressive": {
        "Short-term": {"stocks": 55, "growth_stocks": 25, "reits": 15, "cash": 5},
        "Medium-term": {"stocks": 65, "growth_stocks": 20, "reits": 10, "cash": 5},
        "Long-term": {"stocks": 70, "growth_stocks": 15, "reits": 10, "cash": 5},
    },
}
EXPECTED_RETURNS = {
    "stocks": 8.5,
    "bonds": 3.5,
    "dividend_stocks": 6.5,
    "reits": 7.0,
    "growth_stocks": 12.0,
    "cash": 1.0,
}
RISK_LEVELS = {
    "bonds": "Low",
    "dividend_stocks": "Low-Med",
    "stocks": "Medium",
    "reits": "Medium",
    "growth_stocks": "High",
    "cash": "Low",
}


def portfolio_strategist_node(state: AdvisorState) -> Dict[str, Any]:
    budget = float(state.get("budget") or 0)
    horizon = state.get("time_horizon") or "Medium-term"
    risk = state.get("risk_tolerance") or "Moderate"
    goals = state.get("goals") or "General long-term growth"
    allocation = ALLOCATIONS.get(risk, ALLOCATIONS["Moderate"]).get(horizon, {})
    expected = sum((allocation.get(asset, 0) / 100) * EXPECTED_RETURNS.get(asset, 0) for asset in allocation)
    projected = round(budget * (1 + expected / 100) ** 5, 2)
    portfolio = {
        "budget": budget,
        "allocation": allocation,
        "expected_annual_return": round(expected, 2),
        "projected_value_5y": projected,
        "narrative": {},
    }
    prompt = f"""Write a concise portfolio narrative.
Risk: {risk}. Horizon: {horizon}. Budget: ${budget:,.0f}.
Allocation: {allocation}.
Expected annual return: {portfolio['expected_annual_return']}%.
5-year projected value: ${projected:,.0f}.
Investor goals: {goals}.
This is educational, not personalized financial advice."""
    try:
        narrative = get_llm(0.4).with_structured_output(PortfolioNarrative).invoke(prompt)
        portfolio["narrative"] = narrative.model_dump()
        log = _log(state, f"Portfolio Strategist: {risk} / {horizon} at {portfolio['expected_annual_return']}% expected")
    except Exception as exc:
        portfolio["narrative"] = {
            "summary": "Allocation was computed from the risk and horizon profile.",
            "risk_notes": ["Review the mix if your goals or timeline change."],
            "goal_alignment": goals,
            "rebalancing": "Review the allocation about once a quarter.",
        }
        log = _log(state, "Portfolio Strategist: allocation ready, narrative failed")
        return {
            "portfolio": portfolio,
            "agent_log": log,
            "errors": _errors(state, f"Portfolio Strategist: {exc}"),
        }
    return {"portfolio": portfolio, "agent_log": log}


def _trending_report(state: AdvisorState) -> str:
    research_type = state.get("research_type") or "Trending Stocks"
    sectors = ", ".join(state.get("sectors") or [])
    ratings = state.get("ratings") or []
    lines = [
        f"# {research_type}",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Sectors**: {sectors or 'N/A'}  ",
        f"**Symbols priced**: {len(ratings)}",
        "",
        "## Trending names",
        "",
    ]
    trending = [item for item in ratings if item.get("bucket") != "emerging"]
    emerging = [item for item in ratings if item.get("bucket") == "emerging"]
    if not trending:
        lines.append("No priced trending names were returned.")
    for item in trending:
        quote = item.get("quote") or {}
        lines.extend([
            f"### {item['symbol']} — {item.get('name', item['symbol'])}",
            f"**Rating**: {item['rating']} · **Price**: ${quote.get('current_price', 'N/A')} · **1M**: {quote.get('1m_performance', 0):+.1f}%",
            "",
            f"{item.get('thesis', '')}",
            "",
            f"Catalyst: {item.get('catalyst', 'N/A')}",
            "",
        ])
    lines.extend(["## Emerging ideas", ""])
    if not emerging:
        lines.append("No priced emerging names were returned.")
    for item in emerging:
        quote = item.get("quote") or {}
        lines.extend([
            f"### {item['symbol']} — {item.get('name', item['symbol'])}",
            f"**Rating**: {item['rating']} · **Price**: ${quote.get('current_price', 'N/A')} · **1M**: {quote.get('1m_performance', 0):+.1f}%",
            "",
            f"{item.get('thesis', '')}",
            "",
            f"Catalyst: {item.get('catalyst', 'N/A')}",
            "",
        ])
    return "\n".join(lines)


def _individual_report(state: AdvisorState) -> str:
    symbol = state.get("symbol") or ""
    company = state.get("company") or {}
    sentiment = state.get("sentiment") or {}
    quote = (state.get("market_data") or {}).get(symbol, {})
    rating = (state.get("ratings") or [{}])[0].get("rating", "Hold")
    score = float(sentiment.get("sentiment_score") or 0)
    developments = sentiment.get("recent_developments") or ["No recent developments returned."]
    catalysts = sentiment.get("growth_catalysts") or ["No catalysts returned."]
    risks = sentiment.get("risk_factors") or ["Standard market risk."]
    lines = [
        f"# {symbol} research",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Company**: {company.get('company_name', symbol)}  ",
        f"**Sector**: {company.get('sector') or quote.get('sector', 'Unknown')}  ",
        f"**Price**: ${quote.get('current_price', 0)} · **1M**: {quote.get('1m_performance', 0):+.1f}% · **1Y**: {quote.get('1y_performance', 0):+.1f}%",
        f"**Market cap**: {format_market_cap(quote.get('market_cap') or 0)}  ",
        f"**Sentiment**: {sentiment.get('overall_sentiment', 'NEUTRAL')} ({score:+.1f}/5)  ",
        f"**Rating**: {rating}",
        "",
        "## Company",
        "",
        company.get("business_description") or "Company profile was unavailable.",
        "",
        f"Founded: {company.get('founded_year', 'N/A')}. {company.get('competitive_advantages', '')}",
        "",
        f"Recent company developments: {company.get('recent_developments', 'N/A')}",
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
    lines.extend([
        "",
        "## Outlook",
        "",
        sentiment.get("analyst_outlook") or "Outlook was unavailable.",
    ])
    return "\n".join(lines)


def _portfolio_report(state: AdvisorState) -> str:
    portfolio = state.get("portfolio") or {}
    narrative = portfolio.get("narrative") or {}
    allocation = portfolio.get("allocation") or {}
    budget = float(portfolio.get("budget") or 0)
    lines = [
        "# Portfolio strategy",
        "",
        f"**Date**: {datetime.now():%B %d, %Y}  ",
        f"**Profile**: {state.get('risk_tolerance')} risk, {state.get('time_horizon')} horizon  ",
        f"**Budget**: ${budget:,.2f}  ",
        f"**Expected annual return**: {portfolio.get('expected_annual_return', 0)}%  ",
        f"**5-year projection**: ${portfolio.get('projected_value_5y', 0):,.2f}",
        "",
        narrative.get("summary") or "",
        "",
        "## Allocation",
        "",
    ]
    for asset, percentage in allocation.items():
        amount = budget * percentage / 100
        lines.append(f"- **{asset.replace('_', ' ').title()}**: {percentage}% (${amount:,.0f})")
    lines.extend(["", "## Goals", "", narrative.get("goal_alignment") or state.get("goals") or "", "", "## Risk notes"])
    lines.extend(f"- {note}" for note in (narrative.get("risk_notes") or ["Review this mix against your own constraints."]))
    lines.extend(["", "## Rebalancing", "", narrative.get("rebalancing") or "Review about once a quarter."])
    return "\n".join(lines)


def _quote_report(state: AdvisorState) -> str:
    symbol = state.get("symbol") or ""
    quote = (state.get("market_data") or {}).get(symbol)
    if not quote:
        return f"# {symbol}\n\nNo market data was returned for this symbol."
    return "\n".join([
        f"# {symbol} quote",
        "",
        f"**Price**: ${quote['current_price']}  ",
        f"**1-month**: {quote['1m_performance']:+.1f}%  ",
        f"**1-year**: {quote['1y_performance']:+.1f}%  ",
        f"**52-week range**: ${quote['52w_low']} – ${quote['52w_high']}  ",
        f"**Sector**: {quote.get('sector', 'Unknown')}",
    ])


def report_writer_node(state: AdvisorState) -> Dict[str, Any]:
    task = state.get("task")
    summary_rows: List[Dict[str, Any]] = []
    stable_rows: List[Dict[str, Any]] = []
    if task == "trending":
        report = _trending_report(state)
        for item in state.get("ratings") or []:
            buy_rated = item.get("rating") in {"Strong Buy", "Moderate Buy"}
            row = _quote_row(
                item,
                item.get("quote") or {},
                item.get("rating", "Hold"),
                item.get("kind", "stock"),
                include_market_cap=buy_rated,
            )
            if buy_rated:
                summary_rows.append(row)
            else:
                stable_rows.append(row)
    elif task == "individual":
        report = _individual_report(state)
    elif task == "portfolio":
        report = _portfolio_report(state)
        portfolio = state.get("portfolio") or {}
        budget = float(portfolio.get("budget") or 0)
        for asset, percentage in (portfolio.get("allocation") or {}).items():
            summary_rows.append({
                "Asset Class": asset.replace("_", " ").title(),
                "Allocation": f"{percentage}%",
                "Amount": f"${budget * percentage / 100:,.0f}",
                "Risk Level": RISK_LEVELS.get(asset, "Medium"),
            })
    else:
        report = _quote_report(state)
    return {
        "report": report,
        "summary_rows": summary_rows,
        "stable_rows": stable_rows,
        "agent_log": _log(state, "Report Writer: assembled the report"),
    }


# ================================
# GRAPH
# ================================

def build_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("router", router_node)
    graph.add_node("discovery", discovery_node)
    graph.add_node("emerging", emerging_node)
    graph.add_node("market_data", market_data_node)
    graph.add_node("company_analyst", company_analyst_node)
    graph.add_node("sentiment_analyst", sentiment_analyst_node)
    graph.add_node("risk_agent", risk_agent_node)
    graph.add_node("portfolio_strategist", portfolio_strategist_node)
    graph.add_node("report_writer", report_writer_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_start,
        {
            "portfolio_strategist": "portfolio_strategist",
            "discovery": "discovery",
            "market_data": "market_data",
        },
    )
    graph.add_edge("discovery", "emerging")
    graph.add_edge("emerging", "market_data")
    graph.add_conditional_edges(
        "market_data",
        after_market_data,
        {
            "company_analyst": "company_analyst",
            "risk_agent": "risk_agent",
            "report_writer": "report_writer",
        },
    )
    graph.add_edge("company_analyst", "sentiment_analyst")
    graph.add_edge("sentiment_analyst", "risk_agent")
    graph.add_edge("risk_agent", "report_writer")
    graph.add_edge("portfolio_strategist", "report_writer")
    graph.add_edge("report_writer", END)
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
        "portfolio": {},
        "report": "",
        "summary_rows": [],
        "stable_rows": [],
        "agent_log": [],
        "errors": [],
        "trace_url": "",
    }
    state.update(kwargs)
    return state


@traceable(name="investment_advisor", run_type="chain", tags=["langgraph", "investment-advisor"])
def run_advisor(state: AdvisorState) -> AdvisorState:
    result = ADVISOR.invoke(
        state,
        config={
            "run_name": f"advisor-{state.get('task', 'run')}",
            "tags": ["investment-advisor", state.get("task", "run")],
            "metadata": {
                "task": state.get("task"),
                "research_type": state.get("research_type"),
                "symbol": state.get("symbol"),
                "sectors": state.get("sectors"),
                "risk_tolerance": state.get("risk_tolerance"),
            },
        },
    )
    run = get_current_run_tree()
    trace_url = ""
    if run is not None:
        try:
            trace_url = run.get_url()
        except Exception:
            trace_url = ""
    result["trace_url"] = trace_url
    _persist(result)
    return result


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
                            minimum=1000,
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
                            label="Trending Recommendations (All Ratings)",
                            headers=["Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating", "Market Cap", "P/E Ratio", "PEG Ratio", "ROE", "P/B Ratio", "D/E Ratio"]
                        )
                        stable_table = gr.Dataframe(
                            label="Additional Analysis (Hold/Sell Ratings)",
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
            if budget_val < 1000:
                return "<div class='warning-msg'>Minimum investment is $1,000</div>", create_placeholder_html("Error", "Minimum investment required"), pd.DataFrame()

            try:
                result = run_advisor(_blank_state(
                    task="portfolio",
                    budget=float(budget_val),
                    time_horizon=time_val,
                    risk_tolerance=risk_val,
                    goals=goals_val or "",
                ))
                expected = (result.get("portfolio") or {}).get("expected_annual_return", 0)
                status = f"Portfolio strategy generated! Expected return: {expected}%"
                status_html = f"<div class='success-msg'>Success: {status}</div>"
                strategy_html = convert_markdown_to_html(result.get("report", ""))
                return status_html, strategy_html, pd.DataFrame(result.get("summary_rows") or [])
            except Exception as e:
                error_html = f"<div class='warning-msg'>Error: {str(e)}</div>"
                return error_html, create_placeholder_html("Error", str(e)), pd.DataFrame()

        def research_wrapper(research_type_val, market_focus_val):
            """Handle trending assets research generation"""
            try:
                if not os.getenv('OPENAI_API_KEY'):
                    report = "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
                    status_html = "<div class='success-msg'>Success: API Key Required</div>"
                    return status_html, convert_markdown_to_html(report), pd.DataFrame(), pd.DataFrame()

                result = run_advisor(_blank_state(
                    task="trending",
                    research_type=research_type_val,
                    sectors=list(market_focus_val or ["Technology"]),
                ))
                if result.get("ratings"):
                    status = f"Generated {research_type_val} report with emerging opportunities!"
                else:
                    status = f"No {research_type_val.lower()} discovered"
                status_html = f"<div class='success-msg'>Success: {status}</div>"
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
            inputs=[research_type, market_focus],
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

    try:
        import yfinance
        import pandas
        import markdown
        print("All required packages installed")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install: pip install gradio yfinance pandas markdown python-dotenv openai")
        exit(1)

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
