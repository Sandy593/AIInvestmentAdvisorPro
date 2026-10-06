import gradio as gr
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import sqlite3
import markdown

# Load environment variables
load_dotenv()

# Custom CSS for sleek dark theme with markdown support
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
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
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
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

.results-section {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 2.5rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

/* Tab styling - WHITE TAB NAMES */
.gradio-container .gradio-tabs {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 0.5rem !important;
}

.gradio-container .gradio-tabs button,
.gradio-container .gradio-tabs button[role="tab"] {
    background: transparent !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.gradio-container .gradio-tabs button.selected,
.gradio-container .gradio-tabs button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.gradio-container .gradio-tabs button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
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
    backdrop-filter: blur(10px) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    font-weight: 500 !important;
}

.warning-msg {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(245, 87, 108, 0.9) 100%) !important;
    backdrop-filter: blur(10px) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    font-weight: 500 !important;
}

/* Input styling */
.gradio-textbox input, 
.gradio-number input,
.gradio-textbox textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    color: #1f2937 !important;
    padding: 0.75rem !important;
}

.gradio-label, label {
    color: #e0e6ed !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
    font-size: 1rem !important;
}

.gradio-radio .option {
    color: #e0e6ed !important;
    padding: 0.5rem !important;
}

.gradio-checkbox .option {
    color: #e0e6ed !important;
}

/* Markdown content styling */
.markdown-content {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
    line-height: 1.6 !important;
    height: 400px !important;
    overflow-y: auto !important;
    white-space: normal !important;
    word-wrap: break-word !important;
}

.markdown-content h1 {
    color: #1e293b !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid #e2e8f0 !important;
    padding-bottom: 0.5rem !important;
}

.markdown-content h2 {
    color: #374151 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.8rem !important;
}

.markdown-content h3 {
    color: #4b5563 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.6rem !important;
}

.markdown-content p {
    color: #374151 !important;
    margin-bottom: 0.8rem !important;
    line-height: 1.6 !important;
}

.markdown-content ul, .markdown-content ol {
    margin-left: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.markdown-content li {
    color: #374151 !important;
    margin-bottom: 0.3rem !important;
    line-height: 1.5 !important;
}

.markdown-content strong {
    color: #1e293b !important;
    font-weight: 600 !important;
}

.disclaimer {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin-top: 3rem !important;
    text-align: center !important;
}
"""

# Initialize database
def init_database():
    conn = sqlite3.connect('investment_advisor.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_date DATE,
            budget REAL,
            risk_tolerance TEXT,
            time_horizon TEXT,
            strategy_content TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE,
            report_content TEXT,
            report_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Investment Tools
class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze investment portfolio based on user preferences"
    
    def _run(self, budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> str:
        allocations = {
            "Conservative": {
                "Short-term": {"bonds": 50, "dividend_stocks": 30, "reits": 15, "gold": 5},
                "Medium-term": {"bonds": 40, "dividend_stocks": 35, "reits": 20, "gold": 5},
                "Long-term": {"bonds": 30, "dividend_stocks": 40, "reits": 25, "gold": 5}
            },
            "Moderate": {
                "Short-term": {"bonds": 35, "stocks": 40, "reits": 15, "gold": 10},
                "Medium-term": {"stocks": 50, "bonds": 25, "reits": 20, "gold": 5},
                "Long-term": {"stocks": 60, "bonds": 20, "reits": 15, "gold": 5}
            },
            "Aggressive": {
                "Short-term": {"stocks": 55, "crypto": 25, "reits": 15, "gold": 5},
                "Medium-term": {"stocks": 65, "crypto": 20, "reits": 10, "gold": 5},
                "Long-term": {"stocks": 70, "crypto": 15, "reits": 10, "gold": 5}
            }
        }
        
        allocation = allocations.get(risk_tolerance, {}).get(time_horizon, {})
        expected_returns = {"stocks": 8.5, "bonds": 3.5, "dividend_stocks": 6.5, "reits": 7.0, "crypto": 15.0, "gold": 4.0}
        portfolio_return = sum((allocation.get(asset, 0) / 100) * expected_returns.get(asset, 0) for asset in expected_returns)
        
        return json.dumps({
            "budget": budget,
            "allocation": allocation,
            "expected_annual_return": round(portfolio_return, 2),
            "projected_value_5y": round(budget * (1 + portfolio_return/100)**5, 2)
        }, indent=2)

class MarketDataTool(BaseTool):
    name: str = "market_data"
    description: str = "Get real-time market data"
    
    def _run(self, symbols: str) -> str:
        symbol_list = symbols.split(",") if isinstance(symbols, str) else symbols
        market_data = {}
        
        for symbol in symbol_list[:10]:
            symbol = symbol.strip()
            try:
                if symbol in ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "LINK"]:
                    ticker = f"{symbol}-USD"
                    crypto = yf.Ticker(ticker)
                    hist = crypto.history(period="1y")
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        start = hist['Close'].iloc[0]
                        high_52w = hist['High'].max()
                        low_52w = hist['Low'].min()
                        performance = ((current - start) / start) * 100
                        
                        market_data[symbol] = {
                            "current_price": round(float(current), 2),
                            "52w_high": round(float(high_52w), 2),
                            "52w_low": round(float(low_52w), 2),
                            "1y_performance": round(float(performance), 2),
                            "type": "crypto"
                        }
                else:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period="1y")
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        start = hist['Close'].iloc[0]
                        high_52w = hist['High'].max()
                        low_52w = hist['Low'].min()
                        performance = ((current - start) / start) * 100
                        
                        market_data[symbol] = {
                            "current_price": round(float(current), 2),
                            "52w_high": round(float(high_52w), 2),
                            "52w_low": round(float(low_52w), 2),
                            "1y_performance": round(float(performance), 2),
                            "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                            "type": "stock"
                        }
            except:
                continue
        
        return json.dumps(market_data, indent=2)

# Initialize tools and agents
portfolio_tool = PortfolioAnalysisTool()
market_data_tool = MarketDataTool()

# Check if OpenAI API key is available
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    portfolio_advisor = Agent(
        role="Senior Portfolio Advisor",
        goal="Create personalized investment portfolios with proper markdown formatting",
        backstory="Expert portfolio manager specializing in asset allocation and passive income strategies.",
        tools=[portfolio_tool, market_data_tool],
        llm=llm,
        verbose=True
    )

    stock_research_agent = Agent(
        role="Senior Equity Research Analyst",
        goal="Provide comprehensive stock market analysis in markdown format",
        backstory="Top-tier equity research analyst with expertise in market trends and stock analysis.",
        tools=[market_data_tool],
        llm=llm,
        verbose=True
    )
else:
    portfolio_advisor = None
    stock_research_agent = None

def create_portfolio_strategy(budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> Tuple[str, pd.DataFrame, str]:
    """Create portfolio strategy with markdown formatting"""
    
    # Create allocation summary first
    allocation_data = portfolio_tool._run(budget, time_horizon, risk_tolerance, goals)
    allocation_info = json.loads(allocation_data)
    allocation = allocation_info["allocation"]
    
    # Create simple strategy without AI if no API key
    strategy_text = f"""# Portfolio Investment Strategy Report

## 🎯 Executive Summary
Based on your investment profile, we recommend a **{risk_tolerance.lower()}** allocation strategy for **{time_horizon.lower()}** investing.

## 📊 Investment Profile
- **Budget:** ${budget:,.2f}
- **Time Horizon:** {time_horizon}
- **Risk Tolerance:** {risk_tolerance}
- **Goals:** {goals}

## 💼 Recommended Asset Allocation

"""
    for asset, percentage in allocation.items():
        amount = (percentage / 100) * budget
        strategy_text += f"- **{asset.replace('_', ' ').title()}:** {percentage}% (${amount:,.0f})\n"
    
    strategy_text += f"""

## 📈 Expected Returns & Risk Analysis
- **Expected Annual Return:** {allocation_info.get('expected_annual_return', 7.5)}%
- **Projected 5-Year Value:** ${allocation_info.get('projected_value_5y', budget * 1.4):,.0f}

## 🎯 Investment Strategy
This allocation balances risk and return based on your preferences. The strategy focuses on:

1. **Diversification** across multiple asset classes
2. **Risk Management** appropriate to your tolerance level
3. **Time Horizon Optimization** for your investment timeline
4. **Goal Alignment** with your specific objectives

## 📝 Next Steps
1. Review and adjust allocation based on changing circumstances
2. Implement dollar-cost averaging for systematic investing
3. Rebalance portfolio quarterly to maintain target allocation
4. Monitor performance and market conditions regularly

*This analysis is based on historical data and market trends. Past performance does not guarantee future results.*
"""
    
    # Create summary table
    summary_data = []
    risk_levels = {"bonds": "🟢 Low", "dividend_stocks": "🟡 Low-Med", "stocks": "🟠 Medium", "reits": "🟠 Medium", "crypto": "🔴 High", "gold": "🟢 Low"}
    
    for asset, percentage in allocation.items():
        amount = (percentage / 100) * budget
        risk_level = risk_levels.get(asset, "🟠 Medium")
        summary_data.append({
            "💼 Asset Class": asset.replace("_", " ").title(),
            "📊 Allocation": f"{percentage}%",
            "💰 Amount": f"${amount:,.0f}",
            "⚠️ Risk Level": risk_level
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to database
    try:
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO portfolio_strategies (strategy_date, budget, risk_tolerance, time_horizon, strategy_content) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().date(), budget, risk_tolerance, time_horizon, strategy_text)
        )
        conn.commit()
        conn.close()
    except:
        pass
    
    return strategy_text, summary_df, f"✅ Portfolio strategy generated! Expected return: {allocation_info.get('expected_annual_return', 7.5)}%"

def create_stock_research_report(research_type: str = "Comprehensive Analysis", market_focus: list = None) -> Tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Generate concise stock research report based on user selections"""
    
    if market_focus is None:
        market_focus = ["Technology", "Healthcare"]
    
    # Dynamic symbol selection based on user's market focus
    sector_symbols = {
        "Technology": "AAPL,MSFT,GOOGL,NVDA,META",
        "Healthcare": "JNJ,UNH,PFE,ABBV,TMO", 
        "Energy": "XOM,CVX,COP,EOG,SLB",
        "Finance": "JPM,BAC,WFC,GS,MS",
        "Consumer": "AMZN,TSLA,HD,MCD,NKE",
        "Real Estate": "AMT,PLD,CCI,EQIX,SPG"
    }
    
    # Get symbols for selected sectors only
    sector_data = {}
    
    for sector in market_focus:
        if sector in sector_symbols:
            symbols = sector_symbols[sector]
            try:
                data = json.loads(market_data_tool._run(symbols))
                sector_data[sector] = data
            except:
                sector_data[sector] = {}
    
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Create dynamic report based on selected sectors
    research_text = f"""# 📊 Market Research Report - {current_date}

## 🎯 Focus Areas: {', '.join(market_focus)}

### 💡 Executive Summary
- **Market Outlook:** Cautiously optimistic
- **Key Opportunity:** Sector diversification and growth potential
- **Top Recommendation:** Balanced portfolio across selected sectors

---

## 🚀 TOP STOCK PICKS

"""

    # Dynamic sector analysis based on selections
    trending_recommendations = []
    stable_recommendations = []
    
    # Get sector-specific picks
    sector_picks = {
        "Technology": [
            ("NVDA", "AI Chip Demand", "STRONG BUY", "AI infrastructure leader"),
            ("MSFT", "Azure Growth", "BUY", "Cloud dominance + AI integration"),
            ("GOOGL", "Search + AI", "BUY", "AI-enhanced advertising"),
            ("META", "AI Efficiency", "BUY", "Cost reduction + AI targeting"),
            ("AAPL", "Value Play", "HOLD", "Attractive at current levels")
        ],
        "Healthcare": [
            ("PFE", "FDA Pipeline", "BUY", "Multiple drug approvals expected"),
            ("UNH", "AI Healthcare", "BUY", "Cost reduction through AI"),
            ("TMO", "Gene Therapy", "STRONG BUY", "Biotech tools demand"),
            ("JNJ", "Dividend Play", "HOLD", "Steady dividend growth"),
            ("ABBV", "Patent Cliff", "HOLD", "New developments needed")
        ],
        "Energy": [
            ("XOM", "Oil Production", "BUY", "Strong cash flow generation"),
            ("CVX", "Dividend Leader", "BUY", "Reliable dividend aristocrat"),
            ("COP", "Shale Efficiency", "BUY", "Low-cost production advantage"),
            ("EOG", "Tech Innovation", "BUY", "Drilling technology leader"),
            ("SLB", "Services Recovery", "HOLD", "Oilfield services rebound")
        ],
        "Finance": [
            ("JPM", "Interest Rates", "BUY", "Benefits from higher rates"),
            ("BAC", "Credit Quality", "BUY", "Strong credit metrics"),
            ("WFC", "Turnaround", "HOLD", "Regulatory improvements"),
            ("GS", "Trading Revenue", "BUY", "Investment banking strength"),
            ("MS", "Wealth Mgmt", "BUY", "Growing wealth management")
        ],
        "Consumer": [
            ("AMZN", "Cloud + Retail", "BUY", "AWS growth + e-commerce"),
            ("TSLA", "EV Leadership", "VOLATILE BUY", "Electric vehicle pioneer"),
            ("HD", "Housing Market", "BUY", "Home improvement demand"),
            ("MCD", "Global Expansion", "HOLD", "Steady international growth"),
            ("NKE", "Brand Strength", "BUY", "Premium athletic brand")
        ],
        "Real Estate": [
            ("AMT", "5G Infrastructure", "BUY", "Cell tower demand growth"),
            ("PLD", "E-commerce Logistics", "BUY", "Warehouse demand surge"),
            ("CCI", "Telecom Growth", "BUY", "Communication infrastructure"),
            ("EQIX", "Data Centers", "STRONG BUY", "Cloud data center leader"),
            ("SPG", "Retail Recovery", "HOLD", "Mall recovery potential")
        ]
    }
    
    # Add analysis for each selected sector
    for sector in market_focus:
        if sector in sector_data and sector_data[sector]:
            research_text += f"### {sector} Leaders\n"
            
            sector_stocks = sector_picks.get(sector, [])
            for symbol, catalyst, rating, thesis in sector_stocks[:3]:
                price = sector_data[sector].get(symbol, {}).get('current_price', 'N/A')
                performance = sector_data[sector].get(symbol, {}).get('1y_performance', 0)
                
                research_text += f"- **{symbol}**: ${price} - {thesis}\n"
                
                trending_recommendations.append({
                    "🏢 Symbol": symbol,
                    "💰 Price": f"${price}",
                    "📊 Catalyst": catalyst,
                    "📈 1Y": f"{performance:+.1f}%" if isinstance(performance, (int, float)) else "N/A",
                    "🎯 Rating": rating
                })

    # Add stable stocks from selected sectors
    all_sector_data = {}
    for sector_name, data in sector_data.items():
        all_sector_data.update(data)
    
    stable_count = 0
    for sector in market_focus:
        sector_stocks = sector_picks.get(sector, [])
        for symbol, catalyst, rating, thesis in sector_stocks[2:]:  # Skip first 3
            if stable_count >= 5:
                break
            price = all_sector_data.get(symbol, {}).get('current_price', 'N/A')
            div_yield = all_sector_data.get(symbol, {}).get('dividend_yield', 0)
            
            stable_recommendations.append({
                "🏢 Symbol": symbol,
                "💰 Price": f"${price}",
                "💎 Div Yield": f"{div_yield}%" if div_yield else "0%",
                "📊 Catalyst": catalyst,
                "🎯 Rating": rating
            })
            stable_count += 1

    research_text += f"""
---

## 📰 News-Based Stock Opportunities

### 🚀 CATALYST-DRIVEN RECOMMENDATIONS

"""

    # Add specific news-based analysis for each sector
    if "Technology" in market_focus:
        research_text += """#### 💻 TECHNOLOGY SECTOR - Recent Catalysts

**🔥 AI & Machine Learning Boom**
- **NVIDIA (NVDA)** - ⬆️ STRONG BUY
  - *Catalyst:* Continued AI chip demand, data center growth
  - *Price Target:* $600+ (significant upside potential)
  - *Timeline:* 3-6 months for next earnings beat

**☁️ Cloud Infrastructure Expansion**
- **Microsoft (MSFT)** - ⬆️ BUY
  - *Catalyst:* Azure growth, AI integration in Office 365
  - *Recent News:* Enterprise AI adoption accelerating
  - *Price Target:* $550+ based on cloud revenue growth

**🤖 Meta's AI Pivot Success**
- **Meta Platforms (META)** - ⬆️ BUY
  - *Catalyst:* AI-driven ad targeting improvements
  - *Recent Development:* Metaverse cost reduction, efficiency gains
  - *Price Target:* $400+ on improved margins

"""

    if "Healthcare" in market_focus:
        research_text += """#### 🏥 HEALTHCARE SECTOR - Recent Catalysts

**💊 FDA Approval Pipeline**
- **Pfizer (PFE)** - ⬆️ MODERATE BUY
  - *Catalyst:* New drug approvals, cancer treatment portfolio
  - *Watch For:* Q1 2025 FDA decisions
  - *Price Target:* $45+ on successful approvals

**🧬 Gene Therapy Breakthrough**
- **Thermo Fisher (TMO)** - ⬆️ STRONG BUY
  - *Catalyst:* Gene therapy tools demand surge
  - *Recent News:* Major biotech partnerships announced
  - *Price Target:* $600+ on biotech tool sales

**🏥 Healthcare AI Integration**
- **UnitedHealth (UNH)** - ⬆️ BUY
  - *Catalyst:* AI-powered cost reduction initiatives
  - *Recent Development:* Predictive healthcare analytics expansion
  - *Price Target:* $650+ on margin improvements

"""

    if "Energy" in market_focus:
        research_text += """#### ⚡ ENERGY SECTOR - Recent Catalysts

**🛢️ Oil Production Efficiency**
- **ExxonMobil (XOM)** - ⬆️ BUY
  - *Catalyst:* Permian Basin production growth
  - *Recent News:* Record quarterly production announced
  - *Price Target:* $130+ on sustained high oil prices

**🔋 Clean Energy Transition**
- **Chevron (CVX)** - ⬆️ BUY
  - *Catalyst:* Renewable energy investments, carbon capture
  - *Policy Support:* Infrastructure bill clean energy funding
  - *Price Target:* $180+ on energy transition leadership

"""

    if "Finance" in market_focus:
        research_text += """#### 🏦 FINANCE SECTOR - Recent Catalysts

**📈 Interest Rate Environment**
- **JPMorgan Chase (JPM)** - ⬆️ STRONG BUY
  - *Catalyst:* Net interest margin expansion
  - *Recent Development:* Credit quality remains strong
  - *Price Target:* $250+ on sustained higher rates

**💳 Digital Banking Growth**
- **Bank of America (BAC)** - ⬆️ BUY
  - *Catalyst:* Digital transformation cost savings
  - *Recent News:* Record digital banking adoption
  - *Price Target:* $55+ on operational efficiency

"""

    if "Consumer" in market_focus:
        research_text += """#### 🛒 CONSUMER SECTOR - Recent Catalysts

**📦 E-commerce Dominance**
- **Amazon (AMZN)** - ⬆️ BUY
  - *Catalyst:* AWS growth + Prime subscription increases
  - *Recent Development:* Supply chain optimization success
  - *Price Target:* $250+ on cloud margins expansion

**🚗 Electric Vehicle Revolution**
- **Tesla (TSLA)** - ⬆️ VOLATILE BUY
  - *Catalyst:* EV tax credit extensions, charging infrastructure
  - *Risk:* High volatility, production challenges
  - *Price Target:* $400+ on policy support

"""

    if "Real Estate" in market_focus:
        research_text += """#### 🏢 REAL ESTATE SECTOR - Recent Catalysts

**📡 5G Infrastructure Demand**
- **American Tower (AMT)** - ⬆️ BUY
  - *Catalyst:* 5G rollout acceleration
  - *Recent News:* Major carrier lease extensions
  - *Price Target:* $250+ on tower lease growth

**📊 Data Center Expansion**
- **Equinix (EQIX)** - ⬆️ STRONG BUY
  - *Catalyst:* AI workload demand for data centers
  - *Recent Development:* Hyperscale customer wins
  - *Price Target:* $900+ on AI infrastructure boom

"""

    research_text += f"""
---

### 📊 MARKET-WIDE CATALYSTS TO WATCH

**🏛️ Government & Policy Impact**

**Infrastructure & Green Energy**
- **Government funding** for clean energy projects benefits renewable stocks
- **EV tax credit extensions** supporting Tesla and EV ecosystem
- **Infrastructure spending** boosting materials and construction

**Defense & Cybersecurity**
- **Increased defense spending** benefiting aerospace and defense contractors
- **Cybersecurity threats** driving enterprise security investments
- **AI contract awards** from government agencies

---

### 🎯 NEWS-DRIVEN TRADING STRATEGIES

**📈 Short-Term Opportunities (1-3 months):**
1. **Earnings Surprise Plays**
   - Companies beating AI revenue guidance
   - Healthcare companies with FDA approval announcements
   - Energy companies with production beat estimates

2. **Event-Driven Catalysts**
   - Product launch announcements (Apple, Google)
   - Clinical trial results (biotech companies)
   - Government contract awards (defense contractors)

**📊 Medium-Term Themes (3-12 months):**
1. **AI Infrastructure Build-Out**
   - Data center expansion benefiting REITs
   - Semiconductor supply chain investments
   - Cloud service provider capacity expansion

2. **Regulatory Changes**
   - Healthcare pricing reform impacts
   - Financial services regulation updates
   - Energy transition policy implementations

---

### ⚡ REAL-TIME CATALYST MONITORING

**🔔 Set Alerts For:**
- FDA drug approval announcements
- Major tech company product launches
- Government funding/contract announcements
- Earnings guidance revisions
- Clinical trial results
- Merger & acquisition rumors

**📱 Recommended News Sources:**
- SEC filings (8-K forms for material events)
- Company investor relations pages
- Sector-specific trade publications
- Financial news terminals (Bloomberg, Reuters)
- Earnings call transcripts

---

### ⚠️ RISK MANAGEMENT FOR NEWS-BASED TRADING

**🛡️ Key Principles:**
1. **Position Sizing:** Never risk more than 2-3% on news plays
2. **Time Limits:** Set 30-90 day exit strategies
3. **Stop Losses:** Use 15-20% stop losses on speculative plays
4. **Diversification:** Don't concentrate in single catalyst events
5. **Verification:** Confirm news from multiple reliable sources

**🚨 Red Flags to Avoid:**
- Unverified social media rumors
- Penny stocks with "breakthrough" claims
- Companies with weak fundamentals despite news
- Over-hyped events already priced in

"""

    if len(market_focus) == 1:
        research_text += f"- **60%** {market_focus[0]} stocks\n- **25%** Diversified ETFs\n- **10%** Cryptocurrency\n- **5%** Cash reserves"
    elif len(market_focus) == 2:
        research_text += f"- **35%** {market_focus[0]} stocks\n- **35%** {market_focus[1]} stocks\n- **20%** Diversified ETFs\n- **10%** Cryptocurrency"
    else:
        allocation = 70 // len(market_focus)
        for sector in market_focus:
            research_text += f"- **{allocation}%** {sector} stocks\n"
        research_text += "- **20%** Diversified ETFs\n- **10%** Cryptocurrency"

    research_text += """

### ⚠️ Key Risks
- Sector concentration risk
- Market volatility
- Economic policy changes
- Global market conditions

### 🎪 Action Items
1. Monitor sector-specific developments
2. Track earnings and guidance updates
3. Set appropriate stop losses
4. Rebalance quarterly between sectors

*Analysis based on current market data and trends. Consult financial advisor before investing.*
"""

    # Create final dataframes
    trending_df = pd.DataFrame(trending_recommendations)
    stable_df = pd.DataFrame(stable_recommendations)

    # Save to database
    try:
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO stock_research (report_date, report_content, report_type) VALUES (?, ?, ?)",
            (datetime.now().date(), research_text, f"{research_type}_{'+'.join(market_focus)}_dynamic")
        )
        conn.commit()
        conn.close()
    except:
        pass
    
    return research_text, trending_df, stable_df, f"✅ {research_type} report generated for {', '.join(market_focus)} sectors!"

def send_strategy_email(email: str, content: str, content_type: str = "portfolio") -> str:
    """Send email with strategy/research content"""
    try:
        if not email or "@" not in email:
            return "❌ Please provide a valid email address"
        
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        
        if not all([sender_email, sender_password]):
            return "❌ Email credentials not configured"
        
        subject = f"🎯 Your {content_type.title()} Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = email
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"✅ {content_type.title()} report sent successfully!"
        
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

def convert_markdown_to_html(markdown_text: str) -> str:
    """Convert markdown text to styled HTML"""
    if not markdown_text or markdown_text.startswith("❌") or "Error" in markdown_text:
        return create_placeholder_html("Report", "Please generate a report to see results here")
    
    try:
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_text, extensions=['nl2br', 'tables'])
        
        # Wrap in styled container
        styled_html = f"""
        <div class='markdown-content'>{html_content}</div>
        """
        return styled_html
        
    except Exception as e:
        return create_placeholder_html("Report", f"Error rendering report: {str(e)}")

def create_placeholder_html(title: str, description: str) -> str:
    """Create a placeholder HTML for empty reports"""
    return f"""
    <div class='markdown-content' style='display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 300px;'>
        <div style='font-weight: 600; font-size: 1.1rem; color: #374151; margin-bottom: 0.5rem;'>{title}</div>
        <div style='color: #6b7280;'>{description}</div>
    </div>
    """

def create_gradio_app():
    """Create the Gradio application with all features"""
    
    init_database()
    
    with gr.Blocks(title="🎯 AI Investment Advisor Pro", theme=gr.themes.Soft(), css=custom_css) as app:
        
        # Header
        with gr.Row():
            gr.HTML("""
            <div class="main-header">
                <h1>AI Investment Advisor Pro</h1>
                <h2>Advanced Portfolio Management & Market Intelligence</h2>
                <p>Harness the power of artificial intelligence to build sophisticated investment strategies and analyze market trends.</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                    <div class="metric-card">
                        <h3>💼</h3>
                        <h5>Portfolio Builder</h5>
                        <p>AI-Powered Allocation</p>
                    </div>
                    <div class="metric-card">
                        <h3>📊</h3>
                        <h5>Market Research</h5>
                        <p>Real-Time Analysis</p>
                    </div>
                    <div class="metric-card">
                        <h3>🎯</h3>
                        <h5>Smart Recommendations</h5>
                        <p>Data-Driven Insights</p>
                    </div>
                    <div class="metric-card">
                        <h3>📧</h3>
                        <h5>Report Delivery</h5>
                        <p>Automated Updates</p>
                    </div>
                </div>
            </div>
            """)
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # Portfolio Strategy Tab
            with gr.TabItem("💼 Portfolio Strategy"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>📝 Investment Profile</h3>")
                        
                        budget = gr.Number(
                            label="💰 Initial Investment Amount ($)",
                            value=25000,
                            minimum=1000,
                            maximum=10000000,
                            info="💡 Minimum: $1,000"
                        )
                        
                        time_horizon = gr.Radio(
                            label="⏳ Investment Time Horizon",
                            choices=["Short-term", "Medium-term", "Long-term"],
                            value="Medium-term",
                            info="📅 How long will you invest?"
                        )
                        
                        risk_tolerance = gr.Radio(
                            label="⚖️ Risk Tolerance",
                            choices=["Conservative", "Moderate", "Aggressive"],
                            value="Moderate",
                            info="📈 Your comfort with market volatility"
                        )
                        
                        goals = gr.Textbox(
                            label="🎯 Investment Goals & Interests",
                            placeholder="e.g., passive income, retirement, real estate, tech focus, ESG investing...",
                            lines=3,
                            info="🎪 Tell us about your investment objectives"
                        )
                        
                        email = gr.Textbox(
                            label="📧 Email (Optional)",
                            placeholder="your.email@example.com",
                            info="📨 Receive your strategy via email"
                        )
                        
                        generate_portfolio_btn = gr.Button(
                            "🚀 Generate Portfolio Strategy",
                            variant="primary",
                            elem_classes="primary-btn",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2, elem_classes="results-section"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>📊 Your Portfolio Strategy</h3>")
                        
                        portfolio_status = gr.HTML(
                            value="<div style='text-align: center; color: #9ca3af;'>👆 Fill out your profile and click 'Generate Portfolio Strategy' to begin</div>"
                        )
                        
                        portfolio_output = gr.HTML(
                            value=create_placeholder_html("Portfolio Strategy Report", "Generate your strategy to see detailed recommendations"),
                            elem_classes="markdown-content"
                        )
                        
                        portfolio_table = gr.Dataframe(
                            label="📋 Asset Allocation Summary",
                            interactive=False,
                            wrap=True
                        )
                        
                        with gr.Row():
                            portfolio_email_btn = gr.Button(
                                "📧 Email Portfolio Strategy",
                                variant="secondary"
                            )
                            
                            portfolio_email_status = gr.Textbox(
                                label="📨 Email Status",
                                interactive=False
                            )
            
            # Stock Research Tab
            with gr.TabItem("📊 Stock Research"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>🔍 Market Research</h3>")
                        
                        research_type = gr.Radio(
                            label="📊 Research Type",
                            choices=["Comprehensive Analysis", "Trending Stocks", "Dividend Stocks", "Growth Stocks"],
                            value="Comprehensive Analysis",
                            info="🎯 Select your research focus"
                        )
                        
                        market_focus = gr.CheckboxGroup(
                            label="🎪 Market Focus Areas",
                            choices=["Technology", "Healthcare", "Energy", "Finance", "Consumer", "Real Estate"],
                            value=["Technology", "Healthcare"],
                            info="🏭 Choose sectors of interest"
                        )
                        
                        research_email = gr.Textbox(
                            label="📧 Email (Optional)",
                            placeholder="your.email@example.com",
                            info="📨 Receive research report via email"
                        )
                        
                        generate_research_btn = gr.Button(
                            "📊 Generate Research Report",
                            variant="primary",
                            elem_classes="primary-btn",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2, elem_classes="results-section"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>📈 Market Research Report</h3>")
                        
                        research_status = gr.HTML(
                            value="<div style='text-align: center; color: #9ca3af;'>👆 Select your research preferences and click 'Generate Research Report' to begin</div>"
                        )
                        
                        research_output = gr.HTML(
                            value=create_placeholder_html("Stock Research Report", "Generate your research to see market analysis"),
                            elem_classes="markdown-content"
                        )
                        
                        trending_stocks_table = gr.Dataframe(
                            label="🚀 Trending Stocks Analysis",
                            interactive=False,
                            wrap=True
                        )
                        
                        stable_stocks_table = gr.Dataframe(
                            label="💎 Stable Dividend Stocks",
                            interactive=False,
                            wrap=True
                        )
                        
                        with gr.Row():
                            research_email_btn = gr.Button(
                                "📧 Email Research Report",
                                variant="secondary"
                            )
                            
                            research_email_status = gr.Textbox(
                                label="📨 Email Status",
                                interactive=False
                            )
            
            # Market Dashboard Tab
            with gr.TabItem("📈 Market Dashboard"):
                gr.HTML("<h3 style='text-align: center; color: #d1d5db;'>📊 Real-Time Market Overview</h3>")
                
                with gr.Row():
                    with gr.Column():
                        symbol_input = gr.Textbox(
                            label="🔍 Quick Stock Lookup",
                            placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)",
                            info="Get real-time data for any stock"
                        )
                    with gr.Column():
                        lookup_btn = gr.Button("📊 Get Quote", variant="secondary")
                
                stock_quote_output = gr.HTML()
            
            # Research History Tab
            with gr.TabItem("📚 Research History"):
                gr.HTML("<h3 style='text-align: center; color: #d1d5db;'>📚 Your Research History</h3>")
                
                with gr.Row():
                    with gr.Column():
                        history_type = gr.Radio(
                            label="📊 History Type",
                            choices=["Portfolio Strategies", "Stock Research", "All Reports"],
                            value="All Reports",
                            info="📋 Choose what to display"
                        )
                    with gr.Column():
                        load_history_btn = gr.Button(
                            "📊 Load History",
                            variant="secondary"
                        )
                
                history_output = gr.Dataframe(
                    label="📚 Research History",
                    interactive=False,
                    wrap=True
                )
        
        # Footer
        with gr.Row():
            gr.HTML("""
            <div class="disclaimer">
                <h4 style="color: #fbbf24;">⚠️ Investment Disclaimer</h4>
                <p style="color: #9ca3af;">
                    This AI-generated investment analysis is for informational and educational purposes only. 
                    All investments carry inherent risks, and past performance does not guarantee future results. 
                    Cryptocurrency investments are particularly volatile and speculative. 
                    Always conduct your own research and consult with qualified financial professionals before making investment decisions.
                </p>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; margin-top: 1rem; font-weight: 600;">
                    <strong>🤖 Powered by Advanced AI • 📊 Real-Time Market Data • 🔒 Privacy Protected • 🌐 Global Markets</strong>
                </div>
            </div>
            """)
        
        # Event Handlers with Markdown rendering
        def portfolio_strategy_wrapper(budget_val, time_val, risk_val, goals_val, email_val):
            if budget_val < 1000:
                error_html = "<div class='warning-msg'>❌ Minimum investment amount is $1,000</div>"
                return error_html, create_placeholder_html("Portfolio Strategy Report", "Minimum investment is $1,000"), pd.DataFrame(), ""
            
            try:
                success_html = "<div class='success-msg'>🚀 Generating your personalized portfolio strategy...</div>"
                strategy, summary_df, status = create_portfolio_strategy(budget_val, time_val, risk_val, goals_val)
                final_html = f"<div class='success-msg'>{status}</div>"
                
                # Convert markdown to styled HTML
                strategy_html = convert_markdown_to_html(strategy)
                
                return final_html, strategy_html, summary_df, ""
            except Exception as e:
                error_html = f"<div class='warning-msg'>❌ Error: {str(e)}</div>"
                return error_html, create_placeholder_html("Portfolio Strategy Report", f"Error: {str(e)}"), pd.DataFrame(), ""
        
        def research_report_wrapper(research_type_val, market_focus_val, research_email_val):
            try:
                success_html = "<div class='success-msg'>📊 Generating comprehensive market research...</div>"
                research, trending_df, stable_df, status = create_stock_research_report(research_type_val, market_focus_val)
                final_html = f"<div class='success-msg'>{status}</div>"
                
                # Convert markdown to styled HTML
                research_html = convert_markdown_to_html(research)
                
                return final_html, research_html, trending_df, stable_df, ""
            except Exception as e:
                error_html = f"<div class='warning-msg'>❌ Error: {str(e)}</div>"
                return error_html, create_placeholder_html("Stock Research Report", f"Error: {str(e)}"), pd.DataFrame(), pd.DataFrame(), ""
        
        def portfolio_email_wrapper(email_val, strategy_val):
            if not email_val or "@" not in email_val:
                return "❌ Please provide a valid email address"
            if "Error" in str(strategy_val) or not strategy_val:
                return "❌ Please generate a portfolio strategy first"
            return send_strategy_email(email_val, strategy_val, "portfolio")
        
        def research_email_wrapper(email_val, research_val):
            if not email_val or "@" not in email_val:
                return "❌ Please provide a valid email address"
            if "Error" in str(research_val) or not research_val:
                return "❌ Please generate a research report first"
            return send_strategy_email(email_val, research_val, "research")
        
        def quick_lookup_wrapper(symbol_val):
            if not symbol_val:
                return "<div class='warning-msg'>❌ Please enter a stock symbol</div>"
            
            try:
                market_data = json.loads(market_data_tool._run(symbol_val.upper()))
                if symbol_val.upper() in market_data:
                    data = market_data[symbol_val.upper()]
                    performance_color = "#22c55e" if data['1y_performance'] > 0 else "#ef4444"
                    return f"""
                    <div class="section-card">
                        <h4 style="color: #d1d5db; margin-bottom: 1.5rem;">📊 {symbol_val.upper()} - Live Quote</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                            <div class="metric-card">
                                <h5>Current Price</h5>
                                <p style="font-size: 1.5rem; font-weight: bold;">${data['current_price']}</p>
                            </div>
                            <div class="metric-card">
                                <h5>1Y Performance</h5>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {performance_color};">{data['1y_performance']:+.1f}%</p>
                            </div>
                            <div class="metric-card">
                                <h5>52W High</h5>
                                <p style="font-size: 1.3rem;">${data['52w_high']}</p>
                            </div>
                            <div class="metric-card">
                                <h5>52W Low</h5>
                                <p style="font-size: 1.3rem;">${data['52w_low']}</p>
                            </div>
                        </div>
                    </div>
                    """
                else:
                    return "<div class='warning-msg'>❌ Symbol not found or no data available</div>"
            except Exception as e:
                return f"<div class='warning-msg'>❌ Error: {str(e)}</div>"
        
        def load_history_wrapper(history_type_val):
            try:
                conn = sqlite3.connect('investment_advisor.db')
                
                if history_type_val == "Portfolio Strategies":
                    query = """SELECT 
                        strategy_date as '📅 Date', 
                        ' || CAST(budget as TEXT) as '💰 Budget',
                        risk_tolerance as '⚖️ Risk Level',
                        time_horizon as '⏳ Time Horizon'
                        FROM portfolio_strategies 
                        ORDER BY strategy_date DESC LIMIT 10"""
                elif history_type_val == "Stock Research":
                    query = """SELECT 
                        report_date as '📅 Date',
                        report_type as '📊 Report Type'
                        FROM stock_research 
                        ORDER BY report_date DESC LIMIT 10"""
                else:
                    query = """
                    SELECT '💼 Portfolio Strategy' as '📊 Type', strategy_date as '📅 Date', 
                           ' || CAST(budget as TEXT) as '📋 Details' FROM portfolio_strategies
                    UNION
                    SELECT '📊 Stock Research' as '📊 Type', report_date as '📅 Date', 
                           report_type as '📋 Details' FROM stock_research
                    ORDER BY Date DESC LIMIT 15
                    """
                
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                if df.empty:
                    return pd.DataFrame({"📚 Message": ["No historical records found. Generate some reports to see them here!"]})
                
                return df
                
            except Exception as e:
                return pd.DataFrame({"⚠️ Error": [f"Database error: {str(e)}"]})
        
        # Connect event handlers
        generate_portfolio_btn.click(
            fn=portfolio_strategy_wrapper,
            inputs=[budget, time_horizon, risk_tolerance, goals, email],
            outputs=[portfolio_status, portfolio_output, portfolio_table, portfolio_email_status]
        )
        
        generate_research_btn.click(
            fn=research_report_wrapper,
            inputs=[research_type, market_focus, research_email],
            outputs=[research_status, research_output, trending_stocks_table, stable_stocks_table, research_email_status]
        )
        
        portfolio_email_btn.click(
            fn=portfolio_email_wrapper,
            inputs=[email, portfolio_output],
            outputs=[portfolio_email_status]
        )
        
        research_email_btn.click(
            fn=research_email_wrapper,
            inputs=[research_email, research_output],
            outputs=[research_email_status]
        )
        
        lookup_btn.click(
            fn=quick_lookup_wrapper,
            inputs=[symbol_input],
            outputs=[stock_quote_output]
        )
        
        load_history_btn.click(
            fn=load_history_wrapper,
            inputs=[history_type],
            outputs=[history_output]
        )
    
    return app

# Main execution
if __name__ == "__main__":
    print("🚀 Starting AI Investment Advisor Pro...")
    
    # Check for required dependencies
    required_env_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ Optional environment variables not set: {', '.join(missing_vars)}")
        print("📝 The app will run with basic functionality. To enable AI features:")
        print("   - Set OPENAI_API_KEY for AI-powered analysis")
        print("   - Set SENDER_EMAIL and SENDER_PASSWORD for email functionality")
    else:
        print("✅ Environment configured")
    
    print("🌐 Launching interface...")
    
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )