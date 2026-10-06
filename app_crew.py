"""
AI Investment Advisor - CrewAI Multi-Agent Version
Complete rewrite using CrewAI framework with specialized agents
"""

import gradio as gr
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import yfinance as yf
import sqlite3
import markdown
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import random
import time

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# ================================
# CONFIGURATION & STYLING
# ================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Force light mode for entire application */
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%) !important;
    color: #1e293b !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    padding: 3rem 2rem !important;
    border-radius: 20px !important;
    margin: 2rem 0 !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
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
    color: #374151 !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
    margin-bottom: 1rem !important;
}

.main-header p {
    color: #4b5563 !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
    max-width: 600px !important;
    margin: 0 auto 2rem !important;
}

.section-card {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease !important;
}

.metric-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%) !important;
    backdrop-filter: blur(10px) !important;
    color: white !important;
    padding: 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    margin: 0.5rem !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2) !important;
}

.success-msg {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.9) 0%, rgba(168, 230, 207, 0.9) 100%) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2) !important;
}

.warning-msg {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(245, 87, 108, 0.9) 100%) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    text-align: center !important;
    box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2) !important;
}

.markdown-content {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #1e293b !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 8px !important;
    padding: 1.5rem !important;
    line-height: 1.6 !important;
    height: 500px !important;
    overflow-y: auto !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05) !important;
}

/* Markdown content styling */
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

.markdown-content table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 1rem 0 !important;
    background: white !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.markdown-content table th {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    border-bottom: 2px solid rgba(102, 126, 234, 0.2) !important;
    text-align: left !important;
}

.markdown-content table td {
    color: #374151 !important;
    padding: 0.75rem !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
}

.markdown-content table tr:nth-child(even) {
    background: rgba(102, 126, 234, 0.02) !important;
}

/* Gradio dataframe tables */
.gradio-dataframe {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.gradio-dataframe table {
    background: white !important;
    color: #1e293b !important;
    width: 100% !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

.gradio-dataframe th {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(102, 126, 234, 0.2) !important;
    padding: 0.75rem !important;
    text-align: left !important;
}

.gradio-dataframe td {
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
    color: #374151 !important;
    padding: 0.75rem !important;
}

.primary-btn,
button.primary {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%) !important;
    color: white !important;
    padding: 0.875rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    border: none !important;
}

/* Force light mode for all gradio elements */
.gradio-container * {
    color-scheme: light !important;
}

/* Input elements */
input, textarea, select {
    background: white !important;
    color: #1e293b !important;
    border: 1px solid rgba(0, 0, 0, 0.2) !important;
    border-radius: 6px !important;
}

::placeholder {
    color: #9ca3af !important;
}

/* Labels */
label {
    color: #374151 !important;
    font-weight: 500 !important;
}
"""

# ================================
# DATABASE SETUP
# ================================

def init_database():
    """Initialize SQLite database for storing reports"""
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

# ================================
# CREWAI TOOLS
# ================================

class MarketDataTool(BaseTool):
    name: str = "market_data_tool"
    description: str = "Fetch real-time market data for stocks, ETFs, and cryptocurrencies"
    
    def _run(self, symbols: str) -> str:
        """Fetch market data for given symbols"""
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
            market_data = {}
            
            for symbol in symbol_list[:10]:  # Limit to 10 symbols
                try:
                    ticker = yf.Ticker(symbol)
                    hist_1m = ticker.history(period="1mo")
                    hist_1y = ticker.history(period="1y")
                    info = ticker.info
                    
                    if not hist_1m.empty:
                        current = hist_1m['Close'].iloc[-1]
                        start_1m = hist_1m['Close'].iloc[0]
                        start_1y = hist_1y['Close'].iloc[0] if not hist_1y.empty else current
                        performance_1m = ((current - start_1m) / start_1m) * 100
                        performance_1y = ((current - start_1y) / start_1y) * 100
                        
                        market_data[symbol] = {
                            "current_price": round(float(current), 2),
                            "1m_performance": round(float(performance_1m), 2),
                            "1y_performance": round(float(performance_1y), 2),
                            "market_cap": info.get("marketCap", 0),
                            "sector": info.get("sector", "Unknown"),
                            "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0
                        }
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    continue
            
            return json.dumps(market_data, indent=2)
        except Exception as e:
            return f"Error fetching market data: {str(e)}"

class MarketSentimentTool(BaseTool):
    name: str = "market_sentiment_tool"
    description: str = "Get overall market sentiment from major indices"
    
    def _run(self, input_text: str = "") -> str:
        """Get market sentiment"""
        try:
            indices = ["^GSPC", "^IXIC", "^DJI"]
            market_performance = []
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Close'].iloc[0]
                        performance = ((current - previous) / previous) * 100
                        market_performance.append(performance)
                except:
                    continue
            
            if market_performance:
                avg_performance = sum(market_performance) / len(market_performance)
                if avg_performance > 1:
                    sentiment = "BULLISH"
                elif avg_performance < -1:
                    sentiment = "BEARISH"
                else:
                    sentiment = "NEUTRAL"
            else:
                sentiment = "NEUTRAL"
                avg_performance = 0
            
            return json.dumps({
                "market_sentiment": sentiment,
                "market_performance": round(avg_performance, 2),
                "trending_themes": ["AI & Technology", "Healthcare Innovation", "Energy Transition"]
            })
        except Exception as e:
            return f"Error getting market sentiment: {str(e)}"

# ================================
# CREWAI AGENTS
# ================================

def create_llm():
    """Create OpenAI LLM instance"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

def create_portfolio_strategist():
    """Create Portfolio Strategist Agent"""
    return Agent(
        role='Portfolio Strategist',
        goal='Create personalized investment portfolio strategies based on user risk tolerance, time horizon, and goals',
        backstory="""You are an expert portfolio strategist with 15+ years of experience in asset allocation. 
        You specialize in creating diversified portfolios that balance risk and return based on individual investor profiles.
        You understand modern portfolio theory, risk management, and market dynamics.""",
        verbose=True,
        allow_delegation=False,
        llm=create_llm()
    )

def create_market_researcher():
    """Create Market Research Agent"""
    return Agent(
        role='Market Research Analyst',
        goal='Discover and analyze trending stocks, cryptocurrencies, and ETFs across different sectors',
        backstory="""You are a seasoned market research analyst with deep expertise in identifying trending assets 
        and emerging investment opportunities. You have access to real-time market data and can spot patterns 
        that indicate strong investment potential across stocks, cryptos, and ETFs.""",
        verbose=True,
        allow_delegation=False,
        tools=[MarketDataTool(), MarketSentimentTool()],
        llm=create_llm()
    )

def create_stock_analyst():
    """Create Individual Stock Analyst Agent"""
    return Agent(
        role='Senior Stock Analyst',
        goal='Provide deep analysis of individual stocks including company fundamentals, sentiment, and investment recommendations',
        backstory="""You are a senior equity research analyst with expertise in fundamental analysis, 
        technical analysis, and market sentiment evaluation. You can assess company financials, 
        competitive positioning, and growth prospects to provide actionable investment recommendations.""",
        verbose=True,
        allow_delegation=False,
        tools=[MarketDataTool()],
        llm=create_llm()
    )

def create_risk_analyst():
    """Create Risk Analysis Agent"""
    return Agent(
        role='Risk Assessment Specialist',
        goal='Evaluate investment risks and provide risk-adjusted recommendations',
        backstory="""You are a risk management expert who specializes in identifying and quantifying 
        investment risks. You understand market volatility, sector-specific risks, and can provide 
        risk-adjusted investment recommendations with appropriate position sizing.""",
        verbose=True,
        allow_delegation=False,
        tools=[MarketSentimentTool()],
        llm=create_llm()
    )

def create_report_writer():
    """Create Report Writing Agent"""
    return Agent(
        role='Investment Report Writer',
        goal='Create comprehensive, professional investment reports and strategies',
        backstory="""You are an expert financial writer who specializes in creating clear, actionable 
        investment reports. You can synthesize complex financial analysis into readable reports 
        that help investors make informed decisions.""",
        verbose=True,
        allow_delegation=False,
        llm=create_llm()
    )

# ================================
# CREWAI TASKS
# ================================

def create_portfolio_strategy_task(budget, time_horizon, risk_tolerance, goals):
    """Create portfolio strategy task"""
    return Task(
        description=f"""Create a personalized investment portfolio strategy with the following parameters:
        - Budget: ${budget:,.2f}
        - Time Horizon: {time_horizon}
        - Risk Tolerance: {risk_tolerance}
        - Investment Goals: {goals}
        
        Provide:
        1. Recommended asset allocation percentages
        2. Expected annual returns
        3. 5-year projection
        4. Risk assessment
        5. Rebalancing strategy
        6. Specific investment recommendations
        
        Format as a comprehensive markdown report.""",
        expected_output="A detailed investment portfolio strategy report in markdown format with asset allocation, projections, and recommendations.",
        agent=create_portfolio_strategist()
    )

def create_trending_research_task(research_type, sectors):
    """Create trending assets research task"""
    sectors_text = ", ".join(sectors) if sectors else "Technology, Healthcare"
    
    return Task(
        description=f"""Research and analyze {research_type.lower()} with focus on {sectors_text} sectors.
        
        Requirements:
        1. Identify 8-12 trending {research_type.lower()} with real ticker symbols
        2. Get real-time market data for each asset
        3. Analyze 1-month and 1-year performance
        4. Categorize as Strong Buy, Moderate Buy, Hold, or Sell based on performance percentiles
        5. Include market cap, sector, and founded year information
        6. Identify key catalysts and investment thesis for each
        
        Use the market_data_tool to get current prices and performance data.
        Format as a comprehensive markdown report with tables showing recommendations by rating.""",
        expected_output=f"A detailed {research_type} analysis report with buy/hold/sell ratings, market data, and investment recommendations in markdown format.",
        agent=create_market_researcher()
    )

def create_stock_analysis_task(symbol):
    """Create individual stock analysis task"""
    return Task(
        description=f"""Perform deep analysis of stock symbol {symbol.upper()}.
        
        Provide comprehensive analysis including:
        1. Company overview and business model
        2. Financial performance and metrics
        3. Recent developments and catalysts
        4. Competitive positioning
        5. Growth prospects and risks
        6. Market sentiment analysis
        7. Price targets and recommendations
        8. Investment thesis and rating (Strong Buy/Buy/Hold/Sell)
        
        Use market_data_tool to get current market data.
        Format as a detailed markdown research report.""",
        expected_output=f"A comprehensive stock analysis report for {symbol.upper()} with rating, price targets, and investment recommendations in markdown format.",
        agent=create_stock_analyst()
    )

def create_emerging_opportunities_task(research_type, sectors, exclude_symbols):
    """Create emerging opportunities discovery task"""
    sectors_text = ", ".join(sectors) if sectors else "Technology, Healthcare"
    exclude_text = f"Exclude these symbols: {', '.join(exclude_symbols)}" if exclude_symbols else ""
    
    return Task(
        description=f"""Discover LOW-COST emerging {research_type.lower()} opportunities in {sectors_text} sectors.
        {exclude_text}
        
        Focus on:
        - Stocks under $5 with 3x-10x upside potential (for stocks)
        - Cryptocurrencies under $1 with moon-shot potential (for cryptos)  
        - Innovative ETFs with unique strategies (for ETFs)
        - Recent catalysts: FDA approvals, partnerships, breakthrough tech
        - Small-cap opportunities with high growth potential
        
        Provide 4-6 emerging opportunities with:
        1. Current price and target price
        2. Upside potential calculation
        3. Key catalyst or recent development
        4. Investment thesis
        5. Risk assessment
        
        Format as markdown with a table showing the opportunities.""",
        expected_output=f"A focused report on emerging {research_type.lower()} opportunities with high upside potential in markdown format.",
        agent=create_market_researcher()
    )

# ================================
# CREWAI CREWS
# ================================

def create_portfolio_crew(budget, time_horizon, risk_tolerance, goals):
    """Create crew for portfolio strategy generation"""
    
    # Create agents
    strategist = create_portfolio_strategist()
    risk_analyst = create_risk_analyst()
    report_writer = create_report_writer()
    
    # Create tasks
    strategy_task = Task(
        description=f"""Analyze investment profile and create asset allocation strategy:
        - Budget: ${budget:,.2f}
        - Time Horizon: {time_horizon}
        - Risk Tolerance: {risk_tolerance}  
        - Goals: {goals}
        
        Create optimal asset allocation with percentages for: stocks, bonds, REITs, cash, growth stocks.
        Calculate expected returns and 5-year projections.""",
        expected_output="Asset allocation strategy with percentages and expected returns",
        agent=strategist
    )
    
    risk_task = Task(
        description=f"""Assess risks for the proposed portfolio strategy for {risk_tolerance} investor.
        Evaluate market risks, sector concentration, and provide risk management recommendations.""",
        expected_output="Risk assessment with mitigation strategies",
        agent=risk_analyst
    )
    
    report_task = Task(
        description="""Compile a comprehensive Investment Portfolio Strategy Report combining:
        1. Asset allocation recommendations
        2. Risk assessment
        3. Expected returns and projections
        4. Investment guidelines
        5. Rebalancing strategy
        
        Format as professional markdown report.""",
        expected_output="Complete portfolio strategy report in markdown format",
        agent=report_writer
    )
    
    return Crew(
        agents=[strategist, risk_analyst, report_writer],
        tasks=[strategy_task, risk_task, report_task],
        process=Process.sequential,
        verbose=True
    )

def create_research_crew(research_type, sectors):
    """Create crew for trending assets research"""
    
    researcher = create_market_researcher()
    risk_analyst = create_risk_analyst()
    report_writer = create_report_writer()
    
    research_task = create_trending_research_task(research_type, sectors)
    
    risk_task = Task(
        description=f"""Analyze risks for the identified {research_type.lower()} and provide:
        1. Risk ratings for each asset
        2. Portfolio concentration recommendations
        3. Position sizing guidance
        4. Risk management strategies""",
        expected_output="Risk analysis for trending assets with position sizing recommendations",
        agent=risk_analyst
    )
    
    emerging_task = create_emerging_opportunities_task(research_type, sectors, [])
    
    report_task = Task(
        description=f"""Create comprehensive {research_type} Analysis Report including:
        1. Executive summary with market outlook
        2. Trending assets analysis with buy/hold/sell ratings
        3. Emerging opportunities section
        4. Risk assessment and investment strategy
        5. Professional formatting with tables
        
        Combine all research into one cohesive report.""",
        expected_output=f"Complete {research_type} analysis report in markdown format",
        agent=report_writer
    )
    
    return Crew(
        agents=[researcher, risk_analyst, report_writer],
        tasks=[research_task, risk_task, emerging_task, report_task],
        process=Process.sequential,
        verbose=True
    )

def create_stock_analysis_crew(symbol):
    """Create crew for individual stock analysis"""
    
    analyst = create_stock_analyst()
    risk_analyst = create_risk_analyst()
    report_writer = create_report_writer()
    
    analysis_task = create_stock_analysis_task(symbol)
    
    risk_task = Task(
        description=f"""Assess investment risks for {symbol.upper()} including:
        1. Company-specific risks
        2. Sector and market risks  
        3. Technical risk factors
        4. Risk-adjusted position sizing recommendations""",
        expected_output=f"Risk assessment for {symbol.upper()} with position sizing guidance",
        agent=risk_analyst
    )
    
    report_task = Task(
        description=f"""Create comprehensive stock research report for {symbol.upper()} including:
        1. Company snapshot and business analysis
        2. Financial performance and metrics
        3. Investment recommendation with rating
        4. Risk assessment and position sizing
        5. Price targets and catalysts
        
        Format as professional research report.""",
        expected_output=f"Complete stock analysis report for {symbol.upper()} in markdown format",
        agent=report_writer
    )
    
    return Crew(
        agents=[analyst, risk_analyst, report_writer],
        tasks=[analysis_task, risk_task, report_task],
        process=Process.sequential,
        verbose=True
    )

# ================================
# MAIN FUNCTIONS
# ================================

def create_portfolio_strategy_with_crew(budget, time_horizon, risk_tolerance, goals):
    """Generate portfolio strategy using CrewAI"""
    
    if not os.getenv('OPENAI_API_KEY'):
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.", pd.DataFrame(), "API Key Required"
    
    try:
        print("🚀 Creating CrewAI Portfolio Strategy Crew...")
        crew = create_portfolio_crew(budget, time_horizon, risk_tolerance, goals)
        
        print("⚙️ Executing portfolio strategy tasks...")
        result = crew.kickoff()
        
        # Parse the result and create summary
        strategy_text = str(result)
        
        # Create summary dataframe (simplified allocation)
        allocations = {
            "Conservative": {"bonds": 40, "dividend_stocks": 35, "reits": 20, "cash": 5},
            "Moderate": {"stocks": 50, "bonds": 25, "reits": 20, "cash": 5},
            "Aggressive": {"stocks": 65, "growth_stocks": 20, "reits": 10, "cash": 5}
        }
        
        allocation = allocations.get(risk_tolerance, allocations["Moderate"])
        summary_data = []
        
        for asset, percentage in allocation.items():
            amount = (percentage / 100) * budget
            summary_data.append({
                "Asset Class": asset.replace("_", " ").title(),
                "Allocation": f"{percentage}%",
                "Amount": f"${amount:,.0f}",
                "Risk Level": "Low" if "bond" in asset or "cash" in asset else "Medium"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to database
        save_report_to_db(strategy_text, "CrewAI_Portfolio_Strategy", [risk_tolerance])
        
        return strategy_text, summary_df, "CrewAI portfolio strategy generated successfully!"
        
    except Exception as e:
        error_msg = f"CrewAI Portfolio Strategy error: {str(e)}"
        print(error_msg)
        return error_msg, pd.DataFrame(), error_msg

def create_trending_research_with_crew(research_type, sectors):
    """Generate trending research using CrewAI"""
    
    if not os.getenv('OPENAI_API_KEY'):
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.", pd.DataFrame(), pd.DataFrame(), "API Key Required"
    
    if not sectors:
        sectors = ["Technology", "Healthcare"]
    
    try:
        print(f"🚀 Creating CrewAI {research_type} Research Crew...")
        crew = create_research_crew(research_type, sectors)
        
        print(f"⚙️ Executing {research_type} research tasks...")
        result = crew.kickoff()
        
        report = str(result)
        
        # Create sample dataframes (in real implementation, parse from crew results)
        trending_data = []
        stable_data = []
        
        # Add sample data for demonstration
        if research_type == "Trending Stocks":
            trending_data = [
                {"Symbol": "NVDA", "Name": "NVIDIA Corp", "Price": "$875.50", "Type": "Stock", "Sector": "Technology", "1M Return": "+12.5%", "Rating": "🟢 Strong Buy", "Market Cap": "$2.1T"},
                {"Symbol": "TSLA", "Name": "Tesla Inc", "Price": "$248.90", "Type": "Stock", "Sector": "Consumer", "1M Return": "+8.3%", "Rating": "🟡 Moderate Buy", "Market Cap": "$789B"}
            ]
            stable_data = [
                {"Symbol": "AAPL", "Name": "Apple Inc", "Price": "$182.30", "Type": "Stock", "Sector": "Technology", "1M Return": "+2.1%", "Rating": "🔵 Hold"}
            ]
        
        trending_df = pd.DataFrame(trending_data)
        stable_df = pd.DataFrame(stable_data)
        
        # Save to database
        save_report_to_db(report, f"CrewAI_{research_type}", sectors)
        
        return report, trending_df, stable_df, f"CrewAI {research_type} analysis completed!"
        
    except Exception as e:
        error_msg = f"CrewAI {research_type} Research error: {str(e)}"
        print(error_msg)
        return error_msg, pd.DataFrame(), pd.DataFrame(), error_msg

def create_stock_analysis_with_crew(symbol):
    """Generate individual stock analysis using CrewAI"""
    
    if not symbol:
        return "Please enter a stock symbol"
    
    if not os.getenv('OPENAI_API_KEY'):
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
    
    try:
        print(f"🚀 Creating CrewAI Stock Analysis Crew for {symbol.upper()}...")
        crew = create_stock_analysis_crew(symbol)
        
        print(f"⚙️ Executing stock analysis tasks for {symbol.upper()}...")
        result = crew.kickoff()
        
        analysis_report = str(result)
        
        # Save to database
        save_report_to_db(analysis_report, f"CrewAI_Stock_Analysis_{symbol.upper()}", [])
        
        return analysis_report
        
    except Exception as e:
        return f"CrewAI Stock Analysis error for {symbol.upper()}: {str(e)}"

# ================================
# HELPER FUNCTIONS
# ================================

def save_report_to_db(report, report_type, sectors):
    """Save report to database"""
    try:
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO stock_research (report_date, report_content, report_type) VALUES (?, ?, ?)",
            (datetime.now().date(), report, f"{report_type}_{'+'.join(sectors) if sectors else 'General'}")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

def convert_markdown_to_html(markdown_text):
    """Convert markdown text to styled HTML"""
    if not markdown_text or markdown_text.startswith("❌"):
        return create_placeholder_html("CrewAI Report", "Please generate a report to see CrewAI results here")
    
    try:
        html_content = markdown.markdown(markdown_text, extensions=['nl2br', 'tables'])
        return f"<div class='markdown-content'>{html_content}</div>"
    except Exception as e:
        return create_placeholder_html("CrewAI Report", f"Error rendering report: {str(e)}")

def create_placeholder_html(title, description):
    """Create placeholder HTML for empty reports"""
    return f"""
    <div class='markdown-content' style='display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 300px;'>
        <div style='font-weight: 600; font-size: 1.1rem; color: #374151; margin-bottom: 0.5rem;'>{title}</div>
        <div style='color: #6b7280;'>{description}</div>
    </div>
    """

def get_stock_quote(symbol):
    """Get stock quote for individual lookup"""
    if not symbol:
        return "<div class='warning-msg'>Please enter a stock symbol</div>"
    
    try:
        tool = MarketDataTool()
        market_data_json = tool._run(symbol.upper())
        market_data = json.loads(market_data_json)
        
        if symbol.upper() in market_data:
            data = market_data[symbol.upper()]
            performance_1m_color = "#22c55e" if data['1m_performance'] > 0 else "#ef4444"
            return f"""
            <div class="section-card">
                <h4 style="color: #374151; margin-bottom: 1.5rem;">{symbol.upper()} - CrewAI Live Quote</h4>
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
                        <h5>1Y Performance</h5>
                        <p style="font-size: 1.3rem;">{data.get('1y_performance', 0):+.1f}%</p>
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
        else:
            return "<div class='warning-msg'>Symbol not found or no data available</div>"
    except Exception as e:
        return f"<div class='warning-msg'>Error fetching data: {str(e)}</div>"

# Database functions (simplified versions)
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
        
        subject = f"Your CrewAI {content_type.title()} Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        msg = MIMEText(content, 'html', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = email
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"CrewAI {content_type.title()} report sent successfully to {email}!"
        
    except Exception as e:
        return f"Failed to send email: {str(e)}"

# ================================
# GRADIO APPLICATION
# ================================

def create_crewai_gradio_app():
    """Create the CrewAI Gradio application"""
    
    init_database()
    
    with gr.Blocks(title="AI Investment Advisor Pro - CrewAI", theme="default", css=CUSTOM_CSS) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>AI Investment Advisor Pro</h1>
            <h2>CrewAI Multi-Agent Investment Intelligence</h2>
            <p>Powered by specialized AI agents working together to provide comprehensive financial analysis and portfolio strategies.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
                <div class="metric-card">
                    <h3>Portfolio Strategist</h3>
                    <p>Asset Allocation Expert</p>
                </div>
                <div class="metric-card">
                    <h3>Market Researcher</h3>
                    <p>Trending Assets Discovery</p>
                </div>
                <div class="metric-card">
                    <h3>Stock Analyst</h3>
                    <p>Deep Company Analysis</p>
                </div>
                <div class="metric-card">
                    <h3>Risk Specialist</h3>
                    <p>Risk Assessment</p>
                </div>
            </div>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Portfolio Strategy Tab
            with gr.TabItem("Portfolio Strategy"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #374151; text-align: center;'>Investment Profile</h3>")
                        
                        budget = gr.Number(
                            label="Initial Investment Amount ($)",
                            value=10000,
                            minimum=1000,
                            maximum=10000000
                        )
                        
                        time_horizon = gr.Radio(
                            label="Investment Time Horizon",
                            choices=["Short-term", "Medium-term", "Long-term"],
                            value="Medium-term"
                        )
                        
                        risk_tolerance = gr.Radio(
                            label="Risk Tolerance",
                            choices=["Conservative", "Moderate", "Aggressive"],
                            value="Moderate"
                        )
                        
                        goals = gr.Textbox(
                            label="Investment Goals & Interests",
                            placeholder="e.g., retirement, growth, income...",
                            lines=3
                        )
                        
                        email = gr.Textbox(
                            label="📧 Email (Optional)",
                            placeholder="your.email@example.com"
                        )
                        
                        generate_portfolio_btn = gr.Button(
                            "Generate CrewAI Portfolio Strategy",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        portfolio_status = gr.HTML(
                            value="""
                            <div style='text-align: center;'>
                                <h3 style='color: #374151; margin-bottom: 0.5rem;'>CrewAI Portfolio Strategy</h3>
                                <div style='color: #6b7280;'>CrewAI agents will collaborate to create your personalized strategy</div>
                            </div>
                            """
                        )
                        
                        portfolio_output = gr.HTML(
                            value=create_placeholder_html("CrewAI Portfolio Strategy", "Multiple AI agents will analyze your profile and create an optimal investment strategy")
                        )
                        portfolio_table = gr.Dataframe(label="Asset Allocation Summary")
                        
                        with gr.Row():
                            portfolio_email_btn = gr.Button("Email Report", variant="primary")
                            portfolio_email_status = gr.Textbox(label="Email Status", interactive=False)
            
            # Trending Assets Research Tab
            with gr.TabItem("Trending Assets Report"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #374151; text-align: center;'>Market Research Crew</h3>")
                        
                        research_type = gr.Radio(
                            label="Research Type",
                            choices=["Trending Stocks", "Trending Cryptos", "Trending ETFs"],
                            value="Trending Stocks"
                        )
                        
                        market_focus = gr.CheckboxGroup(
                            label="Market Focus Areas",
                            choices=["Technology", "Healthcare", "Energy", "Finance", "Consumer", "Real Estate"],
                            value=["Technology"]
                        )
                        
                        research_email = gr.Textbox(
                            label="Email (Optional)",
                            placeholder="your.email@example.com"
                        )
                        
                        generate_research_btn = gr.Button(
                            "Deploy CrewAI Research Team",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        research_status = gr.HTML(
                            value="""
                            <div style='text-align: center;'>
                                <h3 style='color: #374151; margin-bottom: 0.5rem;'>CrewAI Market Research</h3>
                                <div style='color: #6b7280;'>Market Research, Risk Analysis, and Report Writing agents will collaborate</div>
                            </div>
                            """
                        )
                        
                        stock_analysis_output = gr.HTML(
                            value=create_placeholder_html("CrewAI Market Research", "Specialized agents will discover trends, assess risks, and create comprehensive reports")
                        )
                        
                        trending_table = gr.Dataframe(
                            label="Trending Recommendations (All Ratings)", 
                            headers=["Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating", "Market Cap"]
                        )
                        stable_table = gr.Dataframe(
                            label="Additional Analysis (Hold/Sell Ratings)", 
                            headers=["Symbol", "Name", "Price", "Type", "Sector", "1M Return", "Rating"]
                        )
                        
                        with gr.Row():
                            research_email_btn = gr.Button("Email Research Report", variant="primary")
                            research_email_status = gr.Textbox(label="Email Status", interactive=False)
            
            # Stock Lookup Tab
            with gr.TabItem("Stock Analysis"):
                gr.HTML("<h3 style='text-align: center; color: #374151;'>CrewAI Stock Analysis Crew</h3>")
                
                with gr.Row():
                    symbol_input = gr.Textbox(
                        label="Enter Stock Symbol",
                        placeholder="e.g., AAPL, TSLA, MSFT, NVDA",
                        scale=3
                    )
                    with gr.Column(scale=1):
                        lookup_btn = gr.Button("Get Quote", variant="primary")
                        analysis_btn = gr.Button("Deploy Analysis Crew", variant="primary")
                
                quote_output = gr.HTML()
                
                analysis_output = gr.HTML(
                    value=create_placeholder_html("CrewAI Stock Analysis", "Stock Analyst, Risk Specialist, and Report Writer will collaborate for deep analysis")
                )
            
            # Research History Tab
            with gr.TabItem("Research History"):
                gr.HTML("<h3 style='text-align: center; color: #374151;'>CrewAI Analysis History</h3>")
                
                with gr.Row():
                    refresh_history_btn = gr.Button("Refresh History", variant="secondary")
                
                with gr.Tabs():
                    with gr.TabItem("Stock Research History"):
                        stock_history_table = gr.Dataframe(
                            label="Recent CrewAI Analysis Reports",
                            headers=["ID", "Date", "Type", "Preview"]
                        )
                        
                        with gr.Row():
                            stock_report_id = gr.Number(label="Report ID", precision=0)
                            view_stock_report_btn = gr.Button("View Report", variant="primary")
                        
                        stock_report_display = gr.HTML(
                            value=create_placeholder_html("Full CrewAI Report", "Select a report ID to view complete analysis")
                        )
                    
                    with gr.TabItem("Portfolio Strategy History"):
                        portfolio_history_table = gr.Dataframe(
                            label="Recent CrewAI Portfolio Strategies", 
                            headers=["ID", "Date", "Budget", "Risk", "Horizon", "Preview"]
                        )
                        
                        with gr.Row():
                            portfolio_report_id = gr.Number(label="Strategy ID", precision=0)
                            view_portfolio_report_btn = gr.Button("View Strategy", variant="primary")
                        
                        portfolio_report_display = gr.HTML(
                            value=create_placeholder_html("Full CrewAI Strategy", "Select a strategy ID to view complete analysis")
                        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(255, 255, 255, 0.8); border-radius: 16px;">
            <h4 style="color: #f59e0b;">⚠️ Investment Disclaimer</h4>
            <p style="color: #6b7280;">
                This CrewAI-generated investment analysis is for informational and educational purposes only. 
                Multiple AI agents collaborate to provide comprehensive analysis, but all investments carry inherent risks. 
                Always conduct your own research and consult with qualified financial professionals.
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; margin-top: 1rem; font-weight: 600;">
                <strong>Powered by CrewAI Multi-Agent Framework • Portfolio Strategist • Market Researcher • Stock Analyst • Risk Specialist • Report Writer</strong>
            </div>
        </div>
        """)
        
        # ================================
        # EVENT HANDLERS
        # ================================
        
        def portfolio_wrapper(budget_val, time_val, risk_val, goals_val):
            """Handle CrewAI portfolio strategy generation"""
            if budget_val < 1000:
                return "<div class='warning-msg'>Minimum investment is $1,000</div>", create_placeholder_html("Error", "Minimum investment required"), pd.DataFrame()
            
            try:
                strategy, summary_df, status = create_portfolio_strategy_with_crew(budget_val, time_val, risk_val, goals_val)
                status_html = f"<div class='success-msg'>CrewAI Success: {status}</div>"
                strategy_html = convert_markdown_to_html(strategy)
                return status_html, strategy_html, summary_df
            except Exception as e:
                error_html = f"<div class='warning-msg'>CrewAI Error: {str(e)}</div>"
                return error_html, create_placeholder_html("CrewAI Error", str(e)), pd.DataFrame()
        
        def research_wrapper(research_type_val, market_focus_val):
            """Handle CrewAI trending research generation"""
            try:
                report, trending_df, stable_df, status = create_trending_research_with_crew(research_type_val, market_focus_val)
                status_html = f"<div class='success-msg'>CrewAI Success: {status}</div>"
                report_html = convert_markdown_to_html(report)
                return status_html, report_html, trending_df, stable_df
            except Exception as e:
                error_html = f"<div class='warning-msg'>CrewAI Error: {str(e)}</div>"
                placeholder = create_placeholder_html("CrewAI Error", str(e))
                return error_html, placeholder, pd.DataFrame(), pd.DataFrame()
        
        def lookup_wrapper(symbol_val):
            """Handle stock quote lookup"""
            return get_stock_quote(symbol_val)
        
        def analysis_wrapper(symbol_val):
            """Handle CrewAI stock analysis"""
            if not symbol_val:
                return "<div class='warning-msg'>Please enter a stock symbol</div>"
            
            try:
                analysis_report = create_stock_analysis_with_crew(symbol_val.upper())
                analysis_html = convert_markdown_to_html(analysis_report)
                return analysis_html
            except Exception as e:
                return f"<div class='warning-msg'>CrewAI analysis error for {symbol_val.upper()}: {str(e)}</div>"
        
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
            return send_email_report(email_val, research_report, "research")
        
        # Connect event handlers
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
        
        analysis_btn.click(
            fn=analysis_wrapper,
            inputs=[symbol_input],
            outputs=[analysis_output]
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
        
        # Load history on startup
        try:
            stock_df, portfolio_df = get_research_history()
            stock_history_table.value = stock_df
            portfolio_history_table.value = portfolio_df
        except:
            pass
    
    return app

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print("Starting AI Investment Advisor Pro with CrewAI...")
    
    # Check dependencies
    try:
        import crewai
        import yfinance
        import pandas
        import markdown
        print("All required packages installed")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install: pip install crewai gradio yfinance pandas markdown python-dotenv langchain-openai")
        exit(1)
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("OpenAI API key detected - CrewAI agents enabled")
    else:
        print("No OPENAI_API_KEY found - Add to .env file to enable CrewAI agents")
    
    app = create_crewai_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )