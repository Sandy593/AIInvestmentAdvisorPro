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

# Load environment variables
load_dotenv()

# Custom CSS for enhanced UI
custom_css = """
/* Main styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.main-header {
    text-align: center;
    background: rgba(255, 255, 255, 0.95);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.main-header h1 {
    background: linear-gradient(45deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.section-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.results-section {
    background: rgba(255, 255, 255, 0.98);
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
}

.tab-nav {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 10px;
    padding: 0.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 0.5rem;
}

.success-msg {
    background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

.warning-msg {
    background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    padding: 12px 30px !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.secondary-btn {
    background: linear-gradient(45deg, #4facfe, #00f2fe) !important;
    border: none !important;
    color: white !important;
    padding: 10px 25px !important;
    border-radius: 20px !important;
    font-weight: bold !important;
}

/* Input styling */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}
"""

# Initialize database for stock research
def init_database():
    conn = sqlite3.connect('investment_advisor.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_research (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE,
            report_content TEXT,
            report_type TEXT
        )
    ''')
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
    conn.commit()
    conn.close()

# Enhanced Investment Tools
class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze investment portfolio based on user preferences"
    
    def _run(self, budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> str:
        try:
            # Enhanced asset allocation with more sophisticated logic
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
            
            # Add expected returns based on historical data
            expected_returns = {
                "stocks": 8.5, "bonds": 3.5, "dividend_stocks": 6.5, 
                "reits": 7.0, "crypto": 15.0, "gold": 4.0
            }
            
            # Calculate weighted expected return
            portfolio_return = sum(
                (allocation.get(asset, 0) / 100) * expected_returns.get(asset, 0)
                for asset in expected_returns
            )
            
            return json.dumps({
                "budget": budget,
                "allocation": allocation,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "goals": goals,
                "expected_annual_return": round(portfolio_return, 2),
                "projected_value_5y": round(budget * (1 + portfolio_return/100)**5, 2)
            }, indent=2)
        except Exception as e:
            return f"Error in portfolio analysis: {str(e)}"

class StockResearchTool(BaseTool):
    name: str = "stock_research"
    description: str = "Comprehensive stock market research and analysis"
    
    def _run(self, query: str = "trending stocks government funding") -> str:
        try:
            # Get trending stocks with government backing
            trending_stocks = [
                {"symbol": "NEE", "name": "NextEra Energy", "sector": "Renewable Energy"},
                {"symbol": "TSLA", "name": "Tesla Inc", "sector": "Electric Vehicles"},
                {"symbol": "QS", "name": "QuantumScape", "sector": "Battery Technology"},
                {"symbol": "RIVN", "name": "Rivian Automotive", "sector": "Electric Vehicles"},
                {"symbol": "ENPH", "name": "Enphase Energy", "sector": "Solar Technology"}
            ]
            
            # Get stable dividend stocks
            stable_stocks = [
                {"symbol": "AAPL", "name": "Apple Inc", "sector": "Technology"},
                {"symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology"},
                {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
                {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Goods"},
                {"symbol": "KO", "name": "Coca-Cola", "sector": "Beverages"}
            ]
            
            # Get crypto recommendations
            crypto_picks = [
                {"symbol": "BTC", "name": "Bitcoin", "category": "Digital Gold"},
                {"symbol": "ETH", "name": "Ethereum", "category": "Smart Contracts"},
                {"symbol": "BNB", "name": "Binance Coin", "category": "Exchange Token"},
                {"symbol": "ADA", "name": "Cardano", "category": "Proof of Stake"},
                {"symbol": "SOL", "name": "Solana", "category": "High Performance"}
            ]
            
            return json.dumps({
                "trending_stocks": trending_stocks,
                "stable_stocks": stable_stocks,
                "crypto_picks": crypto_picks,
                "research_date": datetime.now().strftime("%Y-%m-%d")
            }, indent=2)
        except Exception as e:
            return f"Error in stock research: {str(e)}"

class MarketDataTool(BaseTool):
    name: str = "market_data"
    description: str = "Get real-time market data for stocks and crypto"
    
    def _run(self, symbols: str) -> str:
        try:
            symbol_list = symbols.split(",") if isinstance(symbols, str) else symbols
            market_data = {}
            
            for symbol in symbol_list[:15]:  # Limit to 15 symbols
                symbol = symbol.strip()
                try:
                    if symbol in ["BTC", "ETH", "BNB", "ADA", "SOL"]:
                        # Crypto data
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
                        # Stock data
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
                                "market_cap": info.get("marketCap", "N/A"),
                                "pe_ratio": info.get("trailingPE", "N/A"),
                                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                                "type": "stock"
                            }
                except:
                    continue
            
            return json.dumps(market_data, indent=2)
        except Exception as e:
            return f"Error getting market data: {str(e)}"

# Initialize tools and agents
portfolio_tool = PortfolioAnalysisTool()
stock_research_tool = StockResearchTool()
market_data_tool = MarketDataTool()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Enhanced agents
portfolio_advisor = Agent(
    role="Senior Portfolio Advisor",
    goal="Create personalized investment portfolios optimized for passive income and long-term growth",
    backstory="You are a seasoned portfolio manager with 20+ years of experience in asset allocation, risk management, and passive income strategies. You specialize in creating tax-efficient portfolios.",
    tools=[portfolio_tool, market_data_tool],
    llm=llm,
    verbose=True
)

stock_research_agent = Agent(
    role="Senior Equity Research Analyst",
    goal="Provide comprehensive stock market analysis and identify trending investment opportunities",
    backstory="You are a top-tier equity research analyst with expertise in fundamental analysis, technical analysis, and market trends. You have a proven track record of identifying winning stocks.",
    tools=[stock_research_tool, market_data_tool],
    llm=llm,
    verbose=True
)

def create_portfolio_strategy(budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> Tuple[str, pd.DataFrame, str]:
    """Create comprehensive portfolio strategy"""
    
    strategy_task = Task(
        description=f"""
        Create a comprehensive investment portfolio strategy for:
        - Budget: ${budget:,.2f}
        - Time Horizon: {time_horizon}
        - Risk Tolerance: {risk_tolerance}
        - Goals: {goals}
        
        Your analysis should include:
        1. Optimal asset allocation with rationale
        2. Specific investment recommendations (ETFs, stocks, REITs)
        3. Expected returns and risk metrics
        4. Tax optimization strategies
        5. Rebalancing recommendations
        6. Implementation timeline
        
        Focus on passive income generation through dividends, REITs, and bonds.
        Provide specific ticker symbols and allocation percentages.
        """,
        agent=portfolio_advisor,
        expected_output="Detailed portfolio strategy with specific recommendations and implementation plan"
    )
    
    crew = Crew(
        agents=[portfolio_advisor],
        tasks=[strategy_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        
        # Create allocation summary
        allocation_data = portfolio_tool._run(budget, time_horizon, risk_tolerance, goals)
        allocation_info = json.loads(allocation_data)
        allocation = allocation_info["allocation"]
        
        summary_data = []
        risk_levels = {
            "bonds": "🟢 Low", "dividend_stocks": "🟡 Low-Med", "stocks": "🟠 Medium", 
            "reits": "🟠 Medium", "crypto": "🔴 High", "gold": "🟢 Low"
        }
        
        for asset, percentage in allocation.items():
            amount = (percentage / 100) * budget
            risk_level = risk_levels.get(asset, "🟠 Medium")
            summary_data.append({
                "💼 Asset Class": asset.replace("_", " ").title(),
                "📊 Allocation": f"{percentage}%",
                "💰 Amount": f"${amount:,.0f}",
                "⚠️ Risk Level": risk_level,
                "📈 Expected Return": f"{[8.5, 3.5, 6.5, 7.0, 15.0, 4.0][list(risk_levels.keys()).index(asset)] if asset in risk_levels else 7.0}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to database
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO portfolio_strategies (strategy_date, budget, risk_tolerance, time_horizon, strategy_content) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().date(), budget, risk_tolerance, time_horizon, str(result))
        )
        conn.commit()
        conn.close()
        
        return str(result), summary_df, f"✅ Portfolio strategy generated successfully! Expected annual return: {allocation_info.get('expected_annual_return', 7.5)}%"
        
    except Exception as e:
        error_msg = f"❌ Error generating strategy: {str(e)}"
        empty_df = pd.DataFrame({"⚠️ Error": [error_msg]})
        return error_msg, empty_df, error_msg

def create_stock_research_report() -> Tuple[str, pd.DataFrame, pd.DataFrame, str]:
    """Generate comprehensive stock research report"""
    
    research_task = Task(
        description="""
        Generate a comprehensive stock research report including:
        
        1. Market Overview & Trends
        2. Top 5 Trending Stocks (government-backed or high-growth)
        3. Top 5 Stable Dividend Stocks
        4. Top 5 Cryptocurrency Recommendations
        5. Sector Analysis & Opportunities
        6. Risk Assessment & Market Outlook
        
        For each recommendation, provide:
        - Current price and 52-week range
        - Key financial metrics
        - Investment thesis
        - Risk factors
        - Price target and timeline
        
        Use real market data and provide specific buy/hold/sell recommendations.
        """,
        agent=stock_research_agent,
        expected_output="Comprehensive stock research report with specific recommendations and market analysis"
    )
    
    crew = Crew(
        agents=[stock_research_agent],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        
        # Get market data for trending stocks
        trending_symbols = "NEE,TSLA,QS,RIVN,ENPH"
        stable_symbols = "AAPL,MSFT,JNJ,PG,KO"
        crypto_symbols = "BTC,ETH,BNB,ADA,SOL"
        
        trending_data = json.loads(market_data_tool._run(trending_symbols))
        stable_data = json.loads(market_data_tool._run(stable_symbols))
        crypto_data = json.loads(market_data_tool._run(crypto_symbols))
        
        # Create trending stocks table
        trending_table_data = []
        stock_names = {"NEE": "NextEra Energy", "TSLA": "Tesla", "QS": "QuantumScape", "RIVN": "Rivian", "ENPH": "Enphase"}
        
        for symbol, data in trending_data.items():
            trending_table_data.append({
                "🏢 Company": f"{stock_names.get(symbol, symbol)} ({symbol})",
                "💰 Price": f"${data['current_price']}",
                "📈 52W High": f"${data['52w_high']}",
                "📉 52W Low": f"${data['52w_low']}",
                "📊 1Y Return": f"{data['1y_performance']:+.1f}%",
                "🎯 Recommendation": "🟢 BUY" if data['1y_performance'] > 10 else "🟡 HOLD"
            })
        
        trending_df = pd.DataFrame(trending_table_data)
        
        # Create stable stocks table
        stable_table_data = []
        stable_names = {"AAPL": "Apple", "MSFT": "Microsoft", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "KO": "Coca-Cola"}
        
        for symbol, data in stable_data.items():
            stable_table_data.append({
                "🏢 Company": f"{stable_names.get(symbol, symbol)} ({symbol})",
                "💰 Price": f"${data['current_price']}",
                "📈 52W High": f"${data['52w_high']}",
                "📉 52W Low": f"${data['52w_low']}",
                "💎 Dividend": f"{data.get('dividend_yield', 0):.1f}%",
                "📊 1Y Return": f"{data['1y_performance']:+.1f}%",
                "🎯 Rating": "🟢 STRONG BUY" if data.get('dividend_yield', 0) > 2 else "🟡 BUY"
            })
        
        stable_df = pd.DataFrame(stable_table_data)
        
        # Save to database
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO stock_research (report_date, report_content, report_type) VALUES (?, ?, ?)",
            (datetime.now().date(), str(result), "comprehensive_research")
        )
        conn.commit()
        conn.close()
        
        return str(result), trending_df, stable_df, "✅ Stock research report generated successfully!"
        
    except Exception as e:
        error_msg = f"❌ Error generating research: {str(e)}"
        empty_df = pd.DataFrame({"⚠️ Error": [error_msg]})
        return error_msg, empty_df, empty_df, error_msg

def send_strategy_email(email: str, content: str, content_type: str = "portfolio") -> str:
    """Enhanced email sending with better formatting"""
    try:
        if not email or "@" not in email:
            return "❌ Please provide a valid email address"
        
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
        if not all([sender_email, sender_password]):
            return "❌ Email credentials not configured"
        
        subject = f"🎯 Your {'Portfolio Strategy' if content_type == 'portfolio' else 'Stock Research Report'} - {datetime.now().strftime('%Y-%m-%d')}"
        
        email_body = f"""
🎯 AI Investment Advisor Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{content}

---
📊 This report was generated by your AI Investment Advisor
⚠️ Please consult with a qualified financial advisor before making investment decisions
💡 Past performance does not guarantee future results

Best regards,
Your AI Investment Team 🤖
        """
        
        msg = MIMEText(email_body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = email
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"✅ {content_type.title()} report sent successfully to {email}!"
        
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

def create_enhanced_gradio_app():
    """Create the enhanced Gradio application"""
    
    # Initialize database
    init_database()
    
    with gr.Blocks(
        title="🎯 AI Investment Advisor Pro",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        with gr.Row():
            gr.HTML("""
            <div class="main-header fade-in">
                <h1>🎯 AI Investment Advisor Pro</h1>
                <h2>🚀 Advanced Portfolio Management & Stock Research Platform</h2>
                <p style="font-size: 1.2rem; color: #666;">Get personalized investment strategies and comprehensive market analysis powered by AI</p>
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
                    <div class="metric-card">
                        <h3>💰 Portfolio Builder</h3>
                        <p>Custom Strategies</p>
                    </div>
                    <div class="metric-card">
                        <h3>📊 Stock Research</h3>
                        <p>AI-Powered Analysis</p>
                    </div>
                    <div class="metric-card">
                        <h3>📧 Reports</h3>
                        <p>Email Delivery</p>
                    </div>
                </div>
            </div>
            """)
        
        # Main application tabs
        with gr.Tabs(elem_classes="tab-nav") as tabs:
            
            # Portfolio Strategy Tab
            with gr.TabItem("💼 Portfolio Strategy", elem_id="portfolio-tab"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='text-align: center; color: #667eea;'>📝 Investment Profile</h3>")
                        
                        budget = gr.Number(
                            label="💰 Initial Investment Amount ($)",
                            value=25000,
                            minimum=1000,
                            maximum=10000000,
                            info="💡 Minimum investment: $1,000"
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
                        gr.HTML("<h3 style='text-align: center; color: #667eea;'>📊 Your Personalized Portfolio</h3>")
                        
                        portfolio_status = gr.HTML(
                            visible=True,
                            value="<div style='text-align: center; color: #666;'>👆 Fill out your profile and click 'Generate Portfolio Strategy' to begin</div>"
                        )
                        
                        portfolio_output = gr.Textbox(
                            label="📈 Portfolio Strategy Report",
                            lines=12,
                            max_lines=25,
                            interactive=False,
                            info="🎯 Detailed investment strategy and recommendations"
                        )
                        
                        portfolio_table = gr.Dataframe(
                            label="📋 Asset Allocation Summary",
                            interactive=False,
                            wrap=True
                        )
                        
                        with gr.Row():
                            portfolio_email_btn = gr.Button(
                                "📧 Email Portfolio Strategy",
                                variant="secondary",
                                elem_classes="secondary-btn"
                            )
                            
                            portfolio_email_status = gr.Textbox(
                                label="📨 Email Status",
                                interactive=False,
                                visible=True
                            )
            
            # Stock Research Tab
            with gr.TabItem("📊 Stock Research", elem_id="research-tab"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='text-align: center; color: #667eea;'>🔍 Market Research</h3>")
                        
                        research_type = gr.Radio(
                            label="📊 Research Type",
                            choices=["Comprehensive Analysis", "Trending Stocks", "Dividend Stocks", "Growth Stocks"],
                            value="Comprehensive Analysis",
                            info="🎯 Select your research focus"
                        )
                        
                        market_focus = gr.Checkboxgroup(
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
                        gr.HTML("<h3 style='text-align: center; color: #667eea;'>📈 Market Research Report</h3>")
                        
                        research_status = gr.HTML(
                            visible=True,
                            value="<div style='text-align: center; color: #666;'>👆 Select your research preferences and click 'Generate Research Report' to begin</div>"
                        )
                        
                        research_output = gr.Textbox(
                            label="📊 Stock Research Report",
                            lines=12,
                            max_lines=25,
                            interactive=False,
                            info="🎯 Comprehensive market analysis and stock recommendations"
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
                                variant="secondary",
                                elem_classes="secondary-btn"
                            )
                            
                            research_email_status = gr.Textbox(
                                label="📨 Email Status",
                                interactive=False,
                                visible=True
                            )
            
            # Market Dashboard Tab
            with gr.TabItem("📈 Market Dashboard", elem_id="dashboard-tab"):
                gr.HTML("<h3 style='text-align: center; color: #667eea;'>📊 Real-Time Market Overview</h3>")
                
                with gr.Row():
                    with gr.Column():
                        market_indices = gr.HTML(
                            value="""
                            <div class="section-card">
                                <h4>📊 Major Market Indices</h4>
                                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                                    <div class="metric-card">
                                        <h5>S&P 500</h5>
                                        <p>Loading...</p>
                                    </div>
                                    <div class="metric-card">
                                        <h5>NASDAQ</h5>
                                        <p>Loading...</p>
                                    </div>
                                    <div class="metric-card">
                                        <h5>DOW JONES</h5>
                                        <p>Loading...</p>
                                    </div>
                                </div>
                            </div>
                            """
                        )
                    
                    with gr.Column():
                        sector_performance = gr.HTML(
                            value="""
                            <div class="section-card">
                                <h4>🏭 Sector Performance</h4>
                                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                                    <div style="padding: 0.5rem; background: #e8f5e8; border-radius: 5px;">
                                        <strong>Technology:</strong> +2.3%
                                    </div>
                                    <div style="padding: 0.5rem; background: #e8f5e8; border-radius: 5px;">
                                        <strong>Healthcare:</strong> +1.8%
                                    </div>
                                    <div style="padding: 0.5rem; background: #ffe8e8; border-radius: 5px;">
                                        <strong>Energy:</strong> -0.9%
                                    </div>
                                    <div style="padding: 0.5rem; background: #e8f5e8; border-radius: 5px;">
                                        <strong>Finance:</strong> +1.2%
                                    </div>
                                </div>
                            </div>
                            """
                        )
                
                with gr.Row():
                    symbol_input = gr.Textbox(
                        label="🔍 Quick Stock Lookup",
                        placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)",
                        info="Get real-time data for any stock"
                    )
                    lookup_btn = gr.Button("📊 Get Quote", variant="secondary")
                
                stock_quote_output = gr.HTML()
            
            # History Tab
            with gr.TabItem("📚 Research History", elem_id="history-tab"):
                gr.HTML("<h3 style='text-align: center; color: #667eea;'>📚 Your Research History</h3>")
                
                with gr.Row():
                    history_type = gr.Radio(
                        label="📊 History Type",
                        choices=["Portfolio Strategies", "Stock Research", "All Reports"],
                        value="All Reports"
                    )
                    
                    load_history_btn = gr.Button(
                        "📊 Load History",
                        variant="secondary",
                        elem_classes="secondary-btn"
                    )
                
                history_output = gr.Dataframe(
                    label="📚 Research History",
                    interactive=False,
                    wrap=True
                )
        
        # Footer
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 2rem; background: rgba(255, 255, 255, 0.9); border-radius: 15px;">
                <h4>⚠️ Important Investment Disclaimer</h4>
                <p style="color: #666; line-height: 1.6;">
                    This AI-generated investment advice is for informational and educational purposes only. 
                    All investments carry risk of loss, and past performance does not guarantee future results. 
                    Always consult with a qualified financial advisor before making investment decisions. 
                    The information provided should not be considered as personalized investment advice.
                </p>
                <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; color: white;">
                    <strong>🤖 Powered by Advanced AI • 📊 Real-Time Market Data • 🔒 Secure & Private</strong>
                </div>
            </div>
            """)
        
        # Event Handlers
        def portfolio_strategy_wrapper(budget_val, time_val, risk_val, goals_val, email_val):
            if budget_val < 1000:
                error_html = "<div class='warning-msg'>❌ Minimum investment amount is $1,000</div>"
                return error_html, "❌ Minimum investment amount is $1,000", pd.DataFrame(), ""
            
            try:
                success_html = "<div class='success-msg'>🚀 Generating your personalized portfolio strategy...</div>"
                strategy, summary_df, status = create_portfolio_strategy(budget_val, time_val, risk_val, goals_val)
                final_html = f"<div class='success-msg'>{status}</div>"
                return final_html, strategy, summary_df, ""
            except Exception as e:
                error_html = f"<div class='warning-msg'>❌ Error: {str(e)}</div>"
                return error_html, f"❌ Error generating strategy: {str(e)}", pd.DataFrame(), ""
        
        def research_report_wrapper(research_type_val, market_focus_val, research_email_val):
            try:
                success_html = "<div class='success-msg'>📊 Generating comprehensive market research...</div>"
                research, trending_df, stable_df, status = create_stock_research_report()
                final_html = f"<div class='success-msg'>{status}</div>"
                return final_html, research, trending_df, stable_df, ""
            except Exception as e:
                error_html = f"<div class='warning-msg'>❌ Error: {str(e)}</div>"
                return error_html, f"❌ Error generating research: {str(e)}", pd.DataFrame(), pd.DataFrame(), ""
        
        def portfolio_email_wrapper(email_val, strategy_val):
            if not email_val or "@" not in email_val:
                return "❌ Please provide a valid email address"
            if not strategy_val or strategy_val.startswith("❌"):
                return "❌ Please generate a portfolio strategy first"
            return send_strategy_email(email_val, strategy_val, "portfolio")
        
        def research_email_wrapper(email_val, research_val):
            if not email_val or "@" not in email_val:
                return "❌ Please provide a valid email address"
            if not research_val or research_val.startswith("❌"):
                return "❌ Please generate a research report first"
            return send_strategy_email(email_val, research_val, "research")
        
        def quick_lookup_wrapper(symbol_val):
            if not symbol_val:
                return "<div class='warning-msg'>❌ Please enter a stock symbol</div>"
            
            try:
                market_data = json.loads(market_data_tool._run(symbol_val.upper()))
                if symbol_val.upper() in market_data:
                    data = market_data[symbol_val.upper()]
                    return f"""
                    <div class="section-card">
                        <h4>📊 {symbol_val.upper()} - Quick Quote</h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                            <div class="metric-card">
                                <h5>Current Price</h5>
                                <p>${data['current_price']}</p>
                            </div>
                            <div class="metric-card">
                                <h5>1Y Performance</h5>
                                <p>{data['1y_performance']:+.1f}%</p>
                            </div>
                            <div class="metric-card">
                                <h5>52W High</h5>
                                <p>${data['52w_high']}</p>
                            </div>
                            <div class="metric-card">
                                <h5>52W Low</h5>
                                <p>${data['52w_low']}</p>
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
                    query = "SELECT strategy_date, budget, risk_tolerance, time_horizon FROM portfolio_strategies ORDER BY strategy_date DESC LIMIT 10"
                elif history_type_val == "Stock Research":
                    query = "SELECT report_date, report_type FROM stock_research ORDER BY report_date DESC LIMIT 10"
                else:
                    query = """
                    SELECT 'Portfolio' as type, strategy_date as date, budget as details FROM portfolio_strategies
                    UNION
                    SELECT 'Research' as type, report_date as date, report_type as details FROM stock_research
                    ORDER BY date DESC LIMIT 10
                    """
                
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                if df.empty:
                    return pd.DataFrame({"📚 History": ["No records found"]})
                
                return df
                
            except Exception as e:
                return pd.DataFrame({"⚠️ Error": [f"Error loading history: {str(e)}"]})
        
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

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting AI Investment Advisor Pro...")
    print("=" * 60)
    
    # Check environment setup
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add it to your .env file")
        exit(1)
    
    print("✅ Environment configured")
    print("🌐 Launching enhanced Gradio interface...")
    
    app = create_enhanced_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )