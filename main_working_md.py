import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests
import yfinance as yf
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
import logging
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from twilio.rest import Client

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup for tracking researched companies
def init_database():
    conn = sqlite3.connect('stock_research.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS researched_companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT UNIQUE,
            ticker TEXT,
            research_date DATE,
            category TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS research_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE,
            report_content TEXT,
            best_stock TEXT,
            trending_companies TEXT,
            stable_stocks TEXT,
            crypto_picks TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Custom Tools
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for current news and information"
    
    def _run(self, query: str) -> str:
        try:
            search = DuckDuckGoSearchRun()
            results = search.run(query)
            return results
        except Exception as e:
            return f"Error in web search: {str(e)}"

class StockDataTool(BaseTool):
    name: str = "stock_data"
    description: str = "Get stock price data and financial metrics"
    
    def _run(self, ticker: str, period: str = "1y") -> str:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            info = stock.info
            
            if hist.empty:
                return f"No data found for ticker {ticker}"
            
            # Calculate comprehensive metrics
            start_price = hist['Close'].iloc[0]
            current_price = hist['Close'].iloc[-1]
            max_price_1y = hist['High'].max()
            min_price_1y = hist['Low'].min()
            trend = ((current_price - start_price) / start_price) * 100
            
            # Calculate performance metrics
            max_gain_from_min = ((max_price_1y - min_price_1y) / min_price_1y) * 100
            current_vs_max = ((current_price - max_price_1y) / max_price_1y) * 100
            current_vs_min = ((current_price - min_price_1y) / min_price_1y) * 100
            
            return json.dumps({
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "current_price": round(float(current_price), 2),
                "52_week_high": round(float(max_price_1y), 2),
                "52_week_low": round(float(min_price_1y), 2),
                "52_week_trend": round(float(trend), 2),
                "current_vs_52w_high": round(float(current_vs_max), 2),
                "current_vs_52w_low": round(float(current_vs_min), 2),
                "max_gain_potential_1y": round(float(max_gain_from_min), 2),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "volume": info.get("volume", "N/A"),
                "avg_volume": info.get("averageVolume", "N/A")
            }, indent=2)
        except Exception as e:
            return f"Error getting stock data for {ticker}: {str(e)}"

class CryptoDataTool(BaseTool):
    name: str = "crypto_data"
    description: str = "Get cryptocurrency price data and trends"
    
    def _run(self, crypto_symbol: str) -> str:
        try:
            # Using yfinance for crypto data (USD pairs)
            ticker = f"{crypto_symbol}-USD"
            crypto = yf.Ticker(ticker)
            hist = crypto.history(period="1y")
            
            if hist.empty:
                return f"No data found for {crypto_symbol}"
            
            start_price = hist['Close'].iloc[0]
            current_price = hist['Close'].iloc[-1]
            max_price_1y = hist['High'].max()
            min_price_1y = hist['Low'].min()
            trend = ((current_price - start_price) / start_price) * 100
            
            # Calculate performance metrics
            current_vs_max = ((current_price - max_price_1y) / max_price_1y) * 100
            current_vs_min = ((current_price - min_price_1y) / min_price_1y) * 100
            volatility = hist['Close'].pct_change().std() * 100
            
            return json.dumps({
                "symbol": crypto_symbol,
                "current_price": round(float(current_price), 2),
                "52_week_high": round(float(max_price_1y), 2),
                "52_week_low": round(float(min_price_1y), 2),
                "1_year_trend": round(float(trend), 2),
                "current_vs_52w_high": round(float(current_vs_max), 2),
                "current_vs_52w_low": round(float(current_vs_min), 2),
                "volatility_percent": round(float(volatility), 2),
                "volume": int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else "N/A"
            }, indent=2)
        except Exception as e:
            return f"Error getting crypto data for {crypto_symbol}: {str(e)}"

class CompanyTrackerTool(BaseTool):
    name: str = "company_tracker"
    description: str = "Track researched companies to avoid duplicates"
    
    def _run(self, action: str, company_name: str = "", ticker: str = "", category: str = "") -> str:
        conn = sqlite3.connect('stock_research.db')
        cursor = conn.cursor()
        
        if action == "check":
            cursor.execute("SELECT company_name FROM researched_companies WHERE company_name = ?", (company_name,))
            result = cursor.fetchone()
            conn.close()
            return "exists" if result else "new"
        
        elif action == "add":
            try:
                cursor.execute(
                    "INSERT INTO researched_companies (company_name, ticker, research_date, category) VALUES (?, ?, ?, ?)",
                    (company_name, ticker, datetime.now().date(), category)
                )
                conn.commit()
                conn.close()
                return f"Added {company_name} to tracking database"
            except sqlite3.IntegrityError:
                conn.close()
                return f"{company_name} already exists in database"
        
        elif action == "list":
            cursor.execute("SELECT company_name, ticker, category FROM researched_companies")
            results = cursor.fetchall()
            conn.close()
            return json.dumps([{"name": r[0], "ticker": r[1], "category": r[2]} for r in results])

# Simple Notification Tool
class NotificationTool(BaseTool):
    name: str = "notification_sender"
    description: str = "Send email notifications"
    
    def _run(self, message: str, method: str = "email", subject: str = "Stock Research Alert") -> str:
        if method == "email":
            return self.send_simple_email(message, subject)
        return "Only email method supported"
    
    def send_simple_email(self, message: str, subject: str) -> str:
        try:
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')
            recipient_email = os.getenv('RECIPIENT_EMAIL')
            
            if not all([sender_email, sender_password, recipient_email]):
                return "❌ EMAIL FAILED: Email credentials not configured"
            
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            import socket
            socket.setdefaulttimeout(60)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            return f"✅ EMAIL SUCCESS: Email sent successfully to {recipient_email}"
            
        except Exception as e:
            return f"❌ EMAIL FAILED: Error sending email: {str(e)}"

# Function to send email with attachment (standalone)
def send_email_with_markdown_attachment(filename: str) -> str:
    """Send email with markdown file attachment"""
    try:
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        recipient_email = os.getenv('RECIPIENT_EMAIL')
        
        if not all([sender_email, sender_password, recipient_email]):
            return "❌ EMAIL FAILED: Email credentials not configured"
        
        # Create summary message
        summary_message = f"""📊 Stock Research Report Complete - {datetime.now().strftime('%Y-%m-%d')}

🎯 EXECUTIVE SUMMARY:
Your comprehensive stock research analysis has been completed successfully!

📄 REPORT DETAILS:
• File: {filename}
• Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Content: Complete stock analysis with real market data

📈 WHAT'S INCLUDED:
• Best stock pick with detailed analysis and price data
• Top 5 trending companies (government-funded opportunities)
• Top 5 stable stocks with consistent upward trends
• Top 5 cryptocurrencies with growth potential
• Investment tips across multiple asset classes
• Risk assessment and market outlook
• Actionable recommendations

📎 The complete analysis is attached as a markdown file for easy viewing.
You can open it with any text editor or markdown viewer.

Best regards,
Your Stock Research System
"""
        
        # Create multipart message
        msg = MIMEMultipart()
        msg['Subject'] = f"📊 Stock Research Report - {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        
        # Add body
        body = MIMEText(summary_message, 'plain', 'utf-8')
        msg.attach(body)
        
        # Add markdown file attachment
        if os.path.exists(filename):
            with open(filename, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(filename)}')
            msg.attach(part)
        
        # Send email
        import socket
        socket.setdefaulttimeout(60)
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"✅ EMAIL SUCCESS: Email sent successfully to {recipient_email} with attachment ({os.path.basename(filename)})"
        
    except Exception as e:
        return f"❌ EMAIL FAILED: Error sending email with attachment: {str(e)}"

# Initialize tools
web_search_tool = WebSearchTool()
stock_data_tool = StockDataTool()
crypto_data_tool = CryptoDataTool()
company_tracker_tool = CompanyTrackerTool()
notification_tool = NotificationTool()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define Agents
trending_company_finder = Agent(
    role="Trending Company Finder",
    goal="Find 5 trending companies in the news that are receiving government funds or significant investment, ensuring no duplicates from previous research",
    backstory="You are an expert market researcher with a keen eye for identifying emerging companies that are making headlines. You specialize in finding companies that are receiving government backing or significant funding.",
    tools=[web_search_tool, company_tracker_tool],
    llm=llm,
    verbose=True
)

financial_researcher = Agent(
    role="Senior Financial Researcher",
    goal="Given details of trending companies in the news, you provide comprehensive analysis of each in a report including financial metrics, market position, and growth potential",
    backstory="You are a financial expert with a proven track record of deeply analyzing hot companies and building comprehensive reports. You excel at fundamental analysis and market research.",
    tools=[stock_data_tool, web_search_tool],
    llm=llm,
    verbose=True
)

stock_picker = Agent(
    role="Stock Picker from Research",
    goal="Given a list of researched companies with investment potential, you select the best one for investment and provide detailed analysis. Also identify 5 stable stocks with upward trends and 5 promising cryptocurrencies.",
    backstory="You're a meticulous, skilled financial analyst with a proven track record of equity selection. You have a talent for synthesizing research and picking the best investments across stocks and crypto.",
    tools=[stock_data_tool, crypto_data_tool, company_tracker_tool],
    llm=llm,
    verbose=True
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Provide comprehensive investment advice including tips on trading, mutual funds, real estate, and risk assessment for different investment vehicles",
    backstory="You are a seasoned investment advisor with decades of experience across multiple asset classes. You excel at explaining complex investment concepts and providing actionable advice.",
    tools=[web_search_tool],
    llm=llm,
    verbose=True
)

manager = Agent(
    role="Manager",
    goal="Coordinate the entire research process, ensure quality outputs, and compile the final comprehensive investment report",
    backstory="You are an experienced and highly effective project manager who can delegate tasks to the right people and ensure deliverables meet high standards.",
    tools=[notification_tool],
    llm=llm,
    verbose=True
)

# Define Tasks
task_find_trending_companies = Task(
    description="""
    Search for trending companies in the news that are:
    1. Receiving government funds or grants
    2. Getting significant investment or funding rounds
    3. Making headlines for breakthrough innovations
    4. Involved in emerging technologies or green energy
    
    Use the company tracker tool to ensure you don't select companies that have been researched before.
    Return exactly 5 companies with their ticker symbols, brief description, and why they're trending.
    
    Format your output as a structured list with company name, ticker, and trending reason.
    """,
    agent=trending_company_finder,
    expected_output="List of 5 trending companies with tickers and trending reasons"
)

task_research_companies = Task(
    description="""
    For each company identified by the trending company finder:
    1. Use the stock_data tool to get detailed financial data for each company's ticker
    2. Get current stock price, 52-week high, 52-week low, market cap, P/E ratio
    3. Analyze the company's financial health and growth trajectory
    4. Research recent news and developments using web search
    5. Assess investment potential and risks
    
    MANDATORY: For each company, you MUST call the stock_data tool with the ticker symbol to get:
    - Current price
    - 52-week high and low
    - Performance metrics
    - Market cap and P/E ratio
    
    Provide a comprehensive analysis for each company including:
    - Financial metrics with ACTUAL numbers from stock_data tool
    - Market position and competitive advantages  
    - Recent developments and news
    - Growth potential and risks
    - Investment recommendation (Buy/Hold/Sell)
    
    Example format for each company:
    ## Company Name (TICKER)
    - Current Price: $[actual price from tool]
    - 52-Week High: $[actual high from tool]
    - 52-Week Low: $[actual low from tool]
    - Market Cap: [actual market cap from tool]
    - Analysis: [your detailed analysis]
    """,
    agent=financial_researcher,
    expected_output="Detailed financial analysis for each trending company with real price data from stock_data tool"
)

task_pick_investments = Task(
    description="""
    Based on the research provided:
    1. Select the BEST single stock for investment from the researched companies
    2. Identify 5 additional stable stocks with consistent upward trends (not from the trending list)
    3. Research and recommend 5 cryptocurrencies with stable upward trends
    4. Provide detailed reasoning for each selection
    
    MANDATORY TOOL USAGE:
    - For EVERY stock recommendation, call stock_data tool with ticker symbol
    - For EVERY crypto recommendation, call crypto_data tool with symbol
    - Do NOT use placeholder values like $X.XX - get REAL data from tools
    
    For stocks, analyze:
    - Large cap stocks like AAPL, MSFT, GOOGL, AMZN, TSLA
    - Use stock_data("AAPL") to get Apple's real price data
    - Use stock_data("MSFT") to get Microsoft's real price data
    - Focus on companies with market cap > $100B
    
    For cryptocurrencies, analyze:
    - Major cryptos like BTC, ETH, BNB, ADA, SOL
    - Use crypto_data("BTC") to get Bitcoin's real price data  
    - Use crypto_data("ETH") to get Ethereum's real price data
    - Focus on established cryptocurrencies with strong fundamentals
    
    Required format for each recommendation:
    
    ## Stock Name (TICKER)
    [Call stock_data tool first, then use the actual data]
    - Current Price: $[real price from stock_data tool]
    - 52-Week High: $[real high from stock_data tool]  
    - 52-Week Low: $[real low from stock_data tool]
    - 1-Year Performance: [real percentage from stock_data tool]
    - Analysis: [detailed analysis based on the real data]
    
    ## Crypto Name (SYMBOL)
    [Call crypto_data tool first, then use the actual data]
    - Current Price: $[real price from crypto_data tool]
    - 52-Week High: $[real high from crypto_data tool]
    - 52-Week Low: $[real low from crypto_data tool]  
    - 1-Year Performance: [real percentage from crypto_data tool]
    - Analysis: [detailed analysis based on the real data]
    
    DO NOT USE PLACEHOLDERS - ALWAYS GET REAL DATA FROM TOOLS FIRST!
    """,
    agent=stock_picker,
    expected_output="Best investment pick, 5 stable stocks, and 5 promising cryptocurrencies with REAL price data from tools (no placeholders)"
)

task_investment_advice = Task(
    description="""
    Provide comprehensive investment advice covering:
    1. Trading strategies and tips for current market conditions
    2. Mutual fund recommendations for different risk profiles
    3. Real estate investment insights and opportunities
    4. Risk assessment for different investment types
    5. Diversification strategies
    6. Market outlook and timing considerations
    
    Focus on actionable advice that investors can implement based on current market conditions.
    """,
    agent=investment_advisor,
    expected_output="Comprehensive investment advice across multiple asset classes"
)

task_compile_report = Task(
    description=f"""
    You are the Manager agent. Your ONLY job is to create a complete, detailed markdown report.
    
    You MUST output the ACTUAL report content, not a summary or description of the report.
    
    Take all the research data from previous agents and format it into this EXACT structure:
    
    # Stock Research Report - {datetime.now().strftime('%Y-%m-%d')}
    
    ## Executive Summary
    - Best stock pick: [ACTUAL COMPANY NAME] ([TICKER]) - $[ACTUAL PRICE]
    - Key investment opportunities: [LIST THE ACTUAL OPPORTUNITIES]
    - Overall market outlook: [ACTUAL MARKET ANALYSIS]
    
    ## Best Stock to Invest
    **[ACTUAL COMPANY NAME] ([TICKER])**
    - Current Price: $[ACTUAL PRICE FROM STOCK_DATA TOOL]
    - 52-Week High: $[ACTUAL HIGH FROM STOCK_DATA TOOL]
    - 52-Week Low: $[ACTUAL LOW FROM STOCK_DATA TOOL]
    - 1-Year Performance: [ACTUAL PERCENTAGE]%
    - Market Cap: [ACTUAL MARKET CAP]
    - P/E Ratio: [ACTUAL P/E RATIO]
    - Analysis: [DETAILED ANALYSIS WITH ACTUAL DATA]
    
    ## Top 5 Trending Companies for Future Research
    1. **[COMPANY 1] ([TICKER])** - Current Price: $[ACTUAL PRICE]
       - 52-Week High: $[ACTUAL HIGH] | 52-Week Low: $[ACTUAL LOW]
       - Why Trending: [ACTUAL REASON]
       - Analysis: [ACTUAL ANALYSIS]
    
    2. **[COMPANY 2] ([TICKER])** - Current Price: $[ACTUAL PRICE]
       - 52-Week High: $[ACTUAL HIGH] | 52-Week Low: $[ACTUAL LOW]
       - Why Trending: [ACTUAL REASON]
       - Analysis: [ACTUAL ANALYSIS]
    
    [Continue for all 5 companies]
    
    ## Top 5 Stable Stocks with Upward Trends
    1. **[STOCK 1] ([TICKER])** - Current Price: $[ACTUAL PRICE]
       - 52-Week High: $[ACTUAL HIGH] | 52-Week Low: $[ACTUAL LOW]
       - 1-Year Performance: +[ACTUAL PERCENTAGE]%
       - Analysis: [ACTUAL ANALYSIS]
    
    [Continue for all 5 stocks]
    
    ## Top 5 Cryptocurrencies with Stable Upward Trends
    1. **[CRYPTO 1] ([SYMBOL])** - Current Price: $[ACTUAL PRICE]
       - 52-Week High: $[ACTUAL HIGH] | 52-Week Low: $[ACTUAL LOW]
       - 1-Year Performance: +[ACTUAL PERCENTAGE]%
       - Analysis: [ACTUAL ANALYSIS]
    
    [Continue for all 5 cryptos]
    
    ## Investment Tips and Strategies
    [PASTE THE ACTUAL INVESTMENT ADVICE FROM INVESTMENT_ADVISOR]
    
    ## Risk Assessment
    [PASTE THE ACTUAL RISK ANALYSIS]
    
    ## Conclusion and Recommendations
    [ACTUAL SUMMARY OF KEY TAKEAWAYS AND ACTION ITEMS]
    
    CRITICAL: Replace ALL bracketed placeholders with ACTUAL data from previous agents.
    Do NOT write "The report has been compiled" - OUTPUT THE ACTUAL REPORT CONTENT.
    Your response should be the complete report that can be saved as a markdown file.
    """,
    agent=manager,
    expected_output="The complete, detailed markdown report with all actual data - not a summary or description"
)

# Create the Crew
def create_stock_research_crew():
    return Crew(
        agents=[
            trending_company_finder,
            financial_researcher,
            stock_picker,
            investment_advisor,
            manager
        ],
        tasks=[
            task_find_trending_companies,
            task_research_companies,
            task_pick_investments,
            task_investment_advice,
            task_compile_report
        ],
        process=Process.sequential,
        verbose=True
    )

# Main execution function
def run_stock_research():
    """
    Main function to execute the stock research workflow
    """
    try:
        # Initialize database
        init_database()
        
        # Create and run the crew
        crew = create_stock_research_crew()
        result = crew.kickoff()
        
        # Save report to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_research_report_{timestamp}.md"
        
        # Clean the result to extract only the report content
        result_str = str(result)
        
        # Look for the actual report content (starts with # Stock Research Report)
        lines = result_str.split('\n')
        report_lines = []
        capturing = False
        
        for line in lines:
            # Start capturing when we see the report header
            if line.strip().startswith('# Stock Research Report'):
                capturing = True
            # Stop capturing at task execution logs
            elif capturing and any(keyword in line.lower() for keyword in 
                                 ['task output:', 'agent execution', 'crew execution', 'final answer:']):
                break
            
            if capturing:
                report_lines.append(line)
        
        # If we found the report content, use it; otherwise use the full result
        if report_lines and len(report_lines) > 5:  # Make sure we have substantial content
            clean_report = '\n'.join(report_lines)
        else:
            clean_report = result_str
        
        # Save the cleaned report
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(clean_report)
        
        # Also save to database
        conn = sqlite3.connect('stock_research.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO research_reports (report_date, report_content) VALUES (?, ?)",
            (datetime.now().date(), clean_report)
        )
        conn.commit()
        conn.close()
        
        # Send email with markdown attachment
        print("\n" + "="*60)
        print("📧 EMAIL NOTIFICATION STATUS")
        print("="*60)
        
        print("🔄 Sending markdown file as email attachment...")
        email_result = send_email_with_markdown_attachment(filename)
        
        if "✅ EMAIL SUCCESS" in email_result:
            print("✅ EMAIL WITH ATTACHMENT SENT SUCCESSFULLY!")
            print(f"   {email_result}")
        else:
            print("❌ EMAIL DELIVERY FAILED!")
            print(f"   {email_result}")
        
        print("="*60)
        
        # Print preview of the actual report content
        print("\n📄 Report Preview:")
        print("-" * 50)
        preview = clean_report[:800] + "..." if len(clean_report) > 800 else clean_report
        print(preview)
        
        logger.info(f"Research completed successfully. Report saved as {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in stock research workflow: {str(e)}")
        return f"Error: {str(e)}"

# Configuration and Environment Setup
def setup_environment():
    """
    Setup environment variables and configuration
    """
    print("🔧 Loading environment variables from .env file...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found in current directory")
        print("   Create a .env file with your API keys")
        return False
    
    # Load and verify required variables
    required_vars = ['OPENAI_API_KEY']
    email_vars = ['SENDER_EMAIL', 'SENDER_PASSWORD', 'RECIPIENT_EMAIL']
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Required variables missing from .env: {', '.join(missing_vars)}")
        return False
    
    # Check email configuration
    email_configured = all(os.getenv(var) for var in email_vars)
    
    print("✅ OpenAI API key loaded successfully")
    
    if email_configured:
        print("✅ Email notifications enabled")
        print(f"   Sender: {os.getenv('SENDER_EMAIL')}")
        print(f"   Recipient: {os.getenv('RECIPIENT_EMAIL')}")
    else:
        print("⚠️  Email notifications disabled (credentials missing from .env)")
        print("   Required in .env file:")
        print("   - SENDER_EMAIL=your-email@gmail.com")
        print("   - SENDER_PASSWORD=your-16-char-app-password")
        print("   - RECIPIENT_EMAIL=recipient@email.com")
    
    print("✅ Environment setup complete")
    return True

# Scheduling function (optional)
def schedule_research(interval_hours: int = 24):
    """
    Schedule the research to run at regular intervals
    """
    import schedule
    import time
    
    schedule.every(interval_hours).hours.do(run_stock_research)
    
    print(f"📅 Research scheduled to run every {interval_hours} hours")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    print("🚀 Stock Research Agentic Workflow")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        exit(1)
    
    # Option to run once or schedule
    run_type = input("Run once (1) or schedule recurring (2)? Enter 1 or 2: ")
    
    if run_type == "1":
        print("\n📊 Starting stock research...")
        result = run_stock_research()
        print("\n✅ Research completed!")
        print("\n📄 Report preview:")
        print("-" * 50)
        print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
        
    elif run_type == "2":
        hours = int(input("Enter interval in hours (default 24): ") or "24")
        print(f"\n📅 Scheduling research every {hours} hours...")
        schedule_research(hours)
    
    else:
        print("❌ Invalid option selected")