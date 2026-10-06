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
from twilio.rest import Client

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_date DATETIME,
            recipient_email TEXT,
            subject TEXT,
            message_preview TEXT,
            status TEXT,
            error_message TEXT
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
            
            # Calculate trend
            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            trend = ((end_price - start_price) / start_price) * 100
            
            return json.dumps({
                "ticker": ticker,
                "current_price": float(end_price),
                "52_week_trend": float(trend),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "company_name": info.get("longName", ticker)
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
            end_price = hist['Close'].iloc[-1]
            trend = ((end_price - start_price) / start_price) * 100
            
            return json.dumps({
                "symbol": crypto_symbol,
                "current_price": float(end_price),
                "1_year_trend": float(trend),
                "volatility": float(hist['Close'].pct_change().std() * 100)
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

# Enhanced Notification Tool with Email Support
class NotificationTool(BaseTool):
    name: str = "notification_sender"
    description: str = "Send notifications via Email, WhatsApp, or SMS"
    
    def _run(self, message: str, method: str = "email", subject: str = "Stock Research Alert") -> str:
        try:
            if method == "email":
                return self._send_email(message, subject)
            elif method == "whatsapp":
                return self._send_whatsapp(message)
            elif method == "both":
                email_result = self._send_email(message, subject)
                whatsapp_result = self._send_whatsapp(message)
                return f"Email: {email_result} | WhatsApp: {whatsapp_result}"
            else:
                return "Notification method not supported. Use 'email', 'whatsapp', or 'both'"
        except Exception as e:
            return f"Error sending notification: {str(e)}"
    
    def _send_email(self, message: str, subject: str) -> str:
        """Send email notification"""
        try:
            # Email configuration from environment variables
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')  # App password for Gmail
            recipient_email = os.getenv('RECIPIENT_EMAIL')
            
            if not all([sender_email, sender_password, recipient_email]):
                logger.warning("Email credentials not configured")
                return "Email credentials not configured"
            
            # Create message
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = recipient_email
            
            # Send email
            logger.info(f"Attempting to send email to {recipient_email}")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            
            # Log successful email
            logger.info(f"✅ Email sent successfully to {recipient_email}")
            print(f"📧 Email sent successfully to {recipient_email}")
            
            # Track email in database
            self._track_email_notification(recipient_email, subject, message[:100], "success")
            
            return f"Email sent successfully to {recipient_email}"
            
        except Exception as e:
            error_msg = f"Error sending email: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            
            # Track failed email in database
            recipient_email = os.getenv('RECIPIENT_EMAIL', 'unknown')
            self._track_email_notification(recipient_email, subject, message[:100], "failed", str(e))
            
            return error_msg
    
    def _send_whatsapp(self, message: str) -> str:
        """Send WhatsApp notification via Twilio"""
        try:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN')
            whatsapp_number = os.getenv('WHATSAPP_NUMBER')
            
            if not all([account_sid, auth_token, whatsapp_number]):
                return "WhatsApp credentials not configured"
            
            client = Client(account_sid, auth_token)
            message_obj = client.messages.create(
                body=message,
                from_='whatsapp:+14155238886',  # Twilio sandbox number
                to=f'whatsapp:{whatsapp_number}'
            )
            return f"WhatsApp message sent: {message_obj.sid}"
            
        except Exception as e:
            return f"Error sending WhatsApp: {str(e)}"
    
    def _track_email_notification(self, recipient_email: str, subject: str, message_preview: str, status: str, error_message: str = None):
        """Track email notifications in database"""
        try:
            conn = sqlite3.connect('stock_research.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO email_notifications 
                (sent_date, recipient_email, subject, message_preview, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), recipient_email, subject, message_preview, status, error_message))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error tracking email notification: {str(e)}")

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
    1. Get detailed financial data including current stock price, market cap, P/E ratio
    2. Analyze the company's financial health and growth trajectory
    3. Research recent news and developments
    4. Assess investment potential and risks
    
    Provide a comprehensive analysis for each company including:
    - Financial metrics and ratios
    - Market position and competitive advantages
    - Recent developments and news
    - Growth potential and risks
    - Investment recommendation (Buy/Hold/Sell)
    """,
    agent=financial_researcher,
    expected_output="Detailed financial analysis for each trending company"
)

task_pick_investments = Task(
    description="""
    Based on the research provided:
    1. Select the BEST single stock for investment from the researched companies
    2. Identify 5 additional stable stocks with consistent upward trends (not from the trending list)
    3. Research and recommend 5 cryptocurrencies with stable upward trends
    4. Provide detailed reasoning for each selection
    
    For stocks, focus on:
    - Consistent upward price trends over 1-2 years
    - Strong fundamentals
    - Market stability
    
    For cryptocurrencies, focus on:
    - Price stability and upward trends
    - Market adoption and use cases
    - Development activity and community
    """,
    agent=stock_picker,
    expected_output="Best investment pick, 5 stable stocks, and 5 promising cryptocurrencies with analysis"
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
    description="""
    Compile all research and analysis into a comprehensive markdown report including:
    
    # Stock Research Report - [Date]
    
    ## Executive Summary
    - Best stock pick with clear reasoning
    - Key investment opportunities identified
    - Overall market outlook
    
    ## Best Stock to Invest
    [Detailed analysis of the top pick]
    
    ## Top 5 Trending Companies for Future Research
    [List with brief analysis of each]
    
    ## Top 5 Stable Stocks with Upward Trends
    [Detailed analysis of each stable stock]
    
    ## Top 5 Cryptocurrencies with Stable Upward Trends
    [Analysis of each crypto recommendation]
    
    ## Investment Tips and Strategies
    [Comprehensive advice on trading, mutual funds, real estate, risk management]
    
    ## Risk Assessment
    [Analysis of what's stable vs risky in current market]
    
    ## Conclusion and Recommendations
    [Summary of key takeaways and action items]
    
    After compiling the report, send an email notification with a summary of key findings.
    Use method='email' with subject='Stock Research Report - [Date]' and include:
    - Best stock recommendation
    - Top trending companies
    - Key investment insights
    - Any urgent market alerts
    """,
    agent=manager,
    expected_output="Complete markdown investment research report with email notification sent"
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
        
        # Save report to database
        conn = sqlite3.connect('stock_research.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO research_reports (report_date, report_content) VALUES (?, ?)",
            (datetime.now().date(), str(result))
        )
        conn.commit()
        conn.close()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_research_report_{timestamp}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(result))
        
        logger.info(f"Research completed successfully. Report saved as {filename}")
        return result
        
    except Exception as e:
        logger.error(f"Error in stock research workflow: {str(e)}")
        return f"Error: {str(e)}"

# Configuration and Environment Setup
def check_email_status():
    """Check the status of recent email notifications"""
    try:
        conn = sqlite3.connect('stock_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sent_date, recipient_email, subject, status, error_message 
            FROM email_notifications 
            ORDER BY sent_date DESC 
            LIMIT 10
        ''')
        notifications = cursor.fetchall()
        conn.close()
        
        if not notifications:
            print("📧 No email notifications found in database")
            return
        
        print("📧 Recent Email Notifications:")
        print("-" * 80)
        for notification in notifications:
            sent_date, recipient, subject, status, error = notification
            status_icon = "✅" if status == "success" else "❌"
            print(f"{status_icon} {sent_date} | {recipient} | {subject}")
            if error:
                print(f"   Error: {error}")
        print("-" * 80)
        
    except Exception as e:
        print(f"Error checking email status: {str(e)}")

def setup_environment():
    """
    Setup environment variables and configuration
    """
    required_env_vars = ['OPENAI_API_KEY']
    email_vars = ['SENDER_EMAIL', 'SENDER_PASSWORD', 'RECIPIENT_EMAIL']
    whatsapp_vars = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'WHATSAPP_NUMBER']
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Required variables missing: {', '.join(missing_vars)}")
        return False
    
    # Check email configuration
    email_configured = all(os.getenv(var) for var in email_vars)
    whatsapp_configured = all(os.getenv(var) for var in whatsapp_vars)
    
    if email_configured:
        print("✅ Email notifications enabled")
    else:
        print("⚠️  Email notifications disabled (credentials missing)")
        print("   Required: SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL")
        print("   Optional: SMTP_SERVER (default: smtp.gmail.com), SMTP_PORT (default: 587)")
    
    if whatsapp_configured:
        print("✅ WhatsApp notifications enabled")
    else:
        print("⚠️  WhatsApp notifications disabled (Twilio credentials missing)")
    
    if not email_configured and not whatsapp_configured:
        print("⚠️  No notification methods configured - reports will only be saved locally")
    
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
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check-emails":
        check_email_status()
        exit(0)
    
    print("🚀 Stock Research Agentic Workflow")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        exit(1)
    
    # Option to run once or schedule
    run_type = input("Run once (1), schedule recurring (2), or check email status (3)? Enter 1, 2, or 3: ")
    
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
    
    elif run_type == "3":
        check_email_status()
    
    else:
        print("❌ Invalid option selected")