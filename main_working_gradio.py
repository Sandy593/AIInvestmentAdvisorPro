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

# Load environment variables
load_dotenv()

# Custom Investment Tools
class PortfolioAnalysisTool(BaseTool):
    name: str = "portfolio_analysis"
    description: str = "Analyze investment portfolio based on user preferences"
    
    def _run(self, budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> str:
        try:
            # Define asset allocation based on risk tolerance and time horizon
            allocations = {
                "Conservative": {
                    "Short-term": {"bonds": 60, "dividend_stocks": 25, "gold": 10, "cash": 5},
                    "Medium-term": {"bonds": 50, "dividend_stocks": 30, "real_estate": 15, "gold": 5},
                    "Long-term": {"bonds": 40, "dividend_stocks": 35, "real_estate": 20, "gold": 5}
                },
                "Moderate": {
                    "Short-term": {"bonds": 40, "stocks": 35, "real_estate": 15, "gold": 10},
                    "Medium-term": {"stocks": 50, "bonds": 25, "real_estate": 20, "gold": 5},
                    "Long-term": {"stocks": 60, "bonds": 20, "real_estate": 15, "gold": 5}
                },
                "Aggressive": {
                    "Short-term": {"stocks": 60, "crypto": 20, "real_estate": 15, "gold": 5},
                    "Medium-term": {"stocks": 65, "crypto": 15, "real_estate": 15, "gold": 5},
                    "Long-term": {"stocks": 70, "crypto": 15, "real_estate": 10, "gold": 5}
                }
            }
            
            allocation = allocations.get(risk_tolerance, {}).get(time_horizon, {})
            
            return json.dumps({
                "budget": budget,
                "allocation": allocation,
                "risk_tolerance": risk_tolerance,
                "time_horizon": time_horizon,
                "goals": goals
            }, indent=2)
        except Exception as e:
            return f"Error in portfolio analysis: {str(e)}"

class InvestmentRecommendationTool(BaseTool):
    name: str = "investment_recommendations"
    description: str = "Get specific investment recommendations based on allocation"
    
    def _run(self, allocation_data: str) -> str:
        try:
            data = json.loads(allocation_data)
            budget = data["budget"]
            allocation = data["allocation"]
            
            recommendations = {
                "stocks": {
                    "etfs": ["VTI", "SPY", "QQQ", "VEA"],
                    "dividend_stocks": ["JNJ", "KO", "PG", "MSFT"],
                    "individual_stocks": ["AAPL", "GOOGL", "AMZN", "TSLA"]
                },
                "bonds": {
                    "government": ["TLT", "IEF", "TIPS"],
                    "corporate": ["LQD", "HYG", "VCIT"]
                },
                "real_estate": {
                    "reits": ["VNQ", "SCHH", "FREL", "RWR"]
                },
                "crypto": {
                    "major": ["BTC", "ETH"],
                    "altcoins": ["BNB", "ADA", "SOL"]
                },
                "commodities": {
                    "gold": ["GLD", "IAU"],
                    "silver": ["SLV"],
                    "broad": ["DJP", "PDBC"]
                }
            }
            
            # Calculate dollar amounts for each category
            dollar_allocations = {}
            for category, percentage in allocation.items():
                dollar_amount = (percentage / 100) * budget
                dollar_allocations[category] = {
                    "percentage": percentage,
                    "amount": dollar_amount,
                    "recommendations": recommendations.get(category.replace("_", ""), {})
                }
            
            return json.dumps(dollar_allocations, indent=2)
        except Exception as e:
            return f"Error getting recommendations: {str(e)}"

class MarketDataTool(BaseTool):
    name: str = "market_data"
    description: str = "Get current market data for recommended investments"
    
    def _run(self, symbols: str) -> str:
        try:
            symbol_list = symbols.split(",")
            market_data = {}
            
            for symbol in symbol_list[:10]:  # Limit to 10 symbols
                symbol = symbol.strip()
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period="1y")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        start_price = hist['Close'].iloc[0]
                        performance = ((current_price - start_price) / start_price) * 100
                        
                        market_data[symbol] = {
                            "current_price": round(float(current_price), 2),
                            "1y_performance": round(float(performance), 2),
                            "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                            "expense_ratio": info.get("expenseRatio", 0) * 100 if info.get("expenseRatio") else 0
                        }
                except:
                    continue
            
            return json.dumps(market_data, indent=2)
        except Exception as e:
            return f"Error getting market data: {str(e)}"

# Initialize tools and LLM
portfolio_tool = PortfolioAnalysisTool()
recommendation_tool = InvestmentRecommendationTool()
market_data_tool = MarketDataTool()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define Investment Advisor Agent
investment_advisor_agent = Agent(
    role="Personal Investment Advisor",
    goal="Provide personalized investment advice and portfolio recommendations based on user's budget, risk tolerance, and goals",
    backstory="You are an experienced financial advisor with expertise in portfolio construction, asset allocation, and passive income strategies. You specialize in creating customized investment plans for different risk profiles and investment horizons.",
    tools=[portfolio_tool, recommendation_tool, market_data_tool],
    llm=llm,
    verbose=True
)

def create_investment_strategy(budget: float, time_horizon: str, risk_tolerance: str, goals: str) -> Tuple[str, pd.DataFrame, str]:
    """Create personalized investment strategy using CrewAI"""
    
    # Create task for the investment advisor
    strategy_task = Task(
        description=f"""
        Create a comprehensive, personalized investment strategy for a client with the following profile:
        
        - Investment Budget: ${budget:,.2f}
        - Time Horizon: {time_horizon}
        - Risk Tolerance: {risk_tolerance}
        - Investment Goals: {goals}
        
        Your task:
        1. Use portfolio_analysis tool to determine optimal asset allocation
        2. Use investment_recommendations tool to get specific investment options
        3. Use market_data tool to get current prices and performance data
        4. Create a detailed strategy report
        
        Provide a comprehensive report including:
        - Recommended asset allocation with rationale
        - Specific investment recommendations (ETFs, stocks, REITs, etc.)
        - Expected returns and risk analysis
        - Implementation steps
        - Rebalancing strategy
        - Tax considerations
        
        Focus on passive income generation strategies where applicable.
        """,
        agent=investment_advisor_agent,
        expected_output="Detailed investment strategy report with specific recommendations and implementation plan"
    )
    
    # Create and run crew
    crew = Crew(
        agents=[investment_advisor_agent],
        tasks=[strategy_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        
        # Create summary table
        allocation_data = portfolio_tool._run(budget, time_horizon, risk_tolerance, goals)
        allocation = json.loads(allocation_data)["allocation"]
        
        summary_data = []
        risk_levels = {"bonds": "Low", "dividend_stocks": "Low-Med", "stocks": "Medium", 
                      "real_estate": "Medium", "crypto": "High", "gold": "Low", "cash": "Very Low"}
        
        for asset, percentage in allocation.items():
            amount = (percentage / 100) * budget
            risk_level = risk_levels.get(asset, "Medium")
            summary_data.append({
                "Asset Class": asset.replace("_", " ").title(),
                "Allocation %": f"{percentage}%",
                "Amount ($)": f"${amount:,.2f}",
                "Risk Level": risk_level
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        return str(result), summary_df, "Strategy generated successfully!"
        
    except Exception as e:
        error_msg = f"Error generating strategy: {str(e)}"
        empty_df = pd.DataFrame({"Error": [error_msg]})
        return error_msg, empty_df, error_msg

def send_strategy_email(email: str, strategy_report: str, summary_table: pd.DataFrame) -> str:
    """Send investment strategy via email"""
    try:
        if not email or "@" not in email:
            return "❌ Please provide a valid email address"
        
        # Email configuration
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        
        if not all([sender_email, sender_password]):
            return "❌ Email credentials not configured in environment"
        
        # Create email content
        subject = f"📊 Your Personalized Investment Strategy - {datetime.now().strftime('%Y-%m-%d')}"
        
        email_body = f"""
📊 Your Personalized Investment Strategy Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{strategy_report}

Portfolio Summary:
{summary_table.to_string(index=False)}

---
This report was generated by your AI Investment Advisor.
Please consult with a qualified financial advisor before making investment decisions.
        """
        
        # Send email
        msg = MIMEText(email_body, 'plain', 'utf-8')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = email
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return f"✅ Investment strategy sent successfully to {email}"
        
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

# Create Gradio Interface
def create_gradio_app():
    with gr.Blocks(
        title="🎯 AI Investment Advisor",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .input-section { background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
        .results-section { background: #e8f4fd; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
        """
    ) as app:
        
        with gr.Row():
            gr.HTML("""
            <div class="main-header">
                <h1>🎯 AI Investment Advisor</h1>
                <h3>Get Personalized Portfolio Recommendations for Passive Income</h3>
                <p>Enter your investment preferences below to receive a custom strategy</p>
            </div>
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="input-section">')
                gr.HTML("<h3>📝 Your Investment Profile</h3>")
                
                budget = gr.Number(
                    label="💰 Initial Investment Amount ($)",
                    value=10000,
                    minimum=1000,
                    maximum=10000000,
                    info="How much are you planning to invest initially?"
                )
                
                time_horizon = gr.Radio(
                    label="⏳ Investment Time Horizon",
                    choices=["Short-term (1-2 years)", "Medium-term (3-5 years)", "Long-term (5+ years)"],
                    value="Medium-term (3-5 years)",
                    info="How long do you plan to keep your money invested?"
                )
                
                risk_tolerance = gr.Radio(
                    label="⚖️ Risk Tolerance",
                    choices=["Conservative", "Moderate", "Aggressive"],
                    value="Moderate",
                    info="How comfortable are you with market volatility?"
                )
                
                goals = gr.Textbox(
                    label="🎯 Investment Goals & Interests",
                    placeholder="e.g., passive income, retirement, house down payment, AI/tech focus, sustainability...",
                    lines=3,
                    info="Any specific goals or areas of interest?"
                )
                
                email = gr.Textbox(
                    label="📧 Email (Optional)",
                    placeholder="your.email@example.com",
                    info="Receive your strategy report via email"
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate My Investment Strategy",
                    variant="primary",
                    size="lg"
                )
                
                gr.HTML('</div>')
            
            with gr.Column(scale=2):
                gr.HTML('<div class="results-section">')
                gr.HTML("<h3>📊 Your Personalized Strategy</h3>")
                
                strategy_output = gr.Textbox(
                    label="📈 Investment Strategy Report",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                    info="Detailed analysis and recommendations"
                )
                
                summary_table = gr.Dataframe(
                    label="📋 Portfolio Allocation Summary",
                    interactive=False,
                    wrap=True
                )
                
                with gr.Row():
                    status_msg = gr.Textbox(
                        label="📊 Generation Status",
                        interactive=False,
                        visible=True
                    )
                
                with gr.Row():
                    email_status = gr.Textbox(
                        label="📧 Email Status",
                        interactive=False,
                        visible=True
                    )
                
                send_email_btn = gr.Button(
                    "📧 Send Strategy to Email",
                    variant="secondary"
                )
                
                gr.HTML('</div>')
        
        with gr.Row():
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #fff3cd; border-radius: 10px;">
                <h4>⚠️ Important Disclaimer</h4>
                <p>This AI-generated advice is for informational purposes only. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.</p>
            </div>
            """)
        
        # Event handlers
        def generate_strategy_wrapper(budget_val, time_val, risk_val, goals_val, email_val):
            if budget_val < 1000:
                return "❌ Minimum investment amount is $1,000", pd.DataFrame(), "❌ Error: Minimum investment is $1,000", ""
            
            # Clean up the time horizon value
            time_clean = time_val.split(" (")[0] if " (" in time_val else time_val
            
            try:
                strategy, summary_df, status = create_investment_strategy(
                    budget_val, time_clean, risk_val, goals_val
                )
                return strategy, summary_df, "✅ Strategy generated successfully!", ""
            except Exception as e:
                error_msg = f"❌ Error generating strategy: {str(e)}"
                return error_msg, pd.DataFrame(), error_msg, ""
        
        def send_email_wrapper(email_val, strategy_val, summary_df):
            if not email_val or "@" not in email_val:
                return "❌ Please provide a valid email address"
            
            if not strategy_val or strategy_val.startswith("❌"):
                return "❌ Please generate a strategy first"
            
            result = send_strategy_email(email_val, strategy_val, summary_df)
            return result
        
        generate_btn.click(
            fn=generate_strategy_wrapper,
            inputs=[budget, time_horizon, risk_tolerance, goals, email],
            outputs=[strategy_output, summary_table, status_msg, email_status]
        )
        
        send_email_btn.click(
            fn=send_email_wrapper,
            inputs=[email, strategy_output, summary_table],
            outputs=[email_status]
        )
    
    return app

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting AI Investment Advisor...")
    print("=" * 50)
    
    # Check environment setup
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add it to your .env file")
        exit(1)
    
    print("✅ Environment configured")
    print("🌐 Launching Gradio interface...")
    
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )