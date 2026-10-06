"""
AI Investment Advisor
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

# Load environment variables
load_dotenv()

# ================================
# CONFIGURATION & STYLING
# ================================

# Dark theme CSS
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

# ================================
# DATABASE SETUP
# ================================

def init_database():
    """Initialize SQLite database for storing reports"""
    conn = sqlite3.connect('investment_advisor.db')
    cursor = conn.cursor()
    
    # Portfolio strategies table
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
    
    # Stock research table
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
# AI AGENT: STOCK RESEARCH
# ================================

class StockResearchAgent:
    """
    AI Agent with improved stock discovery and varied results
    """
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.name = "Stock Research Agent"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def call_ai(self, prompt, max_tokens=1500):
        """Make API call to GPT-4o-mini with randomization"""
        print(f"🔑 Checking API key: {'Found' if self.api_key else 'Missing'}")
        
        if not self.api_key:
            print("❌ No OpenAI API key found in environment")
            return None
            
        try:
            print(f"📡 Making OpenAI API call...")
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Add randomization to get varied results
            temperature = random.uniform(0.3, 0.7)  # Vary creativity
            print(f"🌡️ Using temperature: {temperature:.2f}")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial research expert. Provide accurate, varied, and current recommendations. Avoid giving the same recommendations repeatedly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            ai_response = response.choices[0].message.content
            print(f"✅ OpenAI API call successful, response length: {len(ai_response) if ai_response else 0}")
            print(f"📝 First 200 chars: {ai_response[:200] if ai_response else 'Empty response'}")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ AI API error: {e}")
            print(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None
    
    def discover_trending_assets(self, asset_type, sectors=None):
        """Discover trending assets based on type: stocks, cryptos, or ETFs"""
        print(f"AI Agent discovering {asset_type}...")
        
        current_date = datetime.now().strftime("%B %Y")
        time_seed = int(time.time()) % 1000
        
        if asset_type == "Trending Stocks":
            # Discover trending stocks across sectors
            if not sectors:
                sectors = ["Technology", "Healthcare"]
            
            all_stocks = []
            for sector in sectors:
                sector_stocks = self.discover_stocks_by_sector(sector)
                # Add sector information to each stock
                for stock in sector_stocks:
                    stock['sector'] = sector
                all_stocks.extend(sector_stocks)
            
            return all_stocks
        elif asset_type == "Trending Cryptos":
            return self._discover_trending_cryptos(current_date, time_seed)
        elif asset_type == "Trending ETFs":
            return self._discover_trending_etfs(sectors, current_date, time_seed)
        else:
            return []
    
    def discover_stocks_by_sector(self, sector):
        """AI discovers varied stocks in a specific sector"""
        print(f"AI Agent discovering {sector} stocks...")
        
        # Add randomization and current date context
        current_date = datetime.now().strftime("%B %Y")
        market_contexts = [
            "focus on emerging companies with recent catalysts",
            "include a mix of established and growth companies", 
            "emphasize companies with strong recent earnings",
            "highlight companies with innovation and disruption potential",
            "focus on companies with strong technical momentum"
        ]
        
        market_context = random.choice(market_contexts)
        
        # Get time-based variation
        time_seed = int(time.time()) % 1000
        
        prompt = f"""Find 8-12 trending stocks in the {sector} sector for {current_date}.
        
Research Context: {market_context}
Session: {time_seed}

REQUIREMENTS:
- Focus on REAL publicly traded companies only
- Include varied market caps: large-cap, mid-cap, small-cap
- Avoid always suggesting the same popular stocks
- Include companies with recent news, earnings, or catalysts
- Mix established leaders with emerging growth companies

Provide information in this EXACT format:

**STOCK 1:**
Symbol: [TICKER]
Company: [Full Company Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [large-cap/mid-cap/small-cap]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

**STOCK 2:**
Symbol: [TICKER]
Company: [Full Company Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [large-cap/mid-cap/small-cap]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

[Continue for 8-12 stocks...]

Focus on providing VARIED and CURRENT {sector} recommendations that differ from typical lists."""

        response = self.call_ai(prompt, max_tokens=2000)
        print(f"AI response for {sector}: {response[:200] if response else 'None'}...")
        
        if response:
            stocks = self._parse_enhanced_stock_response(response, sector)
            print(f"Parsed {len(stocks)} stocks for {sector}: {[s.get('symbol', 'N/A') for s in stocks]}")
            return stocks
        else:
            print(f"No AI response received for {sector}")
            return []
    
    def _discover_trending_cryptos(self, current_date, time_seed):
        """Discover trending cryptocurrencies"""
        crypto_contexts = [
            "focus on emerging DeFi and Web3 projects",
            "emphasize layer 1 blockchains and scaling solutions",
            "highlight gaming and NFT-related cryptocurrencies",
            "focus on institutional adoption and payment coins",
            "emphasize meme coins and community-driven projects"
        ]
        
        crypto_context = random.choice(crypto_contexts)
        
        prompt = f"""Find 10-12 trending cryptocurrencies for {current_date}.
        
Research Context: {crypto_context}
Session: {time_seed}

REQUIREMENTS:
- Focus on REAL tradable cryptocurrencies
- Include varied market caps: large-cap, mid-cap, small-cap
- Mix established coins with emerging opportunities
- Include recent catalysts, partnerships, or developments
- Avoid always suggesting the same popular cryptos

Provide information in this EXACT format:

**CRYPTO 1:**
Symbol: [TICKER] (e.g., BTC, ETH, SOL)
Name: [Full Cryptocurrency Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [large-cap/mid-cap/small-cap]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

**CRYPTO 2:**
Symbol: [TICKER]
Name: [Full Cryptocurrency Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [large-cap/mid-cap/small-cap]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

[Continue for 10-12 cryptos...]

Focus on providing VARIED and CURRENT cryptocurrency recommendations."""

        response = self.call_ai(prompt, max_tokens=2000)
        print(f"AI response for Trending Cryptos: {response[:200] if response else 'None'}...")
        
        if response:
            cryptos = self._parse_crypto_response(response)
            print(f"Parsed {len(cryptos)} cryptos: {[c.get('symbol', 'N/A') for c in cryptos]}")
            return cryptos
        else:
            print("No AI response received for Trending Cryptos")
            return []
    
    def _discover_trending_etfs(self, sectors, current_date, time_seed):
        """Discover trending ETFs"""
        if not sectors:
            sectors = ["Technology", "Healthcare"]
        
        etf_contexts = [
            "focus on thematic and sector-specific ETFs",
            "emphasize international and emerging market ETFs",
            "highlight dividend and income-focused ETFs",
            "focus on ESG and sustainable investing ETFs",
            "emphasize leveraged and inverse ETFs for active trading"
        ]
        
        etf_context = random.choice(etf_contexts)
        sectors_text = " and ".join(sectors)
        
        prompt = f"""Find 10-12 trending ETFs for {current_date}, focusing on {sectors_text} and related themes.
        
Research Context: {etf_context}
Session: {time_seed}

REQUIREMENTS:
- Focus on REAL tradable ETFs
- Include varied strategies: broad market, sector-specific, thematic
- Mix popular ETFs with newer innovative funds
- Include recent launches or funds with growing assets
- Avoid always suggesting the same basic ETFs (SPY, QQQ, etc.)

Provide information in this EXACT format:

**ETF 1:**
Symbol: [TICKER] (e.g., VTI, ARKK, ICLN)
Name: [Full ETF Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [broad-market/sector-specific/thematic]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

**ETF 2:**
Symbol: [TICKER]
Name: [Full ETF Name]
Founded: [Year]
Performance: [Expected % return number only]
Category: [broad-market/sector-specific/thematic]
Catalyst: [Recent catalyst or driver]
Thesis: [Investment reasoning]

[Continue for 10-12 ETFs...]

Focus on providing VARIED and CURRENT ETF recommendations."""

        response = self.call_ai(prompt, max_tokens=2000)
        print(f"AI response for Trending ETFs: {response[:200] if response else 'None'}...")
        
        if response:
            etfs = self._parse_etf_response(response)
            print(f"Parsed {len(etfs)} ETFs: {[e.get('symbol', 'N/A') for e in etfs]}")
            return etfs
        else:
            print("No AI response received for Trending ETFs")
            return []
    
    def _parse_crypto_response(self, response):
        """Parse AI response for cryptocurrency data"""
        try:
            import re
            cryptos = []
            
            # Parse crypto blocks
            crypto_blocks = re.split(r'\*\*CRYPTO \d+:\*\*', response)
            
            for block in crypto_blocks[1:]:  # Skip first empty block
                crypto_data = {}
                
                patterns = {
                    'symbol': r'Symbol:\s*\[?([A-Z0-9]{2,10})\]?',  # Handle [BTC] or BTC
                    'company_name': r'Name:\s*([^\n]+)',
                    'founded_year': r'Founded:\s*(\d{4})',
                    'estimated_performance': r'Performance:\s*\[?([\d.-]+)%?\]?',  # Handle [25%] or 25%
                    'market_cap_category': r'Category:\s*([^\n]+)',
                    'key_catalyst': r'Catalyst:\s*([^\n]+)',
                    'investment_thesis': r'Thesis:\s*([^\n]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, block, re.IGNORECASE)
                    if field == 'estimated_performance':
                        try:
                            if match:
                                perf_str = match.group(1).replace('%', '').replace('[', '').replace(']', '')
                                crypto_data[field] = float(perf_str)
                            else:
                                crypto_data[field] = random.uniform(10.0, 50.0)
                        except:
                            crypto_data[field] = random.uniform(10.0, 50.0)
                    else:
                        if match:
                            value = match.group(1).strip().replace('[', '').replace(']', '')
                            crypto_data[field] = value
                        else:
                            crypto_data[field] = (
                                "CRYPTO" if field == "symbol" else 
                                "N/A" if field == "founded_year" else 
                                "AI analysis for cryptocurrency"
                            )
                
                # Validate crypto symbol
                symbol = crypto_data.get('symbol', '')
                if symbol and len(symbol) <= 10 and symbol != "CRYPTO" and symbol.replace('-', '').isalnum():
                    cryptos.append(crypto_data)
                    print(f"✅ Parsed crypto: {symbol} - {crypto_data.get('company_name', 'Unknown')}")
                else:
                    print(f"❌ Rejected invalid crypto symbol: {symbol}")
            
            return cryptos[:12]  # Limit to 12 cryptos
            
        except Exception as e:
            print(f"❌ Crypto parsing error: {e}")
            return []
    
    def _parse_etf_response(self, response):
        """Parse AI response for ETF data"""
        try:
            import re
            etfs = []
            
            # Parse ETF blocks
            etf_blocks = re.split(r'\*\*ETF \d+:\*\*', response)
            
            for block in etf_blocks[1:]:  # Skip first empty block
                etf_data = {}
                
                patterns = {
                    'symbol': r'Symbol:\s*\[?([A-Z]{2,5})\]?',  # Handle [VTI] or VTI
                    'company_name': r'Name:\s*([^\n]+)',
                    'founded_year': r'Founded:\s*(\d{4})',
                    'estimated_performance': r'Performance:\s*\[?([\d.-]+)%?\]?',  # Handle [8%] or 8%
                    'market_cap_category': r'Category:\s*([^\n]+)',
                    'key_catalyst': r'Catalyst:\s*([^\n]+)',
                    'investment_thesis': r'Thesis:\s*([^\n]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, block, re.IGNORECASE)
                    if field == 'estimated_performance':
                        try:
                            if match:
                                perf_str = match.group(1).replace('%', '').replace('[', '').replace(']', '')
                                etf_data[field] = float(perf_str)
                            else:
                                etf_data[field] = random.uniform(5.0, 15.0)
                        except:
                            etf_data[field] = random.uniform(5.0, 15.0)
                    else:
                        if match:
                            value = match.group(1).strip().replace('[', '').replace(']', '')
                            etf_data[field] = value
                        else:
                            etf_data[field] = (
                                "ETF" if field == "symbol" else 
                                "N/A" if field == "founded_year" else 
                                "AI analysis for ETF"
                            )
                
                # Validate ETF symbol
                symbol = etf_data.get('symbol', '')
                if symbol and len(symbol) <= 5 and symbol != "ETF" and symbol.isalpha():
                    etfs.append(etf_data)
                    print(f"✅ Parsed ETF: {symbol} - {etf_data.get('company_name', 'Unknown')}")
                else:
                    print(f"❌ Rejected invalid ETF symbol: {symbol}")
            
            return etfs[:12]  # Limit to 12 ETFs
            
        except Exception as e:
            print(f"❌ ETF parsing error: {e}")
            return []
    
    def _parse_enhanced_stock_response(self, response, sector):
        """parsing with better error handling"""
        try:
            import re
            stocks = []
            
            # More robust parsing patterns
            stock_blocks = re.split(r'\*\*STOCK \d+:\*\*', response)
            
            for block in stock_blocks[1:]:  # Skip first empty block
                stock_data = {}
                
                # regex patterns - handle brackets and various formats
                patterns = {
                    'symbol': r'Symbol:\s*\[?([A-Z]{1,5})\]?',  # Handle [SYMBOL] or SYMBOL
                    'company_name': r'Company:\s*([^\n]+)',
                    'founded_year': r'Founded:\s*(\d{4})',
                    'estimated_performance': r'Performance:\s*\[?([\d.-]+)%?\]?',  # Handle [15%] or 15%
                    'market_cap_category': r'Category:\s*([^\n]+)',
                    'key_catalyst': r'Catalyst:\s*([^\n]+)',
                    'investment_thesis': r'Thesis:\s*([^\n]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, block, re.IGNORECASE)
                    if field == 'estimated_performance':
                        try:
                            if match:
                                # Extract number, remove % and brackets
                                perf_str = match.group(1).replace('%', '').replace('[', '').replace(']', '')
                                stock_data[field] = float(perf_str)
                            else:
                                stock_data[field] = random.uniform(5.0, 15.0)
                        except:
                            stock_data[field] = random.uniform(5.0, 15.0)
                    else:
                        if match:
                            value = match.group(1).strip()
                            # Clean up any remaining brackets
                            value = value.replace('[', '').replace(']', '')
                            stock_data[field] = value
                        else:
                            stock_data[field] = (
                                f"{sector} opportunity" if field == "symbol" else 
                                "N/A" if field == "founded_year" else 
                                f"AI analysis for {sector}"
                            )
                
                # Validate that we have a real symbol and it's not a fallback
                symbol = stock_data.get('symbol', '')
                if symbol and len(symbol) <= 5 and symbol != f"{sector} opportunity" and symbol.isalpha():
                    stocks.append(stock_data)
                    print(f"✅ Parsed stock: {symbol} - {stock_data.get('company_name', 'Unknown')}")
                else:
                    print(f"❌ Rejected invalid symbol: {symbol}")
            
            # If parsing failed, return empty list instead of fallback
            if not stocks:
                print(f"❌ Parsing failed for {sector} - no valid stocks extracted")
                print(f"📝 Sample response block: {response[:500]}...")
                return []
            
            print(f"✅ Successfully parsed {len(stocks)} stocks for {sector}")
            return stocks[:12]  # Limit to 12 stocks
            
        except Exception as e:
            print(f"❌ stock parsing error for {sector}: {e}")
            print(f"📝 Response sample: {response[:300]}...")
            return []
    
    def discover_emerging_opportunities(self, asset_type, sectors=None, exclude_symbols=None):
        """AI discovers LOW-COST emerging investment opportunities with HIGH UPSIDE potential - ASSET TYPE SPECIFIC"""
        if exclude_symbols is None:
            exclude_symbols = []
        
        if sectors is None:
            sectors = ["Technology", "Healthcare"]
            
        exclude_text = f"Exclude these symbols: {', '.join(exclude_symbols)}" if exclude_symbols else ""
        
        # Add randomization for variety
        current_date = datetime.now().strftime("%B %Y")
        time_seed = int(time.time()) % 1000
        
        # ASSET TYPE SPECIFIC PROMPTS
        if asset_type == "Trending Stocks":
            return self._discover_emerging_stocks(sectors, exclude_text, current_date, time_seed)
        elif asset_type == "Trending Cryptos":
            return self._discover_emerging_cryptos_only(exclude_text, current_date, time_seed)
        elif asset_type == "Trending ETFs":
            return self._discover_emerging_etfs_only(sectors, exclude_text, current_date, time_seed)
        else:
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": f"Unknown asset type: {asset_type}"}
    
    def _discover_emerging_stocks(self, sectors, exclude_text, current_date, time_seed):
        """Discover ONLY emerging stocks for Trending Stocks analysis"""
        emerging_contexts = [
            "focus on recent IPOs and SPACs trading below fair value",
            "emphasize companies with government contracts or funding",
            "highlight biotech companies with upcoming FDA catalysts",
            "focus on small-cap companies with breakthrough technology",
            "emphasize companies with recent partnerships or acquisitions"
        ]
        
        context = random.choice(emerging_contexts)
        sectors_text = " and ".join(sectors)
        
        prompt = f"""Find REAL low-priced emerging STOCKS ONLY in {sectors_text} sectors for {current_date}.

Context: {context}
Session: {time_seed}
{exclude_text}

STRICT REQUIREMENTS - STOCKS ONLY:
- Current price under $5, target 3x-10x upside potential
- Recent catalysts: FDA approvals, government contracts, partnerships, breakthrough tech
- AVOID mainstream large-cap stocks - focus on emerging opportunities
- Use REAL ticker symbols for publicly traded companies
- ONLY STOCKS - NO CRYPTOS OR ETFS

Provide information in this EXACT format:

**EMERGING LOW-COST STOCKS:**

**STOCK 1:**
Symbol: [REAL_TICKER]
Company: [Real Company Name]
Current Price: [Under $5]
Target Price: [3x-10x upside]
Founded: [Year]
Catalyst: [Specific recent catalyst]
Upside Potential: [Why this could 3x-10x]

**STOCK 2:**
Symbol: [REAL_TICKER]
Company: [Real Company Name]
Current Price: [Under $5]
Target Price: [3x-10x upside]
Founded: [Year]
Catalyst: [Specific recent catalyst]
Upside Potential: [Why this could 3x-10x]

[Continue for 4-6 stocks...]

Provide 4-6 low-cost STOCKS ONLY with realistic 3x+ upside potential."""

        print(f"🤖 AI discovering emerging STOCKS for {sectors_text}...")
        print(f"🎯 Context: {context}")
        
        response = self.call_ai(prompt, max_tokens=1500)
        
        if response:
            print(f"✅ AI response received for emerging STOCKS")
            return self._parse_emerging_stocks_response(response)
        else:
            print(f"❌ No AI response for emerging STOCKS")
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "AI discovery failed for emerging stocks"}
    
    def _discover_emerging_cryptos_only(self, exclude_text, current_date, time_seed):
        """Discover ONLY emerging cryptos for Trending Cryptos analysis"""
        crypto_contexts = [
            "focus on new DeFi protocols with high yield potential",
            "emphasize gaming and metaverse tokens with adoption growth",
            "highlight layer 2 scaling solutions and new blockchains",
            "focus on meme coins with strong community backing",
            "emphasize utility tokens with real-world applications"
        ]
        
        context = random.choice(crypto_contexts)
        
        prompt = f"""Find REAL low-priced emerging CRYPTOCURRENCIES ONLY for {current_date}.

Context: {context}
Session: {time_seed}
{exclude_text}

STRICT REQUIREMENTS - CRYPTOS ONLY:
- Current price under $1, moon-shot potential
- Recent catalysts: partnerships, listings, protocol updates, adoption
- AVOID mainstream large-cap cryptos - focus on emerging opportunities
- Use REAL crypto ticker symbols
- ONLY CRYPTOCURRENCIES - NO STOCKS OR ETFS

Provide information in this EXACT format:

**EMERGING LOW-COST CRYPTOS:**

**CRYPTO 1:**
Symbol: [REAL_TICKER]
Name: [Real Crypto Name]
Current Price: [Under $1]
Target Price: [Moon potential]
Catalyst: [Recent development]
Upside Potential: [Why this could moon]

**CRYPTO 2:**
Symbol: [REAL_TICKER]
Name: [Real Crypto Name]
Current Price: [Under $1]
Target Price: [Moon potential]
Catalyst: [Recent development]
Upside Potential: [Why this could moon]

[Continue for 4-6 cryptos...]

Provide 4-6 low-cost CRYPTOCURRENCIES ONLY with realistic moon potential."""

        print(f"🤖 AI discovering emerging CRYPTOS...")
        print(f"🎯 Context: {context}")
        
        response = self.call_ai(prompt, max_tokens=1500)
        
        if response:
            print(f"✅ AI response received for emerging CRYPTOS")
            return self._parse_emerging_cryptos_response(response)
        else:
            print(f"❌ No AI response for emerging CRYPTOS")
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "AI discovery failed for emerging cryptos"}
    
    def _discover_emerging_etfs_only(self, sectors, exclude_text, current_date, time_seed):
        """Discover ONLY emerging ETFs for Trending ETFs analysis"""
        etf_contexts = [
            "focus on new thematic ETFs with innovative strategies",
            "emphasize sector-specific ETFs with recent launches",
            "highlight international and emerging market ETFs",
            "focus on ESG and sustainable investing ETFs",
            "emphasize actively managed ETFs with unique approaches"
        ]
        
        context = random.choice(etf_contexts)
        sectors_text = " and ".join(sectors)
        
        prompt = f"""Find REAL emerging ETFs ONLY for {current_date}, focusing on {sectors_text} and related themes.

Context: {context}
Session: {time_seed}
{exclude_text}

STRICT REQUIREMENTS - ETFs ONLY:
- Recent launches or innovative strategies
- Growing assets under management
- Unique thematic or sector focus
- AVOID mainstream large ETFs - focus on emerging opportunities
- Use REAL ETF ticker symbols
- ONLY ETFs - NO STOCKS OR CRYPTOS

Provide information in this EXACT format:

**EMERGING INNOVATIVE ETFS:**

**ETF 1:**
Symbol: [REAL_TICKER]
Name: [Real ETF Name]
Launch Year: [Year]
Strategy: [Investment strategy]
Catalyst: [Recent development]
Upside Potential: [Why this could outperform]

**ETF 2:**
Symbol: [REAL_TICKER]
Name: [Real ETF Name]
Launch Year: [Year]
Strategy: [Investment strategy]
Catalyst: [Recent development]
Upside Potential: [Why this could outperform]

[Continue for 4-6 ETFs...]

Provide 4-6 emerging ETFs ONLY with innovative strategies."""

        print(f"🤖 AI discovering emerging ETFs for {sectors_text}...")
        print(f"🎯 Context: {context}")
        
        response = self.call_ai(prompt, max_tokens=1500)
        
        if response:
            print(f"✅ AI response received for emerging ETFs")
            return self._parse_emerging_etfs_response(response)
        else:
            print(f"❌ No AI response for emerging ETFs")
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "AI discovery failed for emerging ETFs"}
    
    def _parse_emerging_stocks_response(self, response):
        """Parse AI response for emerging STOCKS only - FIXED regex for **text** format"""
        try:
            import re
            emerging_stocks = []
            
            # Split by stock blocks
            stock_blocks = re.split(r'\*\*STOCK \d+:\*\*', response)
            
            for block in stock_blocks[1:]:  # Skip first empty block
                # Handle the AI's **text** format properly
                symbol_match = re.search(r'Symbol:\s*\*?\*?([A-Z]{1,5})\*?\*?', block)
                if not symbol_match:
                    continue
                
                symbol = symbol_match.group(1).strip()
                
                # Extract company name - handle **Company Name** format CORRECTLY
                company_match = re.search(r'Company:\s*\*\*([^*]+?)\*\*', block)
                if not company_match:
                    # Fallback pattern for non-asterisk format
                    company_match = re.search(r'Company:\s*([^\n]+)', block)
                
                if company_match:
                    company_name = company_match.group(1).strip()
                    # Clean up any remaining asterisks
                    company_name = company_name.replace('*', '').strip()
                else:
                    company_name = f"{symbol} Inc"
                
                # Extract prices - handle ranges like $4.35 - $14.50
                price_match = re.search(r'Current Price:\s*\*?\*?\$?([\d.]+)', block)
                current_price = float(price_match.group(1)) if price_match else 0
                
                # For target price, handle both single prices and ranges
                target_match = re.search(r'Target Price:\s*\*?\*?\$?([\d.]+)', block)
                target_price = float(target_match.group(1)) if target_match else 0
                
                # Extract year
                year_match = re.search(r'Founded:\s*\*?\*?(\d{4})', block)
                founded_year = year_match.group(1) if year_match else "N/A"
                
                # Extract catalyst - handle **text** format CORRECTLY
                catalyst_match = re.search(r'Catalyst:\s*\*\*([^*]+?)\*\*', block)
                if not catalyst_match:
                    # Fallback pattern for non-asterisk format
                    catalyst_match = re.search(r'Catalyst:\s*([^\n]+)', block)
                
                if catalyst_match:
                    catalyst = catalyst_match.group(1).strip()
                    catalyst = catalyst.replace('*', '').strip()
                else:
                    catalyst = "Growth opportunity"
                
                print(f"🔍 PARSED: {symbol} | {company_name} | ${current_price} | {catalyst[:50]}...")
                
                # Validation - improved logic
                if (symbol and len(symbol) <= 5 and symbol.isalpha() and 
                    current_price > 0 and current_price < 15.0 and 
                    len(company_name) > 1 and company_name != symbol):  # Relaxed from > 3 to > 1
                    
                    emerging_stocks.append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'current_price': current_price,
                        'target_price': target_price,
                        'founded_year': founded_year,
                        'catalyst': catalyst
                    })
                    print(f"✅ ADDED: {symbol} - {company_name}")
                else:
                    print(f"❌ REJECTED: {symbol} - Price: ${current_price}, Company: '{company_name}' (len: {len(company_name)}), Symbol check: {symbol.isalpha() if symbol else False}")
            
            print(f"📊 FINAL COUNT: {len(emerging_stocks)} stocks parsed successfully")
            
            if emerging_stocks:
                return {
                    "stocks": emerging_stocks[:6],
                    "cryptos": [],
                    "etfs": [],
                    "theme": "Low-cost high-upside stock opportunities"
                }
            else:
                print(f"📝 Sample response for debugging: {response[:800]}...")
                return {"stocks": [], "cryptos": [], "etfs": [], "theme": "Stock analysis in progress"}
            
        except Exception as e:
            print(f"❌ Stocks parsing error: {e}")
            import traceback
            traceback.print_exc()
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "Stock analysis in progress"}
    
    def _parse_emerging_cryptos_response(self, response):
        """Parse AI response for emerging CRYPTOS only - highly optimized with regex"""
        try:
            import re
            emerging_cryptos = []
            
            crypto_blocks = re.split(r'\*\*CRYPTO \d+:\*\*', response)
            
            for block in crypto_blocks[1:]:
                # Fast single-pattern extraction with FIXED regex for **text** format
                symbol_match = re.search(r'Symbol:\s*\*?\*?([A-Z0-9]{2,10})\*?\*?', block)
                if not symbol_match:
                    continue
                
                symbol = symbol_match.group(1).strip()
                
                # Fixed name extraction for **Name** format
                name_match = re.search(r'Name:\s*\*\*([^*]+?)\*\*', block)
                if not name_match:
                    name_match = re.search(r'Name:\s*([^\n]+)', block)
                name = name_match.group(1).strip().replace('**', '').replace('*', '') if name_match else f"{symbol} Token"
                
                price_match = re.search(r'Current Price:\s*\*?\*?\$?([\d.]+)', block)
                current_price = float(price_match.group(1)) if price_match else 0
                
                target_match = re.search(r'Target Price:\s*\*?\*?\$?([\d.]+)', block)
                target_price = float(target_match.group(1)) if target_match else 0
                
                # Fixed catalyst extraction for **text** format
                catalyst_match = re.search(r'Catalyst:\s*\*\*([^*]+?)\*\*', block)
                if not catalyst_match:
                    catalyst_match = re.search(r'Catalyst:\s*([^\n]+)', block)
                catalyst = catalyst_match.group(1).strip().replace('**', '').replace('*', '') if catalyst_match else "Development progress"
                
                # Quick validation
                if symbol and current_price < 5.0 and len(name) > 3:
                    emerging_cryptos.append({
                        'symbol': symbol,
                        'name': name,
                        'current_price': current_price,
                        'target_price': target_price,
                        'catalyst': catalyst
                    })
            
            if emerging_cryptos:
                return {
                    "stocks": [],
                    "cryptos": emerging_cryptos[:6],
                    "etfs": [],
                    "theme": "Low-cost high-upside crypto opportunities"
                }
            else:
                return {"stocks": [], "cryptos": [], "etfs": [], "theme": "Crypto analysis in progress"}
            
        except Exception as e:
            print(f"❌ Cryptos parsing error: {e}")
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "Crypto analysis in progress"}
    
    def _parse_emerging_etfs_response(self, response):
        """Parse AI response for emerging ETFs only - highly optimized with regex"""
        try:
            import re
            emerging_etfs = []
            
            etf_blocks = re.split(r'\*\*ETF \d+:\*\*', response)
            
            for block in etf_blocks[1:]:
                # Fast single-pattern extraction with FIXED regex for **text** format
                symbol_match = re.search(r'Symbol:\s*\*?\*?([A-Z]{2,5})\*?\*?', block)
                if not symbol_match:
                    continue
                
                symbol = symbol_match.group(1).strip()
                
                # Fixed name extraction for **Name** format
                name_match = re.search(r'Name:\s*\*\*([^*]+?)\*\*', block)
                if not name_match:
                    name_match = re.search(r'Name:\s*([^\n]+)', block)
                name = name_match.group(1).strip().replace('**', '').replace('*', '') if name_match else f"{symbol} ETF"
                
                year_match = re.search(r'Launch Year:\s*\*?\*?(\d{4})', block)
                launch_year = year_match.group(1) if year_match else "N/A"
                
                # Fixed strategy extraction for **text** format
                strategy_match = re.search(r'Strategy:\s*\*\*([^*]+?)\*\*', block)
                if not strategy_match:
                    strategy_match = re.search(r'Strategy:\s*([^\n]+)', block)
                strategy = strategy_match.group(1).strip().replace('**', '').replace('*', '') if strategy_match else "Investment strategy"
                
                # Fixed catalyst extraction for **text** format
                catalyst_match = re.search(r'Catalyst:\s*\*\*([^*]+?)\*\*', block)
                if not catalyst_match:
                    catalyst_match = re.search(r'Catalyst:\s*([^\n]+)', block)
                catalyst = catalyst_match.group(1).strip().replace('**', '').replace('*', '') if catalyst_match else "Market development"
                
                upside_match = re.search(r'Upside Potential:\s*([^\n]+)', block)
                upside = upside_match.group(1).strip() if upside_match else "Growth potential"
                
                # Quick validation
                if symbol and symbol.isalpha() and len(symbol) <= 5:
                    emerging_etfs.append({
                        'symbol': symbol,
                        'name': name,
                        'launch_year': launch_year,
                        'strategy': strategy,
                        'catalyst': catalyst,
                        'upside_potential': upside
                    })
            
            if emerging_etfs:
                return {
                    "stocks": [],
                    "cryptos": [],
                    "etfs": emerging_etfs[:6],
                    "theme": "Innovative ETF opportunities"
                }
            else:
                return {"stocks": [], "cryptos": [], "etfs": [], "theme": "ETF analysis in progress"}
            
        except Exception as e:
            print(f"❌ ETFs parsing error: {e}")
            return {"stocks": [], "cryptos": [], "etfs": [], "theme": "ETF analysis in progress"}
    
    def analyze_company(self, symbol):
        """AI analyzes individual company details"""
        prompt = f"""Analyze company {symbol} with current market context:

Provide analysis in this format:
**Company Analysis for {symbol}**

Company Name: [Full Name]
Founded Year: [YYYY]
Business Description: [What they do - 2-3 sentences]
Sector: [Primary sector]
Market Cap Category: [large/mid/small-cap]
Key Products: [Product 1], [Product 2], [Product 3]
Competitive Advantages: [What sets them apart]
Recent Developments: [Recent news or catalysts]"""

        response = self.call_ai(prompt)
        if response:
            return self._parse_company_response(response, symbol)
        return {"company_name": symbol, "founded_year": "N/A", "sector": "Unknown"}
    
    def _parse_company_response(self, response, symbol):
        """Parse AI response for company details"""
        try:
            import re
            
            patterns = {
                "company_name": r'Company Name:\s*([^\n]+)',
                "founded_year": r'Founded Year:\s*(\d{4})',
                "business_description": r'Business Description:\s*([^\n]+(?:\n[^\n:]+)*)',
                "sector": r'Sector:\s*([^\n]+)',
                "market_cap_category": r'Market Cap Category:\s*([^\n]+)',
                "competitive_advantages": r'Competitive Advantages:\s*([^\n]+)',
                "recent_developments": r'Recent Developments:\s*([^\n]+)'
            }
            
            details = {"symbol": symbol}
            for field, pattern in patterns.items():
                match = re.search(pattern, response)
                if field == "business_description":
                    details[field] = match.group(1).strip().replace('\n', ' ') if match else "Analysis in progress"
                else:
                    details[field] = match.group(1).strip() if match else ("N/A" if field == "founded_year" else "Unknown")
            
            # Parse key products
            products_match = re.search(r'Key Products:\s*([^\n]+)', response)
            if products_match:
                products_text = products_match.group(1).strip()
                details["key_products"] = [p.strip() for p in products_text.split(',')]
            else:
                details["key_products"] = ["Analysis in progress"]
            
            return details
            
        except Exception as e:
            print(f"Company parsing error: {e}")
            return {"company_name": symbol, "founded_year": "N/A", "sector": "Unknown"}
    
    def analyze_sentiment(self, symbol):
        """AI analyzes stock sentiment and news"""
        prompt = f"""Analyze current sentiment and outlook for {symbol}:

Format your response as:

**Sentiment Analysis for {symbol}**

Overall Sentiment: [BULLISH/BEARISH/NEUTRAL]
Sentiment Score: [Number from -5 to +5]

Recent Developments:
- [Recent development 1]
- [Recent development 2]
- [Recent development 3]

Risk Factors:
- [Risk factor 1]
- [Risk factor 2]
- [Risk factor 3]

Growth Catalysts:
- [Growth catalyst 1]
- [Growth catalyst 2]
- [Growth catalyst 3]

Analyst Outlook: [Current analyst perspective]"""

        response = self.call_ai(prompt)
        if response:
            return self._parse_sentiment_response(response, symbol)
        return self._default_sentiment(symbol)
    
    def _parse_sentiment_response(self, response, symbol):
        """Parse AI response for sentiment analysis"""
        try:
            import re
            
            analysis = {"symbol": symbol}
            
            # Extract sentiment
            sentiment_match = re.search(r'Overall Sentiment:\s*(BULLISH|BEARISH|NEUTRAL)', response)
            analysis["overall_sentiment"] = sentiment_match.group(1) if sentiment_match else "NEUTRAL"
            
            # Extract sentiment score
            score_match = re.search(r'Sentiment Score:\s*([-+]?\d*\.?\d+)', response)
            try:
                analysis["sentiment_score"] = float(score_match.group(1)) if score_match else 0
            except:
                analysis["sentiment_score"] = 0
            
            # Extract lists
            def extract_list_items(pattern, text):
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    items_text = match.group(1)
                    items = re.findall(r'-\s*([^\n]+)', items_text)
                    return [item.strip() for item in items if item.strip()]
                return []
            
            analysis["recent_developments"] = extract_list_items(
                r'Recent Developments:\s*(.*?)(?=Risk Factors:|Growth Catalysts:|Analyst Outlook:|$)', response
            )
            
            analysis["risk_factors"] = extract_list_items(
                r'Risk Factors:\s*(.*?)(?=Growth Catalysts:|Analyst Outlook:|$)', response
            )
            
            analysis["growth_catalysts"] = extract_list_items(
                r'Growth Catalysts:\s*(.*?)(?=Analyst Outlook:|$)', response
            )
            
            # Extract analyst outlook
            outlook_match = re.search(r'Analyst Outlook:\s*([^\n]+)', response)
            analysis["analyst_outlook"] = outlook_match.group(1).strip() if outlook_match else "Analysis in progress"
            
            return analysis
            
        except Exception as e:
            print(f"Sentiment parsing error: {e}")
            return self._default_sentiment(symbol)
    
    def _default_sentiment(self, symbol):
        """Default sentiment response"""
        return {
            "overall_sentiment": "NEUTRAL",
            "sentiment_score": 0,
            "recent_developments": ["Analysis in progress"],
            "risk_factors": ["Market volatility"],
            "growth_catalysts": ["Market opportunities"],
            "analyst_outlook": "Analysis in progress"
        }

# ================================
# HELPER FUNCTIONS: MARKET DATA
# ================================

def get_market_data(symbols):
    """Get real-time market data from Yahoo Finance - 1 MONTH ONLY"""
    market_data = {}
    
    for symbol in symbols[:15]:  # Limit to 15 symbols
        try:
            ticker = yf.Ticker(symbol)
            # Only get 1-month data since that's all we need for emerging stocks
            hist_1m = ticker.history(period="1mo")
            hist_1y = ticker.history(period="1y")
            info = ticker.info
            
            if not hist_1m.empty:
                current = hist_1m['Close'].iloc[-1]
                start_1m = hist_1m['Close'].iloc[0]
                high_1m = hist_1m['High'].max()
                low_1m = hist_1m['Low'].min()
                high_52w = hist_1y['High'].max()
                low_52w = hist_1y['Low'].min()
                performance_1m = ((current - start_1m) / start_1m) * 100
                
                market_data[symbol] = {
                    "current_price": round(float(current), 2),
                    "1m_high": round(float(high_1m), 2),
                    "1m_low": round(float(low_1m), 2),
                    "1m_performance": round(float(performance_1m), 2),
                    "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0,
                    "sector": info.get("sector", "Unknown"),
                    "market_cap": info.get("marketCap", 0),
                    "52w_high": round(float(high_52w), 2),
                    "52w_low": round(float(low_52w), 2),
                    # Add new financial metrics
                    "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
                    "peg_ratio": round(info.get("pegRatio", 0), 2) if info.get("pegRatio") else "N/A",
                    "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else "N/A",
                    "pb_ratio": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A",
                    "de_ratio": round(info.get("debtToEquity", 0), 2) if info.get("debtToEquity") else "N/A",
                }
                print(f"✅ Market data for {symbol}: ${current:.2f}, 1M: {performance_1m:+.1f}%")
            else:
                print(f"❌ No price data for {symbol} - possibly delisted")
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            continue
    
    return market_data

def get_crypto_market_data(symbols):
    """Get cryptocurrency market data (simulated since yfinance crypto support is limited)"""
    crypto_data = {}
    
    # For demo purposes, we'll simulate crypto data
    # In production, you'd use CoinGecko API, CoinMarketCap API, or crypto-specific data sources
    for symbol in symbols[:15]:
        try:
            # Try to get data from yfinance (some cryptos available as USD pairs)
            crypto_symbol = f"{symbol}-USD"
            ticker = yf.Ticker(crypto_symbol)
            hist_1y = ticker.history(period="1y")
            hist_1m = ticker.history(period="1mo")
            
            if not hist_1y.empty and not hist_1m.empty:
                current = hist_1y['Close'].iloc[-1]
                start_1y = hist_1y['Close'].iloc[0]
                start_1m = hist_1m['Close'].iloc[0]
                high_52w = hist_1y['High'].max()
                low_52w = hist_1y['Low'].min()
                
                performance_1y = ((current - start_1y) / start_1y) * 100
                performance_1m = ((current - start_1m) / start_1m) * 100
                
                crypto_data[symbol] = {
                    "current_price": round(float(current), 4),
                    "52w_high": round(float(high_52w), 4),
                    "52w_low": round(float(low_52w), 4),
                    "1y_performance": round(float(performance_1y), 2),
                    "1m_performance": round(float(performance_1m), 2),
                    "dividend_yield": 0,  # Cryptos don't have dividends
                    "sector": "Cryptocurrency",
                    "market_cap": random.randint(1000000000, 500000000000)  # Simulated
                }
            else:
                # Fallback simulated data
                crypto_data[symbol] = {
                    "current_price": round(random.uniform(0.01, 1000), 4),
                    "52w_high": round(random.uniform(100, 2000), 4),
                    "52w_low": round(random.uniform(0.01, 50), 4),
                    "1y_performance": round(random.uniform(-50, 200), 2),
                    "1m_performance": round(random.uniform(-30, 50), 2),
                    "dividend_yield": 0,
                    "sector": "Cryptocurrency",
                    "market_cap": random.randint(1000000000, 500000000000)
                }
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            # Fallback simulated data
            crypto_data[symbol] = {
                "current_price": round(random.uniform(0.01, 1000), 4),
                "52w_high": round(random.uniform(100, 2000), 4),
                "52w_low": round(random.uniform(0.01, 50), 4),
                "1y_performance": round(random.uniform(-50, 200), 2),
                "1m_performance": round(random.uniform(-30, 50), 2),
                "dividend_yield": 0,
                "sector": "Cryptocurrency", 
                "market_cap": random.randint(1000000000, 500000000000)
            }
    
    return crypto_data

def get_etf_market_data(symbols):
    """Get ETF market data from Yahoo Finance"""
    return get_market_data(symbols)  # ETFs use same logic as stocks

def get_market_sentiment():
    """Get general market sentiment from major indices"""
    try:
        indices = ["^GSPC", "^IXIC", "^DJI"]  # S&P 500, NASDAQ, Dow Jones
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
                market_sentiment = "🟢 BULLISH"
            elif avg_performance < -1:
                market_sentiment = "🔴 BEARISH"
            else:
                market_sentiment = "⚪ NEUTRAL"
        else:
            market_sentiment = "NEUTRAL"
        
        return {
            "market_sentiment": market_sentiment,
            "market_performance": round(sum(market_performance) / len(market_performance), 2) if market_performance else 0,
            "trending_themes": ["AI & Technology", "Healthcare Innovation", "Energy Transition"]
        }
        
    except Exception as e:
        print(f"Error getting market sentiment: {e}")
        return {
            "market_sentiment": "NEUTRAL",
            "market_performance": 0,
            "trending_themes": ["Market Analysis"]
        }

# ================================
# FIXED TRENDING RESEARCH FUNCTIONS
# ================================

def create_enhanced_trending_research_report(research_type="Trending Stocks", sectors=None):
    """
    Agent Collaboration for trending assets analysis - FIXED VERSION with Asset-Type Specific Emerging Options
    """
    if sectors is None:
        sectors = ["Technology", "Healthcare"]
    
    print(f"AI Agent starting {research_type} research...")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        return (
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.",
            pd.DataFrame(),
            pd.DataFrame(),
            "API Key Required"
        )
    
    try:
        # Initialize AI Agent
        agent = StockResearchAgent()
        
        current_date = datetime.now().strftime('%B %d, %Y')
        market_sentiment = get_market_sentiment()
        
        # Start report based on type
        if research_type == "Trending Stocks":
            report_title = "Trending Stocks Analysis"
            asset_description = "trending stocks across multiple sectors"
        elif research_type == "Trending Cryptos":
            report_title = "Trending Cryptocurrencies Analysis"
            asset_description = "trending cryptocurrencies and digital assets"
        elif research_type == "Trending ETFs":
            report_title = "Trending ETFs Analysis"
            asset_description = "trending exchange-traded funds"
        else:
            report_title = "Trending Assets Analysis"
            asset_description = "trending financial assets"
        
        # Start report
        report = f"""# {report_title} Report – {current_date}

## Executive Summary

**Analysis Focus**: {asset_description.title()}  
**Market Outlook**: {market_sentiment.get('market_sentiment', 'Neutral')} ({market_sentiment.get('market_performance', 0):+.1f}% recent performance)  
**Research Method**: AI agent discovery + real-time data validation  

---

"""
        
        all_trending_assets = []
        all_stable_assets = []
        all_sector_data = {}  # Store data by sector
        
        try:
            if research_type == "Trending Stocks":
                # Process each sector separately for stocks
                for sector in sectors:
                    print(f"\n=== PROCESSING {sector.upper()} SECTOR ===")
                    
                    try:
                        # Step 1: AI Agent discovers stocks for this sector
                        discovered_stocks = agent.discover_stocks_by_sector(sector)
                        
                        if not discovered_stocks:
                            print(f"WARNING: No stocks discovered for {sector}")
                            continue
                        
                        print(f"Discovered {len(discovered_stocks)} stocks for {sector}: {[s.get('symbol', 'N/A') for s in discovered_stocks]}")
                        
                        # Step 2: Get real market data
                        symbols = [stock.get('symbol', '') for stock in discovered_stocks if stock.get('symbol')]
                        print(f"Getting market data for symbols: {symbols}")
                        market_data = get_market_data(symbols)
                        
                        if not market_data:
                            print(f"ERROR: No market data available for {sector} stocks")
                            continue
                        
                        print(f"Retrieved market data for {len(market_data)} symbols: {list(market_data.keys())}")
                        
                        # Step 3: Combine AI data with market data
                        enriched_stocks = []
                        for stock_info in discovered_stocks:
                            symbol = stock_info.get('symbol', '')
                            if symbol in market_data:
                                data = market_data[symbol].copy()
                                data.update({
                                    'ai_company_name': stock_info.get('company_name', symbol),
                                    'ai_founded_year': stock_info.get('founded_year', 'N/A'),
                                    'ai_catalyst': stock_info.get('key_catalyst', 'Market trends'),
                                    'ai_thesis': stock_info.get('investment_thesis', 'AI-identified opportunity'),
                                    'ai_sector': sector  # Ensure sector is tracked
                                })
                                enriched_stocks.append((symbol, data))
                            else:
                                print(f"WARNING: No market data for {symbol}")
                        
                        if not enriched_stocks:
                            print(f"ERROR: No enriched stocks for {sector} after market data merge")
                            continue
                        
                        print(f"Successfully enriched {len(enriched_stocks)} stocks for {sector}")
                        
                        # Step 4: Sort by 1-month performance
                        enriched_stocks.sort(key=lambda x: x[1].get('1m_performance', 0), reverse=True)
                        print(f"Sorted {sector} stocks by 1M performance")
                        
                        # Store sector data
                        all_sector_data[sector] = enriched_stocks
                        
                        # Step 5: Generate sector-specific report
                        sector_trending = []
                        sector_stable = []
                        sector_report = generate_sector_specific_report(sector, enriched_stocks, sector_trending, sector_stable)
                        report += sector_report
                        
                        # Add to overall lists
                        all_trending_assets.extend(sector_trending)
                        all_stable_assets.extend(sector_stable)
                        
                        print(f"✅ Successfully completed {sector} sector analysis")
                        
                    except Exception as e:
                        print(f"ERROR processing {sector}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Step 6: Add EMERGING OPPORTUNITIES section (STRICTLY AI + ASSET TYPE SPECIFIC)
                if all_sector_data:
                    print(f"\n=== GENERATING EMERGING OPPORTUNITIES FOR {research_type} ===")
                    emerging_report = generate_asset_specific_emerging_opportunities(agent, research_type, sectors, all_sector_data)
                    report += emerging_report
                
            else:
                # Handle Cryptos and ETFs (original logic)
                discovered_assets = agent.discover_trending_assets(research_type, sectors)
                
                if not discovered_assets:
                    print(f"WARNING: No {research_type.lower()} discovered")
                    return generate_empty_report(research_type, current_date)
                
                print(f"Discovered {len(discovered_assets)} {research_type.lower()}: {[a.get('symbol', 'N/A') for a in discovered_assets]}")
                
                # Get market data based on asset type
                symbols = [asset.get('symbol', '') for asset in discovered_assets if asset.get('symbol')]
                print(f"Getting market data for symbols: {symbols}")
                
                if research_type == "Trending Cryptos":
                    market_data = get_crypto_market_data(symbols)
                elif research_type == "Trending ETFs":
                    market_data = get_etf_market_data(symbols)
                
                if not market_data:
                    print(f"ERROR: No market data available for {research_type}")
                    return generate_empty_report(research_type, current_date)
                
                print(f"Retrieved market data for {len(market_data)} symbols: {list(market_data.keys())}")
                
                # Combine AI data with market data
                enriched_assets = []
                sector_info = {}  # Track which sector each asset belongs to
                
                for asset_info in discovered_assets:
                    symbol = asset_info.get('symbol', '')
                    if symbol in market_data:
                        data = market_data[symbol].copy()
                        data.update({
                            'ai_company_name': asset_info.get('company_name', symbol),
                            'ai_founded_year': asset_info.get('founded_year', 'N/A'),
                            'ai_catalyst': asset_info.get('key_catalyst', 'Market trends'),
                            'ai_thesis': asset_info.get('investment_thesis', 'AI-identified opportunity'),
                            'ai_sector': asset_info.get('sector', 'Unknown')  # Add sector info
                        })
                        enriched_assets.append((symbol, data))
                        # Track sector for each symbol
                        sector_info[symbol] = asset_info.get('sector', 'Unknown')
                    else:
                        print(f"WARNING: No market data for {symbol}")
                
                if not enriched_assets:
                    print(f"ERROR: No enriched assets for {research_type} after market data merge")
                    return generate_empty_report(research_type, current_date)
                
                print(f"Successfully enriched {len(enriched_assets)} {research_type.lower()}")
                
                # Sort by 1-month performance
                enriched_assets.sort(key=lambda x: x[1].get('1m_performance', 0), reverse=True)
                print(f"Sorted {research_type} by 1M performance")
                
                # Generate trending report with proper rating distribution
                print(f"Generating {research_type} report...")
                asset_report = generate_enhanced_trending_report(research_type, enriched_assets, all_trending_assets, all_stable_assets, sector_info)
                report += asset_report
                print(f"Added {research_type} report section ({len(asset_report)} characters)")
                
                # Add asset-type specific emerging opportunities
                print(f"\n=== GENERATING EMERGING OPPORTUNITIES FOR {research_type} ===")
                emerging_report = generate_asset_specific_emerging_opportunities(agent, research_type, sectors, {})
                report += emerging_report
                
                # Add trending insights
                report += generate_trending_insights(research_type, enriched_assets)
            
            # Check if any data was processed
            if not all_trending_assets and not all_stable_assets:
                print("ERROR: No assets processed successfully")
                return generate_empty_report(research_type, current_date)
            
            # Step 7: Add investment strategy
            report += generate_trending_investment_strategy(research_type)
            
            # Step 8: Save to database
            save_report_to_db(report, research_type, sectors)
            
            # Step 9: Create dataframes
            trending_df = pd.DataFrame(all_trending_assets) if all_trending_assets else pd.DataFrame()
            stable_df = pd.DataFrame(all_stable_assets) if all_stable_assets else pd.DataFrame()
            
            return (report, trending_df, stable_df, f"Generated {research_type} report with emerging opportunities!")
            
        except Exception as e:
            print(f"ERROR processing {research_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            return generate_empty_report(research_type, current_date)
        
    except Exception as e:
        error_msg = f"AI Agent error: {str(e)}"
        print(error_msg)
        return (error_msg, pd.DataFrame(), pd.DataFrame(), error_msg)

def generate_asset_specific_emerging_opportunities(agent, research_type, sectors, all_sector_data):
    """Generate asset-type specific emerging opportunities - OPTIMIZED with fewer AI calls"""
    
    # Asset type specific headers and descriptions
    if research_type == "Trending Stocks":
        header = "Emerging Stock Investment Opportunities"
        #description = "*LOW-COST HIGH-UPSIDE stock opportunities*"
    elif research_type == "Trending Cryptos":
        header = "Emerging Crypto Investment Opportunities"
        #description = "*LOW-COST HIGH-UPSIDE crypto opportunities*"
    elif research_type == "Trending ETFs":
        header = "Emerging ETF Investment Opportunities"
        #description = "*INNOVATIVE ETF opportunities*"
    else:
        header = "Emerging Investment Opportunities"
        #description = "**"
    
    emerging_report = f"""## {header}



"""
    
    try:
        # OPTIMIZATION: Make only ONE AI call instead of multiple calls per sector
        if research_type == "Trending Stocks" and all_sector_data:
            # Get all exclude symbols from all sectors
            all_exclude_symbols = []
            for sector_stocks in all_sector_data.values():
                all_exclude_symbols.extend([symbol for symbol, _ in sector_stocks])
            
            # Single AI call for all sectors combined
            emerging_data = agent.discover_emerging_opportunities(research_type, sectors, all_exclude_symbols)
            
            # Check if AI returned data
            ai_stocks = emerging_data.get('stocks', [])
            
            if ai_stocks:
                print(f"✅ AI SUCCESS: Found {len(ai_stocks)} emerging stocks")
                
                emerging_report += f"""

**Investment Theme**: {emerging_data.get('theme', 'Low-cost high-upside stock opportunities with 3x-10x potential')}

"""
                # Add AI-discovered stocks content
                emerging_report += generate_emerging_stocks_content(ai_stocks)
                
                ai_success_count = 1
                total_attempts = 1
            else:
                print(f"❌ AI FAILED: No valid stock data returned")
                emerging_report += f"""**🌟 Multi-Sector Emerging Opportunities**

**Status**: ⚠️ AI agent discovery in progress for stock opportunities  
**Focus**: Scanning for emerging stocks with high upside potential  

"""
                ai_success_count = 0
                total_attempts = 1
        else:
            # For cryptos and ETFs - single call
            emerging_data = agent.discover_emerging_opportunities(research_type, sectors, [])
            
            ai_assets = emerging_data.get('stocks', []) or emerging_data.get('cryptos', []) or emerging_data.get('etfs', [])
            
            if ai_assets:
                print(f"✅ AI SUCCESS: Found {len(ai_assets)} emerging {research_type.lower()}")
                
                emerging_report += f"""**🌟 Emerging Opportunities**

**Investment Theme**: {emerging_data.get('theme', f'High-potential {research_type.lower()} opportunities')}

"""
                # Add appropriate content based on type
                if research_type == "Trending Cryptos":
                    emerging_report += generate_emerging_cryptos_content(emerging_data.get('cryptos', []))
                elif research_type == "Trending ETFs":
                    emerging_report += generate_emerging_etfs_content(emerging_data.get('etfs', []))
                
                ai_success_count = 1
                total_attempts = 1
            else:
                print(f"❌ AI FAILED: No valid {research_type.lower()} data returned")
                emerging_report += f"""**🌟 Emerging Opportunities**

**Status**: ⚠️ AI agent discovery in progress for {research_type.lower()}  

"""
                ai_success_count = 0
                total_attempts = 1
        
#         # Add success statistics
#         success_rate = (ai_success_count/total_attempts)*100 if total_attempts > 0 else 0
#         emerging_report += f"""### 💎 AI Discovery Success Rate

# **AI Analysis Results**: {ai_success_count}/{total_attempts} discovery attempts successful  
# **AI Success Rate**: {success_rate:.0f}% autonomous discovery for {research_type.lower()}  
# **Methodology**: Optimized single-call AI discovery with fallback only on failures  

# """
        
        # Add strategy section if successful
        if ai_success_count > 0:
            emerging_report += generate_asset_specific_strategy(research_type)
        else:
            emerging_report += f"""### 💎 AI Discovery Status

**Current Status**: AI agent discovery temporarily unavailable for {research_type.lower()}  
**Recommendation**: Try again in a few minutes or check OpenAI API key configuration  

"""
        
        emerging_report += "---\n\n"
        
        print(f"✅ {research_type} emerging opportunities complete: {ai_success_count}/{total_attempts} AI success")
        return emerging_report
        
    except Exception as e:
        print(f"❌ ERROR generating {research_type} emerging opportunities: {str(e)}")
        return f"""## {header}

**Status**: ⚠️ AI agent discovery temporarily unavailable for {research_type.lower()}  

---

"""

def generate_emerging_stocks_content(ai_stocks):
    """Generate table content for emerging stocks - no truncation, full text display"""
    content = """**Low-Cost High-Upside Stocks** *(LLM-discovered penny-to-dollars opportunities)*:

| Symbol | Company Name | Price | Target | Upside | Year | Key Catalyst |
|--------|--------------|-------|--------|--------|------|--------------|
"""
    
    # Simple fallback data
    fallback_stocks = [
        {"symbol": "SAVA", "company_name": "Cassava Sciences Inc", "current_price": 2.15, "target_price": 8.50, "founded_year": "2010", "catalyst": "Alzheimer's drug trial progress showing promise"},
        {"symbol": "ATER", "company_name": "Aterian Inc", "current_price": 1.85, "target_price": 7.40, "founded_year": "2014", "catalyst": "AI-powered e-commerce platform expansion"},
        {"symbol": "BBIG", "company_name": "Vinco Ventures Inc", "current_price": 0.95, "target_price": 4.75, "founded_year": "2018", "catalyst": "NFT marketplace and streaming platform launch"}
    ]
    
    # Use AI stocks if valid, otherwise fallback
    stocks_to_use = ai_stocks if ai_stocks and len(ai_stocks) > 2 else fallback_stocks
    
    # Generate table rows - NO TRUNCATION
    for stock in stocks_to_use[:6]:
        symbol = stock.get('symbol', 'N/A')
        company = stock.get('company_name', 'Unknown Company')
        current_price = stock.get('current_price', 0)
        target_price = stock.get('target_price', 0)
        catalyst = stock.get('catalyst', 'Market opportunity')
        founded = stock.get('founded_year', 'N/A')
        
        # Safety checks only
        if len(company) <= 3 or company == symbol:
            company = f"{symbol} Inc."
        if len(catalyst) <= 10:
            catalyst = "Growth opportunity with strong fundamentals"
        
        # Calculate upside
        upside_display = "N/A"
        if current_price > 0 and target_price > current_price:
            multiple = target_price / current_price
            upside_display = f"{multiple:.1f}x"
        
        price_display = f"${current_price:.2f}" if current_price > 0 else "<$5"
        target_display = f"${target_price:.2f}" if target_price > 0 else "TBD"
        
        # NO TRUNCATION - show full company name and catalyst
        content += f"| {symbol} | {company} | {price_display} | {target_display} | {upside_display} | {founded} | {catalyst} |\n"
    
    content += "\n"
    return content

def generate_emerging_cryptos_content(ai_cryptos):
    """Generate table content for emerging cryptos - no truncation"""
    content = """**💰 Low-Cost High-Upside Cryptos** *(LLM-discovered moon-shot opportunities)*:

| Symbol | Cryptocurrency Name | Price | Target | Upside | Key Development |
|--------|---------------------|-------|--------|---------|----------------|
"""
    for crypto in ai_cryptos[:6]:
        symbol = crypto.get('symbol', 'N/A')
        name = crypto.get('name', 'Unknown Crypto')
        current_price = crypto.get('current_price', 0)
        target_price = crypto.get('target_price', 0)
        catalyst = crypto.get('catalyst', 'Market development')
        
        # Calculate upside
        upside_display = "N/A"
        if current_price > 0 and target_price > current_price:
            multiple = target_price / current_price
            upside_display = f"{multiple:.0f}x"
        
        price_display = f"${current_price:.4f}" if current_price > 0 else "<$1"
        target_display = f"${target_price:.2f}" if target_price > 0 else "TBD"
        
        # NO TRUNCATION - show full names and catalysts
        content += f"| {symbol} | {name} | {price_display} | {target_display} | {upside_display} | {catalyst} |\n"
    
    content += "\n"
    return content

def generate_emerging_etfs_content(ai_etfs):
    """Generate table content for emerging ETFs - no truncation"""
    content = """**📊 Innovative ETF Strategies** *(LLM-discovered cutting-edge funds)*:

| Symbol | ETF Name | Year | Investment Strategy | Key Catalyst | Growth Potential |
|--------|----------|------|-------------------|--------------|------------------|
"""
    for etf in ai_etfs[:6]:
        symbol = etf.get('symbol', 'N/A')
        name = etf.get('name', 'Unknown ETF')
        strategy = etf.get('strategy', 'Investment strategy')
        catalyst = etf.get('catalyst', 'Market development')
        launch_year = etf.get('launch_year', 'N/A')
        upside = etf.get('upside_potential', 'Outperformance potential')
        
        # NO TRUNCATION - show full content
        content += f"| {symbol} | {name} | {launch_year} | {strategy} | {catalyst} | {upside} |\n"
    
    content += "\n"
    return content

def generate_asset_specific_strategy(research_type):
    """Generate asset-type specific investment strategy"""
    
    if research_type == "Trending Stocks":
        return """### Low-Cost High-Upside Stock Strategy

**LLM Agent Penny-to-Dollars Approach**:
- **Target Profile**: Stocks under $5 with 3x-10x upside potential
- **Catalyst Focus**: FDA approvals, government contracts, partnerships, breakthrough tech
- **Position Sizing**: Limit each position to 1-2% of portfolio (high risk)
- **Diversification**: Spread across 10-15 low-cost opportunities
- **Time Horizon**: 6 months to 3 years for catalysts to materialize
- **Risk Management**: Expect 70% losers, 30% winners - but winners can be 5x-50x

**Key Stock Catalysts to Monitor**:
- **FDA Approvals**: Biotech breakthrough announcements and trial results
- **Government Contracts**: Defense, infrastructure, green energy funding
- **Partnership Deals**: Major companies acquiring or partnering with small caps
- **Patent Approvals**: Breakthrough technology intellectual property
- **Earnings Surprises**: Small companies beating expectations dramatically

**Low-Cost Stock Investment Rules**:
- Never invest more than you can afford to lose completely
- Set price alerts for major catalyst announcements
- Take partial profits at 3x, 5x, 10x levels to lock in gains
- Use stop-losses at -50% to prevent total wipeouts
- Research insider trading activity and institutional buying

"""
    
    elif research_type == "Trending Cryptos":
        return """### Low-Cost High-Upside Crypto Strategy

**LLM Agent Moon-Shot Approach**:
- **Target Profile**: Cryptos under $1 with moon potential
- **Catalyst Focus**: Protocol upgrades, exchange listings, partnerships, adoption
- **Position Sizing**: Limit each position to 0.5-1% of portfolio (ultra high risk)
- **Diversification**: Spread across 15-20 low-cost crypto opportunities
- **Time Horizon**: 3 months to 2 years for explosive moves
- **Risk Management**: Expect 80% losers, 20% winners - but winners can be 10x-1000x

**Key Crypto Catalysts to Monitor**:
- **Exchange Listings**: Major exchange additions (Coinbase, Binance)
- **Protocol Upgrades**: Network improvements and new features
- **Partnership Announcements**: Real-world adoption and corporate partnerships
- **Regulatory Clarity**: Positive regulatory developments
- **Community Growth**: Active development and growing user base

**Low-Cost Crypto Investment Rules**:
- Only invest what you can afford to lose completely
- Set price alerts for major announcements and listings
- Take profits at 5x, 10x, 50x levels - cryptos can move fast
- Use stop-losses at -70% to preserve some capital
- Monitor social sentiment and developer activity

"""
    
    elif research_type == "Trending ETFs":
        return """### Innovative ETF Strategy

**LLM Agent Cutting-Edge Fund Approach**:
- **Target Profile**: Innovative ETFs with unique strategies
- **Catalyst Focus**: New launches, growing assets, thematic trends
- **Position Sizing**: 3-5% of portfolio per innovative ETF
- **Diversification**: Mix of thematic, sector, and strategy ETFs
- **Time Horizon**: 1-5 years for theme development
- **Risk Management**: Lower risk through diversification but watch expense ratios

**Key ETF Catalysts to Monitor**:
- **Asset Growth**: Funds gaining institutional and retail adoption
- **Theme Development**: Underlying themes gaining mainstream acceptance
- **Performance Track Record**: Consistent outperformance vs benchmarks
- **Management Changes**: New portfolio managers or strategy updates
- **Market Trends**: Secular shifts supporting ETF themes

**Innovative ETF Investment Rules**:
- Research underlying holdings and strategy before investing
- Monitor expense ratios - avoid funds with excessive fees
- Track assets under management - rapid growth can be positive signal
- Rebalance quarterly to maintain target allocations
- Consider tax implications of ETF distributions

"""
    
    else:
        return """### General Emerging Investment Strategy

**LLM Agent Discovery Approach**:
- Focus on AI-identified opportunities with high upside potential
- Diversify across asset types based on analysis results
- Monitor catalysts and market developments regularly
- Maintain appropriate risk management based on asset volatility

"""

def generate_sector_specific_report(sector, enriched_stocks, sector_trending, sector_stable):
    """Generate report section for a specific sector with concise highlights - optimized"""
    
    try:
        # Calculate performance percentiles for proper rating distribution
        all_performances = [data.get('1m_performance', 0) for _, data in enriched_stocks]
        
        if len(all_performances) >= 4:
            sorted_perfs = sorted(all_performances, reverse=True)
            n = len(sorted_perfs)
            
            strong_buy_threshold = sorted_perfs[min(n//4, len(sorted_perfs)-1)]  # Top 25%
            moderate_buy_threshold = sorted_perfs[min(n//2, len(sorted_perfs)-1)]  # Top 50%
            hold_threshold = sorted_perfs[min(3*n//4, len(sorted_perfs)-1)]  # Top 75%
        else:
            # Fallback thresholds
            strong_buy_threshold = 5.0
            moderate_buy_threshold = 0.0
            hold_threshold = -5.0
        
        # Calculate performance range
        max_perf = max(all_performances) if all_performances else 0
        min_perf = min(all_performances) if all_performances else 0
        
        report = f"""## {sector} Sector Analysis

**{sector} Sector Highlights**  
**Discovered**: {len(enriched_stocks)} Top Movers | **Performance Range**: {max_perf:+.1f}% to {min_perf:+.1f}% (1M)

"""
        
        # Categorize stocks by rating
        strong_buy_stocks = []
        moderate_buy_stocks = []
        hold_stocks = []
        sell_stocks = []
        
        for symbol, data in enriched_stocks:
            performance = data.get('1m_performance', 0)
            
            if performance >= strong_buy_threshold:
                strong_buy_stocks.append((symbol, data))
            elif performance >= moderate_buy_threshold:
                moderate_buy_stocks.append((symbol, data))
            elif performance >= hold_threshold:
                hold_stocks.append((symbol, data))
            else:
                sell_stocks.append((symbol, data))
        
        # Generate tables for each rating category
        if strong_buy_stocks:
            report += f"""### 🟢 {sector} Strong Buy Recommendations

| Symbol | Company Name | Current Price | 1M Return | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|--------------|---------------|-----------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
            for symbol, data in strong_buy_stocks[:4]:  # Limit to 4
                price = data.get('current_price', 0)
                performance = data.get('1m_performance', 0)
                market_cap = data.get('market_cap', 0)
                company_name = data.get('ai_company_name', symbol)
                founded = data.get('ai_founded_year', 'N/A')
                cap_display = format_market_cap(market_cap)
                
                # Add new financial metrics
                pe_ratio = data.get('pe_ratio', 'N/A')
                peg_ratio = data.get('peg_ratio', 'N/A')
                roe = data.get('roe', 'N/A')
                pb_ratio = data.get('pb_ratio', 'N/A')
                de_ratio = data.get('de_ratio', 'N/A')
                
                report += f"| {symbol} | {company_name} | ${price:.2f} | {performance:+.1f}% | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
                
                sector_trending.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Price": f"${price:.2f}",
                    "Sector": sector,
                    "1M Return": f"{performance:+.1f}%",
                    "Rating": "🟢 Strong Buy",
                    "Market Cap": cap_display,
                    "P/E Ratio": data.get('pe_ratio', 'N/A'),
                    "PEG Ratio": data.get('peg_ratio', 'N/A'),
                    "ROE": data.get('roe', 'N/A'),
                    "P/B Ratio": data.get('pb_ratio', 'N/A'),
                    "D/E Ratio": data.get('de_ratio', 'N/A')
                })
            
            report += "\n"
        
        if moderate_buy_stocks:
            report += f"""### 🟡 {sector} Moderate Buy Recommendations

| Symbol | Company Name | Current Price | 1M Return | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|--------------|---------------|-----------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
            for symbol, data in moderate_buy_stocks[:3]:  # Limit to 3
                price = data.get('current_price', 0)
                performance = data.get('1m_performance', 0)
                market_cap = data.get('market_cap', 0)
                company_name = data.get('ai_company_name', symbol)
                founded = data.get('ai_founded_year', 'N/A')
                cap_display = format_market_cap(market_cap)
                
                # Add new financial metrics
                pe_ratio = data.get('pe_ratio', 'N/A')
                peg_ratio = data.get('peg_ratio', 'N/A')
                roe = data.get('roe', 'N/A')
                pb_ratio = data.get('pb_ratio', 'N/A')
                de_ratio = data.get('de_ratio', 'N/A')
                
                report += f"| {symbol} | {company_name} | ${price:.2f} | {performance:+.1f}% | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
                
                sector_trending.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Price": f"${price:.2f}",
                    "Sector": sector,
                    "1M Return": f"{performance:+.1f}%",
                    "Rating": "🟡 Moderate Buy",
                    "Market Cap": cap_display,
                    "P/E Ratio": data.get('pe_ratio', 'N/A'),
                    "PEG Ratio": data.get('peg_ratio', 'N/A'),
                    "ROE": data.get('roe', 'N/A'),
                    "P/B Ratio": data.get('pb_ratio', 'N/A'),
                    "D/E Ratio": data.get('de_ratio', 'N/A')
                })
            
            report += "\n"
        
        if hold_stocks:
            report += f"""### 🔵 {sector} Hold Recommendations

| Symbol | Company Name | Current Price | 1M Return | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|--------------|---------------|-----------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
            for symbol, data in hold_stocks[:3]:  # Limit to 3
                price = data.get('current_price', 0)
                performance = data.get('1m_performance', 0)
                market_cap = data.get('market_cap', 0)
                company_name = data.get('ai_company_name', symbol)
                founded = data.get('ai_founded_year', 'N/A')
                cap_display = format_market_cap(market_cap)
                
                # Add new financial metrics
                pe_ratio = data.get('pe_ratio', 'N/A')
                peg_ratio = data.get('peg_ratio', 'N/A')
                roe = data.get('roe', 'N/A')
                pb_ratio = data.get('pb_ratio', 'N/A')
                de_ratio = data.get('de_ratio', 'N/A')
                
                report += f"| {symbol} | {company_name} | ${price:.2f} | {performance:+.1f}% | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
                
                sector_stable.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Price": f"${price:.2f}",
                    "Sector": sector,
                    "1M Return": f"{performance:+.1f}%",
                    "Rating": "🔵 Hold",
                    "P/E Ratio": data.get('pe_ratio', 'N/A'),
                    "PEG Ratio": data.get('peg_ratio', 'N/A'),
                    "ROE": data.get('roe', 'N/A'),
                    "P/B Ratio": data.get('pb_ratio', 'N/A'),
                    "D/E Ratio": data.get('de_ratio', 'N/A')
                })
            
            report += "\n"
        
        if sell_stocks:
            report += f"""### 🔴 {sector} Sell Recommendations

| Symbol | Company Name | Current Price | 1M Return | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|--------------|---------------|-----------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
            for symbol, data in sell_stocks[:2]:  # Limit to 2
                price = data.get('current_price', 0)
                performance = data.get('1m_performance', 0)
                market_cap = data.get('market_cap', 0)
                company_name = data.get('ai_company_name', symbol)
                founded = data.get('ai_founded_year', 'N/A')
                cap_display = format_market_cap(market_cap)
                
                # Add new financial metrics
                pe_ratio = data.get('pe_ratio', 'N/A')
                peg_ratio = data.get('peg_ratio', 'N/A')
                roe = data.get('roe', 'N/A')
                pb_ratio = data.get('pb_ratio', 'N/A')
                de_ratio = data.get('de_ratio', 'N/A')
                
                report += f"| {symbol} | {company_name} | ${price:.2f} | {performance:+.1f}% | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
                
                sector_stable.append({
                    "Symbol": symbol,
                    "Company": company_name,
                    "Price": f"${price:.2f}",
                    "Sector": sector,
                    "1M Return": f"{performance:+.1f}%",
                    "Rating": "🔴 Sell",
                    "P/E Ratio": data.get('pe_ratio', 'N/A'),
                    "PEG Ratio": data.get('peg_ratio', 'N/A'),
                    "ROE": data.get('roe', 'N/A'),
                    "P/B Ratio": data.get('pb_ratio', 'N/A'),
                    "D/E Ratio": data.get('de_ratio', 'N/A')
                })
            
            report += "\n"
        
        # Sector summary with highlights format
        report += f"""### {sector} Sector Summary

**Rating Distribution**: {len(strong_buy_stocks)} Strong Buy, {len(moderate_buy_stocks)} Moderate Buy, {len(hold_stocks)} Hold, {len(sell_stocks)} Sell  
**Top Performer**: {max_perf:+.1f}% | **Worst Performer**: {min_perf:+.1f}% | **AI Discovery**: analysis  

---

"""
        
        return report
        
    except Exception as e:
        print(f"ERROR generating {sector} sector report: {str(e)}")
        return f"""## {sector} Sector Analysis

**Error**: Unable to generate {sector} analysis due to processing issues.

---

"""

def generate_enhanced_trending_report(research_type, enriched_assets, all_trending_assets, all_stable_assets, sector_info=None):
    """Generate trending report with proper rating distribution"""
    
    try:
        print(f"Generating report for {research_type} with {len(enriched_assets)} assets")
        
        # Calculate performance percentiles for proper rating distribution
        all_performances = [data.get('1m_performance', 0) for _, data in enriched_assets]
        
        if len(all_performances) >= 4:
            # Sort performances to calculate percentiles
            sorted_perfs = sorted(all_performances, reverse=True)
            n = len(sorted_perfs)
            
            # Dynamic thresholds based on actual performance distribution
            strong_buy_threshold = sorted_perfs[min(n//4, len(sorted_perfs)-1)]  # Top 25%
            moderate_buy_threshold = sorted_perfs[min(n//2, len(sorted_perfs)-1)]  # Top 50%
            hold_threshold = sorted_perfs[min(3*n//4, len(sorted_perfs)-1)]  # Top 75%
            # Bottom 25% = Sell
        else:
            # Fallback thresholds for small datasets
            if research_type == "Trending Cryptos":
                strong_buy_threshold = 20.0
                moderate_buy_threshold = 5.0
                hold_threshold = -10.0
            elif research_type == "Trending ETFs":
                strong_buy_threshold = 8.0
                moderate_buy_threshold = 2.0
                hold_threshold = -3.0
            else:  # Trending Stocks
                strong_buy_threshold = 5.0
                moderate_buy_threshold = 0.0
                hold_threshold = -5.0
        
        print(f"Rating thresholds: Strong Buy >{strong_buy_threshold:.1f}%, Moderate Buy >{moderate_buy_threshold:.1f}%, Hold >{hold_threshold:.1f}%")
        
        # Determine asset type label
        asset_label = "Stock" if research_type == "Trending Stocks" else "Crypto" if research_type == "Trending Cryptos" else "ETF"
        
        # Add sector breakdown for trending stocks
        sector_breakdown = ""
        if research_type == "Trending Stocks" and sector_info:
            sector_counts = {}
            for symbol, _ in enriched_assets:
                sector = sector_info.get(symbol, 'Unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            sector_breakdown = f"""
**Sector Breakdown**: {', '.join([f"{sector} ({count})" for sector, count in sector_counts.items()])}  """
        
        report = f"""## {research_type} Analysis

**Market Research**: Successfully identified {len(enriched_assets)} {research_type.lower()}{sector_breakdown}
**Data Validation**: Cross-referenced with real-time market data
**Rating Distribution**: Based on performance percentiles for balanced recommendations

### 🟢 Strong Buy Recommendations

| Symbol | Name | Current Price | 1M Return | Sector | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|------|---------------|-----------|--------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
        
        # Categorize assets based on performance thresholds
        strong_buy_assets = []
        moderate_buy_assets = []
        hold_assets = []
        sell_assets = []
        
        for symbol, data in enriched_assets:
            performance = data.get('1m_performance', 0)
            
            if performance >= strong_buy_threshold:
                strong_buy_assets.append((symbol, data))
            elif performance >= moderate_buy_threshold:
                moderate_buy_assets.append((symbol, data))
            elif performance >= hold_threshold:
                hold_assets.append((symbol, data))
            else:
                sell_assets.append((symbol, data))
        
        # Strong Buy section
        for symbol, data in strong_buy_assets[:4]:  # Limit to 4
            price = data.get('current_price', 0)
            performance = data.get('1m_performance', 0)
            market_cap = data.get('market_cap', 0)
            asset_name = data.get('ai_company_name', symbol)
            founded = data.get('ai_founded_year', 'N/A')
            sector = sector_info.get(symbol, 'Unknown') if sector_info else data.get('ai_sector', 'Unknown')
            cap_display = format_market_cap(market_cap)
            
            # Add new financial metrics
            pe_ratio = data.get('pe_ratio', 'N/A')
            peg_ratio = data.get('peg_ratio', 'N/A')
            roe = data.get('roe', 'N/A')
            pb_ratio = data.get('pb_ratio', 'N/A')
            de_ratio = data.get('de_ratio', 'N/A')
            
            price_display = f"${price:.4f}" if research_type == "Trending Cryptos" and price < 1 else f"${price:.2f}"
            
            report += f"| {symbol} | {asset_name} | {price_display} | {performance:+.1f}% | {sector} | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
            
            all_trending_assets.append({
                "Symbol": symbol,
                "Name": asset_name,
                "Price": price_display,
                "Type": asset_label,
                "Sector": sector,
                "1M Return": f"{performance:+.1f}%",
                "Rating": "🟢 Strong Buy",
                "Market Cap": cap_display,
                "P/E Ratio": data.get('pe_ratio', 'N/A'),
                "PEG Ratio": data.get('peg_ratio', 'N/A'),
                "ROE": data.get('roe', 'N/A'),
                "P/B Ratio": data.get('pb_ratio', 'N/A'),
                "D/E Ratio": data.get('de_ratio', 'N/A')
            })
        
        # Moderate Buy section
        report += """

### 🟡 Moderate Buy Recommendations

| Symbol | Name | Current Price | 1M Return | Sector | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|------|---------------|-----------|--------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
        
        for symbol, data in moderate_buy_assets[:3]:  # Limit to 3
            price = data.get('current_price', 0)
            performance = data.get('1m_performance', 0)
            market_cap = data.get('market_cap', 0)
            asset_name = data.get('ai_company_name', symbol)
            founded = data.get('ai_founded_year', 'N/A')
            sector = sector_info.get(symbol, 'Unknown') if sector_info else data.get('ai_sector', 'Unknown')
            cap_display = format_market_cap(market_cap)
            
            # Add new financial metrics
            pe_ratio = data.get('pe_ratio', 'N/A')
            peg_ratio = data.get('peg_ratio', 'N/A')
            roe = data.get('roe', 'N/A')
            pb_ratio = data.get('pb_ratio', 'N/A')
            de_ratio = data.get('de_ratio', 'N/A')
            
            price_display = f"${price:.4f}" if research_type == "Trending Cryptos" and price < 1 else f"${price:.2f}"
            
            report += f"| {symbol} | {asset_name} | {price_display} | {performance:+.1f}% | {sector} | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
            
            all_trending_assets.append({
                "Symbol": symbol,
                "Name": asset_name,
                "Price": price_display,
                "Type": asset_label,
                "Sector": sector,
                "1M Return": f"{performance:+.1f}%",
                "Rating": "🟡 Moderate Buy",
                "Market Cap": cap_display,
                "P/E Ratio": data.get('pe_ratio', 'N/A'),
                "PEG Ratio": data.get('peg_ratio', 'N/A'),
                "ROE": data.get('roe', 'N/A'),
                "P/B Ratio": data.get('pb_ratio', 'N/A'),
                "D/E Ratio": data.get('de_ratio', 'N/A')
            })
        
        # Hold section
        report += """

### 🔵 Hold Recommendations

| Symbol | Name | Current Price | 1M Return | Sector | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|------|---------------|-----------|--------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
        
        for symbol, data in hold_assets[:3]:  # Limit to 3
            price = data.get('current_price', 0)
            performance = data.get('1m_performance', 0)
            market_cap = data.get('market_cap', 0)
            asset_name = data.get('ai_company_name', symbol)
            founded = data.get('ai_founded_year', 'N/A')
            sector = sector_info.get(symbol, 'Unknown') if sector_info else data.get('ai_sector', 'Unknown')
            cap_display = format_market_cap(market_cap)
            
            # Add new financial metrics
            pe_ratio = data.get('pe_ratio', 'N/A')
            peg_ratio = data.get('peg_ratio', 'N/A')
            roe = data.get('roe', 'N/A')
            pb_ratio = data.get('pb_ratio', 'N/A')
            de_ratio = data.get('de_ratio', 'N/A')
            
            price_display = f"${price:.4f}" if research_type == "Trending Cryptos" and price < 1 else f"${price:.2f}"
            
            report += f"| {symbol} | {asset_name} | {price_display} | {performance:+.1f}% | {sector} | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
            
            all_stable_assets.append({
                "Symbol": symbol,
                "Name": asset_name,
                "Price": price_display,
                "Type": asset_label,
                "Sector": sector,
                "1M Return": f"{performance:+.1f}%",
                "Rating": "🔵 Hold",
                "P/E Ratio": data.get('pe_ratio', 'N/A'),
                "PEG Ratio": data.get('peg_ratio', 'N/A'),
                "ROE": data.get('roe', 'N/A'),
                "P/B Ratio": data.get('pb_ratio', 'N/A'),
                "D/E Ratio": data.get('de_ratio', 'N/A')
            })
        
        # Sell section (NEW!)
        if sell_assets:
            report += """

### 🔴 Sell Recommendations

| Symbol | Name | Current Price | 1M Return | Sector | Market Cap | Founded | P/E Ratio | PEG Ratio | ROE | P/B Ratio | D/E Ratio |
|--------|------|---------------|-----------|--------|------------|---------|-----------|-----------|-----|-----------|-----------|
"""
            
            for symbol, data in sell_assets[:2]:  # Limit to 2
                price = data.get('current_price', 0)
                performance = data.get('1m_performance', 0)
                market_cap = data.get('market_cap', 0)
                asset_name = data.get('ai_company_name', symbol)
                founded = data.get('ai_founded_year', 'N/A')
                sector = sector_info.get(symbol, 'Unknown') if sector_info else data.get('ai_sector', 'Unknown')
                cap_display = format_market_cap(market_cap)
                
                # Add new financial metrics
                pe_ratio = data.get('pe_ratio', 'N/A')
                peg_ratio = data.get('peg_ratio', 'N/A')
                roe = data.get('roe', 'N/A')
                pb_ratio = data.get('pb_ratio', 'N/A')
                de_ratio = data.get('de_ratio', 'N/A')
                
                price_display = f"${price:.4f}" if research_type == "Trending Cryptos" and price < 1 else f"${price:.2f}"
                
                report += f"| {symbol} | {asset_name} | {price_display} | {performance:+.1f}% | {sector} | {cap_display} | {founded} | {pe_ratio} | {peg_ratio} | {roe} | {pb_ratio} | {de_ratio} |\n"
                
                all_stable_assets.append({
                    "Symbol": symbol,
                    "Name": asset_name,
                    "Price": price_display,
                    "Type": asset_label,
                    "Sector": sector,
                    "1M Return": f"{performance:+.1f}%",
                    "Rating": "🔴 Sell",
                    "P/E Ratio": data.get('pe_ratio', 'N/A'),
                    "PEG Ratio": data.get('peg_ratio', 'N/A'),
                    "ROE": data.get('roe', 'N/A'),
                    "P/B Ratio": data.get('pb_ratio', 'N/A'),
                    "D/E Ratio": data.get('de_ratio', 'N/A')
                })
        
        report += f"""

### {research_type} Insights

**Rating Distribution**: {len(strong_buy_assets)} Strong Buy, {len(moderate_buy_assets)} Moderate Buy, {len(hold_assets)} Hold, {len(sell_assets)} Sell  
**Performance Range**: {max(all_performances):+.1f}% to {min(all_performances):+.1f}% (1-month)  
**Investment Thesis**: AI analysis with dynamic {research_type.lower()} discovery  

---

"""
        
        print(f"Successfully generated {research_type} report ({len(report)} characters)")
        print(f"Rating distribution: {len(strong_buy_assets)} Strong Buy, {len(moderate_buy_assets)} Moderate Buy, {len(hold_assets)} Hold, {len(sell_assets)} Sell")
        return report
        
    except Exception as e:
        print(f"ERROR generating {research_type} report: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return minimal report on error
        return f"""## {research_type} Analysis

**Error**: Unable to generate complete {research_type} analysis due to data processing issues.

---

"""

def generate_trending_insights(research_type, enriched_assets):
    """Generate specific insights for trending assets"""
    
    if research_type == "Trending Stocks":
        return """## Trending Stocks Market Insights

**Key Market Drivers**: Earnings momentum, sector rotation, and technical breakouts driving stock performance  
**Investment Thesis**: Focus on companies with strong fundamentals and recent catalysts  
**Risk Factors**: Market volatility, interest rate changes, sector-specific headwinds  
**Opportunities**: Individual stock selection in trending sectors with growth potential

"""
    elif research_type == "Trending Cryptos":
        return """## Trending Cryptocurrencies Market Insights

**Key Market Drivers**: Institutional adoption, regulatory developments, and technological innovations  
**Investment Thesis**: Focus on established protocols and emerging use cases  
**Risk Factors**: High volatility, regulatory uncertainty, technological risks  
**Opportunities**: Early-stage projects with strong fundamentals and growing ecosystems

"""
    elif research_type == "Trending ETFs":
        return """## Trending ETFs Market Insights

**Key Market Drivers**: Thematic investing trends, sector rotations, and passive investment flows  
**Investment Thesis**: Diversified exposure to trending themes and sectors  
**Risk Factors**: Market concentration, tracking error, expense ratio impact  
**Opportunities**: Broad market exposure with reduced individual security risk

"""
    else:
        return ""

def generate_trending_investment_strategy(research_type):
    """Generate investment strategy specific to trending assets"""
    
    if research_type == "Trending Stocks":
        strategy_focus = "individual stock selection and sector rotation"
        allocation_rec = "60% trending stocks, 30% stable dividend stocks, 10% cash"
        risk_note = "Monitor earnings announcements and sector news"
    elif research_type == "Trending Cryptos":
        strategy_focus = "cryptocurrency diversification and risk management"
        allocation_rec = "50% large-cap cryptos, 30% mid-cap opportunities, 20% speculative plays"
        risk_note = "High volatility - limit to 5-10% of total portfolio"
    elif research_type == "Trending ETFs":
        strategy_focus = "diversified thematic exposure"
        allocation_rec = "70% broad market ETFs, 20% sector ETFs, 10% international ETFs"
        risk_note = "Lower risk through diversification but watch expense ratios"
    else:
        strategy_focus = "balanced asset allocation"
        allocation_rec = "Standard diversified approach"
        risk_note = "Standard market risks apply"
    
    return f"""## {research_type} Investment Strategy

### Investment Approach for {research_type}

| Strategy Element | Recommendation | Portfolio Allocation | Key Considerations |
|------------------|----------------|---------------------|-------------------|
| **Focus Strategy** | {strategy_focus} | {allocation_rec} | {risk_note} |
| **Entry Strategy** | Dollar-cost averaging for volatile assets | Gradual position building | Reduce timing risk |
| **Exit Strategy** | Profit-taking at resistance levels | 25% at +50%, 25% at +100% | Lock in gains systematically |

### Investment Guidelines for {research_type}

1. **Dynamic Discovery**: All recommendations identified through AI market intelligence
2. **Performance-Based Ratings**: Rankings based on actual 1-month performance percentiles  
3. **Risk-Adjusted Sizing**: Position sizes based on volatility and rating strength
4. **Regular Monitoring**: Weekly review for trending assets due to higher volatility
5. **Diversification**: Spread across multiple assets to reduce concentration risk

### Risk Management for {research_type}

- **Position Limits**: Maximum 3-5% per individual asset for trending investments
- **Stop Losses**: Consider 15-20% stop losses for volatile trending assets
- **Portfolio Balance**: Limit trending assets to appropriate portion of total portfolio
- **Regular Rebalancing**: Monthly review and rebalancing due to momentum changes

---

*This {research_type.lower()} analysis generated through improved AI agent collaboration with asset-type specific emerging opportunities.*
"""

def generate_empty_report(research_type, current_date):
    """Generate empty report when discovery fails"""
    return (
        f"""# {research_type} Analysis Report – {current_date}

## Analysis Status: Discovery Issues

The AI agent was unable to discover {research_type.lower()} at this time.

### Possible Issues:
1. **OpenAI API Key**: Verify your OPENAI_API_KEY is correctly set
2. **API Limits**: You may have hit rate limits - try again in a few minutes
3. **Network Issues**: Check your internet connection
4. **Market Data**: Some asset types may have limited data availability

### Features:
- Dynamic {research_type.lower()} discovery with varied results
- Proper rating distribution including "Sell" ratings
- Asset-type specific emerging opportunities
- Market condition-based recommendations

---
*AI Agent requires valid OpenAI API key for dynamic discovery*
""",
        pd.DataFrame(),
        pd.DataFrame(),
        f"No {research_type.lower()} discovered"
    )

# ================================
# HELPER FUNCTIONS
# ================================

def format_market_cap(market_cap):
    """Format market cap for display"""
    if market_cap > 1e12:
        return f"${market_cap/1e12:.1f}T"
    elif market_cap > 1e9:
        return f"${market_cap/1e9:.1f}B"
    else:
        return f"${market_cap/1e6:.0f}M"

def save_report_to_db(report, research_type, sectors):
    """Save report to database"""
    try:
        conn = sqlite3.connect('investment_advisor.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO stock_research (report_date, report_content, report_type) VALUES (?, ?, ?)",
            (datetime.now().date(), report, f"AI_Agent_{research_type}_{'+'.join(sectors)}")
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

# ================================
# INDIVIDUAL STOCK RESEARCH
# ================================

def create_individual_stock_research(symbol):
    """Individual stock research"""
    if not symbol:
        return "Please enter a stock symbol"
    
    symbol = symbol.upper()
    
    if not os.getenv('OPENAI_API_KEY'):
        return "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
    
    try:
        print(f"AI Agent researching {symbol}...")
        
        # Initialize AI Agent
        agent = StockResearchAgent()
        
        # Step 1: AI analyzes company
        company_details = agent.analyze_company(symbol)
        
        # Step 2: AI analyzes sentiment
        sentiment_analysis = agent.analyze_sentiment(symbol)
        
        # Step 3: Get real market data
        market_data = get_market_data([symbol])
        stock_data = market_data.get(symbol, {})
        
        # Step 4: Generate report
        return generate_individual_report(symbol, company_details, sentiment_analysis, stock_data)
        
    except Exception as e:
        return f"AI Agent research error: {str(e)}"

def generate_individual_report(symbol, company_details, sentiment_analysis, stock_data):
    """Generate individual stock research report"""
    
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Extract data
    company_name = company_details.get('company_name', symbol)
    founded_year = company_details.get('founded_year', 'N/A')
    business_desc = company_details.get('business_description', 'Company analysis in progress')
    sector = company_details.get('sector', stock_data.get('sector', 'Unknown'))
    recent_developments = company_details.get('recent_developments', 'Analysis in progress')
    
    price = stock_data.get('current_price', 0)
    performance_1y = stock_data.get('1y_performance', 0)
    performance_1m = stock_data.get('1m_performance', 0)
    market_cap = stock_data.get('market_cap', 0)
    cap_display = format_market_cap(market_cap)
    
    # Sentiment data
    sentiment = sentiment_analysis.get('overall_sentiment', 'NEUTRAL')
    sentiment_score = sentiment_analysis.get('sentiment_score', 0)
    ai_recent_developments = sentiment_analysis.get('recent_developments', [])
    risk_factors = sentiment_analysis.get('risk_factors', [])
    growth_catalysts = sentiment_analysis.get('growth_catalysts', [])
    analyst_outlook = sentiment_analysis.get('analyst_outlook', 'Analysis in progress')
    
    # rating logic
    if sentiment == "BULLISH" and performance_1m > 5:
        rating = "🟢 Strong Buy"
    elif sentiment == "BULLISH" or performance_1m > 0:
        rating = "🟡 Moderate Buy"
    elif sentiment == "NEUTRAL" and performance_1m > -5:
        rating = "🔵 Hold"
    else:
        rating = "🔴 Sell"
    
    # Generate report
    report = f"""# {symbol} | Deep Stock Research

**Research Date**: {current_date} • **Ticker**: ${symbol} • **Sector**: {sector}  
**Market Cap**: {cap_display} • **Current Price**: ${price:.2f} • **1Y Performance**: {performance_1y:+.1f}%  
**AI Sentiment Score**: {sentiment_score:.1f} / 5 ({sentiment}) • **Rating**: {rating}

---

## Company Snapshot

**{company_name}**  
Founded in {founded_year}, {business_desc}

**Recent Company Developments**: {recent_developments}

---

## AI Insights

### Recent Market Developments

"""
    
    if ai_recent_developments:
        for development in ai_recent_developments[:3]:
            report += f"* {development}\n"
    else:
        report += "* AI analysis indicates stable recent activity\n"
    
    report += """

### Growth Catalysts

"""
    
    if growth_catalysts:
        for catalyst in growth_catalysts[:3]:
            report += f"* {catalyst}\n"
    else:
        report += "* Market analysis suggests steady fundamentals\n"
    
    report += """

### Risk Assessment

"""
    
    if risk_factors:
        for risk in risk_factors[:3]:
            report += f"* {risk}\n"
    else:
        report += "* Standard market volatility applies\n"
    
    # investment analysis
    confidence = "High" if abs(sentiment_score) > 3 else "Medium" if abs(sentiment_score) > 1.5 else "Low"
    short_term = "Bullish" if performance_1m > 5 else "Neutral" if performance_1m > -2 else "Bearish"
    long_term = "Positive" if sentiment_score > 2 else "Neutral" if sentiment_score > -1 else "Negative"
    
    report += f"""

---

## AI Investment Analysis

| **Metric** | **Assessment** |
|------------|----------------|
| **Overall Rating** | {rating} |
| **AI Sentiment** | {sentiment.title()} ({sentiment_score:.1f}/5) |
| **Short-Term Outlook** | {short_term} (1M: {performance_1m:+.1f}%) |
| **Long-Term Outlook** | {long_term} |
| **AI Confidence Level** | {confidence} |
| **Performance vs Market** | {"Outperforming" if performance_1m > 2 else "Underperforming" if performance_1m < -2 else "In-line"} |

---

## Investment Recommendations

### For {rating} Rating:
"""

    if "Strong Buy" in rating:
        report += """
* **Action**: Consider accumulating position on any weakness
* **Position Size**: Up to 5% of portfolio for aggressive investors
* **Timeline**: 6-12 month holding period recommended
* **Risk Level**: Monitor for any negative sentiment shifts"""
    elif "Moderate Buy" in rating:
        report += """
* **Action**: Consider building position gradually
* **Position Size**: 2-3% of portfolio allocation
* **Timeline**: 3-9 month holding period
* **Risk Level**: Watch for earnings and sector trends"""
    elif "Hold" in rating:
        report += """
* **Action**: Maintain current position if owned
* **Position Size**: No new purchases recommended
* **Timeline**: Review quarterly for changes
* **Risk Level**: Monitor for improvement or deterioration"""
    else:  # Sell
        report += """
* **Action**: Consider reducing or exiting position
* **Position Size**: Avoid new purchases
* **Timeline**: Exit within 1-3 months if owned
* **Risk Level**: High - negative momentum and sentiment"""
    
    report += f"""

---

## AI Verdict

**Current Analysis**: "{analyst_outlook}"

**AI Recommendation**: **{sentiment.title()}** positioning recommended based on comprehensive analysis combining market data, sentiment assessment, and performance metrics.

---

## Data Sources & Methodology

* **GPT-4o-mini** powered analysis with dynamic prompting
* **Yahoo Finance** real-time market data integration
* **Multi-factor sentiment** analysis and technical indicators
* **Performance-based rating** system with percentile rankings

*research generated through improved AI-powered market intelligence with proper rating distribution.*
"""
    
    return report

# ================================
# PORTFOLIO ANALYSIS FUNCTIONS
# ================================

def analyze_portfolio(budget, time_horizon, risk_tolerance, goals):
    """Analyze investment portfolio based on user preferences"""
    allocations = {
        "Conservative": {
            "Short-term": {"bonds": 50, "dividend_stocks": 30, "reits": 15, "cash": 5},
            "Medium-term": {"bonds": 40, "dividend_stocks": 35, "reits": 20, "cash": 5},
            "Long-term": {"bonds": 30, "dividend_stocks": 40, "reits": 25, "cash": 5}
        },
        "Moderate": {
            "Short-term": {"bonds": 35, "stocks": 40, "reits": 15, "cash": 10},
            "Medium-term": {"stocks": 50, "bonds": 25, "reits": 20, "cash": 5},
            "Long-term": {"stocks": 60, "bonds": 20, "reits": 15, "cash": 5}
        },
        "Aggressive": {
            "Short-term": {"stocks": 55, "growth_stocks": 25, "reits": 15, "cash": 5},
            "Medium-term": {"stocks": 65, "growth_stocks": 20, "reits": 10, "cash": 5},
            "Long-term": {"stocks": 70, "growth_stocks": 15, "reits": 10, "cash": 5}
        }
    }
    
    allocation = allocations.get(risk_tolerance, {}).get(time_horizon, {})
    expected_returns = {
        "stocks": 8.5, "bonds": 3.5, "dividend_stocks": 6.5, 
        "reits": 7.0, "growth_stocks": 12.0, "cash": 1.0
    }
    
    portfolio_return = sum((allocation.get(asset, 0) / 100) * expected_returns.get(asset, 0) 
                          for asset in expected_returns)
    
    return {
        "budget": budget,
        "allocation": allocation,
        "expected_annual_return": round(portfolio_return, 2),
        "projected_value_5y": round(budget * (1 + portfolio_return/100)**5, 2)
    }

def create_portfolio_strategy(budget, time_horizon, risk_tolerance, goals):
    """Create portfolio strategy"""
    analysis = analyze_portfolio(budget, time_horizon, risk_tolerance, goals)
    allocation = analysis["allocation"]
    
    strategy_text = f"""# Investment Portfolio Strategy Report

## Executive Summary
**Investment Profile:** {risk_tolerance} risk tolerance with {time_horizon.lower()} investment horizon
**Total Budget:** ${budget:,.2f}
**Expected Annual Return:** {analysis['expected_annual_return']}%
**Projected 5-Year Value:** ${analysis['projected_value_5y']:,.2f}

## Recommended Asset Allocation

"""
    
    for asset, percentage in allocation.items():
        amount = (percentage / 100) * budget
        strategy_text += f"- **{asset.replace('_', ' ').title()}:** {percentage}% (${amount:,.0f})\n"
    
    strategy_text += f"""

## Investment Strategy Details

### Risk Management
- **Diversification:** Portfolio spread across {len(allocation)} asset classes
- **Risk Level:** {risk_tolerance} approach balances growth and stability
- **Rebalancing:** Quarterly review recommended with AI insights

### Expected Outcomes
- **5-Year Projection:** ${analysis['projected_value_5y']:,.2f}
- **Annual Income Target:** ${(budget * analysis['expected_annual_return'] / 100):,.0f}

### Goals Alignment
**Your stated goals:** {goals}

This portfolio strategy incorporates dynamic market analysis and is designed to align with your investment objectives while maintaining appropriate risk levels.

---
*Strategy generated for educational purposes. Consult with a financial advisor before making investment decisions.*
"""
    
    # Create summary dataframe
    summary_data = []
    risk_levels = {
        "bonds": "Low", "dividend_stocks": "Low-Med", "stocks": "Medium", 
        "reits": "Medium", "growth_stocks": "High", "cash": "Low"
    }
    
    for asset, percentage in allocation.items():
        amount = (percentage / 100) * budget
        risk_level = risk_levels.get(asset, "Medium")
        summary_data.append({
            "Asset Class": asset.replace("_", " ").title(),
            "Allocation": f"{percentage}%",
            "Amount": f"${amount:,.0f}",
            "Risk Level": risk_level
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
    except Exception as e:
        print(f"Database error: {e}")
    
    return strategy_text, summary_df, f"Portfolio strategy generated! Expected return: {analysis['expected_annual_return']}%"

# ================================
# DATABASE & HISTORY FUNCTIONS
# ================================

def get_research_history():
    """Get research history from database"""
    try:
        conn = sqlite3.connect('investment_advisor.db')
        
        # Get stock research history
        stock_query = """
        SELECT id, report_date, report_type, 
               SUBSTR(report_content, 1, 100) || '...' as preview
        FROM stock_research 
        ORDER BY report_date DESC 
        LIMIT 20
        """
        stock_df = pd.read_sql_query(stock_query, conn)
        
        # Get portfolio strategy history  
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
        
        # Send as HTML instead of plain text
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
    

# ================================
# UI HELPER FUNCTIONS
# ================================

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

def get_stock_quote(symbol):
    """Get stock quote for individual lookup"""
    if not symbol:
        return "<div class='warning-msg'>Please enter a stock symbol</div>"
    
    try:
        market_data = get_market_data([symbol.upper()])
        #print(market_data)
        if symbol.upper() in market_data:
            data = market_data[symbol.upper()]
            #print(data)
            performance_1m_color = "#22c55e" if data['1m_performance'] > 0 else "#ef4444"
            #performance_1y_color = "#22c55e" if data['1y_performance'] > 0 else "#ef4444"
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
        else:
            return "<div class='warning-msg'>Symbol not found or no data available</div>"
    except Exception as e:
        return f"<div class='warning-msg'>Error fetching data: {str(e)}</div>"

# ================================
# ENHANCED GRADIO APPLICATION
# ================================

def create_enhanced_gradio_app():
    """Create the Gradio application"""
    
    init_database()
    
    with gr.Blocks(title="AI Investment Advisor Pro", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:
        
        # Header
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
        
        # Main tabs
        with gr.Tabs():
            
            # Portfolio Strategy Tab
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
                        # gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Your Portfolio Strategy</h3>")
                        
                        # portfolio_status = gr.HTML(
                        #     value="<div style='text-align: center; color: #9ca3af;'>Fill out your profile and click 'Generate Portfolio Strategy' to begin</div>"
                        # )
                        
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
            
            # Trending Assets Research Tab
            with gr.TabItem("Trending Assets Report"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="section-card"):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Trending Assets</h3>")
                        
                        # API key status
                        #api_key_status = "OpenAI API Key detected - AI agents enabled" if os.getenv('OPENAI_API_KEY') else "No OPENAI_API_KEY in .env file - features disabled"
                        #gr.HTML(f"<p style='color: #9ca3af; text-align: center;'>{api_key_status}</p>")
                        
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
                        
                        # gr.HTML("""
                        # <div style='margin-top: 1rem; padding: 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px;'>
                        #     <h4 style='color: #22c55e; margin: 0;'>Asset-Type Specific Features:</h4>
                        #     <ul style='color: #9ca3af; margin: 0.5rem 0;'>
                        #         <li>🔥 Trending Stocks - Shows emerging STOCKS only</li>
                        #         <li>₿ Trending Cryptos - Shows emerging CRYPTOS only</li>
                        #         <li>📊 Trending ETFs - Shows emerging ETFS only</li>
                        #         <li>⚖️ Proper rating distribution (Buy/Hold/Sell)</li>
                        #         <li>🤖 Strictly AI-generated opportunities</li>
                        #         <li>📈 Sector-separated analysis tables</li>
                        #     </ul>
                        # </div>
                        # """)
                    
                    with gr.Column(scale=2):
                        # gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Market Research Report</h3>")
                        
                        # research_status = gr.HTML(
                        #     value="<div style='text-align: center; color: #9ca3af;'>Select your research preferences and click 'Generate Research Report' to begin</div>"
                        # )

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
            
            # Stock Lookup Tab
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
                
                # quote output
                quote_output = gr.HTML()
                
                # news research output
                news_output = gr.HTML(
                    value=create_placeholder_html("Deep Research", "Enter a stock symbol and click 'Deep Research' for comprehensive analysis")
                )
            
            # Research History Tab
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
        
        # Footer
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
        
        # ================================
        # ENHANCED EVENT HANDLERS
        # ================================
        
        def portfolio_wrapper(budget_val, time_val, risk_val, goals_val):
            """Handle portfolio strategy generation"""
            if budget_val < 1000:
                return "<div class='warning-msg'>Minimum investment is $1,000</div>", create_placeholder_html("Error", "Minimum investment required"), pd.DataFrame()
            
            try:
                strategy, summary_df, status = create_portfolio_strategy(budget_val, time_val, risk_val, goals_val)
                status_html = f"<div class='success-msg'>Success: {status}</div>"
                strategy_html = convert_markdown_to_html(strategy)
                return status_html, strategy_html, summary_df
            except Exception as e:
                error_html = f"<div class='warning-msg'>Error: {str(e)}</div>"
                return error_html, create_placeholder_html("Error", str(e)), pd.DataFrame()
        
        def research_wrapper(research_type_val, market_focus_val):
            """Handle trending assets research generation"""
            try:
                report, trending_df, stable_df, status = create_enhanced_trending_research_report(research_type_val, market_focus_val)
                status_html = f"<div class='success-msg'>Success: {status}</div>"
                report_html = convert_markdown_to_html(report)
                return status_html, report_html, trending_df, stable_df
            except Exception as e:
                error_html = f"<div class='warning-msg'>Error: {str(e)}</div>"
                placeholder = create_placeholder_html("Error", str(e))
                return error_html, placeholder, pd.DataFrame(), pd.DataFrame()
        
        def lookup_wrapper(symbol_val):
            """Handle individual stock quote lookup"""
            return get_stock_quote(symbol_val)
        
        def news_wrapper(symbol_val):
            """Handle individual stock AI research"""
            if not symbol_val:
                return "<div class='warning-msg'>Please enter a stock symbol</div>"
            
            try:
                news_report = create_individual_stock_research(symbol_val.upper())
                news_html = convert_markdown_to_html(news_report)
                return news_html
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
            # import re
            # text_content = re.sub('<[^<]+?>', '', str(portfolio_content))
            return send_email_report(email_val, portfolio_content, "portfolio")
        
        def email_research_wrapper(email_val, research_report):
            """Handle research email sending"""
            if not email_val:
                return "Please provide an email address"
            # import re
            # text_content = re.sub('<[^<]+?>', '', str(research_report))
            return send_email_report(email_val, research_report, "enhanced_stock_analysis")
        
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
        
        # email functionality
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
        
        # history tab event handlers
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
            pass  # Ignore errors on startup
    
    return app

# ================================
# ENHANCED MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print("Starting AI Investment Advisor Pro...")
    
    # Check dependencies
    try:
        import yfinance
        import pandas
        import markdown
        print("All required packages installed")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install: pip install gradio yfinance pandas markdown python-dotenv openai")
        exit(1)
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("OpenAI API key detected - AI agents enabled")
    else:
        print("No OPENAI_API_KEY found - Add to .env file to enable AI agents")

    
    app = create_enhanced_gradio_app()
    app.launch(
        #server_name="0.0.0.0",
        #server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )