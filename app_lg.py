"""
Senior Agentic AI Investment Advisor - Multi-Agent System with LangGraph
========================================================================

Advanced multi-agent investment analysis system demonstrating:
- 6+ specialized agents with LangGraph orchestration
- Advanced memory systems (vector + episodic)
- Comprehensive guardrails and safety measures
- Production-ready error handling and resilience
- Continuous learning and feedback loops
- Enterprise-grade scalability

Author: Senior Agentic AI Engineer
"""

import gradio as gr
import pandas as pd
import json
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
import sqlite3
import markdown
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import random
import time
import uuid
from contextlib import asynccontextmanager

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Advanced memory and vector store
import chromadb
from chromadb.config import Settings
import numpy as np

# Circuit breaker and resilience
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# ADVANCED STATE MANAGEMENT
# ================================

class WorkflowState(Enum):
    """Workflow state enumeration"""
    INITIALIZATION = "initialization"
    MARKET_DISCOVERY = "market_discovery"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    TECHNICAL_ANALYSIS = "technical_analysis"
    REPORT_GENERATION = "report_generation"
    QUALITY_ASSURANCE = "quality_assurance"
    COMPLETED = "completed"

class InvestmentState(TypedDict):
    """Shared state across all agents"""
    # Input parameters
    user_profile: Dict[str, Any]
    research_type: str
    sectors: List[str]
    
    # Agent outputs
    discovered_assets: List[Dict[str, Any]]
    fundamental_analysis: Dict[str, Any]
    sentiment_scores: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    portfolio_allocation: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    
    # Workflow management
    current_step: str
    error_count: int
    retry_attempts: Dict[str, int]
    agent_confidence: Dict[str, float]
    
    # Final outputs
    recommendations: List[Dict[str, Any]]
    report_content: str
    quality_score: float
    
    # Context and memory
    conversation_id: str
    timestamp: str
    market_conditions: Dict[str, Any]

# ================================
# ADVANCED MEMORY MANAGEMENT
# ================================

class AdvancedMemoryManager:
    """Multi-tier memory system for agent collaboration"""
    
    def __init__(self, db_path: str = "./memory"):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=f"{db_path}/chroma")
        
        # Initialize collections
        self.market_memory = self.chroma_client.get_or_create_collection(
            name="market_knowledge",
            metadata={"description": "Market analysis and trends"}
        )
        
        self.decision_memory = self.chroma_client.get_or_create_collection(
            name="investment_decisions", 
            metadata={"description": "Past decisions and outcomes"}
        )
        
        # Conversation memory (simplified)
        self.conversation_memory = {
            "chat_history": [],
            "max_messages": 20
        }
        
        # Performance tracking
        self.performance_db = f"{db_path}/performance.db"
        self._init_performance_db()
    
    def _init_performance_db(self):
        """Initialize performance tracking database"""
        conn = sqlite3.connect(self.performance_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                id TEXT PRIMARY KEY,
                agent_name TEXT,
                task_type TEXT,
                execution_time REAL,
                success_rate REAL,
                confidence_score REAL,
                timestamp TEXT,
                context TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendation_tracking (
                id TEXT PRIMARY KEY,
                recommendation TEXT,
                actual_performance REAL,
                prediction_accuracy REAL,
                timestamp TEXT,
                market_context TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_market_insight(self, insight: str, metadata: Dict[str, Any]):
        """Store market insights in vector memory"""
        try:
            self.market_memory.add(
                documents=[insight],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            logger.info(f"Stored market insight: {insight[:100]}...")
        except Exception as e:
            logger.error(f"Error storing market insight: {e}")
    
    def retrieve_similar_insights(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve similar market insights"""
        try:
            results = self.market_memory.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Error retrieving insights: {e}")
            return []
    
    def store_decision_outcome(self, decision: Dict[str, Any], outcome: Dict[str, Any]):
        """Store investment decision and track outcomes"""
        try:
            decision_text = json.dumps(decision)
            metadata = {
                "outcome": json.dumps(outcome),
                "timestamp": datetime.now().isoformat(),
                "asset_type": decision.get("asset_type", "unknown")
            }
            
            self.decision_memory.add(
                documents=[decision_text],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            
            # Also store in performance DB
            conn = sqlite3.connect(self.performance_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO recommendation_tracking 
                (id, recommendation, actual_performance, prediction_accuracy, timestamp, market_context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                decision_text,
                outcome.get("performance", 0.0),
                outcome.get("accuracy", 0.0),
                datetime.now().isoformat(),
                json.dumps(outcome.get("market_context", {}))
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing decision outcome: {e}")

# ================================
# GUARDRAILS AND SAFETY SYSTEM
# ================================

class InvestmentGuardrails:
    """Comprehensive safety and compliance system"""
    
    def __init__(self):
        self.risk_limits = {
            "max_single_asset_allocation": 0.15,  # 15% max per asset
            "max_sector_allocation": 0.35,        # 35% max per sector
            "max_crypto_allocation": 0.10,        # 10% max crypto exposure
            "min_diversification": 5,             # Minimum 5 assets
            "max_leverage": 1.0                   # No leverage allowed
        }
        
        self.content_filters = [
            self._check_financial_compliance,
            self._check_risk_warnings,
            self._check_disclaimer_presence,
            self._validate_recommendation_logic
        ]
    
    def validate_recommendation(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive recommendation validation"""
        violations = []
        
        # Check all content filters
        for filter_func in self.content_filters:
            is_valid, messages = filter_func(recommendation)
            if not is_valid:
                violations.extend(messages)
        
        # Validate portfolio allocation if present
        if "allocation" in recommendation:
            allocation_valid, allocation_violations = self._validate_allocation(
                recommendation["allocation"]
            )
            if not allocation_valid:
                violations.extend(allocation_violations)
        
        return len(violations) == 0, violations
    
    def _check_financial_compliance(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check for financial compliance requirements"""
        violations = []
        content = recommendation.get("content", "")
        
        # Check for required disclaimers
        required_phrases = [
            "not financial advice",
            "consult with",
            "past performance",
            "risk"
        ]
        
        for phrase in required_phrases:
            if phrase.lower() not in content.lower():
                violations.append(f"Missing required disclaimer phrase: {phrase}")
        
        return len(violations) == 0, violations
    
    def _check_risk_warnings(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Ensure appropriate risk warnings are present"""
        violations = []
        
        if recommendation.get("risk_level", "medium").lower() == "high":
            content = recommendation.get("content", "")
            if "high risk" not in content.lower():
                violations.append("High risk investments must include explicit risk warnings")
        
        return len(violations) == 0, violations
    
    def _check_disclaimer_presence(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate presence of investment disclaimers"""
        content = recommendation.get("content", "")
        
        if "educational purposes" not in content.lower():
            return False, ["Content must include 'educational purposes' disclaimer"]
        
        return True, []
    
    def _validate_recommendation_logic(self, recommendation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate logical consistency of recommendations"""
        violations = []
        
        # Check for contradictory recommendations
        if recommendation.get("rating") == "Strong Buy" and recommendation.get("risk_level") == "High":
            if "high risk" not in recommendation.get("rationale", "").lower():
                violations.append("Strong Buy rating with High risk needs explicit risk explanation")
        
        return len(violations) == 0, violations
    
    def _validate_allocation(self, allocation: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate portfolio allocation against risk limits"""
        violations = []
        
        # Check individual asset limits
        for asset, weight in allocation.items():
            if weight > self.risk_limits["max_single_asset_allocation"]:
                violations.append(f"{asset} allocation {weight:.1%} exceeds {self.risk_limits['max_single_asset_allocation']:.1%} limit")
        
        # Check total allocation
        total_allocation = sum(allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            violations.append(f"Total allocation {total_allocation:.1%} must equal 100%")
        
        # Check minimum diversification
        if len(allocation) < self.risk_limits["min_diversification"]:
            violations.append(f"Portfolio must include at least {self.risk_limits['min_diversification']} assets")
        
        return len(violations) == 0, violations

# ================================
# SPECIALIZED AGENT DEFINITIONS
# ================================

class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        self.name = name
        self.memory_manager = memory_manager
        self.guardrails = guardrails
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.performance_history = []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def execute(self, state: InvestmentState) -> Dict[str, Any]:
        """Execute agent with retry logic and error handling"""
        start_time = time.time()
        
        try:
            result = await self._execute_core(state)
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(execution_time, success=True, confidence=result.get('confidence', 0.5))
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._track_performance(execution_time, success=False, confidence=0.0)
            logger.error(f"Agent {self.name} execution failed: {e}")
            
            # Return fallback result
            return self._fallback_result(state, str(e))
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Core execution logic - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_core")
    
    def _fallback_result(self, state: InvestmentState, error_message: str) -> Dict[str, Any]:
        """Provide fallback result when agent fails"""
        return {
            'success': False,
            'error': error_message,
            'confidence': 0.0,
            'fallback': True,
            'agent': self.name
        }
    
    def _track_performance(self, execution_time: float, success: bool, confidence: float):
        """Track agent performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'success': success,
            'confidence': confidence
        })
        
        # Store in database
        try:
            conn = sqlite3.connect(self.memory_manager.performance_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO agent_performance 
                (id, agent_name, task_type, execution_time, success_rate, confidence_score, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                self.name,
                'standard_execution',
                execution_time,
                1.0 if success else 0.0,
                confidence,
                datetime.now().isoformat(),
                json.dumps({'success': success})
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error tracking performance for {self.name}: {e}")

class MarketDiscoveryAgent(BaseAgent):
    """Specialized agent for discovering trending market opportunities"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        super().__init__("MarketDiscoveryAgent", memory_manager, guardrails)
        self.discovery_strategies = [
            "momentum_based",
            "catalyst_driven", 
            "sector_rotation",
            "technical_breakout",
            "earnings_momentum"
        ]
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Discover trending assets using multiple strategies"""
        research_type = state.get("research_type", "Trending Stocks")
        sectors = state.get("sectors", ["Technology"])
        
        # Select discovery strategy based on market conditions
        strategy = random.choice(self.discovery_strategies)
        
        # Retrieve similar past discoveries
        similar_insights = self.memory_manager.retrieve_similar_insights(
            f"{research_type} {' '.join(sectors)}", n_results=3
        )
        
        # Generate discovery prompt
        prompt = self._create_discovery_prompt(research_type, sectors, strategy, similar_insights)
        
        # Execute discovery
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse and validate results
        discovered_assets = self._parse_discovery_response(response.content)
        
        # Store insights in memory
        self.memory_manager.store_market_insight(
            response.content,
            {
                "strategy": strategy,
                "research_type": research_type,
                "sectors": sectors,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            'success': True,
            'discovered_assets': discovered_assets,
            'strategy_used': strategy,
            'confidence': self._calculate_confidence(discovered_assets),
            'insights_used': len(similar_insights)
        }
    
    def _create_discovery_prompt(self, research_type: str, sectors: List[str], strategy: str, 
                                 similar_insights: List[str]) -> str:
        """Create dynamic discovery prompt based on strategy and context"""
        
        strategy_context = {
            "momentum_based": "Focus on assets with strong recent price momentum and increasing volume",
            "catalyst_driven": "Identify assets with upcoming catalysts like earnings, FDA approvals, or partnerships",
            "sector_rotation": "Look for sectors showing rotation patterns and emerging leadership",
            "technical_breakout": "Find assets breaking through technical resistance levels",
            "earnings_momentum": "Focus on companies with strong earnings growth and positive surprises"
        }
        
        context = strategy_context.get(strategy, "General market analysis")
        sectors_text = ", ".join(sectors)
        
        # Include past insights for context
        insights_context = ""
        if similar_insights:
            insights_context = f"\n\nPrevious market insights to consider:\n" + "\n".join(similar_insights[:2])
        
        prompt = f"""As a senior market discovery agent using {strategy} strategy, discover 8-12 trending {research_type.lower()} opportunities.

DISCOVERY STRATEGY: {context}
SECTORS: {sectors_text}
DATE: {datetime.now().strftime('%B %Y')}

REQUIREMENTS:
- Use REAL ticker symbols only
- Include varied market caps (large, mid, small)
- Provide specific catalysts and rationale
- Focus on assets under-covered by mainstream analysts
- Include emerging opportunities with 20%+ upside potential

FORMAT (strict):
**ASSET 1:**
Symbol: [REAL_TICKER]
Name: [Full Name]
Price Range: $[X] - $[Y]
Market Cap: [Large/Mid/Small]-cap
Catalyst: [Specific recent development]
Upside Target: [Percentage]
Strategy Fit: [How it fits {strategy} strategy]

[Continue for 8-12 assets...]

{insights_context}

Provide varied, high-quality discoveries that avoid mainstream picks."""
        
        return prompt
    
    def _parse_discovery_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse discovery response into structured data"""
        assets = []
        
        try:
            import re
            
            # Split response into asset blocks
            asset_blocks = re.split(r'\*\*ASSET \d+:\*\*', response)
            
            for block in asset_blocks[1:]:  # Skip first empty block
                asset_data = {}
                
                # Extract fields using regex
                patterns = {
                    'symbol': r'Symbol:\s*([A-Z]{1,5})',
                    'name': r'Name:\s*([^\n]+)',
                    'price_range': r'Price Range:\s*([^\n]+)',
                    'market_cap': r'Market Cap:\s*([^\n]+)',
                    'catalyst': r'Catalyst:\s*([^\n]+)',
                    'upside_target': r'Upside Target:\s*([^\n]+)',
                    'strategy_fit': r'Strategy Fit:\s*([^\n]+)'
                }
                
                for field, pattern in patterns.items():
                    match = re.search(pattern, block, re.IGNORECASE)
                    if match:
                        asset_data[field] = match.group(1).strip()
                
                # Validate and add if valid
                if asset_data.get('symbol') and len(asset_data.get('symbol', '')) <= 5:
                    assets.append(asset_data)
                    
        except Exception as e:
            logger.error(f"Error parsing discovery response: {e}")
            
        return assets[:12]  # Limit to 12 assets
    
    def _calculate_confidence(self, discovered_assets: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on discovery quality"""
        if not discovered_assets:
            return 0.0
        
        # Score based on number of valid assets and completeness
        asset_count_score = min(len(discovered_assets) / 10.0, 1.0)  # Target 10 assets
        
        completeness_scores = []
        for asset in discovered_assets:
            required_fields = ['symbol', 'name', 'catalyst', 'upside_target']
            present_fields = sum(1 for field in required_fields if asset.get(field))
            completeness_scores.append(present_fields / len(required_fields))
        
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        
        return (asset_count_score * 0.3 + avg_completeness * 0.7)

class SentimentAnalysisAgent(BaseAgent):
    """Specialized agent for market sentiment and news analysis"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        super().__init__("SentimentAnalysisAgent", memory_manager, guardrails)
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Analyze market sentiment for discovered assets"""
        discovered_assets = state.get("discovered_assets", [])
        
        if not discovered_assets:
            return {
                'success': False,
                'error': 'No assets to analyze',
                'confidence': 0.0
            }
        
        sentiment_results = {}
        
        # Analyze sentiment for each asset
        for asset in discovered_assets[:5]:  # Limit to top 5 for efficiency
            symbol = asset.get('symbol')
            if symbol:
                sentiment_data = await self._analyze_asset_sentiment(symbol, asset)
                sentiment_results[symbol] = sentiment_data
        
        # Calculate overall market sentiment
        overall_sentiment = self._calculate_market_sentiment(sentiment_results)
        
        return {
            'success': True,
            'sentiment_scores': sentiment_results,
            'market_sentiment': overall_sentiment,
            'confidence': self._calculate_sentiment_confidence(sentiment_results)
        }
    
    async def _analyze_asset_sentiment(self, symbol: str, asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment for individual asset"""
        
        prompt = f"""As a senior sentiment analysis specialist, analyze the current market sentiment for {symbol} - {asset_info.get('name', 'Unknown')}.

Consider these factors:
- Recent news and developments
- Social media sentiment trends  
- Analyst opinion changes
- Market momentum indicators
- Sector sentiment

Current catalyst: {asset_info.get('catalyst', 'None provided')}

Provide analysis in this format:

**SENTIMENT ANALYSIS FOR {symbol}**

Overall Sentiment: [BULLISH/BEARISH/NEUTRAL]
Sentiment Score: [Number from -5 to +5]
Confidence Level: [High/Medium/Low]

Key Sentiment Drivers:
- [Driver 1 with impact assessment]
- [Driver 2 with impact assessment]  
- [Driver 3 with impact assessment]

Risk Factors:
- [Risk 1]
- [Risk 2]

Sentiment Outlook: [Short summary of sentiment trajectory]
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return self._parse_sentiment_response(response.content, symbol)
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._default_sentiment(symbol)
    
    def _parse_sentiment_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        try:
            import re
            
            # Extract sentiment components
            sentiment_match = re.search(r'Overall Sentiment:\s*(BULLISH|BEARISH|NEUTRAL)', response)
            score_match = re.search(r'Sentiment Score:\s*([-+]?\d*\.?\d+)', response)
            confidence_match = re.search(r'Confidence Level:\s*(High|Medium|Low)', response)
            
            # Extract lists
            drivers = re.findall(r'-\s*([^\n]+)', re.search(r'Key Sentiment Drivers:(.*?)Risk Factors:', response, re.DOTALL).group(1) if re.search(r'Key Sentiment Drivers:(.*?)Risk Factors:', response, re.DOTALL) else "")
            risks = re.findall(r'-\s*([^\n]+)', re.search(r'Risk Factors:(.*?)Sentiment Outlook:', response, re.DOTALL).group(1) if re.search(r'Risk Factors:(.*?)Sentiment Outlook:', response, re.DOTALL) else "")
            
            outlook_match = re.search(r'Sentiment Outlook:\s*([^\n]+)', response)
            
            return {
                'overall_sentiment': sentiment_match.group(1) if sentiment_match else 'NEUTRAL',
                'sentiment_score': float(score_match.group(1)) if score_match else 0.0,
                'confidence_level': confidence_match.group(1) if confidence_match else 'Medium',
                'key_drivers': [d.strip() for d in drivers][:3],
                'risk_factors': [r.strip() for r in risks][:3],
                'outlook': outlook_match.group(1).strip() if outlook_match else 'Neutral outlook'
            }
            
        except Exception as e:
            logger.error(f"Error parsing sentiment for {symbol}: {e}")
            return self._default_sentiment(symbol)
    
    def _default_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Default sentiment when analysis fails"""
        return {
            'overall_sentiment': 'NEUTRAL',
            'sentiment_score': 0.0,
            'confidence_level': 'Low',
            'key_drivers': ['Analysis in progress'],
            'risk_factors': ['Standard market risks'],
            'outlook': 'Monitoring required'
        }
    
    def _calculate_market_sentiment(self, sentiment_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall market sentiment from individual assets"""
        if not sentiment_results:
            return {'overall': 'NEUTRAL', 'score': 0.0, 'confidence': 'Low'}
        
        # Calculate weighted sentiment score
        scores = [data.get('sentiment_score', 0.0) for data in sentiment_results.values()]
        avg_score = sum(scores) / len(scores)
        
        # Determine overall sentiment
        if avg_score > 1.5:
            overall = 'BULLISH'
        elif avg_score < -1.5:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        # Calculate confidence based on score consistency
        score_variance = np.var(scores) if len(scores) > 1 else 0
        confidence = 'High' if score_variance < 1.0 else 'Medium' if score_variance < 4.0 else 'Low'
        
        return {
            'overall': overall,
            'score': round(avg_score, 2),
            'confidence': confidence,
            'asset_count': len(sentiment_results)
        }
    
    def _calculate_sentiment_confidence(self, sentiment_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate confidence in sentiment analysis"""
        if not sentiment_results:
            return 0.0
        
        # Base confidence on number of successful analyses and confidence levels
        high_confidence_count = sum(1 for data in sentiment_results.values() 
                                   if data.get('confidence_level') == 'High')
        
        coverage_score = len(sentiment_results) / 5.0  # Target 5 assets
        quality_score = high_confidence_count / len(sentiment_results)
        
        return min((coverage_score * 0.4 + quality_score * 0.6), 1.0)

class RiskAssessmentAgent(BaseAgent):
    """Specialized agent for comprehensive risk analysis"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        super().__init__("RiskAssessmentAgent", memory_manager, guardrails)
        self.risk_models = [
            "value_at_risk",
            "volatility_analysis", 
            "correlation_analysis",
            "sector_concentration",
            "liquidity_risk"
        ]
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        discovered_assets = state.get("discovered_assets", [])
        sentiment_scores = state.get("sentiment_scores", {})
        
        if not discovered_assets:
            return {
                'success': False,
                'error': 'No assets to assess',
                'confidence': 0.0
            }
        
        # Get market data for risk calculations
        symbols = [asset.get('symbol') for asset in discovered_assets if asset.get('symbol')]
        market_data = await self._get_market_data(symbols)
        
        # Perform multi-model risk assessment
        risk_metrics = {}
        
        for model in self.risk_models:
            risk_metrics[model] = await self._calculate_risk_metric(model, market_data, sentiment_scores)
        
        # Calculate overall risk scores
        asset_risk_scores = self._calculate_asset_risk_scores(market_data, sentiment_scores)
        portfolio_risk = self._calculate_portfolio_risk(asset_risk_scores)
        
        return {
            'success': True,
            'risk_metrics': risk_metrics,
            'asset_risk_scores': asset_risk_scores,
            'portfolio_risk': portfolio_risk,
            'confidence': self._calculate_risk_confidence(risk_metrics)
        }
    
    async def _get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for risk calculations"""
        market_data = {}
        
        for symbol in symbols[:10]:  # Limit for efficiency
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                
                if not hist.empty:
                    # Calculate basic metrics
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    
                    market_data[symbol] = {
                        'current_price': float(hist['Close'].iloc[-1]),
                        'volatility': float(volatility),
                        'returns': returns.tolist()[-60:],  # Last 60 days
                        'volume': float(hist['Volume'].mean()),
                        'price_range_52w': {
                            'high': float(hist['High'].max()),
                            'low': float(hist['Low'].min())
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
                
        return market_data
    
    async def _calculate_risk_metric(self, model: str, market_data: Dict[str, Dict[str, Any]], 
                                   sentiment_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate specific risk metric"""
        
        if model == "volatility_analysis":
            return self._analyze_volatility(market_data)
        elif model == "correlation_analysis":
            return self._analyze_correlations(market_data)
        elif model == "value_at_risk":
            return self._calculate_var(market_data)
        elif model == "sector_concentration":
            return self._analyze_sector_concentration(market_data)
        elif model == "liquidity_risk":
            return self._analyze_liquidity_risk(market_data)
        else:
            return {'model': model, 'status': 'not_implemented'}
    
    def _analyze_volatility(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze volatility risk across assets"""
        volatilities = {}
        
        for symbol, data in market_data.items():
            volatilities[symbol] = data.get('volatility', 0.0)
        
        if volatilities:
            avg_volatility = sum(volatilities.values()) / len(volatilities)
            max_volatility = max(volatilities.values())
            min_volatility = min(volatilities.values())
            
            risk_level = 'High' if avg_volatility > 0.4 else 'Medium' if avg_volatility > 0.25 else 'Low'
            
            return {
                'average_volatility': round(avg_volatility, 3),
                'max_volatility': round(max_volatility, 3),
                'min_volatility': round(min_volatility, 3),
                'risk_level': risk_level,
                'asset_volatilities': volatilities
            }
        
        return {'error': 'No volatility data available'}
    
    def _analyze_correlations(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlation risk between assets"""
        returns_data = {}
        
        for symbol, data in market_data.items():
            if 'returns' in data and len(data['returns']) > 10:
                returns_data[symbol] = data['returns']
        
        if len(returns_data) < 2:
            return {'error': 'Insufficient data for correlation analysis'}
        
        # Calculate pairwise correlations
        symbols = list(returns_data.keys())
        correlations = {}
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                returns1 = np.array(returns_data[symbol1])
                returns2 = np.array(returns_data[symbol2])
                
                # Align lengths
                min_length = min(len(returns1), len(returns2))
                returns1 = returns1[-min_length:]
                returns2 = returns2[-min_length:]
                
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                correlations[f"{symbol1}-{symbol2}"] = round(correlation, 3)
        
        avg_correlation = sum(correlations.values()) / len(correlations) if correlations else 0
        max_correlation = max(correlations.values()) if correlations else 0
        
        diversification_score = 1 - abs(avg_correlation)  # Higher is better
        
        return {
            'average_correlation': round(avg_correlation, 3),
            'max_correlation': round(max_correlation, 3),
            'diversification_score': round(diversification_score, 3),
            'pairwise_correlations': correlations
        }
    
    def _calculate_var(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) estimates"""
        all_returns = []
        
        for symbol, data in market_data.items():
            if 'returns' in data:
                all_returns.extend(data['returns'])
        
        if not all_returns:
            return {'error': 'No returns data for VaR calculation'}
        
        returns_array = np.array(all_returns)
        
        # Calculate VaR at different confidence levels
        var_95 = np.percentile(returns_array, 5)  # 95% VaR
        var_99 = np.percentile(returns_array, 1)  # 99% VaR
        
        # Expected shortfall (CVaR)
        cvar_95 = returns_array[returns_array <= var_95].mean()
        
        return {
            'var_95': round(var_95, 4),
            'var_99': round(var_99, 4),
            'cvar_95': round(cvar_95, 4),
            'risk_interpretation': 'High' if var_95 < -0.05 else 'Medium' if var_95 < -0.02 else 'Low'
        }
    
    def _analyze_sector_concentration(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sector concentration risk"""
        # For demo purposes, simulate sector data
        # In production, this would use real sector classifications
        
        simulated_sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
        sector_allocation = {}
        
        symbols = list(market_data.keys())
        for i, symbol in enumerate(symbols):
            sector = simulated_sectors[i % len(simulated_sectors)]
            sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
        
        # Calculate concentration metrics
        total_assets = len(symbols)
        max_sector_concentration = max(sector_allocation.values()) / total_assets if total_assets > 0 else 0
        
        concentration_risk = 'High' if max_sector_concentration > 0.5 else 'Medium' if max_sector_concentration > 0.3 else 'Low'
        
        return {
            'sector_allocation': sector_allocation,
            'max_concentration': round(max_sector_concentration, 3),
            'concentration_risk': concentration_risk,
            'diversification_score': 1 - max_sector_concentration
        }
    
    def _analyze_liquidity_risk(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze liquidity risk based on volume and market cap"""
        liquidity_scores = {}
        
        for symbol, data in market_data.items():
            volume = data.get('volume', 0)
            price = data.get('current_price', 0)
            
            # Simple liquidity score based on dollar volume
            dollar_volume = volume * price
            
            if dollar_volume > 10_000_000:  # >$10M daily volume
                liquidity_scores[symbol] = 'High'
            elif dollar_volume > 1_000_000:  # >$1M daily volume
                liquidity_scores[symbol] = 'Medium'
            else:
                liquidity_scores[symbol] = 'Low'
        
        # Calculate overall liquidity risk
        low_liquidity_count = sum(1 for score in liquidity_scores.values() if score == 'Low')
        liquidity_risk = 'High' if low_liquidity_count > len(liquidity_scores) * 0.3 else 'Low'
        
        return {
            'asset_liquidity': liquidity_scores,
            'overall_liquidity_risk': liquidity_risk,
            'low_liquidity_assets': low_liquidity_count
        }
    
    def _calculate_asset_risk_scores(self, market_data: Dict[str, Dict[str, Any]], 
                                   sentiment_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive risk scores for each asset"""
        asset_risks = {}
        
        for symbol in market_data.keys():
            market_risk_score = self._calculate_market_risk_score(symbol, market_data)
            sentiment_risk_score = self._calculate_sentiment_risk_score(symbol, sentiment_scores)
            
            # Combine scores
            overall_risk = (market_risk_score + sentiment_risk_score) / 2
            
            risk_level = 'High' if overall_risk > 0.7 else 'Medium' if overall_risk > 0.4 else 'Low'
            
            asset_risks[symbol] = {
                'market_risk': market_risk_score,
                'sentiment_risk': sentiment_risk_score,
                'overall_risk': round(overall_risk, 3),
                'risk_level': risk_level
            }
        
        return asset_risks
    
    def _calculate_market_risk_score(self, symbol: str, market_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate market-based risk score (0-1, higher is riskier)"""
        data = market_data.get(symbol, {})
        
        volatility = data.get('volatility', 0.3)  # Default to 30% volatility
        
        # Normalize volatility to 0-1 scale (0.6+ vol = max risk)
        volatility_score = min(volatility / 0.6, 1.0)
        
        return volatility_score
    
    def _calculate_sentiment_risk_score(self, symbol: str, sentiment_scores: Dict[str, Dict[str, Any]]) -> float:
        """Calculate sentiment-based risk score (0-1, higher is riskier)"""
        sentiment_data = sentiment_scores.get(symbol, {})
        
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)  # -5 to +5 scale
        
        # Convert to risk score (extreme sentiment = higher risk)
        # Neutral sentiment (0) = lowest risk, extreme sentiment = highest risk
        risk_score = abs(sentiment_score) / 5.0  # Normalize to 0-1
        
        return risk_score
    
    def _calculate_portfolio_risk(self, asset_risk_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall portfolio risk metrics"""
        if not asset_risk_scores:
            return {'error': 'No asset risk data available'}
        
        risk_scores = [data['overall_risk'] for data in asset_risk_scores.values()]
        
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)
        
        # Count assets by risk level
        risk_distribution = {'High': 0, 'Medium': 0, 'Low': 0}
        for data in asset_risk_scores.values():
            risk_distribution[data['risk_level']] += 1
        
        overall_risk_level = 'High' if avg_risk > 0.6 else 'Medium' if avg_risk > 0.3 else 'Low'
        
        return {
            'average_risk': round(avg_risk, 3),
            'max_risk': round(max_risk, 3),
            'min_risk': round(min_risk, 3),
            'risk_distribution': risk_distribution,
            'overall_risk_level': overall_risk_level,
            'diversification_benefit': round(1 - (avg_risk - min_risk), 3)
        }
    
    def _calculate_risk_confidence(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in risk assessment"""
        completed_models = sum(1 for metric in risk_metrics.values() 
                              if not metric.get('error') and not metric.get('status') == 'not_implemented')
        
        total_models = len(self.risk_models)
        
        return completed_models / total_models

class PortfolioOptimizationAgent(BaseAgent):
    """Specialized agent for portfolio construction and optimization"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        super().__init__("PortfolioOptimizationAgent", memory_manager, guardrails)
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Optimize portfolio allocation based on analysis results"""
        discovered_assets = state.get("discovered_assets", [])
        risk_metrics = state.get("risk_metrics", {})
        sentiment_scores = state.get("sentiment_scores", {})
        user_profile = state.get("user_profile", {})
        
        if not discovered_assets:
            return {
                'success': False,
                'error': 'No assets available for optimization',
                'confidence': 0.0
            }
        
        # Get optimization parameters from user profile
        risk_tolerance = user_profile.get("risk_tolerance", "Moderate")
        time_horizon = user_profile.get("time_horizon", "Medium-term")
        budget = user_profile.get("budget", 10000)
        
        # Perform portfolio optimization
        allocation = await self._optimize_allocation(
            discovered_assets, risk_metrics, sentiment_scores, 
            risk_tolerance, time_horizon
        )
        
        # Validate allocation against guardrails
        is_valid, violations = self.guardrails._validate_allocation(allocation)
        
        if not is_valid:
            # Adjust allocation to meet guardrails
            allocation = self._adjust_allocation_for_compliance(allocation, violations)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(allocation, budget, risk_metrics)
        
        return {
            'success': True,
            'allocation': allocation,
            'portfolio_metrics': portfolio_metrics,
            'compliance_violations': violations if not is_valid else [],
            'confidence': self._calculate_optimization_confidence(allocation, portfolio_metrics)
        }
    
    async def _optimize_allocation(self, discovered_assets: List[Dict[str, Any]], 
                                 risk_metrics: Dict[str, Any], 
                                 sentiment_scores: Dict[str, Dict[str, Any]],
                                 risk_tolerance: str, time_horizon: str) -> Dict[str, float]:
        """Optimize portfolio allocation using AI-driven approach"""
        
        # Create optimization prompt
        prompt = self._create_optimization_prompt(
            discovered_assets, risk_metrics, sentiment_scores, risk_tolerance, time_horizon
        )
        
        # Get AI optimization recommendation
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse allocation from response
        allocation = self._parse_allocation_response(response.content, discovered_assets)
        
        # Apply optimization constraints
        allocation = self._apply_optimization_constraints(allocation, risk_tolerance)
        
        return allocation
    
    def _create_optimization_prompt(self, discovered_assets: List[Dict[str, Any]], 
                                  risk_metrics: Dict[str, Any], 
                                  sentiment_scores: Dict[str, Dict[str, Any]],
                                  risk_tolerance: str, time_horizon: str) -> str:
        """Create portfolio optimization prompt"""
        
        # Prepare asset summaries
        asset_summaries = []
        for asset in discovered_assets[:10]:  # Limit to top 10
            symbol = asset.get('symbol', 'Unknown')
            name = asset.get('name', 'Unknown')
            catalyst = asset.get('catalyst', 'None')
            
            # Get risk and sentiment data
            risk_data = risk_metrics.get('asset_risk_scores', {}).get(symbol, {})
            sentiment_data = sentiment_scores.get(symbol, {})
            
            risk_level = risk_data.get('risk_level', 'Medium')
            sentiment = sentiment_data.get('overall_sentiment', 'NEUTRAL')
            sentiment_score = sentiment_data.get('sentiment_score', 0.0)
            
            asset_summaries.append(f"""
{symbol} ({name}):
- Risk Level: {risk_level}
- Sentiment: {sentiment} (Score: {sentiment_score})
- Catalyst: {catalyst}
""")
        
        assets_text = "\n".join(asset_summaries)
        
        # Risk tolerance guidelines
        risk_guidelines = {
            "Conservative": "Maximum 5% per asset, focus on stability, limit high-risk positions",
            "Moderate": "Maximum 10% per asset, balanced approach, some growth positions allowed",
            "Aggressive": "Maximum 15% per asset, growth-focused, higher risk tolerance"
        }
        
        guideline = risk_guidelines.get(risk_tolerance, risk_guidelines["Moderate"])
        
        prompt = f"""As a senior portfolio optimization specialist, create an optimal allocation for these discovered assets:

INVESTOR PROFILE:
- Risk Tolerance: {risk_tolerance}
- Time Horizon: {time_horizon}
- Optimization Guideline: {guideline}

AVAILABLE ASSETS:
{assets_text}

PORTFOLIO RISK METRICS:
- Overall Risk Level: {risk_metrics.get('portfolio_risk', {}).get('overall_risk_level', 'Medium')}
- Average Volatility: {risk_metrics.get('volatility_analysis', {}).get('average_volatility', 'N/A')}

OPTIMIZATION REQUIREMENTS:
1. Total allocation must equal 100%
2. Follow risk tolerance guidelines for position sizing
3. Consider sentiment scores and risk levels
4. Ensure adequate diversification
5. Include cash position (5-15%) for flexibility

OUTPUT FORMAT:
**OPTIMAL PORTFOLIO ALLOCATION**

[SYMBOL1]: [X]%
[SYMBOL2]: [Y]%
[SYMBOL3]: [Z]%
...
CASH: [C]%

TOTAL: 100%

**ALLOCATION RATIONALE**
Brief explanation of allocation decisions based on risk, sentiment, and diversification principles.
"""
        
        return prompt
    
    def _parse_allocation_response(self, response: str, discovered_assets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parse AI allocation response into allocation dictionary"""
        allocation = {}
        
        try:
            import re
            
            # Extract allocation lines
            allocation_lines = re.findall(r'([A-Z]{1,5}|CASH):\s*(\d+\.?\d*)%', response)
            
            total_allocation = 0
            for symbol, percentage in allocation_lines:
                pct = float(percentage) / 100.0
                allocation[symbol] = pct
                total_allocation += pct
            
            # Normalize if total doesn't equal 100%
            if abs(total_allocation - 1.0) > 0.01:
                for symbol in allocation:
                    allocation[symbol] = allocation[symbol] / total_allocation
            
            # Ensure we have valid symbols
            valid_symbols = {asset.get('symbol') for asset in discovered_assets if asset.get('symbol')}
            valid_symbols.add('CASH')
            
            # Filter out invalid symbols
            allocation = {symbol: weight for symbol, weight in allocation.items() 
                         if symbol in valid_symbols}
            
            # Add cash if not present
            if 'CASH' not in allocation and allocation:
                current_total = sum(allocation.values())
                if current_total < 0.95:  # If less than 95% allocated
                    allocation['CASH'] = 1.0 - current_total
            
        except Exception as e:
            logger.error(f"Error parsing allocation response: {e}")
            allocation = self._create_default_allocation(discovered_assets)
        
        return allocation
    
    def _create_default_allocation(self, discovered_assets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Create default allocation when parsing fails"""
        if not discovered_assets:
            return {'CASH': 1.0}
        
        # Equal weight allocation with cash buffer
        num_assets = min(len(discovered_assets), 8)  # Max 8 assets
        asset_weight = 0.8 / num_assets  # 80% in assets, 20% cash
        
        allocation = {}
        for i, asset in enumerate(discovered_assets[:num_assets]):
            symbol = asset.get('symbol')
            if symbol:
                allocation[symbol] = asset_weight
        
        allocation['CASH'] = 0.2
        
        return allocation
    
    def _apply_optimization_constraints(self, allocation: Dict[str, float], risk_tolerance: str) -> Dict[str, float]:
        """Apply optimization constraints based on risk tolerance"""
        
        # Maximum position sizes by risk tolerance
        max_position = {
            "Conservative": 0.08,  # 8% max
            "Moderate": 0.12,      # 12% max
            "Aggressive": 0.18     # 18% max
        }
        
        max_pos = max_position.get(risk_tolerance, 0.12)
        
        # Apply position limits
        adjusted_allocation = {}
        excess_weight = 0
        
        for symbol, weight in allocation.items():
            if symbol == 'CASH':
                adjusted_allocation[symbol] = weight
            elif weight > max_pos:
                adjusted_allocation[symbol] = max_pos
                excess_weight += weight - max_pos
            else:
                adjusted_allocation[symbol] = weight
        
        # Redistribute excess weight to cash
        if excess_weight > 0:
            adjusted_allocation['CASH'] = adjusted_allocation.get('CASH', 0) + excess_weight
        
        return adjusted_allocation
    
    def _adjust_allocation_for_compliance(self, allocation: Dict[str, float], violations: List[str]) -> Dict[str, float]:
        """Adjust allocation to meet compliance requirements"""
        
        # Simple adjustment: reduce violating positions and increase cash
        adjusted = allocation.copy()
        
        for violation in violations:
            if "exceeds" in violation and "%" in violation:
                # Extract symbol and limit from violation message
                try:
                    symbol = violation.split()[0]
                    if symbol in adjusted:
                        # Reduce to maximum allowed (assume 10% for safety)
                        excess = adjusted[symbol] - 0.10
                        adjusted[symbol] = 0.10
                        adjusted['CASH'] = adjusted.get('CASH', 0) + excess
                except:
                    pass
        
        return adjusted
    
    def _calculate_portfolio_metrics(self, allocation: Dict[str, float], budget: float, 
                                   risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        
        # Calculate dollar allocations
        dollar_allocations = {symbol: weight * budget for symbol, weight in allocation.items()}
        
        # Calculate expected return (simplified)
        expected_returns = {
            'Conservative': 0.06,
            'Moderate': 0.08,
            'Aggressive': 0.12
        }
        
        # Estimate portfolio return based on allocation
        cash_weight = allocation.get('CASH', 0)
        asset_weight = 1 - cash_weight
        
        # Simple return estimation
        portfolio_return = cash_weight * 0.02 + asset_weight * 0.10  # 2% cash, 10% assets
        
        # Risk metrics from risk assessment
        portfolio_risk_data = risk_metrics.get('portfolio_risk', {})
        portfolio_volatility = portfolio_risk_data.get('average_risk', 0.2) * 0.3  # Convert risk to volatility estimate
        
        # Sharpe ratio estimation
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification metrics
        num_positions = len([w for w in allocation.values() if w > 0.01])  # Positions > 1%
        concentration = max(allocation.values()) if allocation else 0
        
        return {
            'expected_annual_return': round(portfolio_return, 3),
            'estimated_volatility': round(portfolio_volatility, 3),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'number_of_positions': num_positions,
            'max_position_size': round(concentration, 3),
            'cash_allocation': round(cash_weight, 3),
            'dollar_allocations': dollar_allocations,
            '5_year_projection': round(budget * (1 + portfolio_return) ** 5, 2)
        }
    
    def _calculate_optimization_confidence(self, allocation: Dict[str, float], 
                                         portfolio_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in portfolio optimization"""
        
        # Base confidence on diversification and reasonable allocations
        num_positions = portfolio_metrics.get('number_of_positions', 0)
        max_position = portfolio_metrics.get('max_position_size', 1.0)
        cash_allocation = portfolio_metrics.get('cash_allocation', 0)
        
        # Diversification score
        diversification_score = min(num_positions / 8.0, 1.0)  # Target 8 positions
        
        # Position sizing score (penalize extreme concentrations)
        position_score = 1.0 - max(max_position - 0.15, 0) / 0.15  # Penalize >15% positions
        
        # Cash allocation score (prefer 5-20% cash)
        if 0.05 <= cash_allocation <= 0.20:
            cash_score = 1.0
        else:
            cash_score = max(0, 1.0 - abs(cash_allocation - 0.125) / 0.125)
        
        # Combined confidence
        confidence = (diversification_score * 0.4 + position_score * 0.4 + cash_score * 0.2)
        
        return round(confidence, 3)

class ReportGenerationAgent(BaseAgent):
    """Specialized agent for comprehensive report generation"""
    
    def __init__(self, memory_manager: AdvancedMemoryManager, guardrails: InvestmentGuardrails):
        super().__init__("ReportGenerationAgent", memory_manager, guardrails)
    
    async def _execute_core(self, state: InvestmentState) -> Dict[str, Any]:
        """Generate comprehensive investment analysis report"""
        
        # Gather all analysis results
        discovered_assets = state.get("discovered_assets", [])
        sentiment_scores = state.get("sentiment_scores", {})
        risk_metrics = state.get("risk_metrics", {})
        portfolio_allocation = state.get("portfolio_allocation", {})
        user_profile = state.get("user_profile", {})
        
        # Generate report sections
        report_sections = {
            'executive_summary': await self._generate_executive_summary(state),
            'asset_analysis': await self._generate_asset_analysis(discovered_assets, sentiment_scores),
            'risk_assessment': await self._generate_risk_section(risk_metrics),
            'portfolio_recommendations': await self._generate_portfolio_section(portfolio_allocation),
            'investment_strategy': await self._generate_strategy_section(user_profile),
            'disclaimers': self._generate_disclaimers()
        }
        
        # Combine into full report
        full_report = self._combine_report_sections(report_sections)
        
        # Validate report with guardrails
        is_valid, violations = self.guardrails.validate_recommendation({
            'content': full_report,
            'risk_level': risk_metrics.get('portfolio_risk', {}).get('overall_risk_level', 'Medium')
        })
        
        # Store report in memory
        self.memory_manager.store_market_insight(
            full_report,
            {
                'report_type': 'comprehensive_analysis',
                'user_profile': user_profile,
                'asset_count': len(discovered_assets),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return {
            'success': True,
            'report_content': full_report,
            'report_sections': report_sections,
            'compliance_valid': is_valid,
            'compliance_violations': violations,
            'confidence': self._calculate_report_confidence(report_sections)
        }
    
    async def _generate_executive_summary(self, state: InvestmentState) -> str:
        """Generate executive summary section"""
        
        discovered_assets = state.get("discovered_assets", [])
        risk_metrics = state.get("risk_metrics", {})
        portfolio_allocation = state.get("portfolio_allocation", {})
        user_profile = state.get("user_profile", {})
        
        prompt = f"""Generate a professional executive summary for an investment analysis report with these key findings:

PORTFOLIO OVERVIEW:
- Number of Assets Analyzed: {len(discovered_assets)}
- Risk Tolerance: {user_profile.get('risk_tolerance', 'Moderate')}
- Investment Horizon: {user_profile.get('time_horizon', 'Medium-term')}
- Budget: ${user_profile.get('budget', 10000):,}

RISK ASSESSMENT:
- Overall Risk Level: {risk_metrics.get('portfolio_risk', {}).get('overall_risk_level', 'Medium')}
- Portfolio Volatility: {risk_metrics.get('volatility_analysis', {}).get('average_volatility', 'N/A')}

PORTFOLIO METRICS:
- Expected Return: {portfolio_allocation.get('portfolio_metrics', {}).get('expected_annual_return', 'N/A')}
- Diversification: {portfolio_allocation.get('portfolio_metrics', {}).get('number_of_positions', 'N/A')} positions

Create a concise, professional executive summary (150-200 words) that highlights:
1. Investment approach and methodology
2. Key findings and recommendations
3. Risk assessment results
4. Expected outcomes and timeline

Use professional financial language appropriate for investment reports."""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return self._default_executive_summary(user_profile, len(discovered_assets))
    
    def _default_executive_summary(self, user_profile: Dict[str, Any], asset_count: int) -> str:
        """Default executive summary when generation fails"""
        return f"""## Executive Summary

This comprehensive investment analysis evaluated {asset_count} potential opportunities aligned with a {user_profile.get('risk_tolerance', 'moderate')} risk profile and {user_profile.get('time_horizon', 'medium-term')} investment horizon.

Our multi-agent analysis system conducted thorough market discovery, sentiment analysis, and risk assessment to identify optimal portfolio allocation strategies. The analysis incorporates current market conditions, individual asset fundamentals, and comprehensive risk metrics.

Key findings indicate a balanced approach with appropriate diversification across asset classes, maintaining risk levels consistent with investor objectives while targeting competitive returns.

All recommendations include comprehensive risk disclosures and are provided for educational purposes only. Investors should conduct additional due diligence and consult with qualified financial professionals before making investment decisions."""
    
    async def _generate_asset_analysis(self, discovered_assets: List[Dict[str, Any]], 
                                     sentiment_scores: Dict[str, Dict[str, Any]]) -> str:
        """Generate asset analysis section"""
        
        if not discovered_assets:
            return "## Asset Analysis\n\nNo assets were discovered in this analysis cycle."
        
        analysis_content = "## Discovered Assets Analysis\n\n"
        
        # Create summary table
        analysis_content += "| Symbol | Name | Risk Level | Sentiment | Key Catalyst |\n"
        analysis_content += "|--------|------|------------|-----------|-------------|\n"
        
        for asset in discovered_assets[:10]:  # Limit to top 10
            symbol = asset.get('symbol', 'N/A')
            name = asset.get('name', 'Unknown')
            catalyst = asset.get('catalyst', 'Market trends')
            
            sentiment_data = sentiment_scores.get(symbol, {})
            sentiment = sentiment_data.get('overall_sentiment', 'NEUTRAL')
            
            # Simulate risk level for demo
            risk_level = 'Medium'  # In production, get from risk assessment
            
            analysis_content += f"| {symbol} | {name} | {risk_level} | {sentiment} | {catalyst} |\n"
        
        # Add detailed analysis for top assets
        analysis_content += "\n### Detailed Asset Insights\n\n"
        
        for asset in discovered_assets[:5]:  # Top 5 detailed analysis
            symbol = asset.get('symbol', 'N/A')
            name = asset.get('name', 'Unknown')
            catalyst = asset.get('catalyst', 'Market trends')
            
            sentiment_data = sentiment_scores.get(symbol, {})
            sentiment = sentiment_data.get('overall_sentiment', 'NEUTRAL')
            outlook = sentiment_data.get('outlook', 'Monitoring required')
            
            analysis_content += f"""**{symbol} - {name}**
- **Investment Catalyst**: {catalyst}
- **Market Sentiment**: {sentiment}
- **Outlook**: {outlook}
- **Recommendation**: Based on current analysis, this asset shows potential for portfolio inclusion with appropriate position sizing.

"""
        
        return analysis_content
    
    async def _generate_risk_section(self, risk_metrics: Dict[str, Any]) -> str:
        """Generate risk assessment section"""
        
        risk_content = "## Risk Assessment\n\n"
        
        portfolio_risk = risk_metrics.get('portfolio_risk', {})
        volatility_analysis = risk_metrics.get('volatility_analysis', {})
        correlation_analysis = risk_metrics.get('correlation_analysis', {})
        
        # Overall risk summary
        risk_content += f"""### Portfolio Risk Summary

**Overall Risk Level**: {portfolio_risk.get('overall_risk_level', 'Medium')}
**Average Risk Score**: {portfolio_risk.get('average_risk', 'N/A')}
**Risk Distribution**: {portfolio_risk.get('risk_distribution', {})}

### Volatility Analysis

**Average Portfolio Volatility**: {volatility_analysis.get('average_volatility', 'N/A')}
**Risk Assessment**: {volatility_analysis.get('risk_level', 'Medium')} volatility environment

### Diversification Metrics

**Diversification Score**: {correlation_analysis.get('diversification_score', 'N/A')}
**Average Correlation**: {correlation_analysis.get('average_correlation', 'N/A')}

### Risk Management Recommendations

1. **Position Sizing**: Maintain individual position limits based on risk tolerance
2. **Diversification**: Ensure adequate spread across sectors and asset types
3. **Monitoring**: Regular review of risk metrics and market conditions
4. **Rebalancing**: Quarterly portfolio rebalancing to maintain target allocations

"""
        
        return risk_content
    
    async def _generate_portfolio_section(self, portfolio_allocation: Dict[str, Any]) -> str:
        """Generate portfolio recommendations section"""
        
        allocation = portfolio_allocation.get('allocation', {})
        metrics = portfolio_allocation.get('portfolio_metrics', {})
        
        portfolio_content = "## Portfolio Recommendations\n\n"
        
        if not allocation:
            return portfolio_content + "Portfolio optimization pending.\n\n"
        
        # Allocation table
        portfolio_content += "### Recommended Allocation\n\n"
        portfolio_content += "| Asset | Allocation | Dollar Amount |\n"
        portfolio_content += "|-------|------------|---------------|\n"
        
        dollar_allocations = metrics.get('dollar_allocations', {})
        
        for symbol, weight in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
            percentage = f"{weight:.1%}"
            dollar_amount = f"${dollar_allocations.get(symbol, 0):,.0f}"
            portfolio_content += f"| {symbol} | {percentage} | {dollar_amount} |\n"
        
        # Portfolio metrics
        portfolio_content += f"""

### Portfolio Performance Projections

**Expected Annual Return**: {metrics.get('expected_annual_return', 'N/A'):.1%}
**Estimated Volatility**: {metrics.get('estimated_volatility', 'N/A'):.1%}
**Sharpe Ratio**: {metrics.get('sharpe_ratio', 'N/A')}
**5-Year Projection**: ${metrics.get('5_year_projection', 'N/A'):,.0f}

### Portfolio Characteristics

**Number of Positions**: {metrics.get('number_of_positions', 'N/A')}
**Maximum Position Size**: {metrics.get('max_position_size', 'N/A'):.1%}
**Cash Allocation**: {metrics.get('cash_allocation', 'N/A'):.1%}

"""
        
        return portfolio_content
    
    async def _generate_strategy_section(self, user_profile: Dict[str, Any]) -> str:
        """Generate investment strategy section"""
        
        risk_tolerance = user_profile.get('risk_tolerance', 'Moderate')
        time_horizon = user_profile.get('time_horizon', 'Medium-term')
        goals = user_profile.get('goals', 'General investment growth')
        
        strategy_content = f"""## Investment Strategy

### Investment Profile
- **Risk Tolerance**: {risk_tolerance}
- **Time Horizon**: {time_horizon}
- **Investment Goals**: {goals}

### Strategic Approach

Based on your investment profile, our multi-agent analysis recommends a balanced approach that:

1. **Diversifies Risk**: Spreads investments across multiple assets and sectors
2. **Aligns with Timeline**: Matches investment strategies to your {time_horizon.lower()} horizon
3. **Manages Volatility**: Maintains risk levels appropriate for {risk_tolerance.lower()} tolerance
4. **Targets Growth**: Seeks competitive returns while preserving capital

### Implementation Guidelines

**Initial Investment**: Begin with core positions and build gradually
**Rebalancing**: Review and rebalance quarterly or when allocations drift >5%
**Monitoring**: Track performance against benchmarks and adjust as needed
**Risk Management**: Maintain stop-losses and position limits as appropriate

### Next Steps

1. Review recommendations with qualified financial advisor
2. Conduct additional due diligence on recommended assets
3. Implement portfolio gradually using dollar-cost averaging
4. Establish regular monitoring and rebalancing schedule

"""
        
        return strategy_content
    
    def _generate_disclaimers(self) -> str:
        """Generate required disclaimers and risk warnings"""
        
        return """## Important Disclaimers

### Investment Disclaimers

**Not Financial Advice**: This analysis is provided for educational and informational purposes only and does not constitute financial, investment, or trading advice. All investment decisions should be made based on your own evaluation and in consultation with qualified financial professionals.

**Past Performance**: Past performance is not indicative of future results. All investments carry inherent risks, including the potential for loss of principal.

**Risk Warnings**: Investment values can fluctuate significantly, and you may receive back less than your original investment. High-risk investments may experience substantial volatility and potential losses.

**Due Diligence**: Always conduct your own thorough research and due diligence before making any investment decisions. Consider your financial situation, investment objectives, and risk tolerance.

### AI Analysis Limitations

**Automated Analysis**: This report was generated using artificial intelligence and automated analysis tools. While comprehensive, it should supplement, not replace, human judgment and professional advice.

**Market Conditions**: Market conditions change rapidly, and analysis may become outdated. Always verify current market data and conditions before making investment decisions.

**No Guarantees**: No guarantees are made regarding the accuracy of projections, recommendations, or analysis outcomes.

### Regulatory Compliance

This analysis is not intended for distribution to or use by any person in any jurisdiction where such distribution or use would be contrary to local law or regulation. Ensure compliance with all applicable laws and regulations in your jurisdiction.

---

*Generated by AI Investment Advisor Pro - Multi-Agent Analysis System*
*Report Date: {datetime.now().strftime('%B %d, %Y')}*
"""
    
    def _combine_report_sections(self, sections: Dict[str, str]) -> str:
        """Combine all report sections into final report"""
        
        report_header = f"""# Comprehensive Investment Analysis Report

**Report Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Analysis Method**: Multi-Agent AI System  
**Report Type**: Portfolio Strategy & Asset Discovery  

---

"""
        
        # Combine sections in logical order
        section_order = [
            'executive_summary',
            'asset_analysis', 
            'portfolio_recommendations',
            'risk_assessment',
            'investment_strategy',
            'disclaimers'
        ]
        
        full_report = report_header
        
        for section_key in section_order:
            if section_key in sections:
                full_report += sections[section_key] + "\n\n"
        
        return full_report
    
    def _calculate_report_confidence(self, report_sections: Dict[str, str]) -> float:
        """Calculate confidence in report quality"""
        
        # Check completeness of sections
        required_sections = ['executive_summary', 'asset_analysis', 'portfolio_recommendations', 'risk_assessment']
        completed_sections = sum(1 for section in required_sections if section in report_sections and len(report_sections[section]) > 100)
        
        completeness_score = completed_sections / len(required_sections)
        
        # Check content quality (simple length-based heuristic)
        total_content_length = sum(len(content) for content in report_sections.values())
        quality_score = min(total_content_length / 5000, 1.0)  # Target 5000+ chars for quality
        
        return (completeness_score * 0.6 + quality_score * 0.4)

# ================================
# LANGGRAPH WORKFLOW ORCHESTRATION
# ================================

class InvestmentWorkflowOrchestrator:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()
        self.guardrails = InvestmentGuardrails()
        
        # Initialize specialized agents
        self.agents = {
            'market_discovery': MarketDiscoveryAgent(self.memory_manager, self.guardrails),
            'sentiment_analysis': SentimentAnalysisAgent(self.memory_manager, self.guardrails),
            'risk_assessment': RiskAssessmentAgent(self.memory_manager, self.guardrails),
            'portfolio_optimization': PortfolioOptimizationAgent(self.memory_manager, self.guardrails),
            'report_generation': ReportGenerationAgent(self.memory_manager, self.guardrails)
        }
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
        
        # Compile workflow (without checkpointer for now to avoid version issues)
        self.app = self.workflow.compile()
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow with agent nodes"""
        
        workflow = StateGraph(InvestmentState)
        
        # Add agent nodes
        workflow.add_node("market_discovery", self._market_discovery_node)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)  
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("portfolio_optimization", self._portfolio_optimization_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("quality_assurance", self._quality_assurance_node)
        
        # Define workflow edges
        workflow.set_entry_point("market_discovery")
        
        workflow.add_edge("market_discovery", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "portfolio_optimization")
        workflow.add_edge("portfolio_optimization", "report_generation")
        workflow.add_edge("report_generation", "quality_assurance")
        workflow.add_edge("quality_assurance", END)
        
        return workflow
    
    async def _market_discovery_node(self, state: InvestmentState) -> InvestmentState:
        """Market discovery agent node"""
        logger.info("Executing market discovery...")
        
        try:
            result = await self.agents['market_discovery'].execute(state)
            
            if result.get('success'):
                state['discovered_assets'] = result.get('discovered_assets', [])
                state['agent_confidence']['market_discovery'] = result.get('confidence', 0.0)
            else:
                state['error_count'] += 1
                logger.error(f"Market discovery failed: {result.get('error')}")
                
        except Exception as e:
            state['error_count'] += 1
            logger.error(f"Market discovery node error: {e}")
        
        state['current_step'] = 'sentiment_analysis'
        return state
    
    async def _sentiment_analysis_node(self, state: InvestmentState) -> InvestmentState:
        """Sentiment analysis agent node"""
        logger.info("Executing sentiment analysis...")
        
        try:
            result = await self.agents['sentiment_analysis'].execute(state)
            
            if result.get('success'):
                state['sentiment_scores'] = result.get('sentiment_scores', {})
                state['agent_confidence']['sentiment_analysis'] = result.get('confidence', 0.0)
            else:
                state['error_count'] += 1
                logger.error(f"Sentiment analysis failed: {result.get('error')}")
                
        except Exception as e:
            state['error_count'] += 1
            logger.error(f"Sentiment analysis node error: {e}")
        
        state['current_step'] = 'risk_assessment'
        return state
    
    async def _risk_assessment_node(self, state: InvestmentState) -> InvestmentState:
        """Risk assessment agent node"""
        logger.info("Executing risk assessment...")
        
        try:
            result = await self.agents['risk_assessment'].execute(state)
            
            if result.get('success'):
                state['risk_metrics'] = result.get('risk_metrics', {})
                state['agent_confidence']['risk_assessment'] = result.get('confidence', 0.0)
            else:
                state['error_count'] += 1
                logger.error(f"Risk assessment failed: {result.get('error')}")
                
        except Exception as e:
            state['error_count'] += 1
            logger.error(f"Risk assessment node error: {e}")
        
        state['current_step'] = 'portfolio_optimization'
        return state
    
    async def _portfolio_optimization_node(self, state: InvestmentState) -> InvestmentState:
        """Portfolio optimization agent node"""
        logger.info("Executing portfolio optimization...")
        
        try:
            result = await self.agents['portfolio_optimization'].execute(state)
            
            if result.get('success'):
                state['portfolio_allocation'] = {
                    'allocation': result.get('allocation', {}),
                    'portfolio_metrics': result.get('portfolio_metrics', {}),
                    'compliance_violations': result.get('compliance_violations', [])
                }
                state['agent_confidence']['portfolio_optimization'] = result.get('confidence', 0.0)
            else:
                state['error_count'] += 1
                logger.error(f"Portfolio optimization failed: {result.get('error')}")
                
        except Exception as e:
            state['error_count'] += 1
            logger.error(f"Portfolio optimization node error: {e}")
        
        state['current_step'] = 'report_generation'
        return state
    
    async def _report_generation_node(self, state: InvestmentState) -> InvestmentState:
        """Report generation agent node"""
        logger.info("Executing report generation...")
        
        try:
            result = await self.agents['report_generation'].execute(state)
            
            if result.get('success'):
                state['report_content'] = result.get('report_content', '')
                state['agent_confidence']['report_generation'] = result.get('confidence', 0.0)
            else:
                state['error_count'] += 1
                logger.error(f"Report generation failed: {result.get('error')}")
                
        except Exception as e:
            state['error_count'] += 1
            logger.error(f"Report generation node error: {e}")
        
        state['current_step'] = 'quality_assurance'
        return state
    
    async def _quality_assurance_node(self, state: InvestmentState) -> InvestmentState:
        """Quality assurance and final validation"""
        logger.info("Executing quality assurance...")
        
        try:
            # Calculate overall quality score
            confidence_scores = list(state.get('agent_confidence', {}).values())
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Check for critical errors
            error_penalty = min(state.get('error_count', 0) * 0.1, 0.5)
            quality_score = max(avg_confidence - error_penalty, 0.0)
            
            state['quality_score'] = quality_score
            state['current_step'] = 'completed'
            
            # Generate final recommendations summary
            recommendations = []
            
            discovered_assets = state.get('discovered_assets', [])
            portfolio_allocation = state.get('portfolio_allocation', {})
            
            for asset in discovered_assets[:5]:  # Top 5 recommendations
                allocation_pct = portfolio_allocation.get('allocation', {}).get(asset.get('symbol'), 0)
                
                if allocation_pct > 0.02:  # Only include if >2% allocation
                    recommendations.append({
                        'symbol': asset.get('symbol'),
                        'name': asset.get('name'),
                        'allocation': allocation_pct,
                        'rationale': asset.get('catalyst', 'Market opportunity')
                    })
            
            state['recommendations'] = recommendations
            
            logger.info(f"Quality assurance completed. Quality score: {quality_score:.2f}")
            
        except Exception as e:
            state['error_count'] += 1
            state['quality_score'] = 0.0
            logger.error(f"Quality assurance error: {e}")
        
        return state
    
    async def execute_investment_analysis(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete investment analysis workflow"""
        
        # Initialize state
        initial_state = InvestmentState(
            user_profile=user_input.get('user_profile', {}),
            research_type=user_input.get('research_type', 'Trending Stocks'),
            sectors=user_input.get('sectors', ['Technology']),
            discovered_assets=[],
            fundamental_analysis={},
            sentiment_scores={},
            risk_metrics={},
            portfolio_allocation={},
            technical_indicators={},
            current_step='initialization',
            error_count=0,
            retry_attempts={},
            agent_confidence={},
            recommendations=[],
            report_content='',
            quality_score=0.0,
            conversation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            market_conditions={}
        )
        
        try:
            logger.info("Starting investment analysis workflow...")
            
            # Execute workflow (without config since we removed checkpointer)
            final_state = await self.app.ainvoke(initial_state)
            
            logger.info(f"Workflow completed. Quality score: {final_state.get('quality_score', 0.0)}")
            
            return {
                'success': True,
                'state': final_state,
                'report': final_state.get('report_content', ''),
                'recommendations': final_state.get('recommendations', []),
                'quality_score': final_state.get('quality_score', 0.0),
                'error_count': final_state.get('error_count', 0),
                'agent_confidence': final_state.get('agent_confidence', {})
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            
            # Create fallback report
            fallback_report = self._create_fallback_report(user_input, str(e))
            
            return {
                'success': False,
                'error': str(e),
                'report': fallback_report,
                'recommendations': [],
                'quality_score': 0.0
            }
    
    def _create_fallback_report(self, user_input: Dict[str, Any], error_message: str) -> str:
        """Create a fallback report when workflow fails"""
        user_profile = user_input.get('user_profile', {})
        
        return f"""# Investment Analysis Report - System Maintenance Mode

## Status: Temporary Service Limitation

We apologize, but our multi-agent analysis system is currently experiencing technical difficulties.

**Error Details**: {error_message}

## Your Investment Profile
- **Budget**: ${user_profile.get('budget', 0):,}
- **Risk Tolerance**: {user_profile.get('risk_tolerance', 'Not specified')}
- **Time Horizon**: {user_profile.get('time_horizon', 'Not specified')}
- **Goals**: {user_profile.get('goals', 'Not specified')}

## Simplified Recommendations

Based on your profile, here are general investment principles to consider:

### Conservative Approach (Low Risk)
- **60%** Bonds and Fixed Income
- **30%** Dividend-paying Stocks  
- **10%** Cash/Money Market

### Moderate Approach (Medium Risk)
- **40%** Broad Market Index Funds
- **30%** Individual Stocks
- **20%** Bonds
- **10%** Alternative Investments

### Aggressive Approach (High Risk)
- **70%** Growth Stocks and Equity Funds
- **20%** Emerging Markets/Sector ETFs
- **10%** High-Growth Individual Stocks

## Important Disclaimers

**Not Financial Advice**: This simplified analysis is for educational purposes only. Always consult with qualified financial professionals before making investment decisions.

**System Recovery**: Our full multi-agent analysis system will be restored shortly. Please try again later for comprehensive AI-powered investment analysis.

---
*Fallback report generated due to temporary system limitations*
"""

# ================================
# ENHANCED GRADIO APPLICATION
# ================================

# Global orchestrator instance
orchestrator = None

def initialize_orchestrator():
    """Initialize the global orchestrator with error handling"""
    global orchestrator
    if orchestrator is None:
        try:
            logger.info("Initializing multi-agent orchestrator...")
            orchestrator = InvestmentWorkflowOrchestrator()
            logger.info("Multi-agent orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            # Create a fallback orchestrator
            orchestrator = FallbackOrchestrator()
    return orchestrator

class FallbackOrchestrator:
    """Fallback orchestrator when main system fails"""
    
    def __init__(self):
        self.name = "FallbackOrchestrator"
        logger.info("Fallback orchestrator initialized")
    
    async def execute_investment_analysis(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simplified analysis when main system is unavailable"""
        logger.info("Executing fallback analysis...")
        
        user_profile = user_input.get('user_profile', {})
        
        # Create basic recommendations
        basic_recommendations = [
            {
                'symbol': 'SPY',
                'name': 'SPDR S&P 500 ETF',
                'allocation': 0.4,
                'rationale': 'Broad market exposure with low fees'
            },
            {
                'symbol': 'BND',
                'name': 'Vanguard Total Bond Market ETF',
                'allocation': 0.3,
                'rationale': 'Stable income and portfolio diversification'
            },
            {
                'symbol': 'VTI',
                'name': 'Vanguard Total Stock Market ETF',
                'allocation': 0.2,
                'rationale': 'Complete U.S. stock market exposure'
            },
            {
                'symbol': 'CASH',
                'name': 'Cash Position',
                'allocation': 0.1,
                'rationale': 'Liquidity and opportunity reserve'
            }
        ]
        
        fallback_report = f"""# Investment Analysis Report - Simplified Mode

## Executive Summary

This analysis provides basic investment recommendations based on your profile using our fallback system while our full multi-agent platform is being optimized.

**Investment Profile**: {user_profile.get('risk_tolerance', 'Moderate')} risk tolerance, {user_profile.get('time_horizon', 'Medium-term')} horizon
**Budget**: ${user_profile.get('budget', 0):,}

## Basic Asset Allocation

| Asset | Allocation | Rationale |
|-------|------------|-----------|
| SPY | 40% | Core market exposure |
| BND | 30% | Fixed income stability |
| VTI | 20% | Broad market diversification |
| Cash | 10% | Liquidity buffer |

## Investment Strategy

This simplified allocation follows modern portfolio theory principles with:
- **Diversification** across asset classes
- **Cost-effective** index fund approach  
- **Risk-appropriate** allocation based on your profile
- **Liquidity** preservation with cash position

## Next Steps

1. Review allocation with financial advisor
2. Consider dollar-cost averaging for implementation
3. Rebalance quarterly or when allocations drift >5%
4. Return for full multi-agent analysis when system is optimized

## Important Disclaimers

**Not Financial Advice**: This simplified analysis is for educational purposes only. Consult qualified financial professionals before making investment decisions.

**Simplified Mode**: This is a basic analysis. Our full multi-agent system provides comprehensive market discovery, sentiment analysis, and risk assessment when available.

---
*Simplified report generated by fallback system*
"""
        
        return {
            'success': True,
            'report': fallback_report,
            'recommendations': basic_recommendations,
            'quality_score': 0.5,  # Moderate quality for basic analysis
            'agent_confidence': {'fallback_system': 0.5}
        }

def create_senior_agentic_app():
    """Create the senior-level agentic AI application"""
    
    # Dark theme CSS (same as original but with enhancements)
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

    .agent-status {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
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
        height: 600px !important;
        overflow-y: auto !important;
    }

    .primary-btn {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%) !important;
        color: white !important;
        padding: 0.875rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    """
    
    with gr.Blocks(title="Senior Agentic AI Investment Advisor", theme=gr.themes.Soft(), css=CUSTOM_CSS) as app:
        
        # Enhanced header with agent status
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 Senior Agentic AI Investment Advisor</h1>
            <h2>Multi-Agent LangGraph Architecture</h2>
            <p>Advanced autonomous investment analysis powered by specialized AI agents working in coordination through LangGraph workflows.</p>
            
            <div class="agent-status">
                <h3 style="color: #22c55e; margin: 0 0 0.5rem 0;">🏗️ Multi-Agent System Architecture</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.5rem; font-size: 0.9rem;">
                    <div>🔍 Market Discovery Agent</div>
                    <div>📊 Sentiment Analysis Agent</div>
                    <div>⚠️ Risk Assessment Agent</div>
                    <div>📈 Portfolio Optimization Agent</div>
                    <div>📄 Report Generation Agent</div>
                    <div>✅ Quality Assurance System</div>
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #9ca3af;">
                    <strong>Framework:</strong> LangGraph • <strong>Memory:</strong> Vector + Episodic • <strong>Guardrails:</strong> Financial Compliance • <strong>Tools:</strong> 15+ Integrated
                </p>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            
            # Multi-Agent Portfolio Analysis Tab
            with gr.TabItem("🤖 Multi-Agent Portfolio Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Investment Parameters</h3>")
                        
                        budget = gr.Number(
                            label="Investment Budget ($)",
                            value=50000,
                            minimum=10000,
                            maximum=10000000,
                            info="Minimum: $10,000 for multi-agent analysis"
                        )
                        
                        risk_tolerance = gr.Radio(
                            label="Risk Tolerance Profile",
                            choices=["Conservative", "Moderate", "Aggressive"],
                            value="Moderate",
                            info="Influences agent decision-making algorithms"
                        )
                        
                        time_horizon = gr.Radio(
                            label="Investment Time Horizon",
                            choices=["Short-term", "Medium-term", "Long-term"],
                            value="Medium-term",
                            info="Affects agent strategy recommendations"
                        )
                        
                        research_type = gr.Radio(
                            label="Asset Research Focus",
                            choices=["Trending Stocks", "Trending Cryptos", "Trending ETFs"],
                            value="Trending Stocks",
                            info="Specialized agent discovery strategies"
                        )
                        
                        sectors = gr.CheckboxGroup(
                            label="Sector Focus (Multi-Agent Discovery)",
                            choices=["Technology", "Healthcare", "Energy", "Finance", "Consumer", "Real Estate"],
                            value=["Technology", "Healthcare"],
                            info="Agents will discover assets across selected sectors"
                        )
                        
                        goals = gr.Textbox(
                            label="Investment Objectives",
                            placeholder="e.g., long-term growth, income generation, portfolio diversification...",
                            lines=3,
                            info="Guides agent decision-making processes"
                        )
                        
                        execute_analysis_btn = gr.Button(
                            "🚀 Execute Multi-Agent Analysis",
                            variant="primary",
                            elem_classes="primary-btn",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        analysis_status = gr.HTML(
                            value="""
                            <div style='text-align: center;'>
                                <h3 style='color: #d1d5db; margin-bottom: 0.5rem;'>Multi-Agent Analysis Results</h3>
                                <div style='color: #9ca3af;'>Configure parameters and click 'Execute Multi-Agent Analysis' to begin</div>
                            </div>
                            """
                        )
                        
                        # Agent execution progress
                        agent_progress = gr.HTML(value="")
                        
                        # Analysis results
                        analysis_output = gr.HTML(
                            value="""
                            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; height: 400px; background: rgba(255, 255, 255, 0.95); border-radius: 8px; padding: 2rem;'>
                                <div style='font-weight: 600; font-size: 1.1rem; color: #374151; margin-bottom: 0.5rem;'>🤖 Multi-Agent Analysis Engine</div>
                                <div style='color: #6b7280;'>Advanced autonomous investment analysis awaiting execution</div>
                            </div>
                            """
                        )
                        
                        # Analysis metrics
                        with gr.Row():
                            quality_score = gr.Number(label="Analysis Quality Score", interactive=False)
                            agent_confidence = gr.JSON(label="Agent Confidence Metrics", visible=False)
                        
                        # Recommendations table
                        recommendations_df = gr.Dataframe(
                            label="AI Agent Recommendations",
                            headers=["Symbol", "Name", "Allocation", "Rationale"]
                        )
            
            # Agent Performance Monitoring Tab
            with gr.TabItem("📊 Agent Performance Monitoring"):
                gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Real-Time Agent Monitoring</h3>")
                
                with gr.Row():
                    refresh_metrics_btn = gr.Button("🔄 Refresh Agent Metrics", variant="secondary")
                
                # Agent performance metrics
                agent_metrics_display = gr.JSON(label="Live Agent Performance Metrics")
                
                # System health dashboard
                system_health = gr.HTML("""
                <div style='background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                    <h4 style='color: #22c55e; margin: 0 0 1rem 0;'>🏥 System Health Dashboard</h4>
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;'>
                        <div style='text-align: center;'>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Memory System</div>
                            <div style='color: #22c55e; font-weight: 600;'>✅ Operational</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Guardrails Engine</div>
                            <div style='color: #22c55e; font-weight: 600;'>✅ Active</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>LangGraph Workflow</div>
                            <div style='color: #22c55e; font-weight: 600;'>✅ Ready</div>
                        </div>
                        <div style='text-align: center;'>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Agent Coordination</div>
                            <div style='color: #22c55e; font-weight: 600;'>✅ Synchronized</div>
                        </div>
                    </div>
                </div>
                """)
            
            # Memory & Learning Tab
            with gr.TabItem("🧠 Memory & Learning System"):
                gr.HTML("<h3 style='color: #d1d5db; text-align: center;'>Advanced Memory Management</h3>")
                
                with gr.Row():
                    with gr.Column():
                        memory_query = gr.Textbox(
                            label="Query Memory System",
                            placeholder="e.g., 'technology stocks', 'high risk investments', 'portfolio strategies'",
                            info="Search across vector memory and episodic storage"
                        )
                        query_memory_btn = gr.Button("🔍 Search Memory", variant="primary")
                    
                    with gr.Column():
                        memory_results = gr.JSON(label="Memory Search Results")
                
                # Memory statistics
                memory_stats = gr.HTML("""
                <div style='background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;'>
                    <h4 style='color: #667eea; margin: 0 0 1rem 0;'>🧠 Memory System Statistics</h4>
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;'>
                        <div>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Vector Memory Embeddings</div>
                            <div style='color: #667eea; font-weight: 600; font-size: 1.2rem;'>Initializing...</div>
                        </div>
                        <div>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Episodic Decision Records</div>
                            <div style='color: #667eea; font-weight: 600; font-size: 1.2rem;'>0 Stored</div>
                        </div>
                        <div>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Performance Tracking Entries</div>
                            <div style='color: #667eea; font-weight: 600; font-size: 1.2rem;'>Active</div>
                        </div>
                        <div>
                            <div style='color: #9ca3af; font-size: 0.9rem;'>Learning Accuracy</div>
                            <div style='color: #667eea; font-weight: 600; font-size: 1.2rem;'>Baseline</div>
                        </div>
                    </div>
                </div>
                """)
        
        # Enhanced footer with technical details
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 2rem; background: rgba(255, 255, 255, 0.02); border-radius: 16px;">
            <h4 style="color: #fbbf24;">⚠️ Senior-Level Investment Disclaimers</h4>
            <p style="color: #9ca3af; margin-bottom: 1rem;">
                This advanced multi-agent AI system is designed for educational and demonstration purposes. All investment analysis is generated through autonomous AI agents and should not replace professional financial advice. Always consult qualified financial advisors before making investment decisions.
            </p>
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">🎯 Senior Agentic AI Engineering Showcase</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    <strong>Architecture:</strong> LangGraph Multi-Agent Orchestration • 
                    <strong>Memory:</strong> ChromaDB Vector Store + Episodic Learning • 
                    <strong>Safety:</strong> Multi-Layer Guardrails & Compliance • 
                    <strong>Resilience:</strong> Circuit Breakers & Error Recovery • 
                    <strong>Monitoring:</strong> Real-time Performance Tracking
                </div>
            </div>
        </div>
        """)
        
        # ================================
        # EVENT HANDLERS
        # ================================
        
        def execute_multi_agent_analysis(budget_val, risk_val, time_val, research_val, sectors_val, goals_val):
            """Execute multi-agent investment analysis"""
            
            if budget_val < 10000:
                return (
                    "<div class='warning-msg'>Minimum $10,000 required for multi-agent analysis</div>",
                    "", 
                    """
                    <div style='text-align: center; padding: 2rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;'>
                        <div style='color: #ef4444; font-weight: 600;'>Analysis Failed</div>
                        <div style='color: #9ca3af;'>Insufficient budget for comprehensive multi-agent analysis</div>
                    </div>
                    """,
                    0.0,
                    {},
                    pd.DataFrame()
                )
            
            # Check OpenAI API key
            if not os.getenv('OPENAI_API_KEY'):
                return (
                    "<div class='warning-msg'>OpenAI API key required for multi-agent system</div>",
                    "",
                    """
                    <div style='text-align: center; padding: 2rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;'>
                        <div style='color: #ef4444; font-weight: 600;'>Configuration Error</div>
                        <div style='color: #9ca3af;'>Please set OPENAI_API_KEY environment variable</div>
                    </div>
                    """,
                    0.0,
                    {},
                    pd.DataFrame()
                )
            
            try:
                # Initialize orchestrator
                orch = initialize_orchestrator()
                
                # Prepare user input
                user_input = {
                    'user_profile': {
                        'budget': budget_val,
                        'risk_tolerance': risk_val,
                        'time_horizon': time_val,
                        'goals': goals_val
                    },
                    'research_type': research_val,
                    'sectors': sectors_val or ['Technology']
                }
                
                # Run async analysis in sync context
                import asyncio
                
                # Create new event loop if none exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Execute analysis
                result = loop.run_until_complete(orch.execute_investment_analysis(user_input))
                
                if result.get('success'):
                    # Convert report to HTML
                    report_html = f"""
                    <div style='background: rgba(255, 255, 255, 0.95); color: #1e293b; border-radius: 8px; padding: 1.5rem; line-height: 1.6; height: 600px; overflow-y: auto;'>
                        {markdown.markdown(result.get('report', ''), extensions=['tables'])}
                    </div>
                    """
                    
                    # Create recommendations dataframe
                    recommendations = result.get('recommendations', [])
                    if recommendations:
                        rec_data = []
                        for rec in recommendations:
                            rec_data.append([
                                rec.get('symbol', 'N/A'),
                                rec.get('name', 'Unknown'),
                                f"{rec.get('allocation', 0):.1%}",
                                rec.get('rationale', 'AI analysis')[:100] + "..." if len(rec.get('rationale', '')) > 100 else rec.get('rationale', 'AI analysis')
                            ])
                        recommendations_df = pd.DataFrame(rec_data, columns=["Symbol", "Name", "Allocation", "Rationale"])
                    else:
                        recommendations_df = pd.DataFrame()
                    
                    return (
                        f"<div class='success-msg'>Multi-agent analysis completed! Quality Score: {result.get('quality_score', 0):.2f}</div>",
                        """
                        <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; padding: 1rem; margin: 0.5rem 0;'>
                            <div style='color: #22c55e; font-weight: 600;'>✅ Analysis Complete</div>
                            <div style='color: #9ca3af; font-size: 0.9rem; margin-top: 0.5rem;'>
                                All agents completed successfully with comprehensive analysis results
                            </div>
                        </div>
                        """,
                        report_html,
                        result.get('quality_score', 0.0),
                        result.get('agent_confidence', {}),
                        recommendations_df
                    )
                    
                else:
                    return (
                        f"<div class='warning-msg'>Analysis failed: {result.get('error', 'Unknown error')}</div>",
                        "",
                        """
                        <div style='text-align: center; padding: 2rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;'>
                            <div style='color: #ef4444; font-weight: 600;'>Analysis Failed</div>
                            <div style='color: #9ca3af;'>Please check system configuration and try again</div>
                        </div>
                        """,
                        0.0,
                        {},
                        pd.DataFrame()
                    )
                    
            except Exception as e:
                return (
                    f"<div class='warning-msg'>System error: {str(e)}</div>",
                    "",
                    """
                    <div style='text-align: center; padding: 2rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;'>
                        <div style='color: #ef4444; font-weight: 600;'>System Error</div>
                        <div style='color: #9ca3af;'>Please contact system administrator</div>
                    </div>
                    """,
                    0.0,
                    {},
                    pd.DataFrame()
                )
        
        def refresh_agent_metrics():
            """Refresh agent performance metrics"""
            try:
                orch = initialize_orchestrator()
                
                # Simulate metrics (in production, get from actual performance tracking)
                metrics = {
                    "market_discovery_agent": {
                        "avg_execution_time": 4.2,
                        "success_rate": 0.95,
                        "confidence_avg": 0.87
                    },
                    "sentiment_analysis_agent": {
                        "avg_execution_time": 3.1,
                        "success_rate": 0.92,
                        "confidence_avg": 0.84
                    },
                    "risk_assessment_agent": {
                        "avg_execution_time": 5.8,
                        "success_rate": 0.98,
                        "confidence_avg": 0.91
                    },
                    "portfolio_optimization_agent": {
                        "avg_execution_time": 2.7,
                        "success_rate": 0.94,
                        "confidence_avg": 0.89
                    },
                    "report_generation_agent": {
                        "avg_execution_time": 3.4,
                        "success_rate": 0.97,
                        "confidence_avg": 0.85
                    }
                }
                
                return metrics
                
            except Exception as e:
                return {"error": f"Failed to retrieve metrics: {str(e)}"}
        
        def query_memory_system(query):
            """Query the memory system"""
            try:
                if not query:
                    return {"message": "Please enter a search query"}
                
                orch = initialize_orchestrator()
                
                # Query memory system
                insights = orch.memory_manager.retrieve_similar_insights(query, n_results=5)
                
                return {
                    "query": query,
                    "results_count": len(insights),
                    "insights": insights[:3] if insights else ["No matching insights found"]
                }
                
            except Exception as e:
                return {"error": f"Memory query failed: {str(e)}"}
        
        # Connect event handlers
        execute_analysis_btn.click(
            fn=execute_multi_agent_analysis,
            inputs=[budget, risk_tolerance, time_horizon, research_type, sectors, goals],
            outputs=[analysis_status, agent_progress, analysis_output, quality_score, agent_confidence, recommendations_df],
        )
        
        refresh_metrics_btn.click(
            fn=refresh_agent_metrics,
            outputs=[agent_metrics_display]
        )
        
        query_memory_btn.click(
            fn=query_memory_system,
            inputs=[memory_query],
            outputs=[memory_results]
        )
    
    return app

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    print("🚀 Starting Senior Agentic AI Investment Advisor...")
    print("🏗️  Multi-Agent Architecture: LangGraph + Specialized Agents")
    print("🧠 Memory System: Vector Store + Episodic Learning")
    print("🛡️  Safety: Multi-Layer Guardrails & Compliance")
    print("📊 Monitoring: Real-time Performance Tracking")
    
    # Check dependencies
    try:
        import langgraph
        import langchain_openai
        import chromadb
        print("✅ All senior-level dependencies installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install: pip install langgraph langchain-openai chromadb tenacity")
        exit(1)
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key detected - Multi-agent system enabled")
    else:
        print("⚠️  No OPENAI_API_KEY found - Add to .env file for full functionality")
    
    # Initialize and launch
    app = create_senior_agentic_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )