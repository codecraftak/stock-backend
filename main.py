from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import yfinance as yf
yf.set_tz_cache_location("/tmp/yfinance_cache")
from bs4 import BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add environment-based server config
import os
if os.getenv("RENDER"):  # Render sets this automatically
    app = FastAPI(
        servers=[
            {"url": "https://stock-backend-l55g.onrender.com"}
        ]
    )
else:
    app = FastAPI()

import requests
requests.packages.urllib3.disable_warnings()

# ===================== CORS CONFIGURATION =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stock-prediction-analysis-report.netlify.app",
        "https://stock-backend-l55g.onrender.com"
        ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== API CONFIGURATION =====================
FREE_API_KEYS = {
    'gemini': os.getenv('GEMINI_API_KEY', ''),
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', ''),
    'finnhub': os.getenv('FINNHUB_KEY', ''),
    'news_api': os.getenv('NEWS_API_KEY', ''),
}

# Common stock name to symbol mapping
COMMON_STOCKS = {
    # Indian Stocks
    'tcs': 'TCS.NS',
    'reliance': 'RELIANCE.NS',
    'infosys': 'INFY.NS',
    'hdfc': 'HDFCBANK.NS',
    'icici': 'ICICIBANK.NS',
    'sbi': 'SBIN.NS',
    'bharti': 'BHARTIARTL.NS',
    'airtel': 'BHARTIARTL.NS',
    'wipro': 'WIPRO.NS',
    'itc': 'ITC.NS',
    'adani': 'ADANIENT.NS',
    'tata': 'TATAMOTORS.NS',
    'bajaj': 'BAJFINANCE.NS',
    'mahindra': 'M&M.NS',
    'hindalco': 'HINDALCO.NS',
    'olaelec': 'OLAELEC.NS',
    'ola': 'OLAELEC.NS',
    
    # US Stocks
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'meta': 'META',
    'facebook': 'META',
    'nvidia': 'NVDA',
    'netflix': 'NFLX',
    'adobe': 'ADBE',
    'intel': 'INTC',
    'amd': 'AMD',
    'cisco': 'CSCO',
    'oracle': 'ORCL',
    'ibm': 'IBM',
    'salesforce': 'CRM',
    'paypal': 'PYPL',
    'visa': 'V',
    'mastercard': 'MA',
    'coca cola': 'KO',
    'pepsi': 'PEP',
    'walmart': 'WMT',
    'disney': 'DIS',
    'nike': 'NKE',
    'starbucks': 'SBUX',
    'mcdonalds': 'MCD',
    'boeing': 'BA',
    'ge': 'GE',
    'general electric': 'GE',
    'ford': 'F',
    'gm': 'GM',
    'general motors': 'GM',
    'jpmorgan': 'JPM',
    'bank of america': 'BAC',
    'wells fargo': 'WFC',
    'goldman sachs': 'GS',
    'morgan stanley': 'MS',
}

print("\n" + "="*80)
print("📊 FREE APIs STATUS")
print("="*80)
for api_name, key in FREE_API_KEYS.items():
    print(f"{api_name}: {'✅' if key else '❌'}")
print("="*80 + "\n")

executor = ThreadPoolExecutor(max_workers=10)

# ===================== MODELS =====================

class StockRequest(BaseModel):
    stock_name: str

class TechnicalIndicators(BaseModel):
    rsi: Optional[str] = None
    macd: Optional[str] = None
    sma_20: Optional[str] = None
    sma_50: Optional[str] = None

class NewsArticle(BaseModel):
    title: str
    source: str
    url: str
    published_at: Optional[str] = None

class AIModelAnalysis(BaseModel):
    model: str
    recommendation: str
    confidence: int
    reasoning: str
    key_points: List[str]

class EnhancedStockAnalysis(BaseModel):
    stockName: str
    symbol: str
    currentPrice: str
    data_sources_used: List[str]
    dayHigh: Optional[str] = None
    dayLow: Optional[str] = None
    week52High: Optional[str] = None
    week52Low: Optional[str] = None
    volume: Optional[str] = None
    marketCap: Optional[str] = None
    peRatio: Optional[str] = None
    pegRatio: Optional[str] = None
    priceToBook: Optional[str] = None
    debtToEquity: Optional[str] = None
    roe: Optional[str] = None
    eps: Optional[str] = None
    dividendYield: Optional[str] = None
    beta: Optional[str] = None
    technical_indicators: TechnicalIndicators
    credible_news: List[NewsArticle]
    overall_sentiment: str
    sentiment_score: Optional[float] = None
    ai_analyses: List[AIModelAnalysis]
    consensus_recommendation: str
    consensus_confidence: int
    bullish_factors: List[str]
    bearish_factors: List[str]
    risk_factors: List[str]
    opportunity_factors: List[str]
    top_institutional_holders: List[str]
    insider_transactions: Optional[str] = None
    summary: str
    actionable_insights: List[str]
    analysis_timestamp: str
    api_calls_used: int
    debug_info: Optional[Dict] = None


# ===================== SYMBOL RESOLVER =====================

def resolve_stock_symbol(user_input: str) -> List[str]:
    """Convert user-friendly stock names to ticker symbols"""
    user_input_lower = user_input.strip().lower()
    
    # Check if it's in common stocks dictionary first
    if user_input_lower in COMMON_STOCKS:
        symbol = COMMON_STOCKS[user_input_lower]
        print(f"   🔍 Resolved '{user_input}' → '{symbol}'")
        return [symbol]
    
    # If it looks like a ticker symbol already
    if user_input.isupper() and len(user_input) <= 5:
        return [user_input, f"{user_input}.NS", f"{user_input}.BO"]
    
    # Try variations of the input as ticker
    variations = [
        user_input.upper(),
        f"{user_input.upper()}.NS",
        f"{user_input.upper()}.BO",
    ]
    
    return variations


# ===================== DATA FETCHERS =====================

def fetch_yfinance_data(symbol: str) -> Dict:
    """Fetch comprehensive data from Yahoo Finance"""
    print(f"📊 Fetching from Yahoo Finance for: {symbol}")

    # Get symbol variations
    if '.' in symbol or symbol.isupper():
        variations = [symbol]
    else:
        # Resolve the symbol
        variations = resolve_stock_symbol(symbol)
    
    for ticker_symbol in variations:
        try:
            print(f"   🔍 Trying: {ticker_symbol}")

            stock = yf.Ticker(ticker_symbol)
            
            try:
                info = stock.info
            except:
                try:
                    info = stock.get_info()
                except:
                    print(f" ⚠️ Could not fetch info for {ticker_symbol}")
                    continue
            
            #validate info
            if not info or (isinstance(info, dict) and len(info) < 3):
                print(f"   ⚠️ invalid/empty info for {ticker_symbol}")
                continue

            #get current price            
            current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or
                info.get('previousClose')
            )
            
            try:
                current_price = float(current_price)
                if current_price <= 0:
                    print(f"No valid price for {ticker_symbol}")
                    continue
            except (TypeError, ValueError):
                print(f"Invalid price format for {ticker_symbol}")
                continue

            #get history
            try:
                hist = stock.history(period="1y")
            except:
                try:
                    hist=stock.history(period="1mo")
                except:
                    hist=None
                
            data = {
                'symbol': ticker_symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'price': current_price,
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'history': hist,
                'info': info,
                'institutional_holders': stock.institutional_holders,
                'news': stock.news[:10] if stock.news else []
            }
            # Try to get additional data
            try:
                data['institutional_holders'] = stock.institutional_holders
            except:
                pass
            
            try:
                data['news'] = stock.news[:10] if stock.news else []
            except:
                pass
                
            print(f"   ✅ Yahoo Finance: {ticker_symbol} - {data['name']}")
            return data
                
        except Exception as e:
            print(f"   ⚠️ {ticker_symbol}: {str(e)[:100]}")
            continue
    
    return None


def fetch_alpha_vantage_indicators(symbol: str) -> Dict:
    """Fetch technical indicators from Alpha Vantage"""
    if not FREE_API_KEYS['alpha_vantage']:
        return {}
    
    print("📈 Fetching technical indicators from Alpha Vantage...")
    
    base_symbol = symbol.split('.')[0]
    
    indicators = {}
    base_url = "https://www.alphavantage.co/query"
    
    try:
        params = {
            'function': 'RSI',
            'symbol': base_symbol,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': FREE_API_KEYS['alpha_vantage']
        }
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Technical Analysis: RSI' in data:
                latest = list(data['Technical Analysis: RSI'].values())[0]
                indicators['rsi'] = f"{float(latest['RSI']):.2f}"
                print(f"   ✅ RSI: {indicators['rsi']}")
        
    except Exception as e:
        print(f"   ⚠️ Alpha Vantage error: {e}")
    
    return indicators


def fetch_finnhub_news(symbol: str) -> List[Dict]:
    """Fetch news from Finnhub"""
    if not FREE_API_KEYS['finnhub']:
        return []
    
    print("📰 Fetching news from Finnhub...")
    
    base_symbol = symbol.split('.')[0]
    
    articles = []
    
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': base_symbol,
            'from': from_date,
            'to': to_date,
            'token': FREE_API_KEYS['finnhub']
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            news_data = response.json()
            
            for article in news_data[:10]:
                articles.append({
                    'title': article.get('headline', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat()
                })
            
            print(f"   ✅ Finnhub: {len(articles)} articles")
    
    except Exception as e:
        print(f"   ⚠️ Finnhub error: {e}")
    
    return articles


def fetch_news_api_articles(query: str) -> List[Dict]:
    """Fetch news from NewsAPI"""
    if not FREE_API_KEYS['news_api']:
        return []
    
    print("📰 Fetching from NewsAPI...")
    
    articles = []
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{query} stock",
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': FREE_API_KEYS['news_api']
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for article in data.get('articles', [])[:10]:
                articles.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', '')
                })
            
            print(f"   ✅ NewsAPI: {len(articles)} articles")
    
    except Exception as e:
        print(f"   ⚠️ NewsAPI error: {e}")
    
    return articles


# ===================== AI ANALYSIS =====================

async def analyze_with_gemini(prompt: str) -> Dict:
    """Analyze with Google Gemini"""
    if not FREE_API_KEYS['gemini']:
        return {"error": "Gemini API not configured"}
    
    try:
        print("🤖 Analyzing with Gemini...")
        
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash-exp:generateContent"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(
            f"{url}?key={FREE_API_KEYS['gemini']}",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            print("   ✅ Gemini analysis complete")
            return {"success": True, "analysis": text}
        else:
            return {"error": f"Status {response.status_code}"}
    
    except Exception as e:
        print(f"   ❌ Gemini error: {e}")
        return {"error": str(e)}


def create_analysis_prompt(stock_data: Dict, news: List[Dict], indicators: Dict) -> str:
    """Create comprehensive prompt for AI analysis"""
    
    news_text = "\n".join([
        f"- {article['title']} ({article['source']})"
        for article in news[:10]
    ])
    
    prompt = f"""You are a professional stock analyst. Analyze {stock_data['name']} ({stock_data['symbol']}).

CURRENT DATA:
- Price: {stock_data['currency']} {stock_data['price']:.2f}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- P/B Ratio: {stock_data.get('price_to_book', 'N/A')}
- ROE: {stock_data.get('roe', 'N/A')}
- Debt/Equity: {stock_data.get('debt_to_equity', 'N/A')}
- Sector: {stock_data.get('sector', 'N/A')}
- Industry: {stock_data.get('industry', 'N/A')}

TECHNICAL INDICATORS:
{json.dumps(indicators, indent=2)}

RECENT NEWS (Last 7 days):
{news_text}

Provide analysis in this EXACT JSON format:
{{
  "recommendation": "BUY/HOLD/SELL",
  "confidence": 75,
  "reasoning": "Detailed reasoning based on data and news",
  "key_points": [
    "Specific point 1",
    "Specific point 2",
    "Specific point 3"
  ],
  "bullish_factors": ["factor1", "factor2", "factor3"],
  "bearish_factors": ["factor1", "factor2"],
  "risk_factors": ["risk1", "risk2"],
  "opportunities": ["opp1", "opp2"]
}}

Return ONLY valid JSON, no markdown, no code blocks."""
    
    return prompt


# ===================== MAIN ENDPOINT =====================

@app.post("/analyze")
async def analyze_stock(request: StockRequest):
    """Complete FREE stock analysis using multiple APIs"""
    
    stock_name_final = request.stock_name.strip()
    if not stock_name_final:
        raise HTTPException(status_code=400, detail="Stock name required")
    
    api_calls = 0
    data_sources = []
    
    try:
        print("\n" + "="*80)
        print(f"🔬 ENHANCED FREE ANALYSIS: {stock_name_final}")
        print("="*80)
        
        # Step 1: Yahoo Finance
        yf_data = fetch_yfinance_data(stock_name_final)
        if not yf_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Stock '{stock_name_final}' not found. Try using the ticker symbol (e.g., AAPL, TCS.NS)"
            )
        
        data_sources.append("Yahoo Finance")
        api_calls += 1
        
        # Step 2: Technical Indicators
        indicators = fetch_alpha_vantage_indicators(yf_data['symbol'])
        if indicators:
            data_sources.append("Alpha Vantage")
            api_calls += 1
        
        # Step 3: News from multiple sources
        news_articles = []
        
        finnhub_news = fetch_finnhub_news(yf_data['symbol'])
        if finnhub_news:
            news_articles.extend(finnhub_news)
            data_sources.append("Finnhub")
            api_calls += 1
        
        news_api_articles = fetch_news_api_articles(yf_data['name'])
        if news_api_articles:
            news_articles.extend(news_api_articles)
            data_sources.append("NewsAPI")
            api_calls += 1
        
        yf_news = [
            {
                'title': article.get('title', ''),
                'source': article.get('publisher', 'Yahoo Finance'),
                'url': article.get('link', ''),
                'published_at': datetime.fromtimestamp(
                    article.get('providerPublishTime', 0)
                ).isoformat()
            }
            for article in yf_data.get('news', [])[:5]
        ]
        news_articles.extend(yf_news)
        
        # Step 4: Multi-AI Analysis
        print("\n🤖 Running Multi-AI Analysis...")
        
        analysis_prompt = create_analysis_prompt(yf_data, news_articles, indicators)
        
        ai_tasks = []
        if FREE_API_KEYS['gemini']:
            ai_tasks.append(analyze_with_gemini(analysis_prompt))
        
        if not ai_tasks:
            raise HTTPException(
                status_code=500,
                detail="No AI API configured. Add GEMINI_API_KEY to .env"
            )
        
        ai_results = await asyncio.gather(*ai_tasks)
        api_calls += len(ai_tasks)
        
        # Step 5: Process AI results
        ai_analyses = []
        recommendations = []
        all_bullish = []
        all_bearish = []
        all_risks = []
        all_opportunities = []
        
        for i, result in enumerate(ai_results):
            if "success" in result:
                try:
                    analysis_text = result['analysis']
                    json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                    if json_match:
                        analysis_json = json.loads(json_match.group())
                        
                        ai_analyses.append(AIModelAnalysis(
                            model="Gemini",
                            recommendation=analysis_json.get('recommendation', 'HOLD'),
                            confidence=analysis_json.get('confidence', 50),
                            reasoning=analysis_json.get('reasoning', '')[:500],
                            key_points=analysis_json.get('key_points', [])[:5]
                        ))
                        
                        recommendations.append(analysis_json.get('recommendation', 'HOLD'))
                        all_bullish.extend(analysis_json.get('bullish_factors', []))
                        all_bearish.extend(analysis_json.get('bearish_factors', []))
                        all_risks.extend(analysis_json.get('risk_factors', []))
                        all_opportunities.extend(analysis_json.get('opportunities', []))
                        
                        data_sources.append("Gemini AI")
                        
                except Exception as e:
                    print(f"   ⚠️ Error parsing AI response: {e}")
        
        # Fallback manual analysis if AI fails
        if not all_bullish and not all_bearish:
            print("   ⚠️ Using fallback manual analysis...")
            
            if yf_data.get('pe_ratio') and yf_data['pe_ratio'] < 25:
                all_bullish.append(f"Reasonable P/E ratio of {yf_data['pe_ratio']:.2f}")
            
            if yf_data.get('roe') and yf_data['roe'] > 0.15:
                all_bullish.append(f"Strong ROE of {yf_data['roe']*100:.2f}%")
            
            if yf_data.get('debt_to_equity') and yf_data['debt_to_equity'] < 1.0:
                all_bullish.append(f"Low debt-to-equity of {yf_data['debt_to_equity']:.2f}")
            
            if yf_data.get('pe_ratio') and yf_data['pe_ratio'] > 40:
                all_bearish.append(f"High P/E ratio of {yf_data['pe_ratio']:.2f}")
            
            if yf_data.get('debt_to_equity') and yf_data['debt_to_equity'] > 2.0:
                all_bearish.append(f"High debt of {yf_data['debt_to_equity']:.2f}")
            
            all_risks.append("Market volatility risks")
            all_opportunities.append("Long-term growth potential")
        
        # Calculate sentiment
        positive_words = ['bullish', 'growth', 'strong', 'positive', 'buy', 'upgrade']
        negative_words = ['bearish', 'decline', 'weak', 'negative', 'sell', 'downgrade']
        
        positive_count = sum(
            1 for article in news_articles
            if any(word in article['title'].lower() for word in positive_words)
        )
        negative_count = sum(
            1 for article in news_articles
            if any(word in article['title'].lower() for word in negative_words)
        )
        
        total_sentiment = positive_count + negative_count
        if total_sentiment > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment
            if sentiment_score > 0.3:
                overall_sentiment = "Positive"
            elif sentiment_score < -0.3:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Neutral"
            sentiment_score = 0.0
        
        # Advanced Scoring
        score = 0
        factors_checked = []
        
        # P/E Ratio (15 points)
        if yf_data.get('pe_ratio') and yf_data['pe_ratio'] > 0:
            pe = yf_data['pe_ratio']
            if pe < 15:
                score += 15
                factors_checked.append(f"✅ Excellent P/E ({pe:.2f})")
            elif pe < 25:
                score += 10
                factors_checked.append(f"✅ Good P/E ({pe:.2f})")
            elif pe < 40:
                score += 5
                factors_checked.append(f"⚠️ Moderate P/E ({pe:.2f})")
            else:
                score += 2
                factors_checked.append(f"❌ High P/E ({pe:.2f})")
        else:
            score += 7
            factors_checked.append("⚠️ P/E not available")
        
        # Debt/Equity (15 points)
        if yf_data.get('debt_to_equity') and yf_data['debt_to_equity'] >= 0:
            de = yf_data['debt_to_equity']
            if de < 0.5:
                score += 15
                factors_checked.append(f"✅ Low Debt ({de:.2f})")
            elif de < 1.0:
                score += 10
                factors_checked.append(f"✅ Good Debt ({de:.2f})")
            elif de < 2.0:
                score += 5
                factors_checked.append(f"⚠️ Moderate Debt ({de:.2f})")
            else:
                score += 2
                factors_checked.append(f"❌ High Debt ({de:.2f})")
        else:
            score += 7
            factors_checked.append("⚠️ Debt data not available")
        
        # ROE (15 points)
        if yf_data.get('roe') is not None:
            roe = yf_data['roe'] * 100
            if roe > 20:
                score += 15
                factors_checked.append(f"✅ Excellent ROE ({roe:.1f}%)")
            elif roe > 15:
                score += 10
                factors_checked.append(f"✅ Good ROE ({roe:.1f}%)")
            elif roe > 10:
                score += 5
                factors_checked.append(f"⚠️ Average ROE ({roe:.1f}%)")
            elif roe > 0:
                score += 3
                factors_checked.append(f"⚠️ Low ROE ({roe:.1f}%)")
            else:
                score += 0
                factors_checked.append(f"❌ Negative ROE ({roe:.1f}%)")
        else:
            score += 7
            factors_checked.append("⚠️ ROE not available")
        
        # Price Movement (15 points)
        if yf_data.get('week_52_high') and yf_data.get('week_52_low') and yf_data.get('price'):
            current = yf_data['price']
            high_52 = yf_data['week_52_high']
            low_52 = yf_data['week_52_low']
            
            if high_52 > low_52:
                position = (current - low_52) / (high_52 - low_52) * 100
                
                if position < 30:
                    score += 15
                    factors_checked.append(f"✅ Near 52W Low ({position:.0f}%)")
                elif position < 50:
                    score += 12
                    factors_checked.append(f"✅ Good range ({position:.0f}%)")
                elif position < 70:
                    score += 8
                    factors_checked.append(f"⚠️ Mid-range ({position:.0f}%)")
                elif position < 90:
                    score += 5
                    factors_checked.append(f"⚠️ Near high ({position:.0f}%)")
                else:
                    score += 3
                    factors_checked.append(f"❌ At 52W high ({position:.0f}%)")
        else:
            score += 7
            factors_checked.append("⚠️ Price range not available")
        
        # News Sentiment (20 points)
        if sentiment_score > 0.3:
            score += 20
            factors_checked.append(f"✅ Very Positive News")
        elif sentiment_score > 0:
            score += 15
            factors_checked.append(f"✅ Positive News")
        elif sentiment_score > -0.3:
            score += 10
            factors_checked.append(f"⚠️ Neutral News")
        else:
            score += 5
            factors_checked.append(f"❌ Negative News")
        
        # AI Consensus (20 points)
        if recommendations:
            buy_count = recommendations.count('BUY')
            hold_count = recommendations.count('HOLD')
            sell_count = recommendations.count('SELL')
            
            if buy_count > sell_count and buy_count >= hold_count:
                score += 20
                factors_checked.append(f"✅ AI Bullish")
            elif hold_count > sell_count:
                score += 15
                factors_checked.append(f"✅ AI Neutral")
            else:
                score += 5
                factors_checked.append(f"❌ AI Bearish")
        else:
            score += 10
            factors_checked.append("⚠️ No AI analysis")
        
        # Final recommendation
        if score >= 75:
            consensus_recommendation = "BUY"
            confidence_level = "Strong"
        elif score >= 60:
            consensus_recommendation = "HOLD"
            confidence_level = "Moderate-Strong"
        elif score >= 45:
            consensus_recommendation = "HOLD"
            confidence_level = "Moderate"
        elif score >= 30:
            consensus_recommendation = "SELL"
            confidence_level = "Moderate"
        else:
            consensus_recommendation = "SELL"
            confidence_level = "Strong"
        
        consensus_confidence = score
        
        print(f"\n📊 SCORE: {score}/100")
        print(f"📊 RECOMMENDATION: {consensus_recommendation}")
        
        # Format currency
        currency = yf_data.get('currency', 'USD')
        currency_symbol = '₹' if currency == 'INR' else '$'
        current_price = f"{currency_symbol}{yf_data['price']:.2f}"
        
        # Prepare holders
        top_holders = []
        if yf_data.get('institutional_holders') is not None:
            holders_df = yf_data['institutional_holders']
            if not holders_df.empty:
                for _, row in holders_df.head(5).iterrows():
                    top_holders.append(
                        f"{row.get('Holder', 'Unknown')}: {row.get('Shares', 0):,} shares"
                    )
        
        # Create final analysis
        analysis = EnhancedStockAnalysis(
            stockName=yf_data['name'],
            symbol=yf_data['symbol'],
            currentPrice=current_price,
            data_sources_used=list(set(data_sources)),
            
            dayHigh=f"{currency_symbol}{yf_data.get('day_high', 0):.2f}" if yf_data.get('day_high') else None,
            dayLow=f"{currency_symbol}{yf_data.get('day_low', 0):.2f}" if yf_data.get('day_low') else None,
            week52High=f"{currency_symbol}{yf_data.get('week_52_high', 0):.2f}" if yf_data.get('week_52_high') else None,
            week52Low=f"{currency_symbol}{yf_data.get('week_52_low', 0):.2f}" if yf_data.get('week_52_low') else None,
            volume=f"{yf_data.get('volume', 0):,}" if yf_data.get('volume') else None,
            marketCap=f"{yf_data.get('market_cap', 0):,}" if yf_data.get('market_cap') else None,
            
            peRatio=f"{yf_data.get('pe_ratio', 0):.2f}" if yf_data.get('pe_ratio') else None,
            pegRatio=f"{yf_data.get('peg_ratio', 0):.2f}" if yf_data.get('peg_ratio') else None,
            priceToBook=f"{yf_data.get('price_to_book', 0):.2f}" if yf_data.get('price_to_book') else None,
            debtToEquity=f"{yf_data.get('debt_to_equity', 0):.2f}" if yf_data.get('debt_to_equity') else None,
            roe=f"{yf_data.get('roe', 0)*100:.2f}%" if yf_data.get('roe') else None,
            eps=f"{yf_data.get('eps', 0):.2f}" if yf_data.get('eps') else None,
            dividendYield=f"{yf_data.get('dividend_yield', 0)*100:.2f}%" if yf_data.get('dividend_yield') else None,
            beta=f"{yf_data.get('beta', 0):.2f}" if yf_data.get('beta') else None,
            
            technical_indicators=TechnicalIndicators(**indicators),
            
            credible_news=[
                NewsArticle(**article) for article in news_articles[:20]
            ],
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            
            ai_analyses=ai_analyses,
            consensus_recommendation=consensus_recommendation,
            consensus_confidence=consensus_confidence,
            
            bullish_factors=list(set(all_bullish))[:10] if all_bullish else ["Company has stable fundamentals"],
            bearish_factors=list(set(all_bearish))[:10] if all_bearish else ["Market risks exist"],
            risk_factors=list(set(all_risks))[:10] if all_risks else ["General market volatility"],
            opportunity_factors=list(set(all_opportunities))[:10] if all_opportunities else ["Long-term potential"],
            
            top_institutional_holders=top_holders,
            insider_transactions="Data in institutional holders",
            
            summary=f"🎯 {yf_data['name']} at {current_price}. Score: {score}/100 ({confidence_level}). "
                   f"Recommendation: {consensus_recommendation}. Based on {len(news_articles)} news articles, "
                   f"market fundamentals, and AI analysis from {len(set(data_sources))} sources.",
            
            actionable_insights=factors_checked + [
                f"📊 Final Score: {score}/100",
                f"💡 {consensus_recommendation} with {confidence_level} confidence"
            ],
            
            analysis_timestamp=datetime.now().isoformat(),
            api_calls_used=api_calls
        )
        
        print("✅ ANALYSIS COMPLETE!")
        print(f"📊 Used {api_calls} API calls")
        print("="*80 + "\n")
        
        return analysis

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "🚀 Stock Analysis API",
        "version": "4.0",
        "status": "running",
        "tip": "Use stock names like 'tcs', 'apple', or ticker symbols like 'AAPL', 'TCS.NS'"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "api_key_configured": bool(FREE_API_KEYS['gemini']),
        "configured_apis": {
            name: bool(key) for name, key in FREE_API_KEYS.items()
        }
    }


@app.get("/rate-limit")
async def check_rate_limit():
    return {
        "is_limited": False,
        "requests_remaining": 60,
        "seconds_remaining": 0,
        "reset_time_ist": None,
        "message": "No rate limit active"
    }

# ADD THIS NEW ENDPOINT HERE:
@app.get("/test-stock/{stock_name}")
async def test_stock(stock_name: str):
    """Test endpoint - Check if yfinance can fetch stock data"""
    try:
        print(f"\n🧪 TEST: Fetching {stock_name}...")
        yf_data = fetch_yfinance_data(stock_name)
        
        if yf_data:
            return {
                "success": True,
                "symbol": yf_data['symbol'],
                "name": yf_data['name'],
                "price": yf_data['price'],
                "currency": yf_data['currency'],
                "message": "✅ Stock data fetched successfully!"
            }
        else:
            return {
                "success": False,
                "error": "Stock not found. Try ticker symbols like AAPL, MSFT, TCS.NS"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }



if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 STOCK ANALYSIS API v4.0")
    print("="*80)
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)