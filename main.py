from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import yfinance as yf
from cachetools import func # Function level caching
from bs4 import BeautifulSoup
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


app = FastAPI(
    title="Stock Analysis API",
    version="4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

import requests
requests.packages.urllib3.disable_warnings()

# ===================== CORS CONFIGURATION =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== API CONFIGURATION =====================
FREE_API_KEYS = {
    'gemini': os.getenv('GEMINI_API_KEY', ''),
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', ''),
    'finnhub': os.getenv('FINNHUB_KEY', '')
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

# CACHE RESULTS FOR 1 HOUR to prevent 429 errors
# @func.ttl_cache(maxsize=100, ttl=3600) 
def fetch_yfinance_data(symbol: str) -> Dict:
    """Fetch comprehensive data with Fallback to Finnhub & Alpha Vantage"""
    print(f"📊 Fetching data for: {symbol}")

    # Get symbol variations
    if '.' in symbol or symbol.isupper():
        variations = [symbol]
    else:
        # Resolve the symbol
        variations = resolve_stock_symbol(symbol)
    
    # 1. TRY YAHOO FINANCE
    for ticker_symbol in variations:
        try:
            print(f"   🔍 Trying Yahoo Finance: {ticker_symbol}")
            
            # Simple ticker creation
            stock = yf.Ticker(ticker_symbol)
            
            # Try fetching info with a specific verification
            info = None
            try:
                # First try fast_info (newer, faster)
                if hasattr(stock, 'fast_info'):
                     # Convert fast_info to dict-like structure for key compatibility
                     fast = stock.fast_info
                     if fast and fast.last_price:
                         print(f"   🚀 Fast Info found price: {fast.last_price}")
                         # We still need metadata from .info for full profile if possible
                         # but fast_info is great for price
            
                info = stock.info
                print(f"   ℹ️ Info keys found: {len(info) if info else 0}")
            except Exception as e:
                print(f"   ⚠️ Stock.info fetch failed: {e}")
            
            # If info is missing/empty, try history fallback
            # We strictly need price to consider it a success
            current_price = 0.0
            
            # 1. Check Info for Price
            if info:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0.0
            
            # 2. If no price in info, try history
            if not current_price:
                 print("   ⚠️ Price missing in info, checking history...")
                 try:
                     hist = stock.history(period="5d")
                     if not hist.empty:
                         current_price = hist['Close'].iloc[-1]
                         if not info:
                             info = {}
                         info['currentPrice'] = current_price
                         info['longName'] = info.get('longName', ticker_symbol)
                         print(f"   ✅ Recovered price from history: {current_price}")
                     else:
                         print("   ❌ History Empty")
                 except Exception as ex:
                     print(f"   ❌ History fetch failed: {ex}")

            if not current_price or float(current_price) == 0:
                print(f"   ❌ Failed to get valid price for {ticker_symbol}")
                continue

    # If we get here, we have a price!
            # Try to populate more data from fast_info if keys are missing in info
            try:
                if hasattr(stock, 'fast_info'):
                    fast = stock.fast_info
                    # Critical checks for missing info keys
                    if not info.get('marketCap'): info['marketCap'] = fast.market_cap
                    if not info.get('fiftyTwoWeekHigh'): info['fiftyTwoWeekHigh'] = fast.year_high
                    if not info.get('fiftyTwoWeekLow'): info['fiftyTwoWeekLow'] = fast.year_low
                    if not info.get('dayHigh'): info['dayHigh'] = fast.day_high
                    if not info.get('dayLow'): info['dayLow'] = fast.day_low
                    if not info.get('volume'): info['volume'] = fast.last_volume
                    print(f"   ✅ Recovered metrics from fast_info")
            except Exception as e:
                print(f"   ⚠️ fast_info extraction error: {e}")

            data = {
                'symbol': ticker_symbol,
                'name': info.get('longName', ticker_symbol),
                'price': float(current_price),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio') or info.get('trailingPegRatio'),
                'eps': info.get('trailingEps') or info.get('forwardEps'),
                'price_to_book': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity'),
                'roe': info.get('returnOnEquity'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'week_52_high': info.get('fiftyTwoWeekHigh'),
                'week_52_low': info.get('fiftyTwoWeekLow'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'volume': info.get('volume'),
                'sector': info.get('sector'),
                'news': stock.news[:5] if stock.news else []
            }
            print(f"   ✅ Yahoo Finance Success: {ticker_symbol} | Price: {data['price']}")
            return data

        except Exception as e:
            print(f"   ⚠️ Yahoo blocked/failed for {ticker_symbol}: {e}")
            continue

    # 2. TRY FINNHUB (Preferred Fallback - Higher Quota)
    print("⚠️ Yahoo failed, trying Finnhub...")
    for ticker_symbol in variations:
        data = fetch_finnhub_quote_fallback(ticker_symbol)
        if data:
            return data

    # 3. TRY ALPHA VANTAGE (Last Resort - Low Quota)
    print("⚠️ Finnhub failed, trying Alpha Vantage...")
    for ticker_symbol in variations:
        data = fetch_alpha_vantage_quote(ticker_symbol)
        if data:
            return data
            
    print("❌ All sources failed.")
    return None


def fetch_finnhub_quote_fallback(symbol: str) -> Dict:
    """Fetch quote from Finnhub as fallback"""  
    if not FREE_API_KEYS['finnhub']:
        print("   ❌ Finnhub key missing")
        return None
        
    try:
        # Finnhub supports some extensions, but mostly US. 
        # For .NS (India), Finnhub often uses .NS.
        # Let's try the original symbol first, then the base.
        targets = [symbol]
        if '.' in symbol:
             targets.append(symbol.split('.')[0])
        
        for s in targets:
             print(f"   🔄 Trying Finnhub: {s}")
             response = requests.get(
                 f"https://finnhub.io/api/v1/quote?symbol={s}&token={FREE_API_KEYS['finnhub']}",
                 timeout=10
             )
             
             if response.status_code == 200:
                 data = response.json()
                 price = data.get('c', 0) # 'c' is current price
                 if price > 0:
                      base_symbol = s
                      break
        else:
             return None # loop finished without break
             
        if response.status_code == 200:
            data = response.json()
            price = data.get('c', 0) # 'c' is current price
            
            if price > 0:
                print(f"   ✅ Finnhub Success: {base_symbol} = ${price}")
                return {
                    'symbol': base_symbol,
                    'name': base_symbol,
                    'price': price,
                    'currency': 'USD',
                    'currency': 'USD',
                    'day_high': data.get('h'),
                    'day_low': data.get('l'),
                    'volume': data.get('v'),
                    'market_cap': None,
                    'pe_ratio': None,
                    'news': []
                }
    except Exception as e:
        print(f"   ⚠️ Finnhub error: {e}")
        
    return None


def fetch_alpha_vantage_indicators(symbol: str) -> Dict:
    """Fetch technical indicators from Alpha Vantage"""
    if not FREE_API_KEYS['alpha_vantage']:
        print("   ❌ Alpha Vantage key missing")    
        return None
    
    try:
        base_symbol = symbol.split('.')[0]

        indicators = {}
        print(f"   📊 Fetching indicators for: {base_symbol}")
        
        url = "https://www.alphavantage.co/query"

        params = {
            'function': 'RSI',
            'symbol': base_symbol,
            'interval': 'daily',
            'time_period': 14,
            'series_type': 'close',
            'apikey': FREE_API_KEYS['alpha_vantage']
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'Technical Analysis: RSI' in data:
                latest = list(data['Technical Analysis: RSI'].values())[0]
                indicators['rsi'] = f"{float(latest['RSI']):.2f}"
                print(f"   ✅ RSI: {indicators['rsi']}")
            else:
                return None
        
    except Exception as e:
        print(f"   ⚠️ Alpha Vantage error: {e}")
    
    return indicators


def fetch_alpha_vantage_quote(symbol: str) -> Dict:
    """Fetch basic stock quote from Alpha Vantage (fallback for Yahoo)"""
    if not FREE_API_KEYS['alpha_vantage']:
        print("   ❌ Alpha Vantage key missing")
        return None
    
    try:
        base_symbol = symbol.split('.')[0]
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': base_symbol,
            'apikey': FREE_API_KEYS['alpha_vantage']
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
            
        data = response.json()
        quote = data.get('Global Quote', {})
        
        if not quote:
            print(f"   ❌ No Alpha Vantage data for {base_symbol}")
            return None
        
        price = float(quote.get('05. price', 0))
        if price <= 0:
            return None
        
        print(f"   ✅ Alpha Vantage Quote: {base_symbol} = ${price:.2f}")
        
        return {
            'symbol': base_symbol,
            'name': base_symbol,
            'price': price,
            'currency': 'USD',
            'day_high': float(quote.get('03. high', 0)),
            'day_low': float(quote.get('04. low', 0)),
            'volume': int(quote.get('06. volume', 0)),
            'market_cap': None,
            'pe_ratio': None,
            'news': []
        }
        
    except Exception as e:
        print(f"   ❌ Alpha Vantage quote error: {e}")
        return None


def fetch_alpha_vantage_overview(symbol: str) -> Dict:
    """Fetch company overview (fundamentals) from Alpha Vantage"""
    if not FREE_API_KEYS['alpha_vantage']:
        return None
        
    try:
        base_symbol = symbol.split('.')[0]
        print(f"   📊 Fetching fundamentals for: {base_symbol}")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': base_symbol,
            'apikey': FREE_API_KEYS['alpha_vantage']
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if not data or 'Symbol' not in data:
                return None
                
            overview = {
                'market_cap': data.get('MarketCapitalization'),
                'pe_ratio': data.get('PERatio'),
                'peg_ratio': data.get('PEGRatio'),
                'price_to_book': data.get('PriceToBookRatio'),
                'dividend_yield': data.get('DividendYield'),
                'eps': data.get('EPS'),
                'beta': data.get('Beta'),
                'week_52_high': data.get('52WeekHigh'),
                'week_52_low': data.get('52WeekLow'),
                'sector': data.get('Sector'),
                'roe': data.get('ReturnOnEquityTTM')
            }
            
            # Filter out None or 'None' strings
            return {k: v for k, v in overview.items() if v and v != 'None'}
            
    except Exception as e:
        print(f"   ⚠️ Alpha Vantage Overview error: {e}")
    
    return None


def fetch_finnhub_news(symbol: str) -> List[Dict]:
    """Fetch news from Finnhub"""
    if not FREE_API_KEYS['finnhub']:
        return []
    
    try:
        # Rate limit check (simple sleep to stay safe)
        time.sleep(1) 
        
        base_symbol = symbol.split('.')[0]
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
            articles = []
            for article in news_data[:10]:
                articles.append({
                    'title': article.get('headline', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'published_at': datetime.fromtimestamp(article.get('datetime', 0)).isoformat()
                })
            return articles
            
    except Exception as e:
        print(f"   ⚠️ Finnhub news error: {e}")
    
    return []



def fetch_finnhub_metrics(symbol: str) -> Dict:
    """Fetch basic financials from Finnhub"""
    if not FREE_API_KEYS['finnhub']:
        return None
        
    try:
        # Try original first, then base
        targets = [symbol]
        if '.' in symbol:
             targets.append(symbol.split('.')[0])
             
        for s in targets:
            print(f"   📊 Fetching Finnhub metrics for: {s}")
            response = requests.get(
                f"https://finnhub.io/api/v1/stock/metric?symbol={s}&metric=all&token={FREE_API_KEYS['finnhub']}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                metric = data.get('metric', {})
                if metric:
                     return {
                        'market_cap': metric.get('marketCapitalization'),
                        'pe_ratio': metric.get('peBasicExclExtraTTM'),
                        'beta': metric.get('beta'),
                        'dividend_yield': metric.get('dividendYieldIndicatedAnnual'),
                        'week_52_high': metric.get('52WeekHigh'),
                        'week_52_low': metric.get('52WeekLow'),
                        'roe': metric.get('roeTTM'),
                        'price_to_book': metric.get('pbAnnual'),
                        'eps': metric.get('epsBasicExclExtraTTM')
                     }
    except Exception as e:
        print(f"   ⚠️ Finnhub Metrics error: {e}")
    
    return None


def fetch_yahoo_news_rss(symbol: str) -> List[Dict]:
    """Fetch news from Yahoo RSS (Bypasses API blocks)"""
    print(f"📰 Fetching Yahoo RSS news for {symbol}...")
    articles = []
    try:
        # Clean symbol for Yahoo RSS (e.g. TCS.NS -> TCS.NS)
        ticker = symbol.upper()
        
        url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            for item in items[:5]:
                try:
                    title = item.title.text
                    link = item.link.text
                    pub_date = item.pubDate.text
                    
                    # Convert pub_date to ISO if possible, else keep as is
                    try:
                        # Yahoo RSS format: "Fri, 27 Dec 2024 10:30:00 GMT"
                        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                        iso_date = dt.isoformat()
                    except:
                        iso_date = str(datetime.now().isoformat())

                    articles.append({
                        'title': title,
                        'source': 'Yahoo Finance',
                        'url': link,
                        'published_at': iso_date
                    })
                except:
                    continue
            
            print(f"   ✅ Yahoo RSS: {len(articles)} articles")
            
    except Exception as e:
        print(f"   ⚠️ Yahoo RSS error: {e}")

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
    debug_log = {"steps": []}
    
    try:
        print("\n" + "!"*80)
        print(f"🔬 ENHANCED FREE ANALYSIS (DEBUG MODE): {stock_name_final}")
        print("!"*80)
        
        # Step 1: Yahoo Finance
        yf_data = fetch_yfinance_data(stock_name_final)
        time.sleep(1) 
        
        if not yf_data:
             debug_log["steps"].append("Yahoo Finance returned None")
             pass 
             
        debug_log["initial_yf"] = {
            "pe": yf_data.get('pe_ratio'),
            "cap": yf_data.get('market_cap'),
            "price": yf_data.get('price')
        }
        print(f"   🔍 DEBUG: Initial Metrics - PE: {yf_data.get('pe_ratio')}, Cap: {yf_data.get('market_cap')}")
        if not yf_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Stock '{stock_name_final}' not found. Try using the ticker symbol (e.g., AAPL, TCS.NS)"
            )
        
        data_sources.append("Yahoo Finance")
        api_calls += 1
        
        # Step 2: Technical Indicators & Fundamentals Check
        indicators = {}
        try:
            fetched_indicators = fetch_alpha_vantage_indicators(yf_data['symbol'])
            if fetched_indicators:
                indicators = fetched_indicators
                data_sources.append("Alpha Vantage (Indicators)")
                api_calls += 1
                
            # If critical metrics are missing, try Alpha Vantage Overview
            if not yf_data.get('pe_ratio') or not yf_data.get('market_cap'):
                print("   ⚠️ Missing metrics, trying Alpha Vantage Overview...")
                av_overview = fetch_alpha_vantage_overview(yf_data['symbol'])
                if av_overview:
                    print("   ✅ Recovered metrics from Alpha Vantage")
                    # Merge data, prioritizing existing yf_data if valid
                    for key, value in av_overview.items():
                        if not yf_data.get(key):
                            yf_data[key] = value
                    
                    data_sources.append("Alpha Vantage (Fundamentals)")
                    api_calls += 1
            
             
            # Additional Fallback: Finnhub Metrics
            if not yf_data.get('pe_ratio') or not yf_data.get('market_cap'):
                 print("   ⚠️ Still missing metrics, trying Finnhub...")
                 debug_log["steps"].append("Triggered Finnhub Fallback")
                 fh_metrics = fetch_finnhub_metrics(yf_data['symbol'])
                 if fh_metrics:
                     print("   ✅ Recovered metrics from Finnhub")
                     debug_log["finnhub_recovery"] = fh_metrics
                     for key, value in fh_metrics.items():
                         if not yf_data.get(key) and value:
                             yf_data[key] = value
                     data_sources.append("Finnhub (Fundamentals)")
                     api_calls += 1
                 else:
                     debug_log["steps"].append("Finnhub Fallback Failed")

            # Additional Fallback: Missing Quote Data (Volume, Day High/Low)
            if not yf_data.get('volume') or not yf_data.get('day_high'):
                print("   ⚠️ Missing quote details, trying Finnhub Quote...")
                fh_quote = fetch_finnhub_quote_fallback(yf_data['symbol'])
                if fh_quote:
                    print("   ✅ Recovered quote data from Finnhub")
                    if not yf_data.get('volume'): yf_data['volume'] = fh_quote.get('volume')
                    if not yf_data.get('day_high'): yf_data['day_high'] = fh_quote.get('day_high')
                    if not yf_data.get('day_low'): yf_data['day_low'] = fh_quote.get('day_low')
                    if "Finnhub" not in data_sources: data_sources.append("Finnhub (Quote)")
                
                # If still missing, try Alpha Vantage Quote as last resort
                if not yf_data.get('volume') and FREE_API_KEYS['alpha_vantage']:
                     print("   ⚠️ Still missing volume, trying Alpha Vantage...")
                     av_quote = fetch_alpha_vantage_quote(yf_data['symbol'])
                     if av_quote:
                         if not yf_data.get('volume'): yf_data['volume'] = av_quote.get('volume')
                         if not yf_data.get('day_high'): yf_data['day_high'] = av_quote.get('day_high')
                         if not yf_data.get('day_low'): yf_data['day_low'] = av_quote.get('day_low')
                         if "Alpha Vantage" not in data_sources: data_sources.append("Alpha Vantage (Quote)")
            
            # Fallback: Calculate EPS if missing
            if not yf_data.get('eps') and yf_data.get('price') and yf_data.get('pe_ratio'):
                try:
                    yf_data['eps'] = float(yf_data['price']) / float(yf_data['pe_ratio'])
                    print(f"   🧮 Calculated EPS from Price/PE: {yf_data['eps']:.2f}")
                except:
                    pass

            debug_log["final_metrics"] = {
                "pe": yf_data.get('pe_ratio'),
                "cap": yf_data.get('market_cap'),
                "eps": yf_data.get('eps')
            }
                    
        except Exception as e:
            print(f"   ⚠️ Indicator/Fundamentals fetching failed: {e}")
            debug_log["error_indicators"] = str(e)
        
        # Step 3: News from multiple sources
        news_articles = []
        
        # 3a. Try Yahoo RSS (Best Quality, usually unblocked)
        yahoo_rss_news = fetch_yahoo_news_rss(yf_data['symbol'])
        if yahoo_rss_news:
            news_articles.extend(yahoo_rss_news)
            data_sources.append("Yahoo News (RSS)")
            api_calls += 1
        
        # 3b. Try Finnhub News
        finnhub_news = fetch_finnhub_news(yf_data['symbol'])
        if finnhub_news:
            news_articles.extend(finnhub_news)
            if "Finnhub" not in data_sources: 
                data_sources.append("Finnhub")
            api_calls += 1
        
        # 3c. Try Yahoo API News (if available/mapped from yfinance result)
        if yf_data.get('news'):
             yf_api_news = [
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
             # Avoid duplicates based on title
             existing_titles = {n['title'] for n in news_articles}
             for item in yf_api_news:
                 if item['title'] not in existing_titles:
                     news_articles.append(item)
        
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
            api_calls_used=api_calls,
            debug_info=debug_log
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
@app.head("/")
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