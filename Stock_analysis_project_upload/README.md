# 🚀 Comprehensive Stock Analysis Tool

A sophisticated Python-based stock analysis tool that uses **5 specialized AI assistants** to provide comprehensive investment recommendations across multiple time horizons.

## 🎯 Overview

This tool combines fundamental analysis, sentiment analysis, options market analysis, and technical analysis to generate actionable investment recommendations with confidence scores. Each analysis module operates independently and contributes to a final synthesized report.

## 🏗️ Architecture

The tool is built with a modular architecture featuring five specialized "assistants":

### 🔍 **Assistant 1: Fundamental Analysis (40% weight)**
- **Data Sources**: SEC EDGAR API, Alpha Vantage Financial Data
- **Analysis**: Financial ratios (P/E, Debt-to-Equity, ROE, ROA), revenue growth trends, profitability analysis
- **Output**: Ratings for 3m, 6m, and 12m horizons based on financial health

### 📰 **Assistant 2: Sentiment Analysis (15% weight)**
- **Data Sources**: News API, Twitter/X API, Reddit API
- **Analysis**: News sentiment, social media sentiment, community sentiment using VADER and TextBlob
- **Output**: Market sentiment ratings with source-specific breakdowns

### 📊 **Assistant 3: Options Market Analysis (25% weight)**
- **Data Sources**: yfinance options data, Alpha Vantage
- **Analysis**: Put/Call ratios, unusual options activity, options flow patterns
- **Output**: Smart money sentiment indicators and market positioning

### 📈 **Assistant 4: Technical Analysis (20% weight)**
- **Data Sources**: yfinance historical data, pandas-ta indicators
- **Analysis**: RSI, MACD, moving averages, Bollinger Bands, volume analysis
- **Output**: Technical trend analysis and support/resistance levels

### 🔬 **Assistant 5: Synthesis & Final Report**
- **Input**: Results from all four analysis modules
- **Process**: Weighted scoring, confidence calculation, contradiction identification
- **Output**: Final actionable recommendations with confidence scores

## 🚀 Features

- **Multi-time Horizon Analysis**: 3-month, 6-month, and 12-month investment horizons
- **Confidence Scoring**: 0-100% confidence levels based on agreement between modules
- **Contradiction Detection**: Identifies conflicting signals between different analysis methods
- **Comprehensive Reporting**: Detailed justifications and key metrics for each analysis
- **Error Handling**: Robust error handling with graceful fallbacks
- **Configurable Weights**: User-adjustable importance weights for each analysis type
- **🤖 LLM Enhancement**: GPT-4o for high-level reasoning, GPT-3.5-turbo for high-volume tasks
- **Enhanced Probabilities**: Sophisticated probabilistic calculations with market context
- **Superior Confidence**: Advanced confidence scoring with data quality assessment
- **Professional Justifications**: Nuanced investment theses and risk assessments
- **Cost Optimization**: Intelligent model selection for cost-effective analysis

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```bash
pip install -r requirements.txt
```

### Key Libraries
- `pandas` - Data manipulation and analysis
- `yfinance` - Yahoo Finance data access
- `alpha-vantage` - Financial data API
- `pandas-ta` - Technical analysis indicators
- `nltk` - Natural language processing
- `textblob` - Sentiment analysis
- `tweepy` - Twitter API access
- `praw` - Reddit API access
- `newsapi-python` - News API access

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd stock_analysis_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit `config.py` and add your API keys:

```python
# Alpha Vantage API (for financial data and options)
ALPHA_VANTAGE_API_KEY = "YOUR_ACTUAL_API_KEY"

# News API (for sentiment analysis)
NEWS_API_KEY = "YOUR_ACTUAL_API_KEY"

# Twitter/X API (for social media sentiment)
TWITTER_BEARER_TOKEN = "YOUR_ACTUAL_BEARER_TOKEN"

# Reddit API (for Reddit sentiment analysis)
REDDIT_CLIENT_ID = "YOUR_ACTUAL_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_ACTUAL_CLIENT_SECRET"
```

### 4. Customize Weights (Optional)
Adjust the importance weights for each analysis type in `config.py`:

```python
ASSISTANT_WEIGHTS = {
    'fundamental': 0.40,    # 40% weight for fundamental analysis
    'sentiment': 0.15,      # 15% weight for sentiment analysis
    'options': 0.25,        # 25% weight for options analysis
    'technical': 0.20       # 20% weight for technical analysis
}
```

### 5. LLM Enhancement Configuration (Optional but Recommended)
Enable AI-powered analysis enhancement by configuring OpenAI:

```python
# Enable LLM enhancement
LLM_ENABLED = True

# Add your OpenAI API key
OPENAI_API_KEY = "your-openai-api-key-here"

# The system automatically selects the optimal model for each task:
# - GPT-4o for complex reasoning, synthesis, and risk assessment
# - GPT-3.5-turbo for high-volume tasks like sentiment analysis
```

**Benefits of LLM Enhancement:**
- **Enhanced Probabilities**: Sophisticated probabilistic calculations with market context
- **Superior Confidence**: Advanced confidence scoring with data quality assessment  
- **Professional Justifications**: Nuanced investment theses and risk assessments
- **Cost Optimization**: Intelligent model selection for cost-effective analysis

## 🎮 Usage

### Basic Usage
```bash
python main.py AAPL
```

### Command Line Options
```bash
python main.py --help
```

### Example Output
```
🚀 COMPREHENSIVE STOCK ANALYSIS TOOL
================================================================================
Analyzing stocks using 5 specialized AI assistants:
🔍 Assistant 1: Fundamental Analysis (40% weight)
📰 Assistant 2: Sentiment Analysis (15% weight)
📊 Assistant 3: Options Market Analysis (25% weight)
📈 Assistant 4: Technical Analysis (20% weight)
🔬 Assistant 5: Synthesis & Final Report
================================================================================

🎯 Starting comprehensive analysis for AAPL...

🔄 Running all analysis modules...
🔍 Performing fundamental analysis for AAPL...
📰 Performing sentiment analysis for AAPL...
📊 Performing options analysis for AAPL...
📈 Performing technical analysis for AAPL...
🔬 Synthesizing analysis results...

✅ All analysis modules completed successfully!

================================================================================
🎯 FINAL COMPREHENSIVE ANALYSIS REPORT
================================================================================
Timestamp: 2024-01-15 14:30:25

📊 FINAL RATINGS BY TIME HORIZON:
--------------------------------------------------
  3m: Buy          (Confidence:  85.2%)
  6m: Buy          (Confidence:  78.9%)
 12m: Hold         (Confidence:  65.4%)

⭐ BEST INVESTMENT HORIZON:
--------------------------------------------------
  Horizon: 3m
  Rating:  Buy
  Confidence: 85.2%
  Composite Score: 2.85

📋 DETAILED ANALYSIS SUMMARY:
--------------------------------------------------
  Based on comprehensive analysis across fundamental, sentiment, options, and technical factors, the overall recommendation for this stock is Buy with highest confidence over the 3m time horizon (85.2% confidence).

  Fundamental analysis (40% weight): Buy - Stock appears undervalued with P/E below 15. Strong balance sheet with low debt levels. Strong revenue growth indicates business momentum.

  Sentiment analysis (15% weight): Buy - Overall market sentiment is positive. Recent news coverage is favorable. Social media sentiment is bullish.

  Options analysis (25% weight): Buy - Low put/call ratio suggests bullish options sentiment. Options flow is heavily weighted toward calls.

  Technical analysis (20% weight): Buy - Stock is in an uptrend with positive moving average alignment. MACD shows bullish momentum with positive histogram.

  Overall confidence across time horizons is 76.5%, indicating moderate agreement between the different analysis methods.

================================================================================
⚠️  DISCLAIMER: This analysis is for informational purposes only.
   Always conduct your own research and consult with financial advisors.
   Past performance does not guarantee future results.
================================================================================

🎉 Analysis completed successfully for AAPL!
```

## 🔧 Configuration

### Time Horizons
Modify the analysis time horizons in `config.py`:
```python
TIME_HORIZONS = [5, 20, 50, 200]  # Trading days
```

### Technical Analysis Parameters
Adjust technical indicator settings:
```python
TECHNICAL_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'sma_periods': [5, 20, 50, 200]
}
```

### Sentiment Analysis Parameters
Customize sentiment analysis settings:
```python
SENTIMENT_PARAMS = {
    'max_tweets': 100,
    'max_reddit_posts': 50,
    'max_news_articles': 20,
    'sentiment_thresholds': {
        'very_negative': -0.5,
        'negative': -0.1,
        'neutral': 0.1,
        'positive': 0.5,
        'very_positive': 1.0
    }
}
```

## 🚨 Error Handling

The tool includes comprehensive error handling:

- **API Failures**: Graceful fallbacks when external APIs are unavailable
- **Data Issues**: Handles missing or corrupted data gracefully
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Network Errors**: Retry logic and timeout handling

## 📊 Output Format

Each analysis module returns a standardized dictionary structure:

```python
{
    'ratings': {
        '3m': 'Buy',
        '6m': 'Buy', 
        '12m': 'Hold'
    },
    'justification': 'Detailed explanation of the rating...',
    'key_metrics': {...},  # Module-specific metrics
    # Additional module-specific fields
}
```

## 🔒 Security & Privacy

- **API Keys**: Store securely and never commit to version control
- **Rate Limiting**: Respects API usage limits
- **Data Privacy**: No personal data is collected or stored
- **Local Processing**: All analysis is performed locally

## 🧪 Testing

### Run Basic Test
```bash
python main.py AAPL
```

### Test Individual Modules
```python
from fundamental_analysis import FundamentalAnalysis

analyzer = FundamentalAnalysis()
result = analyzer.analyze_stock('AAPL')
print(result)
```

### Test LLM Enhancement
Test the AI-powered analysis enhancement:

```bash
# Test LLM capabilities (requires OpenAI API key)
python test_llm_enhancement.py

# This will test:
# - Enhanced probabilistic calculations with GPT-4o
# - Advanced confidence scoring
# - Professional justification generation
# - Comprehensive risk assessment
# - Batch sentiment processing with GPT-3.5-turbo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

**IMPORTANT**: This tool is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or any form of professional financial guidance.

- **Always conduct your own research** before making investment decisions
- **Consult with qualified financial advisors** for personalized advice
- **Past performance does not guarantee future results**
- **Investing involves risk** - you may lose money

## 🆘 Support

For issues, questions, or contributions:

1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information
4. Provide steps to reproduce the problem

## 🔮 Future Enhancements

- **Real-time Data**: Live market data integration
- **Portfolio Analysis**: Multi-stock portfolio analysis
- **Backtesting**: Historical performance validation
- **Machine Learning**: Enhanced prediction models
- **Web Interface**: User-friendly web application
- **Mobile App**: iOS/Android mobile application
- **API Service**: RESTful API for integration
- **Database**: Persistent storage for historical analysis

---

**Happy Analyzing! 📈📊🎯**

