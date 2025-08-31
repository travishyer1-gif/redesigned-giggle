# Safe Configuration File for Stock Analysis Script
# This file contains NO API keys and is safe to commit to git

# LLM Integration Settings
LLM_ENABLED = True  # Set to False to use rule-based only
LLM_PROVIDER = "openai"  # Options: "openai", "anthropic", "google", "local"

# Model Selection Strategy
OPENAI_MODELS = {
    'high_reasoning': {
        'model': 'gpt-4o',           # For complex analysis, final synthesis, risk assessment
        'max_tokens': 4000,
        'temperature': 0.1,
        'use_cases': ['final_synthesis', 'risk_assessment', 'probabilistic_calculation', 'confidence_enhancement']
    },
    'high_volume': {
        'model': 'gpt-3.5-turbo',    # For document analysis, sentiment processing, data extraction
        'max_tokens': 2000,
        'temperature': 0.1,
        'use_cases': ['document_analysis', 'sentiment_processing', 'data_extraction', 'text_summarization']
    }
}

# Default model for backward compatibility
OPENAI_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 4000
OPENAI_TEMPERATURE = 0.1

# Anthropic Configuration (Claude)
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
ANTHROPIC_MAX_TOKENS = 2000

# Google AI Configuration
GOOGLE_MODEL = "gemini-1.5-pro"

# Local LLM Configuration (if using local models)
LOCAL_LLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"
LOCAL_LLM_MODEL = "llama3.1-8b-instruct"

# Assistant Weights for Final Synthesis (must sum to 1.0)
ASSISTANT_WEIGHTS = {
    'fundamental': 0.40,    # 40% weight for fundamental analysis
    'sentiment': 0.15,      # 15% weight for sentiment analysis
    'options': 0.25,        # 25% weight for options analysis
    'technical': 0.20       # 20% weight for technical analysis
}

# Enhanced LLM Analysis Weights (when LLM is enabled)
LLM_ENHANCEMENT_WEIGHTS = {
    'probabilistic_calculation': 0.30,    # 30% weight for LLM probability enhancement
    'confidence_calculation': 0.25,       # 25% weight for LLM confidence enhancement
    'justification_generation': 0.25,     # 25% weight for LLM justification
    'risk_assessment': 0.20               # 20% weight for LLM risk analysis
}

# Time Horizons for Analysis (in trading days)
TIME_HORIZONS = [5, 20, 50, 200]

# Rating Scale for Conversion
RATING_SCALE = {
    'Strong Sell': -2,
    'Sell': -1,
    'Hold': 0,
    'Buy': 1,
    'Strong Buy': 2
}

# Reverse Rating Scale (for converting scores back to ratings)
REVERSE_RATING_SCALE = {v: k for k, v in RATING_SCALE.items()}

# Enhanced Probabilistic Rating Scale (LLM-enhanced)
PROBABILISTIC_RATING_SCALE = {
    'Strong Sell': {'min_prob': 0.0, 'max_prob': 0.2, 'score': -2},
    'Sell': {'min_prob': 0.2, 'max_prob': 0.4, 'score': -1},
    'Hold': {'min_prob': 0.4, 'max_prob': 0.6, 'score': 0},
    'Buy': {'min_prob': 0.6, 'max_prob': 0.8, 'score': 1},
    'Strong Buy': {'min_prob': 0.8, 'max_prob': 1.0, 'score': 2}
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'sma_periods': [5, 20, 50, 200]
}

# Sentiment Analysis Parameters
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

# Options Analysis Parameters
OPTIONS_PARAMS = {
    'max_expiration_days': 30,
    'volume_threshold': 2.0,  # Volume must be 2x open interest to be "unusual"
    'put_call_ratio_thresholds': {
        'very_bullish': 0.5,
        'bullish': 0.8,
        'neutral': 1.2,
        'bearish': 1.5,
        'very_bearish': 2.0
    }
}

# LLM Analysis Parameters
LLM_PARAMS = {
    'max_retries': 3,
    'timeout_seconds': 30,
    'rate_limit_delay': 1.0,  # seconds between API calls
    'context_window_size': 4000,  # tokens for context
    'analysis_depth': 'comprehensive',  # Options: 'basic', 'standard', 'comprehensive'
    'include_market_context': True,
    'include_risk_assessment': True,
    'include_sector_analysis': True,
    
    # Multi-model strategy parameters
    'model_selection': {
        'auto_select': True,           # Automatically select best model for each task
        'cost_optimization': True,     # Use cheaper models when possible
        'fallback_strategy': 'high_reasoning',  # Fallback to GPT-4o if needed
        'batch_processing': True       # Batch simple tasks for GPT-3.5-turbo
    },
    
    # Task-specific model assignments
    'task_model_mapping': {
        'fundamental_analysis': 'high_volume',      # Document processing
        'sentiment_analysis': 'high_volume',        # Text analysis
        'options_analysis': 'high_reasoning',       # Complex patterns
        'technical_analysis': 'high_reasoning',     # Pattern recognition
        'final_synthesis': 'high_reasoning',        # Complex reasoning
        'risk_assessment': 'high_reasoning',        # Risk analysis
        'confidence_calculation': 'high_reasoning'  # Probability assessment
    }
}

# Confidence Parameters
CONFIDENCE_PARAMS = {
    'min_confidence': 0.0,
    'max_confidence': 100.0,
    'confidence_thresholds': {
        'low': 30.0,
        'medium': 60.0,
        'high': 80.0
    }
}

# Enhanced Confidence Calculation Parameters
CONFIDENCE_PARAMS = {
    'base_confidence_multiplier': 1.0,
    'llm_confidence_boost': 1.2,  # LLM analysis increases confidence by 20%
    'agreement_threshold': 0.7,   # Minimum agreement for high confidence
    'volatility_factor': 0.1,     # Market volatility impact on confidence
    'data_quality_weights': {
        'high_quality': 1.0,
        'medium_quality': 0.8,
        'low_quality': 0.6
    }
}

# Risk Assessment Parameters
RISK_PARAMS = {
    'volatility_thresholds': {
        'low_risk': 0.15,      # < 15% volatility
        'medium_risk': 0.30,   # 15-30% volatility
        'high_risk': 0.50      # > 50% volatility
    },
    'correlation_thresholds': {
        'low_correlation': 0.3,
        'medium_correlation': 0.6,
        'high_correlation': 0.8
    },
    'liquidity_thresholds': {
        'low_liquidity': 1000000,    # < 1M daily volume
        'medium_liquidity': 5000000, # 1M-5M daily volume
        'high_liquidity': 10000000   # > 10M daily volume
    }
}

# SEC EDGAR API (no key required, but rate limiting applies)
SEC_EDGAR_USER_AGENT = "StockAnalysisBot/1.0 (your-email@domain.com)"

# Reddit API (user agent only - no keys)
REDDIT_USER_AGENT = "StockAnalysisBot/1.0"
