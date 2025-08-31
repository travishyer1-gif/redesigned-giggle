"""
LLM Enhancer Module for Stock Analysis
Uses GPT-4o for high-level reasoning and GPT-3.5-turbo for high-volume tasks.
Supercharges probabilistic calculations, confidence levels, and analysis quality.
"""

import openai
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from config import (
    LLM_ENABLED, LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODELS, 
    LLM_PARAMS, LLM_ENHANCEMENT_WEIGHTS, CONFIDENCE_PARAMS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEnhancer:
    """Enhances stock analysis using OpenAI's GPT models with intelligent model selection."""
    
    def __init__(self):
        self.client = None
        self.model_usage_stats = {
            'gpt-4o': {'calls': 0, 'tokens': 0, 'cost': 0.0},
            'gpt-3.5-turbo': {'calls': 0, 'tokens': 0, 'cost': 0.0}
        }
        
        if LLM_ENABLED and LLM_PROVIDER == "openai":
            if OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE":
                try:
                    self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    logger.info("LLM Enhancer initialized with OpenAI")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.client = None
            else:
                logger.warning("OpenAI API key not configured. LLM enhancement disabled.")
                self.client = None
        else:
            logger.info("LLM enhancement disabled")
    
    def select_model_for_task(self, task_type: str) -> Tuple[str, Dict[str, Any]]:
        """Intelligently select the best model for a given task."""
        if not LLM_ENABLED:
            return None, {}
        
        # Get task-specific model assignment
        model_category = LLM_PARAMS['task_model_mapping'].get(task_type, 'high_reasoning')
        model_config = OPENAI_MODELS[model_category]
        
        logger.info(f"Selected {model_config['model']} for {task_type} task")
        return model_config['model'], model_config
    
    def enhance_probabilistic_calculation(self, 
                                        fundamental_analysis: Dict[str, Any],
                                        sentiment_analysis: Dict[str, Any],
                                        options_analysis: Dict[str, Any],
                                        technical_analysis: Dict[str, Any],
                                        ticker: str) -> Dict[str, Any]:
        """
        Use GPT-4o to enhance probabilistic calculations with market context and reasoning.
        """
        if not self.client:
            return self._get_fallback_probabilities()
        
        try:
            model, config = self.select_model_for_task('probabilistic_calculation')
            
            # Prepare context for the LLM
            context = self._prepare_analysis_context(
                fundamental_analysis, sentiment_analysis, 
                options_analysis, technical_analysis, ticker
            )
            
            prompt = f"""
You are an expert quantitative analyst specializing in probabilistic stock price predictions. 
Analyze the following data and provide enhanced probability assessments for {ticker}:

{context}

Based on this comprehensive analysis, provide:

1. **Probability Distribution** (0.0 to 1.0) for each time horizon:
   - 3-month horizon: Probability of positive return
   - 6-month horizon: Probability of positive return  
   - 12-month horizon: Probability of positive return

2. **Confidence Intervals** for each probability (90% confidence)

3. **Risk-Adjusted Probabilities** considering market volatility and sector trends

4. **Key Factors** that most significantly influence these probabilities

5. **Probability Score** (-2 to +2) that can be converted to ratings

Respond in JSON format:
{{
    "probabilities": {{
        "3m": {{"prob": 0.75, "confidence_lower": 0.65, "confidence_upper": 0.85, "risk_adjusted": 0.70}},
        "6m": {{"prob": 0.68, "confidence_lower": 0.58, "confidence_upper": 0.78, "risk_adjusted": 0.65}},
        "12m": {{"prob": 0.62, "confidence_lower": 0.52, "confidence_upper": 0.72, "risk_adjusted": 0.58}}
    }},
    "probability_score": 1.2,
    "key_factors": ["Strong fundamentals", "Positive sentiment", "Technical uptrend"],
    "risk_assessment": "Medium risk with strong upside potential"
}}
"""
            
            response = self._call_llm(prompt, model, config)
            enhanced_probs = self._parse_probability_response(response)
            
            # Update usage stats
            self._update_usage_stats(model, response.usage.total_tokens if response.usage else 0)
            
            return enhanced_probs
            
        except Exception as e:
            logger.error(f"Error in probabilistic enhancement: {e}")
            return self._get_fallback_probabilities()
    
    def enhance_confidence_calculation(self, 
                                     base_confidence: Dict[str, float],
                                     analysis_results: Dict[str, Any],
                                     ticker: str) -> Dict[str, float]:
        """
        Use GPT-4o to enhance confidence calculations with market context and data quality assessment.
        """
        if not self.client:
            return base_confidence
        
        try:
            model, config = self.select_model_for_task('confidence_calculation')
            
            prompt = f"""
You are an expert in statistical confidence assessment for financial analysis.
Analyze the confidence levels for {ticker} and provide enhanced confidence scores.

Base confidence scores: {base_confidence}

Analysis quality indicators:
- Fundamental data quality: {self._assess_data_quality(analysis_results.get('fundamental_analysis', {}))}
- Sentiment data quality: {self._assess_data_quality(analysis_results.get('sentiment_analysis', {}))}
- Options data quality: {self._assess_data_quality(analysis_results.get('options_analysis', {}))}
- Technical data quality: {self._assess_data_quality(analysis_results.get('technical_analysis', {}))}

Market context factors to consider:
- Data consistency across analysis methods
- Market volatility impact
- Sector-specific confidence factors
- Data freshness and reliability

Provide enhanced confidence scores (0-100%) for each time horizon, considering:
1. Base confidence from rule-based analysis
2. Data quality adjustments
3. Market context factors
4. Analysis agreement levels

Respond in JSON format:
{{
    "enhanced_confidence": {{
        "3m": 85.5,
        "6m": 78.2,
        "12m": 65.8
    }},
    "confidence_factors": {{
        "data_quality_boost": 5.2,
        "market_context_boost": 3.1,
        "agreement_boost": 2.8
    }},
    "confidence_explanation": "Enhanced confidence due to high data quality and strong agreement across analysis methods"
}}
"""
            
            response = self._call_llm(prompt, model, config)
            enhanced_confidence = self._parse_confidence_response(response, base_confidence)
            
            self._update_usage_stats(model, response.usage.total_tokens if response.usage else 0)
            
            return enhanced_confidence
            
        except Exception as e:
            logger.error(f"Error in confidence enhancement: {e}")
            return base_confidence
    
    def enhance_justification_generation(self, 
                                       analysis_results: Dict[str, Any],
                                       ticker: str) -> str:
        """
        Use GPT-4o to generate comprehensive, nuanced justifications for investment decisions.
        """
        if not self.client:
            return self._get_fallback_justification(analysis_results)
        
        try:
            model, config = self.select_model_for_task('final_synthesis')
            
            context = self._prepare_analysis_context(
                analysis_results.get('fundamental_analysis', {}),
                analysis_results.get('sentiment_analysis', {}),
                analysis_results.get('options_analysis', {}),
                analysis_results.get('technical_analysis', {}),
                ticker
            )
            
            prompt = f"""
You are a senior investment analyst providing comprehensive stock analysis for {ticker}.
Generate a professional, nuanced justification that synthesizes all available data.

Analysis Context:
{context}

Create a comprehensive justification that includes:

1. **Executive Summary**: Clear, actionable recommendation
2. **Key Strengths**: What supports the bullish/bearish case
3. **Key Risks**: What could go wrong
4. **Market Context**: How this fits into broader market trends
5. **Investment Thesis**: The core argument for/against investment
6. **Time Horizon Considerations**: Why different timeframes may have different outlooks
7. **Risk-Reward Profile**: Balanced assessment of potential outcomes

Tone: Professional, balanced, evidence-based
Length: 3-4 paragraphs
Style: Clear, concise, suitable for institutional investors

Focus on actionable insights and balanced risk assessment.
"""
            
            response = self._call_llm(prompt, model, config)
            enhanced_justification = response.choices[0].message.content.strip()
            
            self._update_usage_stats(model, response.usage.total_tokens if response.usage else 0)
            
            return enhanced_justification
            
        except Exception as e:
            logger.error(f"Error in justification enhancement: {e}")
            return self._get_fallback_justification(analysis_results)
    
    def enhance_risk_assessment(self, 
                               analysis_results: Dict[str, Any],
                               ticker: str) -> Dict[str, Any]:
        """
        Use GPT-4o to provide comprehensive risk assessment with market context.
        """
        if not self.client:
            return self._get_fallback_risk_assessment()
        
        try:
            model, config = self.select_model_for_task('risk_assessment')
            
            context = self._prepare_analysis_context(
                analysis_results.get('fundamental_analysis', {}),
                analysis_results.get('sentiment_analysis', {}),
                analysis_results.get('options_analysis', {}),
                analysis_results.get('technical_analysis', {}),
                ticker
            )
            
            prompt = f"""
You are a risk management expert analyzing {ticker}.
Provide a comprehensive risk assessment based on the analysis data.

Analysis Context:
{context}

Assess the following risk categories:

1. **Market Risk**: Overall market exposure and correlation
2. **Sector Risk**: Industry-specific risks and opportunities  
3. **Company Risk**: Firm-specific financial and operational risks
4. **Liquidity Risk**: Trading volume and market depth considerations
5. **Volatility Risk**: Price stability and potential for large swings
6. **Regulatory Risk**: Potential regulatory changes or compliance issues
7. **Geopolitical Risk**: International exposure and political factors

For each risk category, provide:
- Risk level (Low/Medium/High)
- Risk score (1-10)
- Key risk factors
- Mitigation strategies

Respond in JSON format:
{{
    "overall_risk_score": 6.5,
    "risk_level": "Medium-High",
    "risk_categories": {{
        "market_risk": {{"level": "Medium", "score": 6, "factors": ["Market volatility"], "mitigation": "Diversification"}},
        "sector_risk": {{"level": "Low", "score": 3, "factors": ["Stable sector"], "mitigation": "None needed"}}
    }},
    "risk_summary": "Overall medium-high risk with strong upside potential but notable volatility concerns"
}}
"""
            
            response = self._call_llm(prompt, model, config)
            enhanced_risk = self._parse_risk_response(response)
            
            self._update_usage_stats(model, response.usage.total_tokens if response.usage else 0)
            
            return enhanced_risk
            
        except Exception as e:
            logger.error(f"Error in risk assessment enhancement: {e}")
            return self._get_fallback_risk_assessment()
    
    def batch_process_sentiment(self, texts: List[str], task_type: str = "sentiment_analysis") -> List[Dict[str, Any]]:
        """
        Use GPT-3.5-turbo to batch process multiple texts for sentiment analysis.
        """
        if not self.client or not texts:
            return []
        
        try:
            model, config = self.select_model_for_task(task_type)
            
            # Batch texts for efficiency
            batch_size = 10  # Process 10 texts at once
            results = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                prompt = f"""
Analyze the sentiment of the following {len(batch)} texts related to stock market analysis.
For each text, provide:
1. Sentiment score (-1.0 to +1.0)
2. Sentiment category (Very Negative, Negative, Neutral, Positive, Very Positive)
3. Key topics mentioned
4. Confidence in sentiment assessment (0-100%)

Texts:
{chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}

Respond in JSON format:
{{
    "sentiments": [
        {{
            "text_index": 1,
            "sentiment_score": 0.7,
            "sentiment_category": "Positive",
            "key_topics": ["earnings", "growth"],
            "confidence": 85
        }}
    ]
}}
"""
                
                response = self._call_llm(prompt, model, config)
                batch_results = self._parse_batch_sentiment_response(response)
                results.extend(batch_results)
                
                self._update_usage_stats(model, response.usage.total_tokens if response.usage else 0)
                
                # Rate limiting
                time.sleep(LLM_PARAMS['rate_limit_delay'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch sentiment processing: {e}")
            return []
    
    def _call_llm(self, prompt: str, model: str, config: Dict[str, Any]) -> Any:
        """Make API call to OpenAI with retry logic."""
        max_retries = LLM_PARAMS['max_retries']
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config['max_tokens'],
                    temperature=config['temperature'],
                    timeout=LLM_PARAMS['timeout_seconds']
                )
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"LLM call failed, attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _prepare_analysis_context(self, fundamental: Dict, sentiment: Dict, 
                                 options: Dict, technical: Dict, ticker: str) -> str:
        """Prepare analysis context for LLM prompts."""
        context_parts = []
        
        if fundamental:
            context_parts.append(f"Fundamental Analysis: {fundamental.get('justification', 'N/A')}")
        
        if sentiment:
            context_parts.append(f"Sentiment Analysis: {sentiment.get('justification', 'N/A')}")
        
        if options:
            context_parts.append(f"Options Analysis: {options.get('justification', 'N/A')}")
        
        if technical:
            context_parts.append(f"Technical Analysis: {technical.get('justification', 'N/A')}")
        
        return "\n\n".join(context_parts) if context_parts else "Limited analysis data available"
    
    def _assess_data_quality(self, analysis_result: Dict[str, Any]) -> str:
        """Assess the quality of analysis data."""
        if not analysis_result:
            return "No data"
        
        # Simple heuristic for data quality
        if 'error' in analysis_result:
            return "Low quality (errors present)"
        elif 'justification' in analysis_result and len(analysis_result['justification']) > 100:
            return "High quality"
        elif 'key_metrics' in analysis_result and analysis_result['key_metrics']:
            return "Medium quality"
        else:
            return "Low quality (limited data)"
    
    def _parse_probability_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response for probability enhancement."""
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]  # Remove markdown code blocks
            
            data = json.loads(content)
            return data
        except Exception as e:
            logger.error(f"Error parsing probability response: {e}")
            return self._get_fallback_probabilities()
    
    def _parse_confidence_response(self, response: Any, base_confidence: Dict[str, float]) -> Dict[str, float]:
        """Parse LLM response for confidence enhancement."""
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            
            data = json.loads(content)
            return data.get('enhanced_confidence', base_confidence)
        except Exception as e:
            logger.error(f"Error parsing confidence response: {e}")
            return base_confidence
    
    def _parse_risk_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response for risk assessment."""
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            
            data = json.loads(content)
            return data
        except Exception as e:
            logger.error(f"Error parsing risk response: {e}")
            return self._get_fallback_risk_assessment()
    
    def _parse_batch_sentiment_response(self, response: Any) -> List[Dict[str, Any]]:
        """Parse LLM response for batch sentiment processing."""
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            
            data = json.loads(content)
            return data.get('sentiments', [])
        except Exception as e:
            logger.error(f"Error parsing batch sentiment response: {e}")
            return []
    
    def _update_usage_stats(self, model: str, tokens: int):
        """Update usage statistics for cost tracking."""
        if model in self.model_usage_stats:
            self.model_usage_stats[model]['calls'] += 1
            self.model_usage_stats[model]['tokens'] += tokens
            
            # Rough cost estimation (you may want to update these rates)
            if model == 'gpt-4o':
                cost_per_1k_tokens = 0.03  # Approximate cost for GPT-4o
            else:  # gpt-3.5-turbo
                cost_per_1k_tokens = 0.002  # Approximate cost for GPT-3.5-turbo
            
            self.model_usage_stats[model]['cost'] += (tokens / 1000) * cost_per_1k_tokens
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of LLM usage and costs."""
        total_calls = sum(stats['calls'] for stats in self.model_usage_stats.values())
        total_tokens = sum(stats['tokens'] for stats in self.model_usage_stats.values())
        total_cost = sum(stats['cost'] for stats in self.model_usage_stats.values())
        
        return {
            'total_calls': total_calls,
            'total_tokens': total_tokens,
            'total_cost': round(total_cost, 4),
            'model_breakdown': self.model_usage_stats,
            'cost_per_call': round(total_cost / total_calls, 4) if total_calls > 0 else 0
        }
    
    def _get_fallback_probabilities(self) -> Dict[str, Any]:
        """Fallback probabilities when LLM is unavailable."""
        return {
            "probabilities": {
                "3m": {"prob": 0.5, "confidence_lower": 0.4, "confidence_upper": 0.6, "risk_adjusted": 0.5},
                "6m": {"prob": 0.5, "confidence_lower": 0.4, "confidence_upper": 0.6, "risk_adjusted": 0.5},
                "12m": {"prob": 0.5, "confidence_lower": 0.4, "confidence_upper": 0.6, "risk_adjusted": 0.5}
            },
            "probability_score": 0.0,
            "key_factors": ["Rule-based analysis only"],
            "risk_assessment": "Standard risk profile"
        }
    
    def _get_fallback_justification(self, analysis_results: Dict[str, Any]) -> str:
        """Fallback justification when LLM is unavailable."""
        return "Analysis based on rule-based algorithms. For enhanced insights, enable LLM integration."
    
    def _get_fallback_risk_assessment(self) -> Dict[str, Any]:
        """Fallback risk assessment when LLM is unavailable."""
        return {
            "overall_risk_score": 5.0,
            "risk_level": "Medium",
            "risk_categories": {},
            "risk_summary": "Standard risk assessment based on rule-based analysis"
        }
