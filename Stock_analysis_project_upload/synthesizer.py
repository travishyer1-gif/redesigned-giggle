"""
Assistant 5: Synthesizer & Final Report Module
Combines analyses from all four assistants to produce a final, actionable rating and comprehensive report.
Enhanced with LLM integration for superior probabilistic calculations and confidence levels.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from config import (
    ASSISTANT_WEIGHTS, RATING_SCALE, REVERSE_RATING_SCALE, TIME_HORIZONS,
    LLM_ENABLED, LLM_ENHANCEMENT_WEIGHTS
)
from llm_enhancer import LLMEnhancer

class Synthesizer:
    """Combines analyses from all assistants to produce final recommendations."""
    
    def __init__(self):
        self.llm_enhancer = LLMEnhancer() if LLM_ENABLED else None
        if self.llm_enhancer:
            print("🚀 LLM enhancement enabled - using GPT-4o for reasoning, GPT-3.5-turbo for high-volume tasks")
        else:
            print("📊 Using rule-based analysis only")
    
    def synthesize_analysis(self, fundamental_analysis: Dict[str, Any], 
                            sentiment_analysis: Dict[str, Any], 
                            options_analysis: Dict[str, Any], 
                            technical_analysis: Dict[str, Any],
                            ticker: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Synthesize all analyses into a comprehensive final report.
        
        Args:
            fundamental_analysis: Output from fundamental analysis
            sentiment_analysis: Output from sentiment analysis
            options_analysis: Output from options analysis
            technical_analysis: Output from technical analysis
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing final recommendations, confidence scores, and detailed summary
        """
        try:
            print("🔬 Synthesizing analysis results...")
            
            # Convert ratings to numerical scores
            fundamental_scores = self._convert_ratings_to_scores(fundamental_analysis.get('ratings', {}))
            sentiment_scores = self._convert_ratings_to_scores(sentiment_analysis.get('ratings', {}))
            options_scores = self._convert_ratings_to_scores(options_analysis.get('ratings', {}))
            technical_scores = self._convert_ratings_to_scores(technical_analysis.get('ratings', {}))
            
            # Calculate weighted scores for each time horizon
            weighted_scores = self._calculate_weighted_scores(
                fundamental_scores, sentiment_scores, options_scores, technical_scores
            )
            
            # LLM Enhancement for probabilistic calculations
            enhanced_probabilities = None
            if self.llm_enhancer:
                print("🧠 Enhancing probabilistic calculations with GPT-4o...")
                enhanced_probabilities = self.llm_enhancer.enhance_probabilistic_calculation(
                    fundamental_analysis, sentiment_analysis, options_analysis, technical_analysis, ticker
                )
                
                # Use enhanced probabilities if available
                if enhanced_probabilities and 'probability_score' in enhanced_probabilities:
                    weighted_scores = self._apply_enhanced_probabilities(weighted_scores, enhanced_probabilities)
            
            # Convert scores back to ratings
            final_ratings = self._convert_scores_to_ratings(weighted_scores)
            
            # Calculate base confidence scores
            base_confidence_scores = self._calculate_confidence_scores(
                fundamental_scores, sentiment_scores, options_scores, technical_scores
            )
            
            # LLM Enhancement for confidence calculation
            confidence_scores = base_confidence_scores
            if self.llm_enhancer:
                print("🧠 Enhancing confidence calculations with GPT-4o...")
                analysis_results = {
                    'fundamental_analysis': fundamental_analysis,
                    'sentiment_analysis': sentiment_analysis,
                    'options_analysis': options_analysis,
                    'technical_analysis': technical_analysis
                }
                confidence_scores = self.llm_enhancer.enhance_confidence_calculation(
                    base_confidence_scores, analysis_results, ticker
                )
            
            # Generate detailed summary
            detailed_summary = self._generate_detailed_summary(
                fundamental_analysis, sentiment_analysis, options_analysis, technical_analysis,
                final_ratings, confidence_scores
            )
            
            # LLM Enhancement for justification generation
            if self.llm_enhancer:
                print("🧠 Generating enhanced justification with GPT-4o...")
                analysis_results = {
                    'fundamental_analysis': fundamental_analysis,
                    'sentiment_analysis': sentiment_analysis,
                    'options_analysis': options_analysis,
                    'technical_analysis': technical_analysis
                }
                enhanced_justification = self.llm_enhancer.enhance_justification_generation(
                    analysis_results, ticker
                )
                detailed_summary = enhanced_justification
            
            # Determine best time horizon
            best_horizon = self._determine_best_horizon(final_ratings, confidence_scores)
            
            # Prepare result with LLM enhancements
            result = {
                'final_ratings': final_ratings,
                'confidence_scores': confidence_scores,
                'best_horizon': best_horizon,
                'detailed_summary': detailed_summary,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add LLM enhancement data if available
            if enhanced_probabilities:
                result['enhanced_probabilities'] = enhanced_probabilities
            
            # Add LLM usage summary if available
            if self.llm_enhancer:
                result['llm_usage'] = self.llm_enhancer.get_usage_summary()
            
            return result
            
        except Exception as e:
            print(f"❌ Error in synthesis: {str(e)}")
            return self._get_error_response()
    
    def _convert_ratings_to_scores(self, ratings: Dict[str, str]) -> Dict[str, float]:
        """Convert rating strings to numerical scores."""
        try:
            scores = {}
            for horizon, rating in ratings.items():
                if rating in RATING_SCALE:
                    scores[horizon] = RATING_SCALE[rating]
                else:
                    scores[horizon] = 0  # Default to Hold if rating not recognized
            return scores
        except Exception as e:
            print(f"Warning: Error converting ratings to scores: {e}")
            return {'3m': 0, '6m': 0, '12m': 0}
    
    def _calculate_weighted_scores(self, fundamental_scores: Dict[str, float], 
                                  sentiment_scores: Dict[str, float], 
                                  options_scores: Dict[str, float], 
                                  technical_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate weighted scores for each time horizon."""
        try:
            weighted_scores = {}
            
            for horizon in ['3m', '6m', '12m']:
                fundamental_score = fundamental_scores.get(horizon, 0)
                sentiment_score = sentiment_scores.get(horizon, 0)
                options_score = options_scores.get(horizon, 0)
                technical_score = technical_scores.get(horizon, 0)
                
                # Apply weights from config
                weighted_score = (
                    fundamental_score * ASSISTANT_WEIGHTS['fundamental'] +
                    sentiment_score * ASSISTANT_WEIGHTS['sentiment'] +
                    options_score * ASSISTANT_WEIGHTS['options'] +
                    technical_score * ASSISTANT_WEIGHTS['technical']
                )
                
                weighted_scores[horizon] = weighted_score
            
            return weighted_scores
            
        except Exception as e:
            print(f"Warning: Error calculating weighted scores: {e}")
            return {'3m': 0, '6m': 0, '12m': 0}
    
    def _convert_scores_to_ratings(self, weighted_scores: Dict[str, float]) -> Dict[str, str]:
        """Convert weighted scores back to rating categories."""
        try:
            ratings = {}
            
            for horizon, score in weighted_scores.items():
                # Round to nearest integer for rating conversion
                rounded_score = round(score)
                
                # Ensure score is within valid range
                if rounded_score > 2:
                    rounded_score = 2
                elif rounded_score < -2:
                    rounded_score = -2
                
                # Convert to rating
                if rounded_score in REVERSE_RATING_SCALE:
                    ratings[horizon] = REVERSE_RATING_SCALE[rounded_score]
                else:
                    ratings[horizon] = 'Hold'  # Default fallback
            
            return ratings
            
        except Exception as e:
            print(f"Warning: Error converting scores to ratings: {e}")
            return {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
    
    def _calculate_confidence_scores(self, fundamental_scores: Dict[str, float], 
                                   sentiment_scores: Dict[str, float], 
                                   options_scores: Dict[str, float], 
                                   technical_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate confidence scores based on agreement between assistants."""
        try:
            confidence_scores = {}
            
            for horizon in ['3m', '6m', '12m']:
                scores = [
                    fundamental_scores.get(horizon, 0),
                    sentiment_scores.get(horizon, 0),
                    options_scores.get(horizon, 0),
                    technical_scores.get(horizon, 0)
                ]
                
                # Calculate standard deviation to measure agreement
                std_dev = np.std(scores)
                
                # Convert to confidence score (0-100%)
                # Lower std dev = higher confidence
                max_std_dev = 4  # Maximum possible std dev with our rating scale
                confidence = max(0, 100 - (std_dev / max_std_dev) * 100)
                
                confidence_scores[horizon] = round(confidence, 1)
            
            return confidence_scores
            
        except Exception as e:
            print(f"Warning: Error calculating confidence scores: {e}")
            return {'3m': 50.0, '6m': 50.0, '12m': 50.0}
    
    def _determine_best_horizon(self, final_ratings: Dict[str, str], 
                               confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Determine the best investment time horizon based on ratings and confidence."""
        try:
            best_horizon = None
            best_score = -1
            
            # Find horizon with highest confidence and strongest rating
            for horizon in ['3m', '6m', '12m']:
                rating = final_ratings.get(horizon, 'Hold')
                confidence = confidence_scores.get(horizon, 0)
                
                # Calculate composite score (rating strength + confidence)
                rating_score = abs(RATING_SCALE.get(rating, 0))
                composite_score = rating_score + (confidence / 100) * 2
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_horizon = horizon
            
            return {
                'horizon': best_horizon,
                'rating': final_ratings.get(best_horizon, 'Hold'),
                'confidence': confidence_scores.get(best_horizon, 0),
                'composite_score': round(best_score, 2)
            }
            
        except Exception as e:
            print(f"Warning: Error determining best horizon: {e}")
            return {
                'horizon': '3m',
                'rating': 'Hold',
                'confidence': 50.0,
                'composite_score': 0.0
            }
    
    def _generate_detailed_summary(self, fundamental_analysis: Dict[str, Any], 
                                  sentiment_analysis: Dict[str, Any], 
                                  options_analysis: Dict[str, Any], 
                                  technical_analysis: Dict[str, Any], 
                                  final_ratings: Dict[str, str], 
                                  confidence_scores: Dict[str, float]) -> str:
        """Generate a comprehensive summary of all analyses."""
        try:
            summary_parts = []
            
            # Overall recommendation
            best_horizon = self._determine_best_horizon(final_ratings, confidence_scores)
            summary_parts.append(
                f"Based on comprehensive analysis across fundamental, sentiment, options, and technical factors, "
                f"the overall recommendation for this stock is {best_horizon['rating']} with highest confidence "
                f"over the {best_horizon['horizon']} time horizon ({best_horizon['confidence']}% confidence)."
            )
            
            # Fundamental analysis summary
            fundamental_rating = fundamental_analysis.get('ratings', {}).get('3m', 'Hold')
            fundamental_just = fundamental_analysis.get('justification', 'No justification available')
            summary_parts.append(
                f"Fundamental analysis ({ASSISTANT_WEIGHTS['fundamental']*100:.0f}% weight): {fundamental_rating} - {fundamental_just}"
            )
            
            # Sentiment analysis summary
            sentiment_rating = sentiment_analysis.get('ratings', {}).get('3m', 'Hold')
            sentiment_just = sentiment_analysis.get('justification', 'No justification available')
            summary_parts.append(
                f"Sentiment analysis ({ASSISTANT_WEIGHTS['sentiment']*100:.0f}% weight): {sentiment_rating} - {sentiment_just}"
            )
            
            # Options analysis summary
            options_rating = options_analysis.get('ratings', {}).get('3m', 'Hold')
            options_just = options_analysis.get('justification', 'No justification available')
            summary_parts.append(
                f"Options analysis ({ASSISTANT_WEIGHTS['options']*100:.0f}% weight): {options_rating} - {options_just}"
            )
            
            # Technical analysis summary
            technical_rating = technical_analysis.get('ratings', {}).get('3m', 'Hold')
            technical_just = technical_analysis.get('justification', 'No justification available')
            summary_parts.append(
                f"Technical analysis ({ASSISTANT_WEIGHTS['technical']*100:.0f}% weight): {technical_rating} - {technical_just}"
            )
            
            # Confidence analysis
            avg_confidence = np.mean(list(confidence_scores.values()))
            if avg_confidence > 80:
                confidence_desc = "high agreement"
            elif avg_confidence > 60:
                confidence_desc = "moderate agreement"
            else:
                confidence_desc = "low agreement"
            
            summary_parts.append(
                f"Overall confidence across time horizons is {avg_confidence:.1f}%, indicating {confidence_desc} "
                f"between the different analysis methods."
            )
            
            # Contradictory evidence
            contradictory_evidence = self._identify_contradictions(
                fundamental_analysis, sentiment_analysis, options_analysis, technical_analysis
            )
            
            if contradictory_evidence:
                summary_parts.append(
                    f"Contradictory evidence: {contradictory_evidence}"
                )
            
            return " ".join(summary_parts)
            
        except Exception as e:
            return f"Error generating detailed summary: {str(e)}"
    
    def _identify_contradictions(self, fundamental_analysis: Dict[str, Any], 
                                sentiment_analysis: Dict[str, Any], 
                                options_analysis: Dict[str, Any], 
                                technical_analysis: Dict[str, Any]) -> str:
        """Identify contradictory signals between different analysis methods."""
        try:
            contradictions = []
            
            # Get 3-month ratings for comparison
            fundamental_3m = fundamental_analysis.get('ratings', {}).get('3m', 'Hold')
            sentiment_3m = sentiment_analysis.get('ratings', {}).get('3m', 'Hold')
            options_3m = options_analysis.get('ratings', {}).get('3m', 'Hold')
            technical_3m = technical_analysis.get('ratings', {}).get('3m', 'Hold')
            
            # Check for strong contradictions
            ratings = [fundamental_3m, sentiment_3m, options_3m, technical_3m]
            buy_signals = sum(1 for r in ratings if 'Buy' in r)
            sell_signals = sum(1 for r in ratings if 'Sell' in r)
            
            if buy_signals >= 2 and sell_signals >= 2:
                contradictions.append("Strong mixed signals with both bullish and bearish indicators")
            elif buy_signals >= 3 and sell_signals >= 1:
                contradictions.append("Mostly bullish with one bearish signal")
            elif sell_signals >= 3 and buy_signals >= 1:
                contradictions.append("Mostly bearish with one bullish signal")
            
            # Check for specific contradictions
            if fundamental_3m != sentiment_3m:
                contradictions.append("Fundamental and sentiment analysis disagree")
            if technical_3m != options_3m:
                contradictions.append("Technical and options analysis disagree")
            
            if not contradictions:
                return "All analysis methods are generally aligned"
            
            return "; ".join(contradictions)
            
        except Exception as e:
            return f"Unable to identify contradictions: {str(e)}"
    
    def _apply_enhanced_probabilities(self, weighted_scores: Dict[str, float], 
                                    enhanced_probabilities: Dict[str, Any]) -> Dict[str, float]:
        """Apply LLM-enhanced probabilities to weighted scores."""
        try:
            if not enhanced_probabilities or 'probability_score' not in enhanced_probabilities:
                return weighted_scores
            
            # Get the enhanced probability score
            enhanced_score = enhanced_probabilities['probability_score']
            
            # Apply enhancement weights
            enhancement_weight = LLM_ENHANCEMENT_WEIGHTS.get('probabilistic_calculation', 0.3)
            
            # Blend enhanced score with weighted scores
            enhanced_scores = {}
            for horizon in weighted_scores:
                # Combine rule-based score with LLM-enhanced score
                rule_based_score = weighted_scores[horizon]
                blended_score = (rule_based_score * (1 - enhancement_weight)) + (enhanced_score * enhancement_weight)
                enhanced_scores[horizon] = blended_score
            
            return enhanced_scores
            
        except Exception as e:
            print(f"Warning: Error applying enhanced probabilities: {e}")
            return weighted_scores
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Return error response when synthesis fails."""
        return {
            'final_ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'confidence_scores': {'3m': 0.0, '6m': 0.0, '12m': 0.0},
            'best_horizon': {'horizon': '3m', 'rating': 'Hold', 'confidence': 0.0, 'composite_score': 0.0},
            'detailed_summary': 'Unable to synthesize analysis due to errors in individual modules.',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
