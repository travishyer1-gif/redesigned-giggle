"""
Assistant 3: Options Market Analysis Module
Analyzes options chain data to identify potential "smart money" moves and market sentiment.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from config import ALPHA_VANTAGE_API_KEY, OPTIONS_PARAMS

class OptionsAnalysis:
    """Analyzes options chain data to identify smart money moves and market sentiment."""
    
    def __init__(self):
        self.alpha_vantage = TimeSeries(ALPHA_VANTAGE_API_KEY) if ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY_HERE" else None
        
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive options analysis on a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict containing ratings, justification, and put/call ratio
        """
        try:
            print(f"📊 Performing options analysis for {ticker}...")
            
            # Get options chain data
            options_data = self._get_options_chain(ticker)
            
            if options_data is None or options_data.empty:
                return self._get_error_response("No options data available")
            
            # Calculate put/call ratio
            put_call_ratio = self._calculate_put_call_ratio(options_data)
            
            # Identify unusual options activity
            unusual_activity = self._identify_unusual_activity(options_data)
            
            # Analyze options flow patterns
            flow_patterns = self._analyze_options_flow(options_data)
            
            # Determine ratings for different time horizons
            ratings = self._determine_ratings(put_call_ratio, unusual_activity, flow_patterns)
            
            # Generate justification
            justification = self._generate_justification(
                put_call_ratio, unusual_activity, flow_patterns
            )
            
            return {
                'ratings': ratings,
                'justification': justification,
                'put_call_ratio': put_call_ratio
            }
            
        except Exception as e:
            print(f"❌ Error in options analysis: {str(e)}")
            return self._get_error_response(str(e))
    
    def _get_options_chain(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get options chain data for the stock."""
        try:
            # Try to get options data from yfinance (more reliable for options)
            stock = yf.Ticker(ticker)
            
            # Get options expiration dates
            expiration_dates = stock.options
            
            if not expiration_dates:
                print(f"No options available for {ticker}")
                return None
            
            # Get options for the nearest expiration date
            nearest_expiry = expiration_dates[0]
            options = stock.option_chain(nearest_expiry)
            
            # Combine calls and puts
            calls = options.calls
            puts = options.puts
            
            # Add expiration date and type
            calls['expiration'] = nearest_expiry
            calls['type'] = 'call'
            puts['expiration'] = nearest_expiry
            puts['type'] = 'put'
            
            # Combine into single dataframe
            options_chain = pd.concat([calls, puts], ignore_index=True)
            
            return options_chain
            
        except Exception as e:
            print(f"Warning: Could not get options data from yfinance: {e}")
            return None
    
    def _calculate_put_call_ratio(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate put/call ratio based on volume and open interest."""
        try:
            if options_data is None or options_data.empty:
                return {'volume_ratio': None, 'open_interest_ratio': None, 'combined_ratio': None}
            
            # Separate calls and puts
            calls = options_data[options_data['type'] == 'call']
            puts = options_data[options_data['type'] == 'put']
            
            # Calculate volume-based ratio
            total_call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            total_put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else None
            
            # Calculate open interest-based ratio
            total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            
            open_interest_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
            
            # Combined ratio (weighted average)
            combined_ratio = None
            if volume_ratio is not None and open_interest_ratio is not None:
                # Weight volume more heavily as it's more current
                combined_ratio = (volume_ratio * 0.7) + (open_interest_ratio * 0.3)
            
            return {
                'volume_ratio': volume_ratio,
                'open_interest_ratio': open_interest_ratio,
                'combined_ratio': combined_ratio
            }
            
        except Exception as e:
            print(f"Warning: Error calculating put/call ratio: {e}")
            return {'volume_ratio': None, 'open_interest_ratio': None, 'combined_ratio': None}
    
    def _identify_unusual_activity(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify unusual options activity based on volume vs open interest."""
        try:
            if options_data is None or options_data.empty:
                return {'unusual_strikes': [], 'total_unusual': 0}
            
            unusual_strikes = []
            volume_threshold = OPTIONS_PARAMS['volume_threshold']
            
            for _, option in options_data.iterrows():
                try:
                    volume = option.get('volume', 0)
                    open_interest = option.get('openInterest', 0)
                    
                    if open_interest > 0 and volume > (open_interest * volume_threshold):
                        unusual_strikes.append({
                            'strike': option.get('strikePrice', 'N/A'),
                            'type': option.get('type', 'N/A'),
                            'volume': volume,
                            'open_interest': open_interest,
                            'volume_oi_ratio': volume / open_interest if open_interest > 0 else 0
                        })
                except Exception as e:
                    continue
            
            return {
                'unusual_strikes': unusual_strikes,
                'total_unusual': len(unusual_strikes)
            }
            
        except Exception as e:
            print(f"Warning: Error identifying unusual activity: {e}")
            return {'unusual_strikes': [], 'total_unusual': 0}
    
    def _analyze_options_flow(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze options flow patterns and identify key price levels."""
        try:
            if options_data is None or options_data.empty:
                return {'key_levels': [], 'flow_pattern': 'neutral'}
            
            # Find strikes with highest volume
            if 'volume' in options_data.columns:
                high_volume_options = options_data.nlargest(5, 'volume')
                key_levels = []
                
                for _, option in high_volume_options.iterrows():
                    key_levels.append({
                        'strike': option.get('strikePrice', 'N/A'),
                        'type': option.get('type', 'N/A'),
                        'volume': option.get('volume', 0),
                        'significance': 'high_volume'
                    })
            else:
                key_levels = []
            
            # Determine overall flow pattern
            if 'volume' in options_data.columns:
                call_volume = options_data[options_data['type'] == 'call']['volume'].sum()
                put_volume = options_data[options_data['type'] == 'put']['volume'].sum()
                
                if call_volume > put_volume * 1.5:
                    flow_pattern = 'call_heavy'
                elif put_volume > call_volume * 1.5:
                    flow_pattern = 'put_heavy'
                else:
                    flow_pattern = 'balanced'
            else:
                flow_pattern = 'neutral'
            
            return {
                'key_levels': key_levels,
                'flow_pattern': flow_pattern
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing options flow: {e}")
            return {'key_levels': [], 'flow_pattern': 'neutral'}
    
    def _determine_ratings(self, put_call_ratio: Dict[str, float], 
                          unusual_activity: Dict[str, Any], 
                          flow_patterns: Dict[str, Any]) -> Dict[str, str]:
        """Determine ratings based on options analysis."""
        try:
            # Start with neutral rating
            score = 0
            
            # Analyze put/call ratio
            combined_ratio = put_call_ratio.get('combined_ratio')
            if combined_ratio is not None:
                thresholds = OPTIONS_PARAMS['put_call_ratio_thresholds']
                
                if combined_ratio <= thresholds['very_bullish']:
                    score += 2  # Very bullish
                elif combined_ratio <= thresholds['bullish']:
                    score += 1  # Bullish
                elif combined_ratio >= thresholds['very_bearish']:
                    score -= 2  # Very bearish
                elif combined_ratio >= thresholds['bearish']:
                    score -= 1  # Bearish
            
            # Analyze unusual activity
            total_unusual = unusual_activity.get('total_unusual', 0)
            if total_unusual > 5:
                # High unusual activity can indicate smart money moves
                # Analyze the nature of unusual activity
                unusual_strikes = unusual_activity.get('unusual_strikes', [])
                call_unusual = sum(1 for strike in unusual_strikes if strike['type'] == 'call')
                put_unusual = sum(1 for strike in unusual_strikes if strike['type'] == 'put')
                
                if call_unusual > put_unusual * 1.5:
                    score += 1  # Bullish unusual activity
                elif put_unusual > call_unusual * 1.5:
                    score -= 1  # Bearish unusual activity
            
            # Analyze flow patterns
            flow_pattern = flow_patterns.get('flow_pattern', 'neutral')
            if flow_pattern == 'call_heavy':
                score += 1
            elif flow_pattern == 'put_heavy':
                score -= 1
            
            # Convert score to rating
            if score >= 2:
                rating = 'Strong Buy'
            elif score >= 1:
                rating = 'Buy'
            elif score >= -1:
                rating = 'Hold'
            elif score >= -2:
                rating = 'Sell'
            else:
                rating = 'Strong Sell'
            
            # Options analysis is more short-term focused
            ratings = {
                '3m': rating,
                '6m': rating if abs(score) > 1 else 'Hold',  # Less confident for longer term
                '12m': 'Hold'  # Options have limited predictive value for 1 year
            }
            
            return ratings
            
        except Exception as e:
            print(f"Warning: Error determining ratings: {e}")
            return {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
    
    def _generate_justification(self, put_call_ratio: Dict[str, float], 
                               unusual_activity: Dict[str, Any], 
                               flow_patterns: Dict[str, Any]) -> str:
        """Generate justification for the options analysis rating."""
        try:
            justification_parts = []
            
            # Put/call ratio analysis
            combined_ratio = put_call_ratio.get('combined_ratio')
            if combined_ratio is not None:
                thresholds = OPTIONS_PARAMS['put_call_ratio_thresholds']
                
                if combined_ratio <= thresholds['very_bullish']:
                    justification_parts.append("Very low put/call ratio indicates extreme bullish sentiment")
                elif combined_ratio <= thresholds['bullish']:
                    justification_parts.append("Low put/call ratio suggests bullish options sentiment")
                elif combined_ratio >= thresholds['very_bearish']:
                    justification_parts.append("Very high put/call ratio indicates extreme bearish sentiment")
                elif combined_ratio >= thresholds['bearish']:
                    justification_parts.append("High put/call ratio suggests bearish options sentiment")
                else:
                    justification_parts.append("Put/call ratio is within neutral range")
            
            # Unusual activity analysis
            total_unusual = unusual_activity.get('total_unusual', 0)
            if total_unusual > 0:
                unusual_strikes = unusual_activity.get('unusual_strikes', [])
                call_unusual = sum(1 for strike in unusual_strikes if strike['type'] == 'call')
                put_unusual = sum(1 for strike in unusual_strikes if strike['type'] == 'put')
                
                if total_unusual > 5:
                    justification_parts.append(f"High unusual options activity detected with {total_unusual} strikes")
                    
                    if call_unusual > put_unusual:
                        justification_parts.append("Unusual activity favors call options, suggesting bullish sentiment")
                    elif put_unusual > call_unusual:
                        justification_parts.append("Unusual activity favors put options, suggesting bearish sentiment")
                    else:
                        justification_parts.append("Unusual activity is balanced between calls and puts")
            
            # Flow pattern analysis
            flow_pattern = flow_patterns.get('flow_pattern', 'neutral')
            if flow_pattern == 'call_heavy':
                justification_parts.append("Options flow is heavily weighted toward calls")
            elif flow_pattern == 'put_heavy':
                justification_parts.append("Options flow is heavily weighted toward puts")
            else:
                justification_parts.append("Options flow is relatively balanced")
            
            if not justification_parts:
                justification_parts.append("Limited options data available for comprehensive analysis")
            
            return ". ".join(justification_parts) + "."
            
        except Exception as e:
            return f"Error generating justification: {str(e)}"
    
    def _get_error_response(self, error_msg: str = "Unknown error") -> Dict[str, Any]:
        """Return error response when analysis fails."""
        return {
            'ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'justification': f'Unable to perform options analysis: {error_msg}',
            'put_call_ratio': {'volume_ratio': None, 'open_interest_ratio': None, 'combined_ratio': None}
        }
