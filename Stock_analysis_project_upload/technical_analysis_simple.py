"""
Assistant 4: Technical Analysis Module (Simplified)
Analyzes price action, chart patterns, and technical indicators.
This version avoids pandas-ta compatibility issues.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from config import TECHNICAL_PARAMS, TIME_HORIZONS

class TechnicalAnalysisSimple:
    """Simplified technical analysis using basic calculations."""
    
    def __init__(self):
        pass
    
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform technical analysis on a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict containing technical analysis results
        """
        try:
            print(f"📈 Performing technical analysis for {ticker}...")
            
            # Get historical data
            historical_data = self._get_historical_data(ticker)
            if historical_data is None or historical_data.empty:
                return self._get_error_response("Unable to fetch historical data")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(historical_data)
            
            # Analyze patterns and trends
            patterns = self._analyze_patterns(historical_data, indicators)
            
            # Identify key price levels
            key_levels = self._identify_key_levels(historical_data, indicators)
            
            # Determine ratings for different time horizons
            ratings = self._determine_ratings(patterns, indicators, key_levels)
            
            # Generate justification
            justification = self._generate_justification(patterns, indicators, key_levels, ratings)
            
            return {
                'ratings': ratings,
                'justification': justification,
                'indicator_values': indicators,
                'patterns': patterns,
                'key_levels': key_levels,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"❌ Error in technical analysis: {str(e)}")
            return self._get_error_response(str(e))
    
    def _get_historical_data(self, ticker: str) -> pd.DataFrame:
        """Fetch historical price data."""
        try:
            # Get data for the past year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"Warning: No data returned for {ticker}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic technical indicators."""
        try:
            indicators = {}
            
            # Simple Moving Averages
            indicators['sma_5'] = data['Close'].rolling(window=5).mean().iloc[-1]
            indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            indicators['sma_200'] = data['Close'].rolling(window=200).mean().iloc[-1]
            
            # Current price
            current_price = data['Close'].iloc[-1]
            indicators['current_price'] = current_price
            
            # Price relative to moving averages
            indicators['above_sma_5'] = current_price > indicators['sma_5']
            indicators['above_sma_20'] = current_price > indicators['sma_20']
            indicators['above_sma_50'] = current_price > indicators['sma_50']
            indicators['above_sma_200'] = current_price > indicators['sma_200']
            
            # RSI (simplified calculation)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.iloc[-1]
            
            # MACD (simplified)
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
            
            # Volume analysis
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['current_volume'] = data['Volume'].iloc[-1]
            indicators['volume_ratio'] = indicators['current_volume'] / indicators['volume_sma']
            
            # Price momentum
            indicators['price_change_1d'] = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            indicators['price_change_5d'] = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100
            indicators['price_change_20d'] = ((current_price - data['Close'].iloc[-21]) / data['Close'].iloc[-21]) * 100
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}
    
    def _analyze_patterns(self, data: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, Any]:
        """Analyze price patterns and trends."""
        try:
            patterns = {}
            
            # Trend analysis
            current_price = indicators['current_price']
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            sma_200 = indicators['sma_200']
            
            # Determine trend
            if current_price > sma_20 > sma_50 > sma_200:
                patterns['trend'] = 'Strong Uptrend'
                patterns['trend_strength'] = 'Strong'
            elif current_price > sma_20 > sma_50:
                patterns['trend'] = 'Uptrend'
                patterns['trend_strength'] = 'Moderate'
            elif current_price > sma_20:
                patterns['trend'] = 'Weak Uptrend'
                patterns['trend_strength'] = 'Weak'
            elif current_price < sma_20 < sma_50 < sma_200:
                patterns['trend'] = 'Strong Downtrend'
                patterns['trend_strength'] = 'Strong'
            elif current_price < sma_20 < sma_50:
                patterns['trend'] = 'Downtrend'
                patterns['trend_strength'] = 'Moderate'
            elif current_price < sma_20:
                patterns['trend'] = 'Weak Downtrend'
                patterns['trend_strength'] = 'Weak'
            else:
                patterns['trend'] = 'Sideways'
                patterns['trend_strength'] = 'None'
            
            # RSI analysis
            rsi = indicators['rsi']
            if rsi > 70:
                patterns['rsi_condition'] = 'Overbought'
                patterns['rsi_signal'] = 'Bearish'
            elif rsi < 30:
                patterns['rsi_condition'] = 'Oversold'
                patterns['rsi_signal'] = 'Bullish'
            else:
                patterns['rsi_condition'] = 'Neutral'
                patterns['rsi_signal'] = 'Neutral'
            
            # MACD analysis
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_hist = indicators['macd_histogram']
            
            if macd > macd_signal and macd_hist > 0:
                patterns['macd_signal'] = 'Bullish'
            elif macd < macd_signal and macd_hist < 0:
                patterns['macd_signal'] = 'Bearish'
            else:
                patterns['macd_signal'] = 'Neutral'
            
            # Volume analysis
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 1.5:
                patterns['volume_signal'] = 'High Volume'
            elif volume_ratio < 0.5:
                patterns['volume_signal'] = 'Low Volume'
            else:
                patterns['volume_signal'] = 'Normal Volume'
            
            # Support/Resistance levels
            recent_highs = data['High'].rolling(window=20).max()
            recent_lows = data['Low'].rolling(window=20).min()
            
            patterns['resistance'] = recent_highs.iloc[-1]
            patterns['support'] = recent_lows.iloc[-1]
            
            return patterns
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {}
    
    def _identify_key_levels(self, data: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
        """Identify key support and resistance levels."""
        try:
            key_levels = {}
            
            current_price = indicators['current_price']
            
            # Moving averages as key levels
            key_levels['sma_20'] = indicators['sma_20']
            key_levels['sma_50'] = indicators['sma_50']
            key_levels['sma_200'] = indicators['sma_200']
            
            # Recent highs and lows
            key_levels['recent_high'] = data['High'].tail(20).max()
            key_levels['recent_low'] = data['Low'].tail(20).min()
            
            # Fibonacci levels (simplified)
            high = key_levels['recent_high']
            low = key_levels['recent_low']
            diff = high - low
            
            key_levels['fib_38'] = high - (diff * 0.382)
            key_levels['fib_50'] = high - (diff * 0.5)
            key_levels['fib_61'] = high - (diff * 0.618)
            
            # Current price position
            key_levels['above_200_sma'] = current_price > indicators['sma_200']
            key_levels['above_50_sma'] = current_price > indicators['sma_50']
            
            return key_levels
            
        except Exception as e:
            print(f"Error identifying key levels: {e}")
            return {}
    
    def _determine_ratings(self, patterns: Dict[str, Any], indicators: Dict[str, float], 
                          key_levels: Dict[str, float]) -> Dict[str, str]:
        """Determine technical ratings for different time horizons."""
        try:
            ratings = {}
            
            # Calculate composite score
            score = 0
            
            # Trend contribution
            trend_strength = patterns.get('trend_strength', 'None')
            if trend_strength == 'Strong':
                score += 2 if 'Uptrend' in patterns.get('trend', '') else -2
            elif trend_strength == 'Moderate':
                score += 1 if 'Uptrend' in patterns.get('trend', '') else -1
            
            # RSI contribution
            rsi_signal = patterns.get('rsi_signal', 'Neutral')
            if rsi_signal == 'Bullish':
                score += 1
            elif rsi_signal == 'Bearish':
                score -= 1
            
            # MACD contribution
            macd_signal = patterns.get('macd_signal', 'Neutral')
            if macd_signal == 'Bullish':
                score += 1
            elif macd_signal == 'Bearish':
                score -= 1
            
            # Moving average alignment
            if indicators.get('above_sma_200', False):
                score += 1
            if indicators.get('above_sma_50', False):
                score += 1
            if indicators.get('above_sma_20', False):
                score += 0.5
            
            # Volume confirmation
            volume_signal = patterns.get('volume_signal', 'Normal')
            if volume_signal == 'High Volume':
                score += 0.5
            
            # Convert score to ratings
            for horizon in ['3m', '6m', '12m']:
                if score >= 3:
                    ratings[horizon] = 'Strong Buy'
                elif score >= 1.5:
                    ratings[horizon] = 'Buy'
                elif score >= -0.5:
                    ratings[horizon] = 'Hold'
                elif score >= -2:
                    ratings[horizon] = 'Sell'
                else:
                    ratings[horizon] = 'Strong Sell'
                
                # Reduce confidence for longer horizons
                if horizon == '6m':
                    score *= 0.8
                elif horizon == '12m':
                    score *= 0.6
            
            return ratings
            
        except Exception as e:
            print(f"Error determining ratings: {e}")
            return {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
    
    def _generate_justification(self, patterns: Dict[str, Any], indicators: Dict[str, float], 
                               key_levels: Dict[str, float], ratings: Dict[str, str]) -> str:
        """Generate justification for technical analysis."""
        try:
            justification_parts = []
            
            # Overall trend
            trend = patterns.get('trend', 'Unknown')
            trend_strength = patterns.get('trend_strength', 'Unknown')
            justification_parts.append(f"Stock is currently in a {trend_strength.lower()} {trend.lower()}.")
            
            # Moving average analysis
            current_price = indicators.get('current_price', 0)
            sma_200 = indicators.get('sma_200', 0)
            if current_price > sma_200:
                justification_parts.append("Price is above the 200-day moving average, indicating long-term bullish momentum.")
            else:
                justification_parts.append("Price is below the 200-day moving average, indicating long-term bearish pressure.")
            
            # RSI analysis
            rsi = indicators.get('rsi', 50)
            rsi_condition = patterns.get('rsi_condition', 'Neutral')
            if rsi_condition == 'Overbought':
                justification_parts.append(f"RSI at {rsi:.1f} indicates overbought conditions, suggesting potential pullback.")
            elif rsi_condition == 'Oversold':
                justification_parts.append(f"RSI at {rsi:.1f} indicates oversold conditions, suggesting potential bounce.")
            else:
                justification_parts.append(f"RSI at {rsi:.1f} is in neutral territory.")
            
            # MACD analysis
            macd_signal = patterns.get('macd_signal', 'Neutral')
            if macd_signal == 'Bullish':
                justification_parts.append("MACD shows bullish momentum with positive histogram.")
            elif macd_signal == 'Bearish':
                justification_parts.append("MACD shows bearish momentum with negative histogram.")
            else:
                justification_parts.append("MACD is neutral with no clear directional bias.")
            
            # Volume analysis
            volume_signal = patterns.get('volume_signal', 'Normal')
            if volume_signal == 'High Volume':
                justification_parts.append("High volume confirms the current price action.")
            elif volume_signal == 'Low Volume':
                justification_parts.append("Low volume suggests weak conviction in the current move.")
            
            # Key levels
            support = key_levels.get('recent_low', 0)
            resistance = key_levels.get('recent_high', 0)
            justification_parts.append(f"Key support level at ${support:.2f} and resistance at ${resistance:.2f}.")
            
            return " ".join(justification_parts)
            
        except Exception as e:
            return f"Technical analysis completed with some limitations: {str(e)}"
    
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Return error response when analysis fails."""
        return {
            'ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'justification': f'Technical analysis failed: {error_msg}',
            'indicator_values': {},
            'patterns': {},
            'key_levels': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
