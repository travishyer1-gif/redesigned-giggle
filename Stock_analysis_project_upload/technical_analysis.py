"""
Assistant 4: Technical Analysis Module
Analyzes stock price action, chart patterns, and technical indicators to determine directionality and confidence levels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Any, List, Optional
import pandas_ta as ta
from config import TECHNICAL_PARAMS

class TechnicalAnalysis:
    """Analyzes technical aspects of a stock including price patterns and indicators."""
    
    def __init__(self):
        pass
        
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis on a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict containing ratings, justification, and indicator values
        """
        try:
            print(f"📈 Performing technical analysis for {ticker}...")
            
            # Get historical price data
            price_data = self._get_historical_data(ticker)
            
            if price_data is None or price_data.empty:
                return self._get_error_response("No price data available")
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(price_data)
            
            # Analyze price patterns and trends
            patterns = self._analyze_patterns(price_data, indicators)
            
            # Determine support and resistance levels
            levels = self._identify_key_levels(price_data, indicators)
            
            # Determine ratings for different time horizons
            ratings = self._determine_ratings(indicators, patterns, levels)
            
            # Generate justification
            justification = self._generate_justification(indicators, patterns, levels)
            
            return {
                'ratings': ratings,
                'justification': justification,
                'indicator_values': indicators
            }
            
        except Exception as e:
            print(f"❌ Error in technical analysis: {str(e)}")
            return self._get_error_response(str(e))
    
    def _get_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get historical price data for the stock."""
        try:
            # Get data for the last 2 years to ensure enough data for all indicators
            end_date = datetime.now()
            start_date = end_date - timedelta(days=500)
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                print(f"No historical data available for {ticker}")
                return None
            
            # Ensure we have OHLCV data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Missing required price data columns for {ticker}")
                return None
            
            return data
            
        except Exception as e:
            print(f"Warning: Could not get historical data: {e}")
            return None
    
    def _calculate_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators."""
        try:
            indicators = {}
            
            # RSI
            rsi_period = TECHNICAL_PARAMS['rsi_period']
            indicators['rsi'] = ta.rsi(price_data['Close'], length=rsi_period).iloc[-1]
            
            # MACD
            macd_fast = TECHNICAL_PARAMS['macd_fast']
            macd_slow = TECHNICAL_PARAMS['macd_slow']
            macd_signal = TECHNICAL_PARAMS['macd_signal']
            
            macd = ta.macd(price_data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            indicators['macd'] = macd['MACD_12_26_9'].iloc[-1]
            indicators['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
            indicators['macd_histogram'] = macd['MACDh_12_26_9'].iloc[-1]
            
            # Moving Averages
            sma_periods = TECHNICAL_PARAMS['sma_periods']
            for period in sma_periods:
                indicators[f'sma_{period}'] = ta.sma(price_data['Close'], length=period).iloc[-1]
            
            # Bollinger Bands
            bb = ta.bbands(price_data['Close'], length=20)
            indicators['bb_upper'] = bb['BBU_20_2.0'].iloc[-1]
            indicators['bb_middle'] = bb['BBM_20_2.0'].iloc[-1]
            indicators['bb_lower'] = bb['BBL_20_2.0'].iloc[-1]
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            
            # Volume indicators
            indicators['volume_sma'] = ta.sma(price_data['Volume'], length=20).iloc[-1]
            indicators['current_volume'] = price_data['Volume'].iloc[-1]
            indicators['volume_ratio'] = indicators['current_volume'] / indicators['volume_sma']
            
            # Price position relative to moving averages
            current_price = price_data['Close'].iloc[-1]
            indicators['price_vs_sma_20'] = (current_price - indicators['sma_20']) / indicators['sma_20'] * 100
            indicators['price_vs_sma_50'] = (current_price - indicators['sma_50']) / indicators['sma_50'] * 100
            indicators['price_vs_sma_200'] = (current_price - indicators['sma_200']) / indicators['sma_200'] * 100
            
            # Trend strength
            indicators['adx'] = ta.adx(price_data['High'], price_data['Low'], price_data['Close'], length=14).iloc[-1]
            
            return indicators
            
        except Exception as e:
            print(f"Warning: Error calculating indicators: {e}")
            return {}
    
    def _analyze_patterns(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price patterns and trends."""
        try:
            patterns = {}
            
            # Trend analysis
            current_price = price_data['Close'].iloc[-1]
            
            # Moving average trends
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            sma_200 = indicators.get('sma_200', 0)
            
            if sma_20 > sma_50 > sma_200:
                patterns['trend'] = 'strong_uptrend'
            elif sma_20 > sma_50 and sma_50 > sma_200:
                patterns['trend'] = 'uptrend'
            elif sma_20 < sma_50 < sma_200:
                patterns['trend'] = 'strong_downtrend'
            elif sma_20 < sma_50 and sma_50 < sma_200:
                patterns['trend'] = 'downtrend'
            else:
                patterns['trend'] = 'sideways'
            
            # Golden Cross / Death Cross detection
            if sma_20 > sma_50 and price_data['Close'].iloc[-2] <= price_data['Close'].iloc[-2]:
                patterns['cross_signal'] = 'golden_cross'
            elif sma_20 < sma_50 and price_data['Close'].iloc[-2] >= price_data['Close'].iloc[-2]:
                patterns['cross_signal'] = 'death_cross'
            else:
                patterns['cross_signal'] = 'none'
            
            # RSI conditions
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                patterns['rsi_condition'] = 'overbought'
            elif rsi < 30:
                patterns['rsi_condition'] = 'oversold'
            else:
                patterns['rsi_condition'] = 'neutral'
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > 0:
                patterns['macd_signal'] = 'bullish'
            elif macd < macd_signal and macd_histogram < 0:
                patterns['macd_signal'] = 'bearish'
            else:
                patterns['macd_signal'] = 'neutral'
            
            # Volume analysis
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                patterns['volume_pattern'] = 'high_volume'
            elif volume_ratio < 0.5:
                patterns['volume_pattern'] = 'low_volume'
            else:
                patterns['volume_pattern'] = 'normal_volume'
            
            # Support/Resistance breakouts
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            if current_price > bb_upper:
                patterns['bb_position'] = 'above_upper'
            elif current_price < bb_lower:
                patterns['bb_position'] = 'below_lower'
            else:
                patterns['bb_position'] = 'within_bands'
            
            return patterns
            
        except Exception as e:
            print(f"Warning: Error analyzing patterns: {e}")
            return {}
    
    def _identify_key_levels(self, price_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Identify key support and resistance levels."""
        try:
            levels = {}
            
            # Moving averages as support/resistance
            levels['sma_20'] = indicators.get('sma_20', 0)
            levels['sma_50'] = indicators.get('sma_50', 0)
            levels['sma_200'] = indicators.get('sma_200', 0)
            
            # Bollinger Bands
            levels['bb_upper'] = indicators.get('bb_upper', 0)
            levels['bb_lower'] = indicators.get('bb_lower', 0)
            
            # Recent highs and lows
            recent_high = price_data['High'].tail(20).max()
            recent_low = price_data['Low'].tail(20).min()
            
            levels['recent_high'] = recent_high
            levels['recent_low'] = recent_low
            
            # Fibonacci retracement levels (if in downtrend)
            if indicators.get('price_vs_sma_50', 0) < -5:  # Potential downtrend
                high = price_data['High'].tail(50).max()
                low = price_data['Low'].tail(50).min()
                diff = high - low
                
                levels['fib_38'] = high - (diff * 0.382)
                levels['fib_50'] = high - (diff * 0.5)
                levels['fib_61'] = high - (diff * 0.618)
            else:
                levels['fib_38'] = None
                levels['fib_50'] = None
                levels['fib_61'] = None
            
            return levels
            
        except Exception as e:
            print(f"Warning: Error identifying key levels: {e}")
            return {}
    
    def _determine_ratings(self, indicators: Dict[str, Any], patterns: Dict[str, Any], 
                          levels: Dict[str, Any]) -> Dict[str, str]:
        """Determine ratings based on technical analysis."""
        try:
            # Start with neutral rating
            score = 0
            
            # Trend analysis
            trend = patterns.get('trend', 'sideways')
            if trend == 'strong_uptrend':
                score += 2
            elif trend == 'uptrend':
                score += 1
            elif trend == 'strong_downtrend':
                score -= 2
            elif trend == 'downtrend':
                score -= 1
            
            # RSI analysis
            rsi_condition = patterns.get('rsi_condition', 'neutral')
            if rsi_condition == 'oversold':
                score += 1  # Potential reversal
            elif rsi_condition == 'overbought':
                score -= 1  # Potential reversal
            
            # MACD analysis
            macd_signal = patterns.get('macd_signal', 'neutral')
            if macd_signal == 'bullish':
                score += 1
            elif macd_signal == 'bearish':
                score -= 1
            
            # Moving average crossovers
            cross_signal = patterns.get('cross_signal', 'none')
            if cross_signal == 'golden_cross':
                score += 1
            elif cross_signal == 'death_cross':
                score -= 1
            
            # Volume confirmation
            volume_pattern = patterns.get('volume_pattern', 'normal_volume')
            if volume_pattern == 'high_volume':
                # Volume confirms the trend
                if score > 0:
                    score += 1
                elif score < 0:
                    score -= 1
            
            # Price position relative to moving averages
            price_vs_sma_20 = indicators.get('price_vs_sma_20', 0)
            price_vs_sma_50 = indicators.get('price_vs_sma_50', 0)
            
            if price_vs_sma_20 > 0 and price_vs_sma_50 > 0:
                score += 1  # Above key moving averages
            elif price_vs_sma_20 < 0 and price_vs_sma_50 < 0:
                score -= 1  # Below key moving averages
            
            # Convert score to rating
            if score >= 3:
                rating = 'Strong Buy'
            elif score >= 1:
                rating = 'Buy'
            elif score >= -1:
                rating = 'Hold'
            elif score >= -3:
                rating = 'Sell'
            else:
                rating = 'Strong Sell'
            
            # Technical analysis is more short-term focused
            ratings = {
                '3m': rating,
                '6m': rating if abs(score) > 1 else 'Hold',  # Less confident for longer term
                '12m': rating if abs(score) > 2 else 'Hold'   # Even less confident for 1 year
            }
            
            return ratings
            
        except Exception as e:
            print(f"Warning: Error determining ratings: {e}")
            return {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
    
    def _generate_justification(self, indicators: Dict[str, Any], patterns: Dict[str, Any], 
                               levels: Dict[str, Any]) -> str:
        """Generate justification for the technical analysis rating."""
        try:
            justification_parts = []
            
            # Trend analysis
            trend = patterns.get('trend', 'sideways')
            if trend == 'strong_uptrend':
                justification_parts.append("Stock is in a strong uptrend with all moving averages aligned")
            elif trend == 'uptrend':
                justification_parts.append("Stock is in an uptrend with positive moving average alignment")
            elif trend == 'strong_downtrend':
                justification_parts.append("Stock is in a strong downtrend with all moving averages aligned")
            elif trend == 'downtrend':
                justification_parts.append("Stock is in a downtrend with negative moving average alignment")
            else:
                justification_parts.append("Stock is moving sideways with mixed moving average signals")
            
            # RSI analysis
            rsi_condition = patterns.get('rsi_condition', 'neutral')
            if rsi_condition == 'overbought':
                justification_parts.append("RSI indicates overbought conditions, suggesting potential pullback")
            elif rsi_condition == 'oversold':
                justification_parts.append("RSI indicates oversold conditions, suggesting potential bounce")
            else:
                justification_parts.append("RSI is in neutral territory")
            
            # MACD analysis
            macd_signal = patterns.get('macd_signal', 'neutral')
            if macd_signal == 'bullish':
                justification_parts.append("MACD shows bullish momentum with positive histogram")
            elif macd_signal == 'bearish':
                justification_parts.append("MACD shows bearish momentum with negative histogram")
            else:
                justification_parts.append("MACD signals are mixed")
            
            # Moving average crossovers
            cross_signal = patterns.get('cross_signal', 'none')
            if cross_signal == 'golden_cross':
                justification_parts.append("Golden cross formation suggests bullish momentum")
            elif cross_signal == 'death_cross':
                justification_parts.append("Death cross formation suggests bearish momentum")
            
            # Volume analysis
            volume_pattern = patterns.get('volume_pattern', 'normal_volume')
            if volume_pattern == 'high_volume':
                justification_parts.append("High volume confirms the current price action")
            elif volume_pattern == 'low_volume':
                justification_parts.append("Low volume suggests weak conviction in current moves")
            
            # Key levels
            current_price = indicators.get('sma_20', 0)  # Use SMA20 as proxy for current price
            sma_200 = levels.get('sma_200', 0)
            if sma_200 > 0:
                if current_price > sma_200:
                    justification_parts.append("Price is above the 200-day moving average, indicating long-term bullish trend")
                else:
                    justification_parts.append("Price is below the 200-day moving average, indicating long-term bearish trend")
            
            if not justification_parts:
                justification_parts.append("Limited technical data available for comprehensive analysis")
            
            return ". ".join(justification_parts) + "."
            
        except Exception as e:
            return f"Error generating justification: {str(e)}"
    
    def _get_error_response(self, error_msg: str = "Unknown error") -> Dict[str, Any]:
        """Return error response when analysis fails."""
        return {
            'ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'justification': f'Unable to perform technical analysis: {error_msg}',
            'indicator_values': {}
        }
