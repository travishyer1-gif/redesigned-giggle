"""
Assistant 1: Fundamental Analysis Module
Analyzes company financial health and performance using SEC filings and financial data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional
import yfinance as yf
from alpha_vantage.fundamentaldata import FundamentalData
from config import ALPHA_VANTAGE_API_KEY, SEC_EDGAR_USER_AGENT

class FundamentalAnalysis:
    """Analyzes fundamental aspects of a stock including financial ratios and trends."""
    
    def __init__(self):
        self.alpha_vantage = FundamentalData(ALPHA_VANTAGE_API_KEY)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': SEC_EDGAR_USER_AGENT})
        
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis on a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict containing ratings, justification, and key metrics
        """
        try:
            print(f"🔍 Performing fundamental analysis for {ticker}...")
            
            # Get company overview
            company_info = self._get_company_overview(ticker)
            
            # Get financial statements
            income_stmt = self._get_income_statement(ticker)
            balance_sheet = self._get_balance_sheet(ticker)
            cash_flow = self._get_cash_flow(ticker)
            
            # Calculate key ratios
            ratios = self._calculate_financial_ratios(company_info, income_stmt, balance_sheet)
            
            # Analyze trends
            trends = self._analyze_financial_trends(income_stmt, balance_sheet, cash_flow)
            
            # Determine ratings for different time horizons
            ratings = self._determine_ratings(ratios, trends, company_info)
            
            # Generate justification
            justification = self._generate_justification(ratios, trends, company_info)
            
            return {
                'ratings': ratings,
                'justification': justification,
                'key_metrics': ratios
            }
            
        except Exception as e:
            print(f"❌ Error in fundamental analysis: {str(e)}")
            return self._get_error_response()
    
    def _get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage."""
        try:
            overview, _ = self.alpha_vantage.get_company_overview(ticker)
            time.sleep(0.1)  # Rate limiting
            return overview
        except Exception as e:
            print(f"Warning: Could not get company overview: {e}")
            return {}
    
    def _get_income_statement(self, ticker: str) -> pd.DataFrame:
        """Get annual income statements."""
        try:
            income, _ = self.alpha_vantage.get_income_statement_annual(ticker)
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame(income)
        except Exception as e:
            print(f"Warning: Could not get income statement: {e}")
            return pd.DataFrame()
    
    def _get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Get annual balance sheets."""
        try:
            balance, _ = self.alpha_vantage.get_balance_sheet_annual(ticker)
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame(balance)
        except Exception as e:
            print(f"Warning: Could not get balance sheet: {e}")
            return pd.DataFrame()
    
    def _get_cash_flow(self, ticker: str) -> pd.DataFrame:
        """Get annual cash flow statements."""
        try:
            cash_flow, _ = self.alpha_vantage.get_cash_flow_annual(ticker)
            time.sleep(0.1)  # Rate limiting
            return pd.DataFrame(cash_flow)
        except Exception as e:
            print(f"Warning: Could not get cash flow: {e}")
            return pd.DataFrame()
    
    def _calculate_financial_ratios(self, company_info: Dict, income_stmt: pd.DataFrame, 
                                  balance_sheet: pd.DataFrame) -> Dict[str, float]:
        """Calculate key financial ratios."""
        ratios = {}
        
        try:
            # P/E Ratio
            if 'PERatio' in company_info and company_info['PERatio']:
                ratios['pe_ratio'] = float(company_info['PERatio'])
            else:
                ratios['pe_ratio'] = None
            
            # Debt-to-Equity
            if not balance_sheet.empty and 'totalShareholderEquity' in balance_sheet.columns:
                try:
                    total_debt = float(balance_sheet.iloc[0].get('totalDebt', 0))
                    total_equity = float(balance_sheet.iloc[0].get('totalShareholderEquity', 1))
                    ratios['debt_to_equity'] = total_debt / total_equity if total_equity > 0 else None
                except:
                    ratios['debt_to_equity'] = None
            else:
                ratios['debt_to_equity'] = None
            
            # Revenue Growth (YoY)
            if not income_stmt.empty and 'totalRevenue' in income_stmt.columns:
                try:
                    revenues = income_stmt['totalRevenue'].astype(float)
                    if len(revenues) >= 2:
                        current_revenue = revenues.iloc[0]
                        previous_revenue = revenues.iloc[1]
                        ratios['revenue_growth_yoy'] = ((current_revenue - previous_revenue) / previous_revenue) * 100
                    else:
                        ratios['revenue_growth_yoy'] = None
                except:
                    ratios['revenue_growth_yoy'] = None
            else:
                ratios['revenue_growth_yoy'] = None
            
            # Dividend Payout Ratio
            if 'DividendYield' in company_info and company_info['DividendYield']:
                ratios['dividend_yield'] = float(company_info['DividendYield'])
            else:
                ratios['dividend_yield'] = None
            
            # Return on Equity
            if 'ReturnOnEquityTTM' in company_info and company_info['ReturnOnEquityTTM']:
                ratios['roe'] = float(company_info['ReturnOnEquityTTM'])
            else:
                ratios['roe'] = None
            
            # Return on Assets
            if 'ReturnOnAssetsTTM' in company_info and company_info['ReturnOnAssetsTTM']:
                ratios['roa'] = float(company_info['ReturnOnAssetsTTM'])
            else:
                ratios['roa'] = None
                
        except Exception as e:
            print(f"Warning: Error calculating ratios: {e}")
        
        return ratios
    
    def _analyze_financial_trends(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, 
                                 cash_flow: pd.DataFrame) -> Dict[str, str]:
        """Analyze trends in financial metrics over the last 3-5 years."""
        trends = {}
        
        try:
            # Revenue trend
            if not income_stmt.empty and 'totalRevenue' in income_stmt.columns:
                revenues = income_stmt['totalRevenue'].astype(float).head(5)
                if len(revenues) >= 3:
                    if revenues.iloc[0] > revenues.iloc[-1]:
                        trends['revenue_trend'] = 'increasing'
                    elif revenues.iloc[0] < revenues.iloc[-1]:
                        trends['revenue_trend'] = 'decreasing'
                    else:
                        trends['revenue_trend'] = 'stable'
                else:
                    trends['revenue_trend'] = 'insufficient_data'
            else:
                trends['revenue_trend'] = 'no_data'
            
            # Profitability trend
            if not income_stmt.empty and 'netIncome' in income_stmt.columns:
                net_income = income_stmt['netIncome'].astype(float).head(5)
                if len(net_income) >= 3:
                    if net_income.iloc[0] > net_income.iloc[-1]:
                        trends['profitability_trend'] = 'improving'
                    elif net_income.iloc[0] < net_income.iloc[-1]:
                        trends['profitability_trend'] = 'declining'
                    else:
                        trends['profitability_trend'] = 'stable'
                else:
                    trends['profitability_trend'] = 'insufficient_data'
            else:
                trends['profitability_trend'] = 'no_data'
                
        except Exception as e:
            print(f"Warning: Error analyzing trends: {e}")
            trends = {'revenue_trend': 'error', 'profitability_trend': 'error'}
        
        return trends
    
    def _determine_ratings(self, ratios: Dict[str, float], trends: Dict[str, str], 
                          company_info: Dict) -> Dict[str, str]:
        """Determine ratings for different time horizons based on fundamental analysis."""
        ratings = {}
        
        try:
            # Calculate a composite score
            score = 0
            
            # P/E Ratio analysis
            if ratios.get('pe_ratio'):
                if ratios['pe_ratio'] < 15:
                    score += 1  # Undervalued
                elif ratios['pe_ratio'] > 25:
                    score -= 1  # Overvalued
            
            # Debt-to-Equity analysis
            if ratios.get('debt_to_equity'):
                if ratios['debt_to_equity'] < 0.5:
                    score += 1  # Low debt
                elif ratios['debt_to_equity'] > 1.0:
                    score -= 1  # High debt
            
            # Revenue growth analysis
            if ratios.get('revenue_growth_yoy'):
                if ratios['revenue_growth_yoy'] > 10:
                    score += 1  # Strong growth
                elif ratios['revenue_growth_yoy'] < 0:
                    score -= 1  # Declining revenue
            
            # ROE analysis
            if ratios.get('roe'):
                if ratios['roe'] > 15:
                    score += 1  # Strong ROE
                elif ratios['roe'] < 5:
                    score -= 1  # Weak ROE
            
            # Trend analysis
            if trends.get('revenue_trend') == 'increasing':
                score += 1
            elif trends.get('revenue_trend') == 'decreasing':
                score -= 1
            
            if trends.get('profitability_trend') == 'improving':
                score += 1
            elif trends.get('profitability_trend') == 'declining':
                score -= 1
            
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
            
            # Apply to all time horizons (fundamentals are longer-term)
            ratings = {
                '3m': rating,
                '6m': rating,
                '12m': rating
            }
            
        except Exception as e:
            print(f"Warning: Error determining ratings: {e}")
            ratings = {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
        
        return ratings
    
    def _generate_justification(self, ratios: Dict[str, float], trends: Dict[str, str], 
                               company_info: Dict) -> str:
        """Generate justification for the fundamental analysis rating."""
        try:
            justification_parts = []
            
            # P/E analysis
            if ratios.get('pe_ratio'):
                if ratios['pe_ratio'] < 15:
                    justification_parts.append("Stock appears undervalued with P/E below 15")
                elif ratios['pe_ratio'] > 25:
                    justification_parts.append("Stock may be overvalued with P/E above 25")
                else:
                    justification_parts.append("P/E ratio is within reasonable range")
            
            # Debt analysis
            if ratios.get('debt_to_equity'):
                if ratios['debt_to_equity'] < 0.5:
                    justification_parts.append("Strong balance sheet with low debt levels")
                elif ratios['debt_to_equity'] > 1.0:
                    justification_parts.append("High debt levels may pose financial risk")
            
            # Growth analysis
            if ratios.get('revenue_growth_yoy'):
                if ratios['revenue_growth_yoy'] > 10:
                    justification_parts.append("Strong revenue growth indicates business momentum")
                elif ratios['revenue_growth_yoy'] < 0:
                    justification_parts.append("Declining revenue raises concerns about business health")
            
            # ROE analysis
            if ratios.get('roe'):
                if ratios['roe'] > 15:
                    justification_parts.append("Strong return on equity shows efficient capital utilization")
                elif ratios['roe'] < 5:
                    justification_parts.append("Low ROE suggests inefficient capital allocation")
            
            # Trend analysis
            if trends.get('revenue_trend') == 'increasing':
                justification_parts.append("Revenue trend is positive over recent years")
            elif trends.get('revenue_trend') == 'decreasing':
                justification_parts.append("Revenue trend is concerning over recent years")
            
            if trends.get('profitability_trend') == 'improving':
                justification_parts.append("Profitability is improving over time")
            elif trends.get('profitability_trend') == 'declining':
                justification_parts.append("Profitability is declining over time")
            
            if not justification_parts:
                justification_parts.append("Limited financial data available for comprehensive analysis")
            
            return ". ".join(justification_parts) + "."
            
        except Exception as e:
            return f"Error generating justification: {str(e)}"
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Return error response when analysis fails."""
        return {
            'ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'justification': 'Unable to perform fundamental analysis due to data retrieval errors.',
            'key_metrics': {}
        }
