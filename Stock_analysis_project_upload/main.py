"""
Main Stock Analysis Script
Orchestrates all five analysis modules to provide comprehensive stock analysis.
"""

import sys
import argparse
from datetime import datetime
from typing import Dict, Any

# Import our analysis modules
from fundamental_analysis import FundamentalAnalysis
from sentiment_analysis import SentimentAnalysis
from options_analysis import OptionsAnalysis
from technical_analysis_simple import TechnicalAnalysisSimple as TechnicalAnalysis
from synthesizer import Synthesizer

def print_banner():
    """Print a welcome banner for the stock analysis tool."""
    print("=" * 80)
    print("🚀 COMPREHENSIVE STOCK ANALYSIS TOOL")
    print("=" * 80)
    print("Analyzing stocks using 5 specialized AI assistants:")
    print("🔍 Assistant 1: Fundamental Analysis (40% weight)")
    print("📰 Assistant 2: Sentiment Analysis (15% weight)")
    print("📊 Assistant 3: Options Market Analysis (25% weight)")
    print("📈 Assistant 4: Technical Analysis (20% weight)")
    print("🔬 Assistant 5: Synthesis & Final Report")
    print("=" * 80)
    print()

def print_final_report(synthesis_result: Dict[str, Any]):
    """Print the final comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("🎯 FINAL COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Timestamp: {synthesis_result.get('timestamp', 'N/A')}")
    print()
    
    # Final Ratings
    print("📊 FINAL RATINGS BY TIME HORIZON:")
    print("-" * 50)
    final_ratings = synthesis_result.get('final_ratings', {})
    confidence_scores = synthesis_result.get('confidence_scores', {})
    
    for horizon, rating in final_ratings.items():
        confidence = confidence_scores.get(horizon, 0)
        print(f"  {horizon:>3}: {rating:<12} (Confidence: {confidence:>5.1f}%)")
    
    print()
    
    # Best Investment Horizon
    best_horizon = synthesis_result.get('best_horizon', {})
    print("⭐ BEST INVESTMENT HORIZON:")
    print("-" * 50)
    print(f"  Horizon: {best_horizon.get('horizon', 'N/A')}")
    print(f"  Rating:  {best_horizon.get('rating', 'N/A')}")
    print(f"  Confidence: {best_horizon.get('confidence', 0):.1f}%")
    print(f"  Composite Score: {best_horizon.get('composite_score', 0):.2f}")
    print()
    
    # Detailed Summary
    print("📋 DETAILED ANALYSIS SUMMARY:")
    print("-" * 50)
    detailed_summary = synthesis_result.get('detailed_summary', 'No summary available')
    
    # Split long summary into readable paragraphs
    summary_parts = detailed_summary.split('. ')
    for i, part in enumerate(summary_parts):
        if part.strip():
            print(f"  {part.strip()}")
            if i < len(summary_parts) - 1:
                print()
    
    print("\n" + "=" * 80)
    print("⚠️  DISCLAIMER: This analysis is for informational purposes only.")
    print("   Always conduct your own research and consult with financial advisors.")
    print("   Past performance does not guarantee future results.")
    print("=" * 80)

def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on a stock using all five assistants.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Dict containing the complete analysis results
    """
    try:
        print(f"🎯 Starting comprehensive analysis for {ticker.upper()}...")
        print()
        
        # Initialize all analysis modules
        fundamental_analyzer = FundamentalAnalysis()
        sentiment_analyzer = SentimentAnalysis()
        options_analyzer = OptionsAnalysis()
        technical_analyzer = TechnicalAnalysis()
        synthesizer = Synthesizer()
        
        # Run all analyses in parallel (or sequentially for now)
        print("🔄 Running all analysis modules...")
        print()
        
        # Fundamental Analysis
        fundamental_result = fundamental_analyzer.analyze_stock(ticker)
        
        # Sentiment Analysis
        sentiment_result = sentiment_analyzer.analyze_stock(ticker)
        
        # Options Analysis
        options_result = options_analyzer.analyze_stock(ticker)
        
        # Technical Analysis
        technical_result = technical_analyzer.analyze_stock(ticker)
        
        print()
        print("✅ All analysis modules completed successfully!")
        print()
        
        # Synthesize results
        synthesis_result = synthesizer.synthesize_analysis(
            fundamental_result,
            sentiment_result,
            options_result,
            technical_result,
            ticker
        )
        
        return synthesis_result
        
    except Exception as e:
        print(f"❌ Critical error during analysis: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    """Main function to run the stock analysis tool."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Stock Analysis Tool using 5 AI Assistants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py MSFT
  python main.py --help
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to analyze (e.g., AAPL, MSFT)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate ticker
    ticker = args.ticker.upper().strip()
    if not ticker:
        print("❌ Error: Please provide a valid stock ticker symbol.")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Perform analysis
    try:
        result = analyze_stock(ticker)
        
        if 'error' in result:
            print(f"❌ Analysis failed: {result['error']}")
            sys.exit(1)
        
        # Print final report
        print_final_report(result)
        
        print(f"\n🎉 Analysis completed successfully for {ticker}!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
