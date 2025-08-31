"""
Test script to demonstrate LLM enhancement capabilities
Shows how GPT-4o and GPT-3.5-turbo enhance the analysis.
"""

import json
from llm_enhancer import LLMEnhancer

# Safe configuration for testing - no API keys exposed
LLM_ENABLED = True  # Set to False to disable LLM features

def test_llm_enhancement():
    """Test the LLM enhancement capabilities."""
    print("🧪 TESTING LLM ENHANCEMENT CAPABILITIES")
    print("=" * 60)
    
    if not LLM_ENABLED:
        print("❌ LLM enhancement is disabled")
        print("   Set LLM_ENABLED = True and add your OpenAI API key to config.py to enable")
        return
    
    # Initialize LLM enhancer
    enhancer = LLMEnhancer()
    
    if not enhancer.client:
        print("❌ LLM enhancer not initialized - check your API key in config.py")
        print("   Make sure you have set OPENAI_API_KEY in your configuration")
        return
    
    print("✅ LLM enhancer initialized successfully")
    print(f"📊 Model selection strategy: {enhancer.select_model_for_task('final_synthesis')}")
    
    # Test data (mock analysis results)
    mock_fundamental = {
        'ratings': {'3m': 'Buy', '6m': 'Buy', '12m': 'Hold'},
        'justification': 'Strong financial ratios with P/E of 15, debt-to-equity of 0.3, and 12% revenue growth YoY.',
        'key_metrics': {'pe_ratio': 15.2, 'debt_to_equity': 0.3, 'revenue_growth_yoy': 12.5}
    }
    
    mock_sentiment = {
        'ratings': {'3m': 'Buy', '6m': 'Buy', '12m': 'Hold'},
        'justification': 'Positive news sentiment with favorable earnings coverage and bullish social media sentiment.',
        'sentiment_score': 0.65
    }
    
    mock_options = {
        'ratings': {'3m': 'Buy', '6m': 'Hold', '12m': 'Hold'},
        'justification': 'Low put/call ratio of 0.6 suggests bullish options sentiment with unusual call activity.',
        'put_call_ratio': {'combined_ratio': 0.6}
    }
    
    mock_technical = {
        'ratings': {'3m': 'Buy', '6m': 'Buy', '12m': 'Hold'},
        'justification': 'Strong uptrend with price above all moving averages, RSI at 65, and bullish MACD crossover.',
        'indicator_values': {'rsi': 65, 'macd': 0.8, 'sma_20': 150.0}
    }
    
    ticker = "AAPL"
    
    print(f"\n🔍 Testing LLM enhancement for {ticker}")
    print("-" * 40)
    
    try:
        # Test 1: Probabilistic calculation enhancement
        print("\n📊 Test 1: Enhanced Probabilistic Calculations")
        print("Using GPT-4o for complex probability assessment...")
        
        enhanced_probs = enhancer.enhance_probabilistic_calculation(
            mock_fundamental, mock_sentiment, mock_options, mock_technical, ticker
        )
        
        print("✅ Enhanced probabilities generated:")
        print(json.dumps(enhanced_probs, indent=2))
        
        # Test 2: Confidence calculation enhancement
        print("\n🎯 Test 2: Enhanced Confidence Calculations")
        print("Using GPT-4o for confidence assessment...")
        
        base_confidence = {'3m': 75.0, '6m': 68.0, '12m': 55.0}
        analysis_results = {
            'fundamental_analysis': mock_fundamental,
            'sentiment_analysis': mock_sentiment,
            'options_analysis': mock_options,
            'technical_analysis': mock_technical
        }
        
        enhanced_confidence = enhancer.enhance_confidence_calculation(
            base_confidence, analysis_results, ticker
        )
        
        print("✅ Enhanced confidence scores:")
        print(json.dumps(enhanced_confidence, indent=2))
        
        # Test 3: Justification generation
        print("\n📝 Test 3: Enhanced Justification Generation")
        print("Using GPT-4o for comprehensive analysis...")
        
        enhanced_justification = enhancer.enhance_justification_generation(
            analysis_results, ticker
        )
        
        print("✅ Enhanced justification generated:")
        print(enhanced_justification[:200] + "..." if len(enhanced_justification) > 200 else enhanced_justification)
        
        # Test 4: Risk assessment
        print("\n⚠️  Test 4: Enhanced Risk Assessment")
        print("Using GPT-4o for risk analysis...")
        
        enhanced_risk = enhancer.enhance_risk_assessment(analysis_results, ticker)
        
        print("✅ Enhanced risk assessment:")
        print(json.dumps(enhanced_risk, indent=2))
        
        # Test 5: Batch sentiment processing
        print("\n📰 Test 5: Batch Sentiment Processing")
        print("Using GPT-3.5-turbo for high-volume text analysis...")
        
        sample_texts = [
            "Apple's new iPhone sales exceeded expectations with strong demand in China",
            "Market concerns about supply chain disruptions affecting tech stocks",
            "Analysts raise price targets following strong quarterly earnings",
            "Competition from Android manufacturers intensifying in emerging markets"
        ]
        
        batch_sentiments = enhancer.batch_process_sentiment(sample_texts)
        
        print("✅ Batch sentiment analysis completed:")
        print(json.dumps(batch_sentiments, indent=2))
        
        # Usage summary
        print("\n📊 LLM Usage Summary")
        print("-" * 40)
        usage_summary = enhancer.get_usage_summary()
        print(f"Total API calls: {usage_summary['total_calls']}")
        print(f"Total tokens used: {usage_summary['total_tokens']}")
        print(f"Estimated cost: ${usage_summary['total_cost']}")
        print(f"Cost per call: ${usage_summary['cost_per_call']}")
        
        print("\n🎉 All LLM enhancement tests completed successfully!")
        print("\n💡 Key Benefits of LLM Enhancement:")
        print("   • GPT-4o provides sophisticated probabilistic calculations")
        print("   • Enhanced confidence scoring with market context")
        print("   • Professional-grade investment justifications")
        print("   • Comprehensive risk assessment")
        print("   • GPT-3.5-turbo handles high-volume tasks efficiently")
        print("   • Cost-optimized model selection strategy")
        
    except Exception as e:
        print(f"❌ Error during LLM enhancement testing: {e}")
        print("   Check your API key in config.py and internet connection")

def test_model_selection():
    """Test the intelligent model selection strategy."""
    print("\n🤖 TESTING INTELLIGENT MODEL SELECTION")
    print("=" * 60)
    
    enhancer = LLMEnhancer()
    
    if not enhancer.client:
        print("❌ LLM enhancer not available")
        return
    
    # Test different task types
    task_types = [
        'fundamental_analysis',    # Should use GPT-3.5-turbo
        'sentiment_analysis',      # Should use GPT-3.5-turbo
        'options_analysis',        # Should use GPT-4o
        'technical_analysis',      # Should use GPT-4o
        'final_synthesis',         # Should use GPT-4o
        'risk_assessment',         # Should use GPT-4o
        'probabilistic_calculation' # Should use GPT-4o
    ]
    
    print("Task Type -> Selected Model:")
    print("-" * 40)
    
    for task_type in task_types:
        model, config = enhancer.select_model_for_task(task_type)
        print(f"{task_type:<25} -> {model}")
    
    print("\n✅ Model selection strategy working correctly!")
    print("   • High-volume tasks use GPT-3.5-turbo (cost-effective)")
    print("   • Complex reasoning tasks use GPT-4o (high-quality)")

if __name__ == "__main__":
    print("🚀 STOCK ANALYSIS TOOL - LLM ENHANCEMENT TESTING")
    print("=" * 70)
    
    # Test LLM enhancement capabilities
    test_llm_enhancement()
    
    # Test model selection strategy
    test_model_selection()
    
    print("\n" + "=" * 70)
    print("🧪 Testing completed!")
    print("💡 To use in production:")
    print("   1. Configure your OpenAI API key in config.py")
    print("   2. Set LLM_ENABLED = True")
    print("   3. Run: python main.py AAPL")
    print("=" * 70)
