"""
Assistant 2: Sentiment Analysis Module
Analyzes public and market sentiment using social media, news, and other text sources.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional
import re
from textblob import TextBlob
import nltk
from newsapi import NewsApiClient
import tweepy
import praw
from config import (NEWS_API_KEY, TWITTER_BEARER_TOKEN, REDDIT_CLIENT_ID, 
                   REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, SENTIMENT_PARAMS)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except:
    VADER_AVAILABLE = False
    print("Warning: VADER sentiment analyzer not available, using TextBlob instead")

class SentimentAnalysis:
    """Analyzes sentiment from various sources including news, social media, and Reddit."""
    
    def __init__(self):
        self.news_api = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY != "YOUR_NEWS_API_KEY_HERE" else None
        self.twitter_client = self._setup_twitter_client()
        self.reddit_client = self._setup_reddit_client()
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict containing ratings, justification, and sentiment score
        """
        try:
            print(f"📰 Performing sentiment analysis for {ticker}...")
            
            # Collect data from various sources
            news_sentiment = self._analyze_news_sentiment(ticker)
            social_sentiment = self._analyze_social_sentiment(ticker)
            reddit_sentiment = self._analyze_reddit_sentiment(ticker)
            
            # Combine sentiment scores
            combined_sentiment = self._combine_sentiment_scores(
                news_sentiment, social_sentiment, reddit_sentiment
            )
            
            # Determine ratings for different time horizons
            ratings = self._determine_ratings(combined_sentiment)
            
            # Generate justification
            justification = self._generate_justification(
                combined_sentiment, news_sentiment, social_sentiment, reddit_sentiment
            )
            
            return {
                'ratings': ratings,
                'justification': justification,
                'sentiment_score': combined_sentiment
            }
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {str(e)}")
            return self._get_error_response()
    
    def _setup_twitter_client(self):
        """Setup Twitter client if credentials are available."""
        if TWITTER_BEARER_TOKEN != "YOUR_TWITTER_BEARER_TOKEN_HERE":
            try:
                return tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
            except Exception as e:
                print(f"Warning: Could not setup Twitter client: {e}")
        return None
    
    def _setup_reddit_client(self):
        """Setup Reddit client if credentials are available."""
        if (REDDIT_CLIENT_ID != "YOUR_REDDIT_CLIENT_ID_HERE" and 
            REDDIT_CLIENT_SECRET != "YOUR_REDDIT_CLIENT_SECRET_HERE"):
            try:
                return praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
            except Exception as e:
                print(f"Warning: Could not setup Reddit client: {e}")
        return None
    
    def _analyze_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        if not self.news_api:
            return {'score': 0, 'articles_analyzed': 0, 'error': 'No API key'}
        
        try:
            # Search for news articles about the company
            company_name = self._get_company_name_from_ticker(ticker)
            search_query = f"{ticker} OR {company_name}"
            
            articles = self.news_api.get_everything(
                q=search_query,
                language='en',
                sort_by='relevancy',
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                page_size=SENTIMENT_PARAMS['max_news_articles']
            )
            
            if not articles.get('articles'):
                return {'score': 0, 'articles_analyzed': 0, 'error': 'No articles found'}
            
            # Analyze sentiment of each article
            sentiments = []
            for article in articles['articles']:
                if article.get('title') and article.get('description'):
                    text = f"{article['title']} {article['description']}"
                    sentiment = self._analyze_text_sentiment(text)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return {'score': 0, 'articles_analyzed': 0, 'error': 'No valid articles'}
            
            avg_sentiment = np.mean(sentiments)
            return {
                'score': avg_sentiment,
                'articles_analyzed': len(sentiments),
                'error': None
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing news sentiment: {e}")
            return {'score': 0, 'articles_analyzed': 0, 'error': str(e)}
    
    def _analyze_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Twitter/X posts."""
        if not self.twitter_client:
            return {'score': 0, 'tweets_analyzed': 0, 'error': 'No API access'}
        
        try:
            # Search for tweets mentioning the ticker
            query = f"${ticker} -is:retweet"
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=SENTIMENT_PARAMS['max_tweets'],
                tweet_fields=['created_at', 'lang']
            )
            
            if not tweets.data:
                return {'score': 0, 'tweets_analyzed': 0, 'error': 'No tweets found'}
            
            # Analyze sentiment of each tweet
            sentiments = []
            for tweet in tweets.data:
                if tweet.lang == 'en':  # Only analyze English tweets
                    sentiment = self._analyze_text_sentiment(tweet.text)
                    sentiments.append(sentiment)
            
            if not sentiments:
                return {'score': 0, 'tweets_analyzed': 0, 'error': 'No valid tweets'}
            
            avg_sentiment = np.mean(sentiments)
            return {
                'score': avg_sentiment,
                'tweets_analyzed': len(sentiments),
                'error': None
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing social sentiment: {e}")
            return {'score': 0, 'tweets_analyzed': 0, 'error': str(e)}
    
    def _analyze_reddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze sentiment from Reddit posts and comments."""
        if not self.reddit_client:
            return {'score': 0, 'posts_analyzed': 0, 'error': 'No API access'}
        
        try:
            # Search for posts mentioning the ticker
            subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
            all_posts = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    search_results = subreddit.search(f"{ticker}", limit=20, sort='hot')
                    all_posts.extend(list(search_results))
                except Exception as e:
                    print(f"Warning: Could not search subreddit {subreddit_name}: {e}")
                    continue
            
            if not all_posts:
                return {'score': 0, 'posts_analyzed': 0, 'error': 'No posts found'}
            
            # Analyze sentiment of posts and comments
            sentiments = []
            for post in all_posts[:SENTIMENT_PARAMS['max_reddit_posts']]:
                # Analyze post title and body
                if post.title:
                    title_sentiment = self._analyze_text_sentiment(post.title)
                    sentiments.append(title_sentiment)
                
                if post.selftext:
                    body_sentiment = self._analyze_text_sentiment(post.selftext)
                    sentiments.append(body_sentiment)
                
                # Analyze top comments
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:5]:  # Top 5 comments
                        if comment.body:
                            comment_sentiment = self._analyze_text_sentiment(comment.body)
                            sentiments.append(comment_sentiment)
                except Exception as e:
                    continue
            
            if not sentiments:
                return {'score': 0, 'posts_analyzed': 0, 'error': 'No valid content'}
            
            avg_sentiment = np.mean(sentiments)
            return {
                'score': avg_sentiment,
                'posts_analyzed': len(sentiments),
                'error': None
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing Reddit sentiment: {e}")
            return {'score': 0, 'posts_analyzed': 0, 'error': str(e)}
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of a single text using VADER or TextBlob."""
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            if self.vader_analyzer:
                # Use VADER sentiment analyzer
                scores = self.vader_analyzer.polarity_scores(cleaned_text)
                return scores['compound']  # Returns value between -1 and 1
            else:
                # Fallback to TextBlob
                blob = TextBlob(cleaned_text)
                return blob.sentiment.polarity  # Returns value between -1 and 1
                
        except Exception as e:
            print(f"Warning: Error analyzing text sentiment: {e}")
            return 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing URLs, special characters, etc."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_company_name_from_ticker(self, ticker: str) -> str:
        """Get company name from ticker symbol."""
        # This is a simple mapping - in a real implementation, you might use an API
        ticker_to_name = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc',
            'AMZN': 'Amazon.com Inc',
            'TSLA': 'Tesla Inc',
            'META': 'Meta Platforms Inc',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc',
            'JPM': 'JPMorgan Chase & Co',
            'JNJ': 'Johnson & Johnson'
        }
        return ticker_to_name.get(ticker, ticker)
    
    def _combine_sentiment_scores(self, news_sentiment: Dict, social_sentiment: Dict, 
                                 reddit_sentiment: Dict) -> float:
        """Combine sentiment scores from different sources with weights."""
        try:
            # Weight the different sources (news is most reliable, social media less so)
            weights = {'news': 0.5, 'social': 0.3, 'reddit': 0.2}
            
            combined_score = 0.0
            total_weight = 0.0
            
            if news_sentiment.get('score') is not None and not news_sentiment.get('error'):
                combined_score += news_sentiment['score'] * weights['news']
                total_weight += weights['news']
            
            if social_sentiment.get('score') is not None and not social_sentiment.get('error'):
                combined_score += social_sentiment['score'] * weights['social']
                total_weight += weights['social']
            
            if reddit_sentiment.get('score') is not None and not reddit_sentiment.get('error'):
                combined_score += reddit_sentiment['score'] * weights['reddit']
                total_weight += weights['reddit']
            
            if total_weight > 0:
                return combined_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: Error combining sentiment scores: {e}")
            return 0.0
    
    def _determine_ratings(self, sentiment_score: float) -> Dict[str, str]:
        """Determine ratings based on sentiment score."""
        try:
            thresholds = SENTIMENT_PARAMS['sentiment_thresholds']
            
            if sentiment_score >= thresholds['very_positive']:
                rating = 'Strong Buy'
            elif sentiment_score >= thresholds['positive']:
                rating = 'Buy'
            elif sentiment_score >= thresholds['neutral']:
                rating = 'Hold'
            elif sentiment_score >= thresholds['negative']:
                rating = 'Sell'
            else:
                rating = 'Strong Sell'
            
            # Sentiment is more short-term, so ratings vary by time horizon
            ratings = {
                '3m': rating,
                '6m': rating if abs(sentiment_score) > 0.3 else 'Hold',  # Less confident for longer term
                '12m': 'Hold'  # Sentiment has limited predictive value for 1 year
            }
            
            return ratings
            
        except Exception as e:
            print(f"Warning: Error determining ratings: {e}")
            return {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'}
    
    def _generate_justification(self, combined_sentiment: float, news_sentiment: Dict, 
                               social_sentiment: Dict, reddit_sentiment: Dict) -> str:
        """Generate justification for the sentiment analysis rating."""
        try:
            justification_parts = []
            
            # Overall sentiment
            if combined_sentiment > 0.3:
                justification_parts.append("Overall market sentiment is positive")
            elif combined_sentiment < -0.3:
                justification_parts.append("Overall market sentiment is negative")
            else:
                justification_parts.append("Overall market sentiment is neutral")
            
            # News sentiment
            if news_sentiment.get('score') and not news_sentiment.get('error'):
                if news_sentiment['score'] > 0.2:
                    justification_parts.append("Recent news coverage is favorable")
                elif news_sentiment['score'] < -0.2:
                    justification_parts.append("Recent news coverage is unfavorable")
                else:
                    justification_parts.append("Recent news coverage is neutral")
            
            # Social media sentiment
            if social_sentiment.get('score') and not social_sentiment.get('error'):
                if social_sentiment['score'] > 0.2:
                    justification_parts.append("Social media sentiment is bullish")
                elif social_sentiment['score'] < -0.2:
                    justification_parts.append("Social media sentiment is bearish")
                else:
                    justification_parts.append("Social media sentiment is mixed")
            
            # Reddit sentiment
            if reddit_sentiment.get('score') and not reddit_sentiment.get('error'):
                if reddit_sentiment['score'] > 0.2:
                    justification_parts.append("Reddit community sentiment is positive")
                elif reddit_sentiment['score'] < -0.2:
                    justification_parts.append("Reddit community sentiment is negative")
                else:
                    justification_parts.append("Reddit community sentiment is neutral")
            
            if not justification_parts:
                justification_parts.append("Limited sentiment data available for analysis")
            
            return ". ".join(justification_parts) + "."
            
        except Exception as e:
            return f"Error generating justification: {str(e)}"
    
    def _get_error_response(self) -> Dict[str, Any]:
        """Return error response when analysis fails."""
        return {
            'ratings': {'3m': 'Hold', '6m': 'Hold', '12m': 'Hold'},
            'justification': 'Unable to perform sentiment analysis due to data retrieval errors.',
            'sentiment_score': 0.0
        }
