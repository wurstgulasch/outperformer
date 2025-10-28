"""
Sentiment Analysis Module
==========================
Sentiment analysis using Hugging Face transformers.
"""

from transformers import pipeline
from typing import List, Dict, Optional
from loguru import logger
import torch


class SentimentAnalyzer:
    """Sentiment analyzer for financial news and social media."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None
    ):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        try:
            # Initialize sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if device == 'cuda' else -1
            )
            logger.info(f"SentimentAnalyzer initialized with {model_name} on {device}")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {e}")
            self.sentiment_pipeline = None

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        if not self.sentiment_pipeline:
            logger.warning("Sentiment pipeline not initialized")
            return {'label': 'neutral', 'score': 0.0}

        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
            return {
                'label': result['label'].lower(),
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.0}

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment dictionaries
        """
        if not self.sentiment_pipeline:
            logger.warning("Sentiment pipeline not initialized")
            return [{'label': 'neutral', 'score': 0.0}] * len(texts)

        try:
            # Truncate texts
            truncated_texts = [text[:512] for text in texts]
            results = self.sentiment_pipeline(truncated_texts)
            
            return [
                {
                    'label': result['label'].lower(),
                    'score': result['score']
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error analyzing batch sentiment: {e}")
            return [{'label': 'neutral', 'score': 0.0}] * len(texts)

    def get_sentiment_score(self, text: str) -> float:
        """
        Get numerical sentiment score.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1 to 1, where -1 is negative, 0 is neutral, 1 is positive)
        """
        sentiment = self.analyze_text(text)
        
        # Map sentiment to score
        if sentiment['label'] == 'positive':
            return sentiment['score']
        elif sentiment['label'] == 'negative':
            return -sentiment['score']
        else:
            return 0.0

    def aggregate_sentiments(self, texts: List[str]) -> Dict:
        """
        Aggregate sentiment from multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with aggregated sentiment metrics
        """
        sentiments = self.analyze_batch(texts)
        
        # Count labels
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0.0
        
        for sentiment in sentiments:
            label = sentiment['label']
            score = sentiment['score']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if label == 'positive':
                total_score += score
            elif label == 'negative':
                total_score -= score

        # Calculate metrics
        total = len(sentiments)
        avg_score = total_score / total if total > 0 else 0.0
        
        return {
            'positive_count': label_counts['positive'],
            'negative_count': label_counts['negative'],
            'neutral_count': label_counts['neutral'],
            'positive_ratio': label_counts['positive'] / total if total > 0 else 0.0,
            'negative_ratio': label_counts['negative'] / total if total > 0 else 0.0,
            'neutral_ratio': label_counts['neutral'] / total if total > 0 else 0.0,
            'average_score': avg_score,
            'total_texts': total
        }


class MarketSentimentTracker:
    """Track market sentiment over time."""

    def __init__(self, analyzer: Optional[SentimentAnalyzer] = None):
        """
        Initialize market sentiment tracker.

        Args:
            analyzer: SentimentAnalyzer instance
        """
        self.analyzer = analyzer or SentimentAnalyzer()
        self.sentiment_history = []
        
        logger.info("MarketSentimentTracker initialized")

    def add_sentiment_data(self, texts: List[str], timestamp: Optional[str] = None):
        """
        Add sentiment data for a time period.

        Args:
            texts: List of texts from news/social media
            timestamp: Timestamp for the data
        """
        aggregated = self.analyzer.aggregate_sentiments(texts)
        
        self.sentiment_history.append({
            'timestamp': timestamp,
            'sentiment': aggregated
        })
        
        logger.info(f"Added sentiment data: avg_score={aggregated['average_score']:.3f}")

    def get_current_sentiment(self) -> Optional[Dict]:
        """
        Get most recent sentiment data.

        Returns:
            Dictionary with current sentiment or None
        """
        if not self.sentiment_history:
            return None
        return self.sentiment_history[-1]['sentiment']

    def get_sentiment_trend(self, periods: int = 10) -> Dict:
        """
        Get sentiment trend over recent periods.

        Args:
            periods: Number of recent periods to analyze

        Returns:
            Dictionary with trend metrics
        """
        if not self.sentiment_history:
            return {
                'trend': 'neutral',
                'strength': 0.0,
                'average_score': 0.0
            }

        recent = self.sentiment_history[-periods:]
        scores = [s['sentiment']['average_score'] for s in recent]
        
        avg_score = sum(scores) / len(scores)
        
        # Determine trend
        if len(scores) >= 2:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            change = avg_second - avg_first
            
            if change > 0.1:
                trend = 'improving'
            elif change < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
                
            strength = abs(change)
        else:
            trend = 'neutral'
            strength = 0.0

        return {
            'trend': trend,
            'strength': strength,
            'average_score': avg_score,
            'periods_analyzed': len(recent)
        }

    def clear_old_data(self, keep_last: int = 100):
        """
        Clear old sentiment data.

        Args:
            keep_last: Number of recent entries to keep
        """
        if len(self.sentiment_history) > keep_last:
            self.sentiment_history = self.sentiment_history[-keep_last:]
            logger.info(f"Cleared old sentiment data, kept last {keep_last}")
