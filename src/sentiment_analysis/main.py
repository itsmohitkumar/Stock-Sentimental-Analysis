import logging
import uvicorn
from src.config import Config
from .sentiment_analyzer import SentimentAnalyzer
from .tweet_fetcher import TweetFetcher
from .api import SentimentAPI

def create_app():
    config = Config()
    sentiment_analyzer = SentimentAnalyzer(config)
    tweet_fetcher = TweetFetcher(config)

    # Train or load the model
    sentiment_analyzer.train_or_load_model()

    # Fetch and analyze tweets
    df = tweet_fetcher.fetch_tweets()
    if not df.empty:
        df['sentiment'] = sentiment_analyzer.predict_sentiment(df['full_text'])
        df['sentiment_score'] = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 0.5})
        avg_score = sentiment_analyzer.calculate_average_sentiment(df['sentiment'])
        logging.info(f"Average Sentiment Score: {avg_score:.2f}")

    # Create and return the FastAPI app
    sentiment_api = SentimentAPI(sentiment_analyzer)
    return sentiment_api.app

def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)