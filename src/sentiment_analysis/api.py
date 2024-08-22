from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

class SentimentAPI:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.app = FastAPI()

        # Define API routes
        @self.app.get("/")
        def read_root():
            return {"message": "Welcome to the Sentiment Analysis API"}

        @self.app.post("/analyze/sentiments/")
        async def analyze_sentiments(request: TweetRequest):
            try:
                sentiments = self.sentiment_analyzer.predict_sentiment(request.tweets)
                return {"sentiments": sentiments.tolist()}
            except Exception as e:
                logging.error(f"Error in sentiment analysis: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

        @self.app.post("/analyze/average_score/")
        async def analyze_average_score(request: TweetRequest):
            try:
                sentiments = self.sentiment_analyzer.predict_sentiment(request.tweets)
                avg_score = self.sentiment_analyzer.calculate_average_sentiment(sentiments)
                return {"average_sentiment_score": avg_score}
            except Exception as e:
                logging.error(f"Error in sentiment analysis: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

# Request model for FastAPI
class TweetRequest(BaseModel):
    tweets: list[str]
