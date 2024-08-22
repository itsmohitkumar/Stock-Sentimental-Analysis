import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.apify_api_token = os.getenv("APIFY_API_TOKEN")
        if not self.apify_api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables.")
        self.models_dir = os.getenv("MODELS_DIR", 'model')
        self.dataset_path = os.getenv("DATASET_PATH", os.path.join('data', 'Sentiment.csv'))
        self.model_path = os.path.join(self.models_dir, 'sentiment_model.joblib')
        self.vectorizer_path = os.path.join(self.models_dir, 'vectorizer.joblib')
        self.labelencoder_path = os.path.join(self.models_dir, 'labelencoder.joblib')
        self.tweet_handles = os.getenv("TWEET_HANDLES", "Apify").split(',')
        self.tweets_desired = int(os.getenv("TWEETS_DESIRED", 10))
        os.makedirs(self.models_dir, exist_ok=True)
