import os
import re
import pandas as pd
import nltk
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import dump, load
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from apify_client import ApifyClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configuration
class Config:
    def __init__(self):
        self.apify_api_token = os.getenv("APIFY_API_TOKEN")
        if not self.apify_api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment variables.")
        self.models_dir = 'model'
        self.dataset_path = os.path.join('data', 'Sentiment.csv')
        self.model_path = os.path.join(self.models_dir, 'sentiment_model.joblib')
        self.vectorizer_path = os.path.join(self.models_dir, 'vectorizer.joblib')
        self.labelencoder_path = os.path.join(self.models_dir, 'labelencoder.joblib')
        os.makedirs(self.models_dir, exist_ok=True)

# Text Processor
class TextProcessor:
    def __init__(self):
        nltk.data.path.append('/nltk_data')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.pattern = "(#\w+)|(RT\s@\w+:)|(http.*)|(@\w+)"

    def clean_text(self, text):
        try:
            sentence = re.sub(self.pattern, '', text)
            words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence.split() if word.lower() not in self.stop_words]
            return ' '.join(words)
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return ""

# Sentiment Analyzer
class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor()
        self.model = None
        self.vectorizer = None
        self.labelencoder = None

    def train_or_load_model(self):
        if self._model_exists():
            self._load_model()
        else:
            self._train_model()

    def _model_exists(self):
        return all(os.path.exists(path) for path in [
            self.config.model_path, 
            self.config.vectorizer_path, 
            self.config.labelencoder_path
        ])

    def _load_model(self):
        try:
            self.model = load(self.config.model_path)
            self.vectorizer = load(self.config.vectorizer_path)
            self.labelencoder = load(self.config.labelencoder_path)
            logging.info("Loaded existing model, vectorizer, and label encoder.")
        except Exception as e:
            logging.error(f"Error loading model or vectorizer: {e}")
            raise

    def _train_model(self):
        try:
            dataset = pd.read_csv(self.config.dataset_path)
            dataset = dataset[['text', 'sentiment']]
            dataset = dataset[dataset['sentiment'] != 'Neutral']
            logging.info("Loaded and preprocessed dataset.")
        except FileNotFoundError as e:
            logging.error(f"Dataset file not found: {e}")
            raise

        processed_data = self._process_data(dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data['tweets'], processed_data['sentiments'], 
            test_size=0.1, random_state=42
        )

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        X_train_vect = self.vectorizer.fit_transform(X_train)
        X_test_vect = self.vectorizer.transform(X_test)

        param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
        grid_search.fit(X_train_vect, y_train)
        logging.info("Completed grid search for hyperparameter tuning.")

        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model Accuracy: {accuracy:.2f}")

        dump(self.model, self.config.model_path)
        dump(self.vectorizer, self.config.vectorizer_path)
        dump(self.labelencoder, self.config.labelencoder_path)
        logging.info("Saved model, vectorizer, and label encoder.")

    def _process_data(self, dataset):
        cleaned_tweets = dataset['text'].apply(self.text_processor.clean_text)
        self.labelencoder = LabelEncoder()
        encoded_sentiments = self.labelencoder.fit_transform(dataset['sentiment'])
        return pd.DataFrame({'tweets': cleaned_tweets, 'sentiments': encoded_sentiments})

    def predict_sentiment(self, tweets):
        cleaned_tweets = [self.text_processor.clean_text(tweet) for tweet in tweets]
        X_new_tweets = self.vectorizer.transform(cleaned_tweets)
        predictions = self.model.predict(X_new_tweets)
        return self.labelencoder.inverse_transform(predictions)

    def calculate_average_sentiment(self, sentiments):
        sentiment_scores = {'Positive': 1, 'Negative': 0, 'Neutral': 0.5}
        scores = [sentiment_scores.get(sentiment, 0.5) for sentiment in sentiments]
        return sum(scores) / len(scores) if scores else None

# Tweet Fetcher
class TweetFetcher:
    def __init__(self, config):
        self.client = ApifyClient(config.apify_api_token)

    def fetch_tweets(self, handles, tweets_desired=10):
        run_input = {
            "handles": handles,
            "tweetsDesired": tweets_desired,
            "proxyConfig": {"useApifyProxy": True},
        }

        try:
            run = self.client.actor("quacker/twitter-scraper").call(run_input=run_input)
            dataset_id = run["defaultDatasetId"]
            tweets = [item for item in self.client.dataset(dataset_id).iterate_items()]
            logging.info("Fetched new tweets using Apify.")
            return pd.DataFrame(tweets)
        except Exception as e:
            logging.error(f"Error fetching tweets: {e}")
            raise

# FastAPI Integration
class SentimentAPI:
    def __init__(self, sentiment_analyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.app = FastAPI()

        # Define API routes
        @self.app.post("/analyze/")
        async def analyze_tweets(request: TweetRequest):
            try:
                sentiments = self.sentiment_analyzer.predict_sentiment(request.tweets)
                avg_score = self.sentiment_analyzer.calculate_average_sentiment(sentiments)
                return {"sentiments": sentiments.tolist(), "average_sentiment_score": avg_score}
            except Exception as e:
                logging.error(f"Error in sentiment analysis: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")

# Request model for FastAPI
class TweetRequest(BaseModel):
    tweets: list[str]

# Main Function
def main():
    config = Config()
    sentiment_analyzer = SentimentAnalyzer(config)
    tweet_fetcher = TweetFetcher(config)

    # Train or load the model
    sentiment_analyzer.train_or_load_model()

    # Fetch and analyze tweets
    df = tweet_fetcher.fetch_tweets(handles=["Apify"], tweets_desired=10)
    if not df.empty:
        df['sentiment'] = sentiment_analyzer.predict_sentiment(df['full_text'])
        df['sentiment_score'] = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 0.5})
        avg_score = sentiment_analyzer.calculate_average_sentiment(df['sentiment'])
        logging.info(f"Average Sentiment Score: {avg_score:.2f}")

        # Display the results
        print(df[['conversation_id', 'full_text', 'sentiment', 'sentiment_score']])
        print(f"Average Sentiment Score: {avg_score:.2f}")

    # Start the FastAPI server
    sentiment_api = SentimentAPI(sentiment_analyzer)
    import uvicorn
    uvicorn.run(sentiment_api.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
