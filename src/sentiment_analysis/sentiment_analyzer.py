import os
import re
import pandas as pd
import nltk
import logging
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class SentimentAnalyzer:
    def __init__(self, config: Config):
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
