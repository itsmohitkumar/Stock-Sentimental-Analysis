import pytest
from unittest.mock import MagicMock
from src.sentiment_analysis.tweet_fetcher import TweetFetcher
from src.config import Config

@pytest.fixture
def tweet_fetcher():
    config = Config()
    return TweetFetcher(config)

def test_fetch_tweets(tweet_fetcher):
    tweet_fetcher.client = MagicMock()
    tweet_fetcher.client.actor.return_value.call.return_value = {
        "defaultDatasetId": "dummy_dataset_id"
    }
    tweet_fetcher.client.dataset.return_value.iterate_items.return_value = [{"full_text": "Sample tweet"}]
    df = tweet_fetcher.fetch_tweets()
    assert not df.empty
    assert 'full_text' in df.columns
