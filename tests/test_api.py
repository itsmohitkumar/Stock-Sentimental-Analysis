import pytest
from fastapi.testclient import TestClient
from src.sentiment_analysis.api import SentimentAPI
from src.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from src.config import Config

@pytest.fixture
def client():
    config = Config()
    sentiment_analyzer = SentimentAnalyzer(config)
    api = SentimentAPI(sentiment_analyzer)
    return TestClient(api.app)

def test_analyze_sentiments(client):
    response = client.post("/analyze/sentiments/", json={"tweets": ["I love this!", "I hate this!"]})
    assert response.status_code == 200
    assert "sentiments" in response.json()
