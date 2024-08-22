import logging
import pandas as pd
from apify_client import ApifyClient
from src.config import Config

class TweetFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.client = ApifyClient(config.apify_api_token)

    def fetch_tweets(self):
        run_input = {
            "handles": self.config.tweet_handles,
            "tweetsDesired": self.config.tweets_desired,
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
