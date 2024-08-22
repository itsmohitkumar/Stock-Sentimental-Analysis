# %% [markdown]
# Step 1: Install Required Packages

# %%
#%pip install apify-client python-dotenv pandas numpy nltk scikit-learn

# %% [markdown]
# Step 2: Import Libraries

# %%
from apify_client import ApifyClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os

# %% [markdown]
# Step 3: Load Environment Variables and Initialize ApifyClient

# %%
# Load environment variables from .env file
load_dotenv()

# Initialize the ApifyClient with your Apify API token
apify_api_token = os.getenv("APIFY_API_TOKEN")
client = ApifyClient(apify_api_token)

# Test the connection by printing the client object
print(client)

# %% [markdown]
# Step 4: Prepare the Actor Input and Run the Actor

# %%
run_input = {
    "handles": ["Apify"],
    "tweetsDesired": 10,
    "proxyConfig": {"useApifyProxy": True},
}

# Print the prepared input to verify
print(run_input)

# Run the actor
run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
print(run)

# %% [markdown]
# Step 5: Fetch and Save Actor Results

# %%
dataset_id = run["defaultDatasetId"]
print(f"💾 Check your data here: https://console.apify.com/storage/datasets/{dataset_id}")

# Create a list to hold the tweet data
tweets = []

# Fetch the dataset items and append them to the list
for item in client.dataset(dataset_id).iterate_items():
    tweets.append(item)

# Convert the list of tweets into a pandas DataFrame
df = pd.DataFrame(tweets)

# Check if the expected columns are present
if 'conversation_id' in df.columns and 'full_text' in df.columns:
    filtered_df = df[['conversation_id', 'full_text']]
else:
    print("Expected columns are not in the dataset")
    filtered_df = pd.DataFrame()

# Save the DataFrame to a CSV file
filtered_df.to_csv('tweets_data.csv', index=False)
print(filtered_df.head())

# %% [markdown]
# Step 6: Set the NLTK Data Path (if needed)

# %%
nltk.download('stopwords')
nltk.download('wordnet')

# %% [markdown]
# Step 7: Load the Tweets Data and Process

# %%
# Load the tweets data from the CSV file created earlier
tweets_df = pd.read_csv('tweets_data.csv')
print(tweets_df.head())

# %% [markdown]
# Step 8: Clean and Preprocess the Text Data

# %%
pattern = "(#\w+)|(RT\s@\w+:)|(http.*)|(@\w+)"
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(data):
    tweets = []
    for text in data:
        sentence = re.sub(pattern, '', text)
        words = [e.lower() for e in sentence.split()]
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        words = ' '.join(words)
        tweets.append(words)
    return tweets

# Apply the cleaning function to the tweets
cleaned_tweets = clean_text(tweets_df['full_text'])

# Create a DataFrame for the processed data
processed_data = pd.DataFrame({'tweets': cleaned_tweets})

# Display the processed data
print(processed_data.head())

# %% [markdown]
# Step 9: Placeholder for Model Training and Evaluation

# Since we don't have sentiment labels for training, we'll skip this part. 
# You can use your own labeled sentiment dataset to train the model.

# %% [markdown]
# Example of how you might proceed with training if you had sentiment labels
# %% [markdown]
# Step 10: Example Code for Sentiment Analysis (if labels were available)

# %%
# Uncomment and modify the following steps if you have a labeled dataset
# dataset = pd.read_csv('Sentiment.csv') # Load your labeled sentiment dataset
# dataset = dataset[['text', 'sentiment']]
# dataset = dataset[dataset['sentiment'] != 'Neutral']

# train, test = train_test_split(dataset, test_size=0.1)

# train_tweets, train_sentiments = clean_text(train['text']), train['sentiment']
# processed_train_data = pd.DataFrame({'tweets': train_tweets, 'sentiments': train_sentiments})

# labelencoder = LabelEncoder()
# processed_train_data['sentiments'] = labelencoder.fit_transform(processed_train_data['sentiments'])

# cv = CountVectorizer(ngram_range=(1, 3))
# cv.fit(processed_train_data['tweets'])
# X_train = cv.transform(processed_train_data['tweets'])

# classifier = MultinomialNB()
# classifier.fit(X_train.toarray(), processed_train_data['sentiments'])

# test_tweets, test_sentiments = clean_text(test['text']), test['sentiment']
# final_test_data = pd.DataFrame({'tweets': test_tweets, 'sentiments': test_sentiments})
# X_test = cv.transform(final_test_data['tweets'])

# y_pred = classifier.predict(X_test.toarray())

# final_test_data['sentiments'] = labelencoder.transform(final_test_data['sentiments'])
# accuracy = accuracy_score(y_pred, final_test_data['sentiments'])
# print(f"Accuracy: {accuracy}")

# print(final_test_data)
