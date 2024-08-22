# Sentiment Analysis

## Overview

The **Sentiment Analysis** package provides tools to analyze the sentiment of tweets using machine learning techniques. The package includes functionalities for fetching tweets, preprocessing text, training a sentiment analysis model, and exposing an API for sentiment analysis.

## Project Structure

```plaintext
sentiment_analysis/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── sentiment_analysis/
│       ├── __init__.py
│       ├── sentiment_analyzer.py
│       ├── tweet_fetcher.py
│       ├── api.py
│       └── main.py
│
├── tests/
│   ├── __init__.py
│   ├── test_tweet_fetcher.py
│   └── test_api.py
│
├── install_nltk_data.py
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/itsmohitkumar/Stock-Sentimental-Analysis.git
    cd Stock-Sentimental-Analysis
    ```

2. **Install Dependencies**:

    Create a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

    Alternatively, install the package using `setup.py`:

    ```sh
    python setup.py install
    ```

3. **Install NLTK Data**:

    Before using the package, install the required NLTK data:

    ```sh
    python install_nltk_data.py
    ```

## Usage

### Running the Application

To start the FastAPI server and perform sentiment analysis:

```sh
python -m sentiment_analysis.main
```

The server will be available at `http://localhost:8000`. You can use the following endpoints:

- `POST /analyze/sentiments/`: Analyze sentiment of provided tweets.
- `POST /analyze/average_score/`: Get the average sentiment score of provided tweets.

### Example API Requests

1. **Analyze Sentiments**:

    ```sh
    curl -X POST "http://localhost:8000/analyze/sentiments/" -H "Content-Type: application/json" -d '{"tweets": ["I love this!", "I hate this!"]}'
    ```

2. **Average Sentiment Score**:

    ```sh
    curl -X POST "http://localhost:8000/analyze/average_score/" -H "Content-Type: application/json" -d '{"tweets": ["I love this!", "I hate this!"]}'
    ```

## Configuration

The configuration is managed through environment variables. Create a `.env` file in the root directory with the following format:

```
APIFY_API_TOKEN=your_apify_api_token
```

## Testing

To run tests, use:

```sh
pytest
```

This will run unit tests for components in the `tests/` directory.

## Docker

To build and run the project using Docker:

1. **Build the Docker Image**:

    ```sh
    docker build -t sentiment_analysis .
    ```

2. **Run the Docker Container**:

    ```sh
    docker run -p 8000:8000 sentiment_analysis
    ```

### Code Components

#### `src/__init__.py`

- This file is used to mark the directory as a Python package. It can be left empty or used to initialize package-level variables.

#### `src/config.py`

- **Purpose**: Manages configuration settings and environment variables.
- **Key Components**:
  - Uses `python-dotenv` to load environment variables from a `.env` file.
  - Defines `Config` class with attributes for API tokens, file paths, and other settings.

#### `src/logger.py`

- **Purpose**: Sets up the logging configuration.
- **Key Components**:
  - `setup_logger()` function configures logging with a specific format and log level (`INFO`).

#### `src/sentiment_analysis/__init__.py`

- Similar to `src/__init__.py`, it marks the `sentiment_analysis` directory as a package. Can be left empty.

#### `src/sentiment_analysis/sentiment_analyzer.py`

- **Purpose**: Contains the logic for training and using a sentiment analysis model.
- **Key Components**:
  - `TextProcessor` class: Handles text preprocessing like removing stop words and lemmatizing.
  - `SentimentAnalyzer` class: Manages model training, loading, prediction, and sentiment score calculation.
  - Uses libraries such as `scikit-learn`, `joblib`, and `nltk` for machine learning and text processing.

#### `src/sentiment_analysis/tweet_fetcher.py`

- **Purpose**: Fetches tweets using the Apify API.
- **Key Components**:
  - `TweetFetcher` class: Uses Apify's client to call a Twitter scraper actor and fetch tweets.
  - `fetch_tweets()` method: Fetches tweets based on the configuration and returns them as a DataFrame.

#### `src/sentiment_analysis/api.py`

- **Purpose**: Defines FastAPI endpoints for sentiment analysis.
- **Key Components**:
  - `SentimentAPI` class: Sets up the FastAPI application and routes for analyzing tweet sentiments and calculating average sentiment scores.
  - `TweetRequest` class: Defines the request body schema using Pydantic.

#### `src/sentiment_analysis/main.py`

- **Purpose**: Main entry point for the application.
- **Key Components**:
  - `main()` function: Sets up logging, initializes configuration, sentiment analyzer, and tweet fetcher. Fetches tweets, performs sentiment analysis, and starts the FastAPI server using Uvicorn.

#### `install_nltk_data.py`

- **Purpose**: Installs necessary NLTK data for text processing.
- **Key Components**:
  - `install_nltk_data()` function: Downloads and unzips NLTK datasets if not already present.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**: Create a fork of this repository on GitHub.
2. **Create a Branch**: Create a new branch for your changes.
3. **Make Changes**: Implement your changes and write tests if applicable.
4. **Submit a Pull Request**: Push your changes to your fork and submit a pull request with a detailed description of the changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **NLTK** for natural language processing.
- **FastAPI** for building the API.
- **Apify** for scraping tweets.

Feel free to open issues or contact me for any queries or suggestions!

---

**Author**: Mohit Kumar  
**Email**: mohitpanghal12345@gmail.com  
**GitHub**: [itsmohitkumar](https://github.com/itsmohitkumar)
**LinkedIn**: [mohit-kumar](https://www.linkedin.com/in/itsmohitkumar/)