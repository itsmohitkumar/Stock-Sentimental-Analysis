import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of files and directories to create
list_of_files = [
    "src/__init__.py",
    "src/logger.py",
    "src/data_loader.py",
    "src/api/sentiment_analyzer.py",
    "src/stock_predictor.py",
    "src/visualization.py",
    "src/twitter_api.py",
    "src/config.py",
    "notebooks/analysis.ipynb",
    "app.py",
    "requirements.txt",
    "static/.gitkeep",
    "templates/index.html"
]

for filepath in list_of_files:
    path = Path(filepath)
    filedir, filename = path.parent, path.name

    # Create directories if needed
    if filedir != "":
        if not filedir.exists():
            filedir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {filedir}")

    # Create empty file if it doesn't exist or is empty
    if not path.exists() or path.stat().st_size == 0:
        path.touch()  # Create an empty file
        logging.info(f"Created empty file: {path}")
    else:
        logging.info(f"File already exists: {path}")
