import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

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
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
