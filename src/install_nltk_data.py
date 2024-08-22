import os
import subprocess

def install_nltk_data():
    nltk_data_dir = os.path.expanduser('~/nltk_data/corpora')
    if not os.path.exists(nltk_data_dir):
        print("NLTK data not found. Installing...")
        os.makedirs(nltk_data_dir, exist_ok=True)
        # Download and unzip NLTK data
        urls = {
            'stopwords': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip',
            'wordnet': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'
        }
        for name, url in urls.items():
            zip_file = os.path.join(nltk_data_dir, f'{name}.zip')
            subprocess.run(['curl', url, '-o', zip_file], check=True)
            subprocess.run(['unzip', zip_file, '-d', nltk_data_dir], check=True)
            os.remove(zip_file)

if __name__ == "__main__":
    install_nltk_data()
