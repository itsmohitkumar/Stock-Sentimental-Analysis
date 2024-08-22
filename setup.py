from setuptools import setup, find_packages

setup(
    name='sentiment_analysis',
    version='0.1.0',
    description='A package for sentiment analysis of tweets',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    url='https://github.com/itsmohitkumar/Stock-Sentimental-Analysis',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'nltk',
        'fastapi',
        'pydantic',
        'joblib',
        'pandas',
        'scikit-learn',
        'apify-client',
        'python-dotenv',
        'uvicorn',
    ],
    entry_points={
        'console_scripts': [
            'install_nltk_data=install_nltk_data:install_nltk_data',
            'run_app=app:main',  # Adjust this line to point to the function you want to run
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
