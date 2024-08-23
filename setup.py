from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='Stock-Sentimental-Analysis',
    version='0.1.0',
    description='A package for sentiment analysis of tweets',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
            'run_app=app:main', 
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
