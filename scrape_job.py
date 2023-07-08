import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_sqlalchemy import SQLAlchemy
from dateutil.parser import parse
import os
import urllib.parse
from app import db, NewsItem  

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


async def scrape_news():
    ...

if __name__ == "__main__":
    asyncio.run(scrape_news())
