import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_sqlalchemy import SQLAlchemy
from dateutil.parser import parse
import os
import urllib.parse
from app import db, NewsItem
from app import app
from urllib.parse import urlparse

# nltk packages
nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']

# download only if not already downloaded
for package in nltk_packages:
    if not os.path.exists(f'/app/nltk_data/{package}'):
        nltk.download(package)

async def fetch(url, session):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all(urls):
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch(url, session))
        responses = await asyncio.gather(*tasks)
        return responses

async def scrape_news():
    urls = [
        "https://www.blackwallstreet-pa.com/feed/",
        "https://www.pennlive.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://feeds.npr.org/1001/rss.xml"
    ]
    responses = await fetch_all(urls)

    for response in responses:
        soup = BeautifulSoup(response, "xml")
        items = soup.find_all("item")
        article_urls = [item.link.text for item in items]
        article_responses = await fetch_all(article_urls)

        for item, article_response in zip(items, article_responses):
            # Extract source from URL
            parsed_url = urlparse(item.link.text)
            source = parsed_url.netloc

            # Process article content
            article_soup = BeautifulSoup(article_response, "html.parser")
            article_content = article_soup.get_text()

            # New code to get the main image in the article
            main_image = article_soup.find('meta', attrs={'property': 'og:image'})
            if main_image and 'content' in main_image.attrs:
                image = main_image['content']
            else:
                image = None

            # Tokenize text
            tokens = word_tokenize(article_content)
            
            # Define additional stopwords that you want to ignore
            additional_stopwords = ['npr', 'pennlive', '2023', 'site', 'get', 'said', 'look', 'etc']
            
            # Filter out short and numeric tokens
            tokens = [token for token in tokens if len(token) > 2 and not token.isnumeric()]
    
            # Apply POS tagging
            tagged_tokens = pos_tag(tokens)
    
            # Keep only nouns, adjectives, and verbs
            tokens = [word for word, pos in tagged_tokens if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]

            # Remove stopwords, lemmatize, and convert to lowercase
            stop_words = set(stopwords.words('english') + additional_stopwords)
            lemmatizer = WordNetLemmatizer()
            processed_words = [lemmatizer.lemmatize(word.lower()) for word in tokens if not word.lower() in stop_words]

            all_words = ' '.join(processed_words)

            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Include unigrams and bi-grams
            vectors = vectorizer.fit_transform([' '.join(tokens)])
            names = vectorizer.get_feature_names_out()
            data = vectors.todense().tolist()

            # Get top10 keywords based on tf-idf score
            tfidf_scores = sorted(list(zip(names, data[0])), key=lambda x: x[1], reverse=True)[:10]
            top_keywords = ', '.join([word for word, score in tfidf_scores])

            # Check if the news item already exists in the database
            news_item = NewsItem.query.filter_by(link=item.link.text).first()
            if not NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords):
                news_item = NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords)

async def update_keywords():
    # Get all news items from the database
    news_items = NewsItem.query.all()

    # Define additional stopwords that you want to ignore
    additional_stopwords = ['npr', 'pennlive', '2023', 'site', 'get', 'said', 'look', 'etc']

    # Iterate over all news items
    for news_item in news_items:
        # Process article content
        article_content = news_item.all_words

        # Tokenize text
        tokens = word_tokenize(article_content)

        # Filter out short and numeric tokens
        tokens = [token for token in tokens if len(token) > 2 and not token.isnumeric()]

        # Apply POS tagging
        tagged_tokens = pos_tag(tokens)

        # Keep only nouns, adjectives, and verbs
        tokens = [word for word, pos in tagged_tokens if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]

        # Remove stopwords, lemmatize, and convert to lowercase
        stop_words = set(stopwords.words('english') + additional_stopwords)
        lemmatizer = WordNetLemmatizer()
        processed_words = [lemmatizer.lemmatize(word.lower()) for word in tokens if not word.lower() in stop_words]

        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Include unigrams and bi-grams
        vectors = vectorizer.fit_transform([' '.join(processed_words)])
        names = vectorizer.get_feature_names_out()
        data = vectors.todense().tolist()

        # Get top10 keywords based on tf-idf score
        tfidf_scores = sorted(list(zip(names, data[0])), key=lambda x: x[1], reverse=True)[:10]
        top_keywords = ', '.join([word for word, score in tfidf_scores])

        # Update the news item's keywords in the database
        news_item.keywords = top_keywords
        db.session.commit()


if __name__ == "__main__":
    with app.app_context():
        asyncio.run(scrape_news())
        asyncio.run(update_keywords())
