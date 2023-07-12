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

def custom_tokenizer(sentence):
    # use nltk's word_tokenize
    token_list = nltk.word_tokenize(sentence)
    # filter out short tokens
    token_list = [token for token in token_list if len(token) > 2]
    return token_list

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
            
            # Filter out short and numeric tokens
            tokens = [token for token in tokens if len(token) > 2 and not token.isnumeric()]
    
            # Apply POS tagging
            tagged_tokens = pos_tag(tokens)
    
            # Keep only nouns, adjectives, and verbs
            tokens = [word for word, pos in tagged_tokens if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]

            # Lemmatize with POS tag
            from nltk.corpus import wordnet
            def get_wordnet_pos(treebank_tag):
                if treebank_tag.startswith('J'):
                    return wordnet.ADJ
                elif treebank_tag.startswith('V'):
                    return wordnet.VERB
                elif treebank_tag.startswith('N'):
                    return wordnet.NOUN
                elif treebank_tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return wordnet.NOUN
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)) for word, pos in tagged_tokens]

            # Define additional stopwords that you want to ignore
            additional_stopwords = ['npr', 'pennlive', '2023', 'site', 'get', 'said', 'look', 'etc', 'was', 'were', 'has', 'the', 'privacy', 'medium', 'say', 'may', 'give', 'since', 'choice']
            
            # Remove stopwords
            stop_words = set(stopwords.words('english') + additional_stopwords)
            processed_words = [word for word in lemmatized_words if not word in stop_words]

            all_words = ' '.join(processed_words)

            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=custom_tokenizer)  # Include unigrams and bi-grams
            vectors = vectorizer.fit_transform([' '.join(processed_words)])  # Use processed_words instead of tokens
            names = vectorizer.get_feature_names_out()
            data = vectors.todense().tolist()
            
            # Get top10 keywords based on tf-idf score
            tfidf_scores = sorted(list(zip(names, data[0])), key=lambda x: x[1], reverse=True)[:10]
            single_word_keywords = []
            two_word_keywords = []
            for word, score in tfidf_scores:
                if ' ' in word:
                    two_word_keywords.append(word)
                else:
                    single_word_keywords.append(word)
            
            # Remove single-word keywords that are part of a two-word keyword
            for keyword in two_word_keywords:
                word1, word2 = keyword.split()
                if word1 in single_word_keywords:
                    single_word_keywords.remove(word1)
                if word2 in single_word_keywords:
                    single_word_keywords.remove(word2)
            
            top_keywords = ', '.join(single_word_keywords + two_word_keywords)

            # Check if the news item already exists in the database
            news_item = NewsItem.query.filter_by(link=item.link.text).first()
            if not NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords):
                news_item = NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords)

'''
def remove_keyword(session, keyword):
    # Get all NewsItems from the database
    news_items = session.query(NewsItem).all()
    
    for news_item in news_items:
        # Get the current keywords
        keywords = news_item.keywords.split(', ')
        
        # Create a new list to store the updated keywords
        new_keywords = []
        
        for k in keywords:
            # Split the keyword into words
            words = k.split()
            
            # If the specified keyword is not in the words, add the keyword to the new list
            if keyword not in words:
                new_keywords.append(k)
        
        # Set the NewsItem's keywords to the updated list
        news_item.keywords = ', '.join(new_keywords)
        
        # Add the NewsItem back to the session
        session.add(news_item)
    
    # Commit the changes to the database
    session.commit()
'''


if __name__ == "__main__":
    with app.app_context():
        asyncio.run(scrape_news())
        # remove_keyword(db.session, "give")

