import asyncio
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from app import db, NewsItem
from app import app

# nltk packages
nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']

# download only if not already downloaded
for package in nltk_packages:
    if not nltk.data.find(f'tokenizers/{package}'):
        nltk.download(package)

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
        asyncio.run(update_keywords())
