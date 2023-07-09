import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, url_for, session
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from dateutil.parser import parse
from wordcloud import WordCloud
from sqlalchemy import or_, and_
import os
import urllib.parse
import nltk

# function to remove keywords from the database
"""
def clean_keywords(keywords):
    # Define words to remove
    words_to_remove = ['npr', 'pennlive']

    # Tokenize keywords
    keywords = keywords.split(', ')

    # Remove unwanted words
    keywords = [word for word in keywords if word not in words_to_remove]

    # Join words back into a comma-separated string
    return ', '.join(keywords)

with app.app_context():
    # Get all news items
    all_news_items = NewsItem.query.all()

    for news_item in all_news_items:
        # Clean the keywords
        cleaned_keywords = clean_keywords(news_item.keywords)

        # Update the keywords field for this news item
        news_item.keywords = cleaned_keywords

    # Commit all changes to the database
    db.session.commit()
"""


# nltk packages
nltk_packages = ['punkt', 'stopwords', 'wordnet']

# download only if not already downloaded
for package in nltk_packages:
    try:
        if not os.path.exists(f'/app/nltk_data/{package}'):
            nltk.download(package)
    except FileExistsError:
        pass

app = Flask(__name__)
app.secret_key = 'it_is_a_secret_key'

# Get Heroku Postgres DB URL
DATABASE_URL = os.environ['DATABASE_URL']

# Parse the URL and add the connect_args for psycopg2
url = urllib.parse.urlparse(DATABASE_URL)
dbname = url.path[1:]
user = url.username
password = url.password
host = url.hostname
port = url.port

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # To suppress warning

db = SQLAlchemy(app)

class NewsItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    link = db.Column(db.String(200), nullable=False)
    published_date = db.Column(db.DateTime, nullable=False)
    source = db.Column(db.String(200), nullable=False)
    image = db.Column(db.String(200))
    all_words = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.Text, nullable=False)

    @classmethod
    def get_or_create(cls, title, link, published_date, source, image, all_words, keywords):
        exists = db.session.query(NewsItem.id).filter_by(link=link).scalar() is not None
        if exists:
            return db.session.query(NewsItem).filter_by(link=link).first()
        else:
            instance = cls(title=title, link=link, published_date=published_date, source=source, image=image, all_words=all_words, keywords=keywords)
            db.session.add(instance)
            db.session.commit()
            return instance

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def index():
    PAGE_SIZE = 15
    page = request.args.get('page', 1, type=int)  # Get the current page number from the query parameters
    wordcloud_filename = None
    if request.method == 'POST':
        keyword = request.form['keyword'].lower()
        source = request.form.get('source')
        session['keyword'] = keyword
        session['source'] = source
    else:
        keyword = session.get('keyword', '')
        source = session.get('source', 'all')

    if source == 'all':
        related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword))
    else:
        related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword), NewsItem.source == source)

    all_keywords = ' '.join([item.keywords for item in related_news.all()])
    if all_keywords.strip():
        wordcloud = WordCloud(width = 900, height = 400, background_color ='#ffffff', stopwords = None, min_font_size = 10).generate(all_keywords)
        wordcloud_filename = 'wordcloud.png'
        wordcloud.to_file(f'static/{wordcloud_filename}')

    # Add pagination to your query
    related_news = related_news.paginate(page=page, per_page=PAGE_SIZE, error_out=False)

    next_url = url_for('index', page=related_news.next_num) if related_news.has_next else None
    prev_url = url_for('index', page=related_news.prev_num) if related_news.has_prev else None

    related_news = [item.as_dict() for item in related_news.items]

    return render_template('index.html', news=related_news, wordcloud_filename=wordcloud_filename, next_url=next_url, prev_url=prev_url)



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

            # Remove stopwords, lemmatize, and convert to lowercase
            stop_words = set(stopwords.words('english') + additional_stopwords)
            lemmatizer = WordNetLemmatizer()
            processed_words = [lemmatizer.lemmatize(word.lower()) for word in tokens if not word.lower() in stop_words]

            all_words = ' '.join(processed_words)

            # Calculate TF-IDF
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([all_words])
            names = vectorizer.get_feature_names_out()
            data = vectors.todense().tolist()

            # Get top10 keywords based on tf-idf score
            tfidf_scores = sorted(list(zip(names, data[0])), key=lambda x: x[1], reverse=True)[:10]
            top_keywords = ', '.join([word for word, score in tfidf_scores])

            # Check if the news item already exists in the database
            news_item = NewsItem.query.filter_by(link=item.link.text).first()
            if not NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords):
                news_item = NewsItem.get_or_create(title=item.title.text, link=item.link.text, published_date=parse(item.pubDate.text), source=source, image=image, all_words=all_words, keywords=top_keywords)


if __name__ == "__main__":
    asyncio.run(scrape_news())
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
