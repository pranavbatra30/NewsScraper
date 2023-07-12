import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template, jsonify
from urllib.parse import urlparse
from flask_sqlalchemy import SQLAlchemy
from dateutil.parser import parse
from wordcloud import WordCloud
from sqlalchemy import or_, and_
import os
import urllib.parse
import nltk

# nltk packages
nltk_packages = ['punkt', 'stopwords', 'wordnet']

for package in nltk_packages:
    try:
        if not os.path.exists(f'/app/nltk_data/{package}'):
            nltk.download(package)
    except FileExistsError:
        pass

app = Flask(__name__)

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

@app.route('/load_more', methods=['POST'])
def load_more():
    start = int(request.form['start'])
    keyword = request.form['keyword'].lower()
    source = request.form.get('source')
    if source == 'all':
        related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword)).order_by(NewsItem.published_date.desc()).all()
    else:
        related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword), NewsItem.source == source).order_by(NewsItem.published_date.desc()).all()

    # Pagination
    related_news = related_news[start:start+15]
    
    # Check if there are more news to load
    has_more = len(related_news) < len(NewsItem.query.filter(NewsItem.all_words.contains(keyword)).order_by(NewsItem.published_date.desc()).all()) - start
    
    return jsonify({
        'news': related_news,  # return the 15 items
        'has_more': has_more  # If there's more data
    })


@app.route('/', methods=['GET', 'POST'])
def index():
    wordcloud_filename = None
    has_more = False
    if request.method == 'POST':
        start = int(request.form.get('start', 0))  # Get the start value from the form
        keyword = request.form['keyword'].lower()
        source = request.form.get('source')
        if source == 'all':
            related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword)).order_by(NewsItem.published_date.desc()).all()
        else:
            related_news = NewsItem.query.filter(NewsItem.all_words.contains(keyword), NewsItem.source == source).order_by(NewsItem.published_date.desc()).all()

        # Check if there are more news to load
        has_more = len(related_news) > start + 15
        
        # Pagination
        related_news = related_news[start:start+15]

        # Combine keywords from all articles into a single string
        all_keywords = ' '.join([item.keywords for item in related_news])

        # Generate a word cloud only if there is at least one keyword
        if all_keywords.strip():
            wordcloud = WordCloud(width = 900, height = 400,
                        background_color ='#ffffff',
                        stopwords = None,
                        min_font_size = 10).generate(all_keywords)
            # Save the word cloud as an image in a static directory
            wordcloud_filename = 'wordcloud.png'
            wordcloud.to_file(f'static/{wordcloud_filename}')
        else:
            wordcloud_filename = None

        related_news = [item.as_dict() for item in related_news]
        return render_template('index.html', news=related_news, has_more=has_more, wordcloud_filename=wordcloud_filename)
    else:
        trending_news = NewsItem.query.order_by(NewsItem.published_date.desc()).limit(12).all()
        return render_template('index.html', news=[], trending_news=trending_news, wordcloud_filename=wordcloud_filename)


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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
