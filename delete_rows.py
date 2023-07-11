from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
import urllib.parse

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

with app.app_context():
    num_rows_deleted = db.session.query(NewsItem).delete()
    db.session.commit()

    print(f"Deleted {num_rows_deleted} rows from NewsItem table.")
