import eventlet
eventlet.monkey_patch()
import os
import time
import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np
import pandas as pd
import bleach
from typing import Dict, Union, Optional
from eventlet.semaphore import Semaphore
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session, send_from_directory, \
    Blueprint
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import bcrypt
import joblib
import html
from pymongo import MongoClient
import markdown2
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from articles_data import articles_metadata
from database import get_db_connection
import uuid
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from urllib.parse import urlparse
import pymysql
from google.generativeai import configure as genai_configure, embed_content, GenerativeModel
import random
from flask import Flask, Response
from datetime import datetime, timezone
import requests
from zoneinfo import ZoneInfo


# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='pages')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=int(os.environ.get('SESSION_LIFETIME_MINUTES', 30)))
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join('static', 'Uploads'))
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp','svg'}

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app, resources={r"/api/*": {"origins": "*"}})
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.environ.get('RATE_LIMIT_DAILY', '200 per day'),
                    os.environ.get('RATE_LIMIT_HOURLY', '50 per hour')],
    storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)
limiter.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'




@app.route("/robots.txt")
def robots():
    

    """
    Returns the contents of the robots.txt file.

    This function responds to HTTP requests for the /robots.txt path and returns a
    plain text response with the contents of the robots.txt file. The response is
    cached for 24 hours.

    :return: A plain text response with the contents of the robots.txt file.
    """

    return Response("User-agent: *\nDisallow:", mimetype="text/plain")

# Custom Jinja filters
def custom_markdown(text):
    return markdown2.markdown(text, extras=["fenced-code-blocks", "tables", "spoiler", "header-ids", "code-friendly",
                                            "pyshell", "markdown-in-html", "footnotes", "cuddled-lists"])


def datetimeformat(value, format='%d/%m/%Y %H:%M'):

    """
    Format a datetime object or string in a custom format.

    If the value is `None`, return an empty string. If the value is a string, attempt to parse it as a datetime object.
    If parsing fails, return the original string.

    The format string is passed to `strftime` to format the datetime object. Default format is '%d/%m/%Y %H:%M'.
    """

    if value is None:
        return ''
    if isinstance(value, str):
        try:
            if 'T' in value:
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return value
    return value.strftime(format)


app.jinja_env.filters['datetimeformat'] = datetimeformat
app.jinja_env.filters['custom_markdown'] = custom_markdown


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):

        """
        Initialize a User object.

        :param id: User ID.
        :param username: Username.
        """

        self.id = id
        self.username = username


@login_manager.user_loader
def load_user(user_id):

    """
    Loads a user from the database by ID.

    This function is used by Flask-Login to load a user by ID. It is called when the user is logged in and when the user is
    accessed from the session.

    :param user_id: User ID.
    :return: User object or None if the user is not found.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, username FROM users WHERE id = %s', (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(id=user_data['id'], username=user_data['username'])
    finally:
        conn.close()
    return None


def get_client_ip(request):

    """
    Returns the client's IP address.

    The client's IP address can be retrieved from the request object's
    `remote_addr` attribute. However, if the request is behind a proxy,
    the `X-Forwarded-For` header should be used instead.

    :param request: The request object.
    :return: The client's IP address as a string.
    """
    if request.headers.get("X-Forwarded-For"):
        ip = request.headers.get("X-Forwarded-For").split(",")[0].strip()
    else:
        ip = request.remote_addr or "127.0.0.1"
    return ip




def get_local_time_from_ip(ip):

    """
    Récupère le fuseau horaire et l'heure exacte selon l'IP.

    :param ip: L'adresse IP
    :return: Un tuple contenant le fuseau horaire (par exemple "Europe/Paris")
             et l'heure exacte (par exemple "2025-08-28T17:22:34.123456+03:00")
    """

    try:
        # Récupérer timezone depuis ipinfo.io
        res = requests.get(f"http://ipinfo.io/{ip}/json", timeout=2).json()
        tz = res.get("timezone", "UTC")

        # Obtenir l'heure exacte via worldtimeapi
        time_res = requests.get(f"http://worldtimeapi.org/api/timezone/{tz}", timeout=2).json()
        local_time = time_res.get("datetime")

        if not local_time:
            local_time = datetime.now(timezone.utc).isoformat()

        return tz, local_time
    except Exception:
        return "UTC", datetime.now(timezone.utc).isoformat()



@app.route('/python-demo', methods=['GET'])
def python_demo():
    """
    Render the Python Demo page with an interactive code example.
    """
    logger.info("Rendering Python Demo page")
    try:
        # Sample data for the demo (mirroring the JavaScript example)


        # define a seed for reproducibility
        np.random.seed(42)

        # number of customers
        num_customers = 1000

        # Générer des données
        recency = np.random.randint(2, 46, size=num_customers)  # Recency between 2 and 45
        frequency = np.random.randint(3, 21, size=num_customers)  # Frequency entre 3 et 20
        avg_order_value = np.full(num_customers, 100)  # Valeur constante à 100
        monetary = frequency * avg_order_value  # Monetary = frequency * avg_order_value
        orders = frequency  # Orders is the same as frequency

        # Créer un DataFrame
        sample_data = {
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'orders': orders,
            'avg_order_value': avg_order_value
        }

       
        df = pd.DataFrame(sample_data)
        df['total_spent'] = df['orders'] * df['avg_order_value']
        scaler = StandardScaler()
        features = ['recency', 'frequency', 'monetary']
        scaled_features = scaler.fit_transform(df[features])
        result = {
            'high_value': len(df[df['total_spent'] > df['total_spent'].quantile(0.8)]),
            'avg_clv': df['total_spent'].mean(),
            'retention_rate': (df['recency'] < 30).mean() * 100
        }

        return render_template('python_demo.html', sample_data=sample_data, result=result)
    except Exception as e:
        logger.error(f"Error rendering Python Demo page: {str(e)}")
        return render_template('500.html', message=str(e)), 500


# Model and data handling
models: Dict[str, Optional[Union[object]]] = {
    'churn': None,
    'recommendation': None,
    'matrix': None,
    'price_optimizer': None
}
model_lock = Semaphore()

MODEL_FILES = {
    'churn': 'models_trains/random_forest_model.pkl'
}

DATA_FILES = {
    'retail_price': 'datasets/retail_price.csv',
    'movies': 'datasets/movies.csv'
}

datasets = {
    'retail_price': None,
    'movies': None
}



def initialize_models():
    for model_type in models:
        try:
            load_model(model_type)
        except Exception as e:
            logger.error(f"Failed to initialize {model_type} model at startup: {str(e)}")



def load_model(model_type: str) -> Optional[Union[object]]:

    """
    Load a model from file.

    Args:
        model_type (str): Model type to load.

    Returns:
        Optional[Union[object]]: Loaded model, or None on error.

    Raises:
        ValueError: If model type is unknown.
        IOError: If model file is not found.
    """

    if model_type not in models:
        logger.error(f"Invalid model type: {model_type}")
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type not in MODEL_FILES:
        logger.error(f"No model file defined for model type: {model_type}")
        raise ValueError(f"No model file defined for model type: {model_type}")
    with model_lock:
        if models[model_type] is not None:
            logger.debug(f"{model_type.capitalize()} model already loaded from cache")
            return models[model_type]
        try:
            local_file = MODEL_FILES[model_type]
            logger.info(f"Loading {model_type} model from {local_file}")
            if not os.path.exists(local_file):
                logger.error(f"Model file not found: {local_file}")
                raise IOError(f"Model file not found: {local_file}")
            models[model_type] = joblib.load(local_file)
            logger.info(f"{model_type.capitalize()} model loaded successfully from {local_file}")
            return models[model_type]
        except (IOError, joblib.JoblibException) as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            models[model_type] = None
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {model_type} model: {str(e)}")
            models[model_type] = None
            return None



def load_dataset(dataset_name):

    """
    Load a dataset from file.

    Args:
        dataset_name (str): Name of dataset to load.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        ValueError: If dataset name is unknown.
        IOError: If dataset file is not found.
        Exception: If any other error occurs during loading.
    """


    if dataset_name not in datasets:
        logger.error(f"Invalid dataset name: {dataset_name}")
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    if datasets[dataset_name] is not None:
        logger.debug(f"{dataset_name} dataset already loaded from cache")
        return datasets[dataset_name]
    try:
        file_path = DATA_FILES[dataset_name]
        logger.info(f"Loading {dataset_name} dataset from {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            raise IOError(f"Dataset file not found: {file_path}")
        df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
        if dataset_name == 'retail_price':
            df['month_year'] = pd.to_datetime(df['month_year'], format='%m-%d-%Y')
            df = df.dropna()
        datasets[dataset_name] = df
        logger.info(f"{dataset_name} dataset loaded successfully.")
        return datasets[dataset_name]
    except Exception as e:
        logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
        datasets[dataset_name] = pd.DataFrame()
        return datasets[dataset_name]












# Database functions
def get_db_connection():

    """
    Establish a connection to JawsDB MySQL or local database.

    Returns a connection object.

    Raises:
        pymysql.MySQLError: If the connection fails.
    """

    jawsdb_url = os.getenv('JAWSDB_CHARCOAL_URL')
    try:
        if jawsdb_url:
            url = urlparse(jawsdb_url)
            config = {
                'host': url.hostname,
                'user': url.username,
                'password': url.password,
                'database': url.path[1:],
                'port': url.port or 3306,
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
                'autocommit': False
            }
        else:
            config = {
                'host': 'localhost',
                'user': 'root',
                'password': os.getenv('LOCAL_MYSQL_PASSWORD', ''),
                'database': 'datacraft_db',
                'port': 3306,
                'charset': 'utf8mb4',
                'cursorclass': pymysql.cursors.DictCursor,
                'autocommit': False
            }
        conn = pymysql.connect(**config)
        logger.info("Successfully connected to database")
        return conn
    except pymysql.MySQLError as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def init_db():

    """
    Initialize the database with the required tables and default values.

    Creates the `users` and `articles` tables with the required columns and indexes.
    Generates UUIDs for existing articles without a UUID.
    Adds the `updated_at` column to the `articles` table if it doesn't exist.
    Initializes the default admin user with the environment variables `DEFAULT_ADMIN_USERNAME`, `DEFAULT_ADMIN_PASSWORD`, and `DEFAULT_ADMIN_EMAIL`.
    Synchronizes the articles metadata.

    Returns:
        bool: `True` if the initialization was successful, `False` otherwise.

    Raises:
        ValueError: If the `DEFAULT_ADMIN_PASSWORD` environment variable is not set.
        pymysql.MySQLError: If a database error occurs during initialization.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_login DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_username (username),
                    INDEX idx_email (email)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')

            # Create articles table with uuid and updated_at
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    uuid VARCHAR(36) UNIQUE,
                    title VARCHAR(255) NOT NULL,
                    content TEXT,
                    category VARCHAR(50),
                    description TEXT,
                    tags VARCHAR(255),
                    image VARCHAR(255),
                    read_time INT,
                    hidden BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME,
                    views INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    created_by INT,
                    updated_by INT,
                    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
                    FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')

            # Ensure uuid column exists
            cursor.execute("SHOW COLUMNS FROM articles LIKE 'uuid'")
            if not cursor.fetchone():
                cursor.execute('ALTER TABLE articles ADD COLUMN uuid VARCHAR(36) UNIQUE')
                logger.info("Added uuid column to articles table")
                cursor.execute('SELECT id FROM articles WHERE uuid IS NULL')
                articles = cursor.fetchall()
                for article in articles:
                    article_id = article['id']
                    new_uuid = str(uuid.uuid4())
                    cursor.execute('UPDATE articles SET uuid = %s WHERE id = %s', (new_uuid, article_id))
                logger.info(f"Generated UUIDs for {len(articles)} existing articles")

            # Ensure updated_at column exists
            cursor.execute("SHOW COLUMNS FROM articles LIKE 'updated_at'")
            if not cursor.fetchone():
                cursor.execute('ALTER TABLE articles ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')
                logger.info("Added updated_at column to articles table")

            # Initialize default admin user
            cursor.execute("SELECT COUNT(*) AS count FROM users")
            if cursor.fetchone()['count'] == 0:
                default_admin = {
                    'username': os.getenv('DEFAULT_ADMIN_USERNAME', 'admin'),
                    'password': os.getenv('DEFAULT_ADMIN_PASSWORD')
                }
                if not default_admin['password']:
                    raise ValueError("DEFAULT_ADMIN_PASSWORD environment variable is not set")
                password_hash = bcrypt.hashpw(
                    default_admin['password'].encode('utf-8'),
                    bcrypt.gensalt()
                ).decode('utf-8')
                cursor.execute('''
                    INSERT INTO users (username, password_hash, email, is_active)
                    VALUES (%s, %s, %s, TRUE)
                ''', (
                    default_admin['username'],
                    password_hash,
                    os.getenv('DEFAULT_ADMIN_EMAIL', 'timotheenkwar16@gmail.com')
                ))
                logger.warning(f"Default admin user created with username: {default_admin['username']}. PLEASE CHANGE THE PASSWORD!")

            conn.commit()
            logger.info("Database initialization completed successfully.")

            # Sync articles metadata
            sync_success = sync_articles_metadata()
            if sync_success:
                logger.info("Article synchronization completed successfully.")
            else:
                logger.error("Article synchronization failed.")
            return sync_success
    except pymysql.MySQLError as e:
        logger.error(f"Database error during initialization: {str(e)}", exc_info=True)
        conn.rollback()
        return False
    finally:
        conn.close()







def sync_articles_metadata():

    """
    Sync articles metadata from articles_metadata dictionary with the database.

    Iterate over each article in articles_metadata and insert or update its
    data in the database. If the article doesn't exist in the database, insert
    it with default values for views, hidden, and timestamp. If the article does
    exist in the database, update its fields with the values from the metadata
    dictionary. If any field is missing from the metadata dictionary, use a
    default value.

    If any error occurs during the sync process, log the error and close the
    database connection. If the sync process is successful, log a success
    message and close the database connection.

    Returns:
        bool: True if sync is successful, False if an error occurs.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for article_uuid, metadata in articles_metadata.items():
                tags_value = metadata.get('tags', [])
                if isinstance(tags_value, list):
                    tags_value = ', '.join(tags_value)
                cursor.execute('SELECT title, content FROM articles WHERE uuid = %s', (article_uuid,))
                existing = cursor.fetchone()
                timestamp = metadata.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')))
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now(pytz.timezone('Asia/Nicosia'))
                if existing and existing['title'] and existing['content']:
                    logger.info(f"Article {article_uuid} exists with data, skipping update.")
                    continue
                cursor.execute('''
                               INSERT INTO articles (uuid, title, content, category, description, tags, image,
                                                     read_time, timestamp, views, created_at, created_by)
                               ON DUPLICATE KEY UPDATE title       = COALESCE(NULLIF(VALUES(title), ''), title),
                                                       content     = COALESCE(NULLIF(VALUES(content), ''), content),
                                                       category    = VALUES(category),
                                                       description = VALUES(description),
                                                       tags        = VALUES(tags),
                                                       image       = VALUES(image),
                                                       read_time   = VALUES(read_time),
                                                       timestamp   = VALUES(timestamp)
                               ''', (
                                   article_uuid,
                                   metadata.get('title', 'Untitled Article'),
                                   metadata.get('content', 'No content available.'),
                                   metadata.get('category', 'uncategorized'),
                                   metadata.get('description', 'No description available.'),
                                   tags_value,
                                   metadata.get('image',
                                                'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
                                   metadata.get('read_time', 1),
                                   timestamp
                               ))
            conn.commit()
            logger.info("Articles metadata synced successfully.")
            return True
    except Exception as e:
        logger.error(f"Error syncing articles_metadata: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()




# API Blueprint
api = Blueprint('api', __name__)

# Other routes
@app.route('/favicon.ico')
def favicon():

    """
    Serves the favicon from the static/images directory.

    If the favicon is not found, it renders the 404.html template with a 404 status code.

    :return: The favicon image or a 404 error page.
    :rtype: flask.Response
    """

    try:
        return send_from_directory(os.path.join(app.root_path, 'static/images'),
                                   'icon.png', mimetype='image/png')
    except Exception as e:
        logger.warning(f"Favicon not found: {str(e)}")
        return render_template('404.html', message="Favicon not found"), 404







@app.route('/', defaults={'page': 'front'}, methods=['GET', 'POST'])
@app.route('/<page>', methods=['GET', 'POST'])
def show_page(page):

    """
    Main route for the application, handles all page requests.

    If the page is not found, it renders the 404.html template with a 404 status code.

    If the page is the blog page, it loads the articles from the database and renders the blog_insights.html template.

    If the page is the front page, it redirects to the homepage.

    If the page is any other page, it renders the corresponding template.

    :param page: The page to render.
    :type page: str
    :return: The rendered template.
    :rtype: flask.Response
    """

    valid_pages = {'front','homepage', 'case_studies', 'about', 'skills_tools', 'blog_insights', 'contact_collaboration'}
    if page not in valid_pages:
        logger.warning(f"Page not found: {page}")
        return render_template('404.html'), 404
    prediction = None
    recommendation = None
    price_result = None
    articles = []
    if page == 'blog_insights':
        logger.info("Loading articles for blog page")
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute('''
                               SELECT id,
                                      title,
                                      category,
                                      description,
                                      tags,
                                      image,
                                      read_time,
                                      timestamp,
                                      views
                               FROM articles
                               WHERE hidden = FALSE
                               ORDER BY timestamp DESC
                               ''')
                db_articles = cursor.fetchall()
                for db_article in db_articles:
                    timestamp = db_article['timestamp']
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except ValueError:
                            timestamp = datetime.now(pytz.timezone('Asia/Nicosia'))
                    formatted_timestamp = (
                        timestamp.astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")
                        if timestamp else ""
                    )
                    tags = db_article.get('tags', '')
                    if isinstance(tags, str):
                        tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
                    article = {
                        'id': str(db_article['id']),
                        'title': db_article.get('title', 'Untitled Article'),
                        'category': db_article.get('category', 'uncategorized').lower(),
                        'description': db_article.get('description', 'No description available.'),
                        'tags': tags,
                        'image': db_article.get('image',
                                                'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
                        'read_time': int(db_article.get('read_time', 5)),
                        'timestamp': formatted_timestamp,
                        'views': int(db_article.get('views', 0)),
                        'content': db_article.get('content', 'No content available.')
                    }
                    articles.append(article)
            logger.info(f"Successfully loaded {len(articles)} articles")
        except Exception as e:
            logger.error(f"Error loading articles: {str(e)}")
            flash('Unable to load articles at this time.', 'error')
        finally:
            conn.close()
    template_name = f"{page}.html"
    logger.debug(f"Rendering template: {template_name}")
    try:
        return render_template(template_name,
                               prediction=prediction,
                               recommendation=recommendation,
                               price_result=price_result,
                               articles=articles)
    except Exception as e:
        logger.error(f"Error rendering template {template_name}: {str(e)}")
        return render_template('500.html', message=str(e)), 500







# Load articles from articles_metadata.json
def load_articles_metadata():

    """
    Load articles metadata from articles_metadata.json
    
    Returns a dictionary containing metadata about each article, where each key is
    the article ID and each value is another dictionary containing the article's
    metadata. If the file is invalid or doesn't exist, returns an empty dictionary.
    """

    try:
        with open('articles_metadata.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                logger.error("articles_metadata.json must be a dictionary")
                return {}
            return data
    except FileNotFoundError:
        logger.error("articles_metadata.json not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding articles_metadata.json: {e}")
        return {}

articles_metadata = load_articles_metadata()

# Sync articles_metadata with database
def sync_articles_to_db():

    """
    Sync articles from articles_metadata to the database.

    Iterate over each article in articles_metadata and insert or update its
    data in the database. If the article doesn't exist in the database, insert
    it with default values for views, hidden, and timestamp. If the article does
    exist in the database, update its fields with the values from the metadata
    dictionary. If any field is missing from the metadata dictionary, use a
    default value.

    If any error occurs during the sync process, log the error and close the
    database connection. If the sync process is successful, log a success
    message and close the database connection.
    """

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for article_id, metadata in articles_metadata.items():
            cursor.execute('''
                INSERT INTO articles (uuid, title, description, category, tags, image, read_time, content, views, hidden, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    description = VALUES(description),
                    category = VALUES(category),
                    tags = VALUES(tags),
                    image = VALUES(image),
                    read_time = VALUES(read_time),
                    content = VALUES(content),
                    timestamp = VALUES(timestamp)
            ''', (
                article_id,
                metadata.get('title', 'Untitled Article'),
                metadata.get('description', 'No description available.'),
                metadata.get('category', 'Uncategorized'),
                ','.join(metadata.get('tags', [])),
                metadata.get('image', 'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
                metadata.get('read_time', 5),
                metadata.get('content', 'No content available.'),
                0,
                False,
                metadata.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')).strftime('%Y-%m-%d %H:%M:%S'))
            ))
        conn.commit()
        logger.info("Synced articles_metadata.json to database")
    except pymysql.MySQLError as e:
        logger.error(f"Error syncing articles to database: {e}")
    finally:
        if conn:
            conn.close()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):   
    """
    Load a user from the database by ID.

    Called by Flask-Login to load the current user from the database using
    the user ID stored in the session.

    Args:
        user_id (int): The ID of the user to load.

    Returns:
        User: The loaded user object, or None if the user ID doesn't exist in
            the database.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, username FROM users WHERE id = %s', (user_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(id=user_data['id'], username=user_data['username'])
    finally:
        conn.close()
    return None

@app.route('/article/<id>')
def show_article(id):

    """
    Fetch an article from the database by ID and render the article page.

    Args:
        id (str): The ID of the article to fetch. Can be either a numeric ID or a UUID.

    Returns:
        A rendered HTML page with the article content.
    """

    logger.debug(f"Fetching article with ID: {id}")
    
    # Normalize ID
    article_id = id
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # Check if id is numeric (database id)
        if id.isdigit():
            cursor.execute('SELECT uuid FROM articles WHERE id = %s AND hidden = FALSE', (id,))
            result = cursor.fetchone()
            if result and f"article_id_{int(id) - 1}" in articles_metadata:
                article_id = f"article_id_{int(id) - 1}"  # Map id=1 to article_id_0
            elif result:
                article_id = result['uuid']  # Fallback to UUID
        # Check if id is a UUID
        elif len(id) == 36 and '-' in id:
            cursor.execute('SELECT id FROM articles WHERE uuid = %s AND hidden = FALSE', (id,))
            result = cursor.fetchone()
            if result and f"article_id_{result['id'] - 1}" in articles_metadata:
                article_id = f"article_id_{result['id'] - 1}"
        logger.debug(f"Normalized article ID: {article_id}")

        # Fetch metadata
        metadata = articles_metadata.get(article_id)
        if not metadata:
            logger.warning(f"Article with ID {article_id} not found in articles_metadata.json")
            return render_template('404.html', message="Article not found"), 404

        # Ensure metadata fields
        metadata = {
            'title': metadata.get('title', 'Untitled Article'),
            'description': metadata.get('description', 'No description available.'),
            'category': metadata.get('category', 'Uncategorized'),
            'tags': metadata.get('tags', []) if isinstance(metadata.get('tags', []), list) else [],
            'image': metadata.get('image', 'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
            'read_time': int(metadata.get('read_time', 5)) if str(metadata.get('read_time', 5)).isdigit() else 5,
            'timestamp': metadata.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia'))),
            'content': metadata.get('content', 'No content available.')
        }

        # Update or insert in database
        cursor.execute('SELECT id, views, hidden FROM articles WHERE uuid = %s', (article_id,))
        article_db = cursor.fetchone()
        views = 1
        if article_db:
            if article_db['hidden']:
                logger.warning(f"Article with ID {article_id} is hidden in DB")
                return render_template('404.html', message="Article not found"), 404
            try:
                cursor.execute('UPDATE articles SET views = views + 1 WHERE uuid = %s', (article_id,))
                views = int(article_db['views']) + 1 if article_db.get('views') else 1
            except pymysql.MySQLError as update_e:
                logger.error(f"Error updating views for article {article_id}: {update_e}")
                views = int(article_db['views']) if article_db.get('views') else 1
        else:
            try:
                cursor.execute('''
                    INSERT INTO articles (uuid, title, description, category, tags, image, read_time, content, views, hidden, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s)
                ''', (
                    article_id,
                    metadata['title'],
                    metadata['description'],
                    metadata['category'],
                    ','.join(metadata['tags']),
                    metadata['image'],
                    metadata['read_time'],
                    metadata['content'],
                    1,
                    metadata['timestamp'] if isinstance(metadata['timestamp'], str) else metadata['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                ))
                views = 1
            except pymysql.MySQLError as insert_e:
                logger.error(f"Error inserting article {article_id} in DB: {insert_e}")
                views = 1
        
        conn.commit()
    except pymysql.MySQLError as e:
        logger.error(f"Database error for article {article_id}: {e}")
        views = 1
    finally:
        if conn:
            try:
                conn.close()
            except pymysql.MySQLError:
                pass

    # Parse timestamp
    timestamp = metadata.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')))
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            logger.warning(f"Invalid timestamp format for article {article_id}, fallback to now")
            timestamp = datetime.now(pytz.timezone('Asia/Nicosia'))
    formatted_timestamp = timestamp.astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")

    # Convert markdown to HTML
    content_md = metadata.get('content', 'No content available.')
    try:
        content_html = markdown2.markdown(
            content_md,
            extras=["fenced-code-blocks", "tables", "strike", "footnotes", "code-friendly"]
        )
    except Exception as md_e:
        logger.error(f"Markdown conversion error for article {article_id}: {md_e}")
        content_html = '<p>No content available.</p>'

    # Sanitize HTML
    allowed_tags = [
        'p', 'pre', 'code', 'img', 'h1', 'h2', 'h3', 'table', 'thead',
        'tbody', 'tr', 'th', 'td', 'div', 'span', 'a', 'b', 'i', 'u',
        'strong', 'em', 'ul', 'ol', 'li', 'br'
    ]
    allowed_attrs = {
        'a': ['href', 'title', 'target', 'rel'],
        'img': ['src', 'alt', 'title', 'width', 'height'],
        'div': ['class'],
        'span': ['class']
    }
    try:
        content_html = bleach.clean(content_html, tags=allowed_tags, attributes=allowed_attrs)
    except Exception as bleach_e:
        logger.error(f"Bleach sanitization error for article {article_id}: {bleach_e}")
        content_html = '<p>No content available.</p>'

    # Build article object
    article = {
        'id': str(article_id),
        'title': metadata['title'],
        'description': metadata['description'],
        'category': metadata['category'],
        'tags': metadata['tags'],
        'image': metadata['image'],
        'read_time': metadata['read_time'],
        'timestamp': formatted_timestamp,
        'views': views,
        'content': content_html
    }

    return render_template('article.html', article=article)

@app.route('/')
def index():
    """\
    Render the homepage with 3 most recent articles.

    Fetch 3 most recent articles from the database, and render the homepage
    with them. If there is an error fetching the articles, fall back to static
    sample data from articles_metadata.json.
    """

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute('SELECT id, uuid, title, description, category, tags, image, read_time, timestamp, views FROM articles WHERE hidden = FALSE ORDER BY timestamp DESC LIMIT 3')
        articles = [
            {
                'id': str(row['id']),  # Use integer id (e.g., '1')
                'uuid': row['uuid'],   # Keep UUID for reference
                'title': row['title'],
                'description': row['description'],
                'category': row['category'],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'image': row['image'],
                'read_time': row['read_time'],
                'timestamp': row['timestamp'].strftime("%B %d, %Y %H:%M:%S") if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'views': row['views']
            } for row in cursor.fetchall()
        ]
        conn.commit()
        return render_template('index.html', articles=articles)  # Fixed to index.html
    except pymysql.MySQLError as e:
        logger.error(f"Error fetching articles: {e}")
        articles = [
            {
                'id': str(idx + 1),  # Map article_id_X to integer (e.g., article_id_0 -> 1)
                'uuid': f"article_id_{idx}",  # Fallback UUID
                'title': article_data['title'],
                'description': article_data['description'],
                'category': article_data.get('category', 'Uncategorized'),
                'tags': article_data['tags'],
                'image': article_data['image'],
                'read_time': article_data['read_time'],
                'timestamp': article_data.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")),
                'views': 0
            } for idx, (article_id, article_data) in enumerate(list(articles_metadata.items())[:3])
        ]
        return render_template('index.html', articles=articles)
    finally:
        if conn:
            conn.close()

@app.route('/blog')
def blog():

    """
    Display all articles in the blog.

    Fetches articles from the database and renders the blog_insights.html template with the article data.
    If an error occurs while fetching articles, it falls back to using the pre-defined articles stored in
    the articles_metadata dictionary.
    """

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute('SELECT id, uuid, title, description, category, tags, image, read_time, timestamp, views FROM articles WHERE hidden = FALSE ORDER BY timestamp DESC')
        articles = [
            {
                'id': str(row['id']),  # Use integer id
                'uuid': row['uuid'],   # Keep UUID
                'title': row['title'],
                'description': row['description'],
                'category': row['category'],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'image': row['image'],
                'read_time': row['read_time'],
                'timestamp': row['timestamp'].strftime("%B %d, %Y %H:%M:%S") if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'views': row['views']
            } for row in cursor.fetchall()
        ]
        conn.commit()
        return render_template('blog_insights.html', articles=articles)
    except pymysql.MySQLError as e:
        logger.error(f"Error fetching blog articles: {e}")
        articles = [
            {
                'id': str(idx + 1),  # Map article_id_X to integer
                'uuid': article_id,
                'title': article_data['title'],
                'description': article_data['description'],
                'category': article_data.get('category', 'Uncategorized'),
                'tags': article_data['tags'],
                'image': article_data['image'],
                'read_time': article_data['read_time'],
                'timestamp': article_data.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")),
                'views': 0
            } for idx, (article_id, article_data) in enumerate(articles_metadata.items())
        ]
        return render_template('blog_insights.html', articles=articles)
    finally:
        if conn:
            conn.close()

@app.route('/api/articles', methods=['GET'])
def api_articles():

    """
    API endpoint to fetch articles from the database.

    The endpoint accepts the following query parameters:

    - `page`: The page number to fetch (default: 1)
    - `per_page`: The number of articles to fetch per page (default: 5)
    - `search`: A search query to filter articles by title or description
    - `status`: The article status to filter by, either 'all', 'published' or 'hidden' (default: 'all')

    The endpoint returns a JSON response containing the following keys:

    - `articles`: A list of article objects with the following keys:
        - `id`: The article ID
        - `title`: The article title
        - `description`: The article description
        - `category`: The article category
        - `tags`: A list of tags associated with the article
        - `image`: The article image URL
        - `read_time`: The article read time in minutes
        - `content`: The article content
        - `hidden`: A boolean indicating whether the article is hidden or not
        - `views`: The article view count
        - `timestamp`: The article timestamp in ISO 8601 format
    - `total`: The total number of articles matching the search query
    - `pages`: The total number of pages in the paginated list

    If an error occurs while fetching articles, the endpoint returns a JSON response containing an
    error message and a 500 status code.
    """

    conn = None
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 5))
        search = request.args.get('search', '')
        status = request.args.get('status', 'all')
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        query = 'SELECT uuid, title, description, category, tags, image, read_time, content, views, hidden, timestamp FROM articles WHERE 1=1'
        params = []
        if status != 'all':
            query += ' AND hidden = %s'
            params.append(status == 'hidden')
        if search:
            query += ' AND (title LIKE %s OR description LIKE %s)'
            params.extend([f'%{search}%', f'%{search}%'])
        query += ' ORDER BY timestamp DESC LIMIT %s OFFSET %s'
        params.extend([per_page, (page - 1) * per_page])
        cursor.execute(query, params)
        articles = [
            {
                'id': row['uuid'],
                'title': row['title'],
                'description': row['description'],
                'category': row['category'],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'image': row['image'],
                'read_time': row['read_time'],
                'content': row['content'],
                'hidden': row['hidden'],
                'views': row['views'],
                'timestamp': row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row['timestamp'], datetime) else row['timestamp']
            } for row in cursor.fetchall()
        ]
        cursor.execute('SELECT COUNT(*) AS total FROM articles WHERE 1=1' + (' AND hidden = %s' if status != 'all' else '') + (' AND (title LIKE %s OR description LIKE %s)' if search else ''), params[:-2] if search else params[:-2] if status != 'all' else [])
        total = cursor.fetchone()['total']
        conn.commit()
        return jsonify({
            'articles': articles,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        })
    except pymysql.MySQLError as e:
        logger.error(f"Error fetching API articles: {e}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        if conn:
            conn.close()









@api.route('/articles', methods=['GET'])
def get_articles():

    """
    GET /articles
    Fetches articles

    Parameters:
    page (int): Page number
    per_page (int): Number of articles per page
    category (string): Article category
    status (string): Article status: visible, hidden or all

    Returns:
    {
        "articles": [
            {
                "id": string,
                "title": string,
                "category": string,
                "hidden": boolean,
                "description": string,
                "tags": [
                    string
                ],
                "image": string,
                "read_time": int,
                "timestamp": string,
                "views": int
            }
        ],
        "total": int,
        "pages": int
    }
    """
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 5))
        category = bleach.clean(request.args.get('category', ''))
        status = request.args.get('status', 'all')

        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = 'SELECT id, uuid, title, category, hidden, description, tags, image, read_time, timestamp, views FROM articles WHERE hidden = %s'
            params = []

            if status == 'visible':
                params.append(False)
            elif status == 'hidden':
                params.append(True)
            else:
                query = query.replace('hidden = %s', '1=1')

            if category:
                query += ' AND category = %s'
                params.append(category)

            query += ' ORDER BY timestamp DESC LIMIT %s OFFSET %s'
            params.extend([per_page, (page - 1) * per_page])

            cursor.execute(query, params)
            articles = cursor.fetchall()

            cursor.execute(
                'SELECT COUNT(*) AS total FROM articles WHERE 1=1' + (' AND category = %s' if category else ''),
                [category] if category else [])
            total = cursor.fetchone()['total']

        conn.close()
        return jsonify({
            'articles': [
                {
                    'id': article['uuid'],
                    'title': article['title'],
                    'category': article['category'],
                    'hidden': article['hidden'],
                    'description': article['description'],
                    'tags': article['tags'].split(',') if article['tags'] else [],
                    'image': article['image'],
                    'read_time': article['read_time'],
                    'timestamp': article['timestamp'].isoformat() if article['timestamp'] else None,
                    'views': article['views']
                } for article in articles
            ],
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }), 200
    except Exception as e:
        logger.error(f"Error fetching articles: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles/<article_id>', methods=['GET'])
def get_article(article_id):

    """Get an article by its uuid.

    Args:
        article_id (str): The uuid of the article to fetch.

    Returns:
        A JSON object containing the article's data, or a 404 error if the article is not found.
    """

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                'SELECT id, uuid, title, category, hidden, description, tags, image, read_time, content FROM articles WHERE uuid = %s',
                (article_id,))
            article = cursor.fetchone()
        conn.close()
        if not article:
            return jsonify({'error': 'Article not found'}), 404
        return jsonify({
            'id': article['uuid'],
            'title': article['title'],
            'category': article['category'],
            'hidden': article['hidden'],
            'description': article['description'],
            'tags': article['tags'].split(',') if article['tags'] else [],
            'image': article['image'],
            'read_time': article['read_time'],
            'content': article['content']
        }), 200
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def create_article():

    """
    Create a new article.

    Args:
        title (str): The article title
        category (str): The article category
        description (str): The article description
        tags (list[str]): The article tags
        image (str): The article image URL
        read_time (int): The article read time in minutes
        content (str): The article content

    Returns:
        A JSON object containing the article ID, or a 400 error if the title is missing, or a 500 error if an internal error occurs.

    Rate limit: 5 requests per minute
    """

    try:
        data = request.get_json()
        if not data or not data.get('title'):
            return jsonify({'error': 'Title is required'}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            article_id = str(uuid.uuid4())
            tags = ','.join([tag.strip() for tag in data.get('tags', [])])
            cursor.execute('''
                           INSERT INTO articles (uuid, title, category, description, tags, image, read_time, content,
                                                 created_by)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                           ''', (
                               article_id,
                               data.get('title', 'Untitled Article'),
                               data.get('category', 'uncategorized'),
                               data.get('description', ''),
                               tags,
                               data.get('image', ''),
                               data.get('read_time', 5),
                               data.get('content', ''),
                               current_user.id
                           ))
            conn.commit()
        conn.close()
        socketio.emit('article_updated', {'article': {'id': article_id}})
        return jsonify({'article': {'id': article_id}}), 201
    except Exception as e:
        logger.error(f"Error creating article: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles/<article_id>', methods=['PUT'])
@limiter.limit("5 per minute")
@login_required
def update_article(article_id):
    try:
        data = request.get_json()
        if not data or not data.get('title'):
            return jsonify({'error': 'Title is required'}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            tags = ','.join([tag.strip() for tag in data.get('tags', [])])
            cursor.execute('''
                           UPDATE articles
                           SET title       = %s,
                               category    = %s,
                               description = %s,
                               tags        = %s,
                               image       = %s,
                               read_time   = %s,
                               content     = %s,
                               updated_by  = %s,
                               updated_at  = CURRENT_TIMESTAMP
                           WHERE uuid = %s
                           ''', (
                               data.get('title', 'Untitled Article'),
                               data.get('category', 'uncategorized'),
                               data.get('description', ''),
                               tags,
                               data.get('image', ''),
                               data.get('read_time', 5),
                               data.get('content', ''),
                               current_user.id,
                               article_id
                           ))
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'error': 'Article not found'}), 404
            conn.commit()
        conn.close()
        socketio.emit('article_updated', {'article': {'id': article_id}})
        return jsonify({'article': {'id': article_id}}), 200
    except Exception as e:
        logger.error(f"Error updating article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles/<article_id>', methods=['DELETE'])
@limiter.limit("5 per minute")
@login_required
def delete_article(article_id):
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM articles WHERE uuid = %s', (article_id,))
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'error': 'Article not found'}), 404
            conn.commit()
        conn.close()
        socketio.emit('article_deleted', {'articleId': article_id})
        return jsonify({'message': 'Article deleted'}), 200
    except Exception as e:
        logger.error(f"Error deleting article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles/<article_id>/toggle-visibility', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def toggle_article_visibility(article_id):
    try:
        data = request.get_json()
        hidden = data.get('hidden', False)
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute('UPDATE articles SET hidden = %s WHERE uuid = %s', (hidden, article_id))
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'error': 'Article not found'}), 404
            conn.commit()
        conn.close()
        socketio.emit('article_visibility_changed', {'articleId': article_id, 'hidden': hidden})
        return jsonify({'message': 'Visibility toggled'}), 200
    except Exception as e:
        logger.error(f"Error toggling visibility for article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('show_page', page='homepage'))
    if request.method == 'POST':
        username = bleach.clean(request.form.get('username', ''))
        password = request.form.get('password', '')
        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('login'))
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, username, password_hash FROM users WHERE username = %s AND is_active = TRUE',
                               (username,))
                user = cursor.fetchone()
                if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                    user_obj = User(id=user['id'], username=user['username'])
                    login_user(user_obj)
                    cursor.execute('UPDATE users SET last_login = NOW() WHERE id = %s', (user['id'],))
                    conn.commit()
                    flash('Login successful!', 'success')
                    return redirect(url_for('admin'))
                else:
                    flash('Invalid username or password.', 'error')
        finally:
            conn.close()
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))



@app.route('/api/categories', methods=['GET'])
def get_categories():
    logger.info("Received request for categories from MySQL table articles")
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Requête SQL pour récupérer les valeurs uniques de la colonne category dans la table articles
            cursor.execute("SELECT DISTINCT category FROM articles")
            categories = [row['category'] for row in cursor.fetchall() if row['category'] is not None]
        
        conn.close()

        if not categories:
            logger.warning("No categories found in articles table")
            return jsonify({'error': 'No categories available'}), 404

        logger.info(f"Returning {len(categories)} categories from articles table")
        return jsonify(categories), 200
    except pymysql.Error as e:
        logger.error(f"Database error fetching categories: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching categories: {e}")
        return jsonify({'error': 'Internal server error'}), 500



@app.route('/admin')
@login_required
def admin():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                           SELECT a.*, u1.username as created_by_username, u2.username as updated_by_username
                           FROM articles a
                                    LEFT JOIN users u1 ON a.created_by = u1.id
                                    LEFT JOIN users u2 ON a.updated_by = u2.id
                           ORDER BY a.updated_at DESC
                           ''')
            articles = cursor.fetchall()
            cursor.execute('SELECT DISTINCT category FROM articles')
            categories = [row['category'] for row in cursor.fetchall()]
            return render_template('admin.html', articles=articles, categories=categories)
    except Exception as e:
        logger.error(f"Error in admin view: {str(e)}")
        flash(f'Error retrieving articles: {str(e)}', 'error')
        return render_template('admin.html', articles=[], categories=[])
    finally:
        conn.close()


@app.errorhandler(Exception)
def handle_error(e):
    code = getattr(e, 'code', 500)
    return render_template('error.html', error_code=code, error_message=str(e)), code





@app.route('/upload_file', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def upload_file():

    """
    Upload a file to the server. The file is saved in the
    `UPLOAD_FOLDER` directory and the filename is returned in the
    response.

    The endpoint is rate-limited to 5 requests per minute to prevent
    abuse.

    :return: A JSON response with the filename and URL of the uploaded file.
    """

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        logger.info(f"File uploaded: {file_path}")
        return jsonify({'filename': filename, 'url': url_for('static', filename=f'Uploads/{filename}')})
    return jsonify({'error': 'File type not allowed'}), 400







# Configuration MongoDB
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    logger.error("MongoDB Atlas URI not found")
    raise ValueError("MongoDB Atlas URI not configured")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['moviesDB']
    pricing_analyses_collection = db['pricing_analyses']
    sentiment_analyses_collection = db['sentiment_analyses']
    user_searches_collection = db['user_searches']
    churn_predictions_collection = db['churn_predictions']   # Nouvelle collection pour les analyses de tarification
    client.admin.command('ping')  # Test connection
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("Gemini API key not found")
    raise ValueError("Gemini API key not configured")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    SENTIMENT_MODEL = "gemini-1.5-flash"
    EMBEDDING_MODEL = "embedding-001"
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise


# Paramètres des catégories (à définir globalement)
category_settings = {
    "electronics": {"price_range": (10.0, 100.0), "inventory_range": (50, 200), "demand_range": (3.0, 9.0)},
    "clothing": {"price_range": (5.0, 50.0), "inventory_range": (100, 300), "demand_range": (2.0, 8.0)},
    "books": {"price_range": (5.0, 30.0), "inventory_range": (50, 150), "demand_range": (2.0, 7.0)},
    "home": {"price_range": (10.0, 80.0), "inventory_range": (75, 250), "demand_range": (3.0, 8.0)}
}

def generate_mock_data(category):
    settings = category_settings.get(category, category_settings["electronics"])  # Par défaut : electronics
    competitor_price = round(random.uniform(*settings["price_range"]), 2)
    inventory_level = random.randint(*settings["inventory_range"])
    demand_score = round(random.uniform(*settings["demand_range"]), 1)
    
    return {
        "competitorPrice": competitor_price,
        "inventoryLevel": inventory_level,
        "demandScore": demand_score,
        "productCategory": category
    }

# Route pour obtenir les données simulées
@app.route('/api/pricing-data/<category>', methods=['GET'])
def get_pricing_data(category):
    

    """
    Simulate pricing data for the given category and store it in the database.

    Parameters
    ----------
    category : str
        The category of the product (e.g. electronics, clothing, books, home)

    Returns
    -------
    JSON
        A JSON object containing the simulated data and a success flag

    """

    try:
        mock_data = generate_mock_data(category)

        # Récupérer IP & timezone locale
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        # Store the pricing data in pricing_analyses collection
        pricing_data = {
            "type": "pricing_data",
            "category": category,
            "data": mock_data,
            "timestamp": local_time,
            "ip_address":ip_address,
            "timezone": tz
        }
        pricing_analyses_collection.insert_one(pricing_data)
        logger.info(f"Stored pricing data in pricing_analyses: {pricing_data}")
        return jsonify({
            "success": True,
            "data": mock_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Error generating data",
            "error": str(e)
        }), 500

# Route pour calculer le prix optimal
@app.route('/api/optimize-price', methods=['POST'])
def optimize_price():
    """
    Route pour calculer le prix optimal en fonction des données de référence :
        - prix concurrent, 
        - niveau d'inventaire, 
        - score de demande, 
        - catégorie de produit.

    Vérifie que les données soient numériques et non NaN.

    Applique les multiplieurs de catégorie et les ajustements de prix en fonction de la demande et de l'inventaire :
        - augmentation pour les produits premium (electronics),
        - réduction pour la compétitivité (clothing),
        - neutre (books),
        - légère augmentation (home);

    Applique les règles de prix en fonction de la demande et de l'inventaire :
        - augmentation pour la demande élevée (>7),
        - réduction pour la faible demande (<5),
        - augmentation pour la faible quantité d'inventaire (<100);

    Enregistre les données d'optimisation dans la collection pricing_analyses :
        - type : "optimize_price",
        - input : données de référence,
        - output : prix optimal, augmentation de revenu, stratégie de prix,
        - timestamp : horodate actuelle,
        - ip_address : adresse IP du client,
        - timezone : fuseau horaire du client;

    Retourne les données d'optimisation en JSON :
        - success : True,
        - data : prix optimal, augmentation de revenu, stratégie de prix.
    """

    try:
        data = request.get_json()
        competitor_price = float(data.get('competitorPrice'))
        inventory_level = float(data.get('inventoryLevel'))
        demand_score = float(data.get('demandScore'))
        product_category = data.get('productCategory')

        # Validation des entrées
        if any(x is None or x != x for x in [competitor_price, inventory_level, demand_score]):  # Vérifie NaN
            return jsonify({
                "success": False,
                "message": "Please provide valid numeric values"
            }), 400

        # Logique d'optimisation (identique au frontend)
        optimal_price = competitor_price
        high_demand_multiplier = 1.05
        low_inventory_multiplier = 1.03
        low_demand_multiplier = 0.95

        # Ajustements spécifiques par catégorie (optionnel)
        category_multipliers = {
            "electronics": 1.02,  # Légère augmentation pour produits premium
            "clothing": 0.98,     # Légère réduction pour compétitivité
            "books": 1.0,         # Neutre
            "home": 1.01          # Légère augmentation
        }
        category_multiplier = category_multipliers.get(product_category, 1.0)
        optimal_price *= category_multiplier

        if demand_score > 7:
            optimal_price *= high_demand_multiplier
        if inventory_level < 100:
            optimal_price *= low_inventory_multiplier
        if demand_score < 5:
            optimal_price *= low_demand_multiplier

        revenue_lift = round((optimal_price / competitor_price - 1) * 100, 1)
        
        if optimal_price > competitor_price:
            price_strategy = "Increased price justified by high demand and low inventory. Monitor the competition."
        elif optimal_price < competitor_price:
            price_strategy = "Reduced price to gain market share. Priority to volume over margin."
        else:
            price_strategy = "Same price as competitor. Stay aligned with the market."


        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        # Store the optimization data in pricing_analyses collection
        optimization_data = {
            "type": "optimize_price",
            "input": {
                "competitorPrice": competitor_price,
                "inventoryLevel": inventory_level,
                "demandScore": demand_score,
                "productCategory": product_category
            },
            "output": {
                "optimalPrice": round(optimal_price, 2),
                "revenueLift": revenue_lift,
                "priceStrategy": price_strategy
            },
            "timestamp": local_time,
            "ip_address":ip_address,
            "timezone": tz
        }
        pricing_analyses_collection.insert_one(optimization_data)
        logger.info(f"Stored optimization data in pricing_analyses: {optimization_data}")

        return jsonify({
            "success": True,
            "data": {
                "optimalPrice": round(optimal_price, 2),
                "revenueLift": revenue_lift,
                "priceStrategy": price_strategy
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": "Erreur lors du calcul du prix optimal",
            "error": str(e)
        }), 500








def allowed_file(filename):

    """Check if a file has an allowed extension.

    :param filename: File name to check.
    :return: True if the file has an allowed extension, False otherwise.
    """

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

churn_feature_names = ['monthly_charges', 'tenure', 'total_charges', 'contract_type']

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():

    """Predict customer churn based on input data.

    POST parameters:
        - monthly_charges (float): Monthly charges of the customer.
        - tenure (int): Number of months the customer has been subscribed.
        - total_charges (float): Total charges of the customer.
        - contract_type (string): Contract type of the customer. Can be "month-to-month", "one-year" or "two-year".

    Returns a JSON object with the following keys:
        - probability (float): Churn probability between 0 and 1.
        - recommendation (string): Recommendation based on the churn probability.
    """

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        monthly_charges = float(data.get('monthly_charges', 0))
        tenure = int(data.get('tenure', 0))
        total_charges = float(data.get('total_charges', 0))
        contract_type = bleach.clean(data.get('contract_type', ''))
        contract_map = {"month-to-month": 0, "one-year": 1, "two-year": 2}
        if contract_type not in contract_map:
            return jsonify({'error': 'Invalid contract type'}), 400
        input_data = pd.DataFrame([[monthly_charges, tenure, total_charges, contract_map[contract_type]]],
                                  columns=churn_feature_names)
        input_data = input_data.rename(columns={
            'contract_type': 'Contract',
            'monthly_charges': 'MonthlyCharges',
            'total_charges': 'TotalCharges',
            'tenure': 'tenure'
        })
        if models['churn'] is None:
            try:
                load_model('churn')
            except Exception as e:
                logger.error(f"Failed to load churn model: {str(e)}")
                return jsonify({'error': 'Churn prediction unavailable'}), 503
        churn_model = models['churn']
        if churn_model is None:
            logger.warning("Churn model not loaded, using mock prediction")
            prediction = np.random.uniform(0.1, 0.9)
        else:
            prediction = churn_model.predict_proba(input_data)[0][1]
        recommendation = (
            "High churn risk. Offer discount or contract upgrade." if prediction >= 0.7 else
            "Medium churn risk. Monitor engagement." if prediction >= 0.5 else
            "Low churn risk. Customer is likely loyal."
        )
          # Récupérer IP & timezone locale
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        # Store the prediction in churn_predictions collection
        prediction_data = {
            "input": {
                "monthly_charges": monthly_charges,
                "tenure": tenure,
                "total_charges": total_charges,
                "contract_type": contract_type
            },
            "probability": round(prediction, 4),
            "recommendation": recommendation,
            "timestamp": local_time,
            "ip_address": ip_address,
            "timezone": tz
        }
        churn_predictions_collection.insert_one(prediction_data)
        logger.info(f"Stored churn prediction in churn_predictions: {prediction_data}")

        return jsonify({
            'probability': round(prediction, 4),
            'recommendation': recommendation
        }), 200
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500






@app.route("/")
def home():
    logger.info("Serving home page")
    try:
        return render_template("skills_tools.html")
    except Exception as e:
        logger.error(f"Template error: {e}")
        return jsonify({"error": "Failed to load page"}), 500




@app.route("/skills_tools")
def skills_tools():
    logger.info("Serving skills_tools page")
    try:
        return render_template("skills_tools.html")
    except Exception as e:
        logger.error(f"Template error: {e}")
        return jsonify({"error": "Failed to load skills page"}), 500







@app.route('/api/analyze-sentiment', methods=['POST'])
@limiter.limit("5 per minute")
def analyze_sentiment_gemini():
    """
    Analyze the sentiment of a given text using the Gemini API.

    Example input:
    {
        "text": "I love this product",
        "language": "en"
    }

    Example output:
    {
        "sentiment": "Positive",
        "positive": 80,
        "neutral": 10,
        "negative": 10,
        "insights": "The customer is very satisfied with the product."
    }

    :param text: The text to analyze
    :type text: str
    :param language: The language of the text (optional, default is auto)
    :type language: str
    :return: A JSON object with the sentiment analysis
    :rtype: dict
    """
    logger.info("Received sentiment analysis request")
    try:
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        text = data.get('text', '').strip()
        language = data.get('language', 'auto').lower()

        if not text:
            logger.warning("Empty text received")
            return jsonify({'error': 'Text is required'}), 400
        if len(text) > 10000:
            logger.warning("Input text too long")
            return jsonify({'error': 'Input text is too long (max 10,000 characters)'}), 400

        escaped_text = json.dumps(html.escape(text))[1:-1]
        prompt = f"""
        Analyze the sentiment of the following {language if language != 'auto' else ''} text and return ONLY a valid JSON object (no Markdown):
        - sentiment (Positive, Negative, Neutral)
        - positive (0-100)
        - neutral (0-100)
        - negative (0-100)
        - insights (string)
        Ensure positive + neutral + negative = 100.
        Text: "{escaped_text}"
        """

        logger.info(f"Sending prompt to Gemini API: {prompt[:100]}...")
        model = GenerativeModel(SENTIMENT_MODEL)
        result = model.generate_content(prompt, request_options={'timeout': 30})

        if not result.text:
            logger.error("Empty response from Gemini API")
            return jsonify({'error': 'Empty response from Gemini API'}), 502

        cleaned_text = result.text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:-3].strip()
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:-3].strip()

        try:
            analysis = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Gemini API: {cleaned_text[:100]}... Error: {e}")
            return jsonify({'error': 'Invalid JSON response from Gemini API'}), 502

        required_keys = ['sentiment', 'positive', 'neutral', 'negative', 'insights']
        if not all(k in analysis for k in required_keys):
            logger.error(f"Incomplete response from Gemini API: {analysis}")
            return jsonify({'error': 'Incomplete response from Gemini API'}), 502

        scores = [analysis['positive'], analysis['neutral'], analysis['negative']]
        if not all(isinstance(s, (int, float)) and 0 <= s <= 100 for s in scores):
            logger.error(f"Invalid score values: {scores}")
            return jsonify({'error': 'Invalid score values from Gemini API'}), 502
        if abs(sum(scores) - 100) > 0.01:
            logger.error(f"Scores do not sum to 100: {scores}")
            return jsonify({'error': 'Scores do not sum to 100'}), 502

        # Récupérer IP & timezone locale
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        # Store the sentiment analysis in sentiment_analyses collection
        sentiment_data = {
            'text': escaped_text,
            'language': language,
            'timestamp': local_time,
            'analysis': analysis,
            'ip_address': ip_address,
            'timezone': tz
        }
        sentiment_analyses_collection.insert_one(sentiment_data)
        logger.info(f"Stored sentiment analysis in sentiment_analyses: {sentiment_data}")

        logger.info(f"Sentiment analysis completed: {analysis}")
        return jsonify(analysis), 200

    except Exception as e:
        logger.exception("Error in sentiment analysis")
        return jsonify({'error': 'Internal server error'}), 500









# Configuration MongoDB
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    logger.error("MongoDB Atlas URI not found")
    raise ValueError("MongoDB Atlas URI not configured")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['moviesDB']
    collection = db['movies1']
    user_searches_collection = db['user_searches']  # Nouvelle collection pour les recherches
    client.admin.command('ping')  # Test connection
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise




def calculate_dynamic_budget_thresholds(collection):
    """
    Calculate dynamic budget thresholds based on percentiles of budget values.
    Returns a dictionary with thresholds for 'budget', 'mid-range', and 'premium'.
    """
    try:
        # Retrieve all non-null, positive budget values
        budgets = [
            doc['Budget'] for doc in collection.find(
                {"Budget": {"$exists": True, "$gt": 0}},
                {"Budget": 1, "_id": 0}
            )
        ]
        if not budgets:
            logger.warning("No valid budget data found. Using default thresholds.")
            return {
                'budget': {'$lte': 30_000_000},
                'mid-range': {'$gt': 30_000_000, '$lte': 50_000_000},
                'premium': {'$gt': 50_000_000}
            }

        # Calculate 33rd and 66th percentiles
        budgets = sorted(budgets)
        p33, p66 = np.percentile(budgets, [33, 66])
        logger.info(f"Dynamic budget thresholds calculated: p33={p33:.2f}, p66={p66:.2f}")

        # Define dynamic filters
        budget_filters = {
            'budget': {'$lte': p33},
            'mid-range': {'$gt': p33, '$lte': p66},
            'premium': {'$gt': p66}
        }
        return budget_filters
    except Exception as e:
        logger.error(f"Error calculating dynamic budget thresholds: {e}")
        return {
            'budget': {'$lte': 30_000_000},
            'mid-range': {'$gt': 30_000_000, '$lte': 50_000_000},
            'premium': {'$gt': 50_000_000}
        }

# Calculate thresholds at startup
budget_filters = calculate_dynamic_budget_thresholds(collection)

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("5 per minute")
def recommend_movies():
    """
    Handle a POST request to /api/recommend.
    Expects a JSON object with:
    - `prompt`: a string describing the movie style or genre to recommend.
    - `budget`: a string describing the budget category ("budget", "mid-range", "premium").
    Returns up to 5 movie recommendations based on vector search and budget filter.
    """
    logger.info("Received movie recommendation request")
    try:
        # Validate JSON request
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        budget = data.get('budget', 'budget').lower()

        # Input validation
        if not prompt:
            logger.warning("Empty prompt received")
            return jsonify({'error': 'Prompt is required'}), 400
        if len(prompt) > 10000:
            logger.warning("Input prompt too long")
            return jsonify({'error': 'Input prompt is too long (max 10,000 characters)'}), 400
        if budget not in ['budget', 'mid-range', 'premium']:
            logger.warning(f"Invalid budget category: {budget}")
            return jsonify({'error': 'Invalid budget category. Use budget, mid-range, or premium'}), 400

        sanitized_prompt = html.escape(prompt)

        # Use dynamic budget filters
        price_filter = budget_filters.get(budget, budget_filters['budget'])

        # Generate embedding using Gemini API
        try:
            logger.info(f"Generating embedding for prompt: {sanitized_prompt[:50]}...")
            embedding_response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=sanitized_prompt,
                task_type="retrieval_query"
            )
            embedding = embedding_response['embedding']
            logger.info(f"Embedding generated, length: {len(embedding)}")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return jsonify({'error': 'Failed to generate embedding'}), 500

        # MongoDB aggregation pipeline
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'vector_index',
                    'path': 'plot_embeddings',
                    'queryVector': embedding,
                    'numCandidates': 100,
                    'limit': 5
                }
            },
            {'$match': {'Budget': price_filter}},
            {
                '$project': {
                    'title': 1,
                    'genres': 1,
                    'Budget': 1,
                    '_id': 0,
                    'score': {'$meta': 'vectorSearchScore'}
                }
            }
        ]

        logger.info(f"Executing MongoDB pipeline: {pipeline}")
        results = list(collection.aggregate(pipeline))
        logger.info(f"Found {len(results)} raw results from MongoDB")

        # Map budget to price
        def map_budget_to_price(budget_value):
            if not isinstance(budget_value, (int, float)) or budget_value <= 0:
                return 0
            if budget_value <= budget_filters['budget']['$lte']:
                return 0
            elif budget_value <= budget_filters['mid-range']['$lte']:
                return 50
            return 200

        # Format recommendations
        recommendations = [
            {
                'name': doc.get('title', 'Unknown Title'),
                'price': map_budget_to_price(doc.get('Budget', 0)),
                'genres': doc.get('genres', '').split('|') if doc.get('genres') else [],
                'score': min(float(doc.get('score', 0)) * 100, 100)
            } for doc in results
        ]
        # Récupérer IP & timezone locale
        # Récupérer IP & heure locale exacte
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)

        # Store the search in user_searches collection
        search_data = {
            'prompt': sanitized_prompt,
            'budget': budget,
            'timestamp': local_time,
            'recommendations': recommendations,
            'ip_address': ip_address,
            'timezone': tz 
        }
        user_searches_collection.insert_one(search_data)
        logger.info(f"Stored search in user_searches: {search_data}")

        if not results:
            logger.warning(f"No recommendations found for prompt: {sanitized_prompt[:50]}..., budget: {budget}")
            return jsonify({'recommendations': [], 'warning': 'No movies matched the prompt or budget'}), 200

        logger.info(f"Returning {len(recommendations)} recommendations for prompt: {sanitized_prompt[:50]}...")
        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        logger.error(f"Error in movie recommendation: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


















@app.route('/static/<path:filename>')
def serve_static(filename):

    """
    Serve a static file from the static folder.

    This function is a simple wrapper around flask.send_from_directory,
    which is used to serve static files from the static folder.

    :param filename: The name of the file to serve, relative to the static folder.
    :return: The contents of the file as a response object.
    """

    logger.info(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)



# SocketIO handlers
@socketio.on('track_view')
def handle_track_view(data):

    """
    Handle a 'track_view' event from the client.

    This event is triggered when the user views an article. We update the article's view count in the database and
    emit an 'article_update' event to the client so that the view count can be updated in the UI.

    :param data: A dictionary containing the article ID in the 'article_id' key.
    :return: None
    """

    article_id = data.get('article_id')
    if article_id:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute('UPDATE articles SET views = views + 1 WHERE uuid = %s AND hidden = FALSE',
                               (article_id,))
                conn.commit()
                cursor.execute('SELECT views FROM articles WHERE uuid = %s', (article_id,))
                result = cursor.fetchone()
                new_views = result['views'] if result else 0
                socketio.emit('article_update', {'id': article_id, 'views': new_views})
                logger.info(f"Tracked view for article {article_id}, new views: {new_views}")
        except Exception as e:
            logger.error(f"Error tracking view for article {article_id}: {str(e)}")
        finally:
            conn.close()


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


# Custom rate limit exceeded response
@limiter.request_filter
def rate_limit_exceeded():
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


# Initialize models and datasets at startup
try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize models at startup: {str(e)}")
df_price = load_dataset('retail_price')
MOVIES_DATA = load_dataset('movies').set_index('movieId').to_dict('index') if not load_dataset('movies').empty else {}

# Register API blueprint
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    try:
        def init_db():            

            """
            Initialize the database tables if they don't exist.

            This function will create two tables, 'articles' and 'users', if they don't already exist in the
            database. The 'articles' table stores the article data while the 'users' table stores the user data.

            :return: None
            """

            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS articles (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            uuid VARCHAR(255) UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            description TEXT,
                            category VARCHAR(100),
                            tags TEXT,
                            image TEXT,
                            read_time INT,
                            content TEXT,
                            views INT DEFAULT 0,,
                            created_by INT,
                            updated_by INT,
                            hidden BOOLEAN DEFAULT FALSE,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                            
                        )
                    ''')
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS users (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            username VARCHAR(255) UNIQUE NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            is_active BOOLEAN DEFAULT TRUE,
                            last_login DATETIME,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                            
                            
                        )
                    ''')
                    conn.commit()
            finally:
                conn.close()
        init_db()
        sync_articles_to_db()
        socketio.run(
            app,
            debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
            use_reloader=True,
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 5000))
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")










