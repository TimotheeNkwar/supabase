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
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from urllib.parse import urlparse
from google.generativeai import configure as genai_configure, embed_content, GenerativeModel
import random
from flask import Flask, Response
from datetime import datetime, timezone
import requests
from zoneinfo import ZoneInfo
from supabase import create_client, Client


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

# 1. Ajoute l'import du client Supabase et configure-le
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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


# --- USER LOADING ---
@login_manager.user_loader
def load_user(user_id):
    try:
        response = supabase.table('users').select('id, username').eq('id', int(user_id)).single().execute()
        user_data = response.data
        if user_data:
            return User(id=user_data['id'], username=user_data['username'])
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}")
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
            response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).execute()
            articles = response.data or []
            for article in articles:
                article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
                if isinstance(article.get('timestamp'), str):
                    article['timestamp'] = article['timestamp']
                elif article.get('timestamp'):
                    article['timestamp'] = str(article['timestamp'])
            logger.info(f"Successfully loaded {len(articles)} articles")
        except Exception as e:
            logger.error(f"Error loading articles: {str(e)}")
            flash('Unable to load articles at this time.', 'error')
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
        response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).limit(3).execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
            if isinstance(article.get('timestamp'), str):
                article['timestamp'] = article['timestamp']
            elif article.get('timestamp'):
                article['timestamp'] = str(article['timestamp'])
        return render_template('index.html', articles=articles)  # Fixed to index.html
    except Exception as e:
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
        response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
            if isinstance(article.get('timestamp'), str):
                article['timestamp'] = article['timestamp']
            elif article.get('timestamp'):
                article['timestamp'] = str(article['timestamp'])
        return render_template('blog_insights.html', articles=articles)
    except Exception as e:
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
        query = supabase.table('articles').select('*')
        if status != 'all':
            query = query.eq('hidden', status == 'hidden')
        if search:
            query = query.ilike('title', f'%{search}%')
        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.order('timestamp', desc=True).range(start, end)
        response = query.execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
        # Pour le total :
        total = supabase.table('articles').select('id', count='exact').execute().count or 0
        return jsonify({
            'articles': articles,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error fetching API articles: {e}")
        return jsonify({'error': 'Database error'}), 500
    finally:
        if conn:
            conn.close()

# --- ARTICLES POUR LE BLOG ---
@app.route('/blog')
def blog():
    try:
        response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
            if isinstance(article.get('timestamp'), str):
                article['timestamp'] = article['timestamp']
            elif article.get('timestamp'):
                article['timestamp'] = str(article['timestamp'])
        return render_template('blog_insights.html', articles=articles)
    except Exception as e:
        logger.error(f"Error fetching blog articles: {e}")
        return render_template('blog_insights.html', articles=[])

# --- PAGE D'ACCUEIL ---
@app.route('/')
def index():
    try:
        response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).limit(3).execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
            if isinstance(article.get('timestamp'), str):
                article['timestamp'] = article['timestamp']
            elif article.get('timestamp'):
                article['timestamp'] = str(article['timestamp'])
        return render_template('index.html', articles=articles)
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        return render_template('index.html', articles=[])

# --- AFFICHAGE D'UN ARTICLE ---
@app.route('/article/<id>')
def show_article(id):
    try:
        # Recherche par uuid ou id numérique
        if id.isdigit():
            response = supabase.table('articles').select('*').eq('id', int(id)).single().execute()
        else:
            response = supabase.table('articles').select('*').eq('uuid', id).single().execute()
        article = response.data
        if not article or article.get('hidden'):
            return render_template('404.html', message="Article not found"), 404
        article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
        # Markdown, bleach, etc. (reprends ta logique existante ici)
        # ...existing code for markdown/bleach...
        return render_template('article.html', article=article)
    except Exception as e:
        logger.error(f"Error fetching article {id}: {e}")
        return render_template('404.html', message="Article not found"), 404

# --- API ARTICLES ---
@app.route('/api/articles', methods=['GET'])
def api_articles():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 5))
        search = request.args.get('search', '')
        status = request.args.get('status', 'all')
        query = supabase.table('articles').select('*')
        if status != 'all':
            query = query.eq('hidden', status == 'hidden')
        if search:
            query = query.ilike('title', f'%{search}%')
        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.order('timestamp', desc=True).range(start, end)
        response = query.execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
        # Pour le total :
        total = supabase.table('articles').select('id', count='exact').execute().count or 0
        return jsonify({
            'articles': articles,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error fetching API articles: {e}")
        return jsonify({'error': 'Database error'}), 500

# --- CRUD ARTICLES (API) ---
@api.route('/articles', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def create_article():
    try:
        data = request.get_json()
        if not data or not data.get('title'):
            return jsonify({'error': 'Title is required'}), 400
        article_id = str(uuid.uuid4())
        tags = ','.join([tag.strip() for tag in data.get('tags', [])])
        supabase.table('articles').insert({
            'uuid': article_id,
            'title': data.get('title', 'Untitled Article'),
            'category': data.get('category', 'uncategorized'),
            'description': data.get('description', ''),
            'tags': tags,
            'image': data.get('image', ''),
            'read_time': data.get('read_time', 5),
            'content': data.get('content', ''),
            'created_by': current_user.id
        }).execute()
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
        tags = ','.join([tag.strip() for tag in data.get('tags', [])])
        result = supabase.table('articles').update({
            'title': data.get('title', 'Untitled Article'),
            'category': data.get('category', 'uncategorized'),
            'description': data.get('description', ''),
            'tags': tags,
            'image': data.get('image', ''),
            'read_time': data.get('read_time', 5),
            'content': data.get('content', ''),
            'updated_by': current_user.id
        }).eq('uuid', article_id).execute()
        if not result.data:
            return jsonify({'error': 'Article not found'}), 404
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
        result = supabase.table('articles').delete().eq('uuid', article_id).execute()
        if not result.data:
            return jsonify({'error': 'Article not found'}), 404
        socketio.emit('article_deleted', {'articleId': article_id})
        return jsonify({'message': 'Article deleted'}), 200
    except Exception as e:
        logger.error(f"Error deleting article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# --- LOGIN ---
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
        try:
            response = supabase.table('users').select('id, username, password_hash').eq('username', username).eq('is_active', True).single().execute()
            user = response.data
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                user_obj = User(id=user['id'], username=user['username'])
                login_user(user_obj)
                supabase.table('users').update({'last_login': datetime.now(timezone.utc).isoformat()}).eq('id', user['id']).execute()
                flash('Login successful!', 'success')
                return redirect(url_for('admin'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login failed.', 'error')
    return render_template('login.html')

# --- ADMIN ---
@app.route('/admin')
@login_required
def admin():
    try:
        response = supabase.table('articles').select('*').order('updated_at', desc=True).execute()
        articles = response.data or []
        for article in articles:
            article['tags'] = article.get('tags', '').split(',') if article.get('tags') else []
        categories = list({a.get('category', '') for a in articles if a.get('category')})
        return render_template('admin.html', articles=articles, categories=categories)
    except Exception as e:
        logger.error(f"Error in admin view: {str(e)}")
        flash(f'Error retrieving articles: {str(e)}', 'error')
        return render_template('admin.html', articles=[], categories=[])

# --- CATEGORIES ---
@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        response = supabase.table('articles').select('category').neq('category', None).execute()
        categories = list({row['category'] for row in (response.data or []) if row.get('category')})
        if not categories:
            return jsonify({'error': 'No categories available'}), 404
        return jsonify(categories), 200
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# --- SOCKETIO VIEWS ---
@socketio.on('track_view')
def handle_track_view(data):
    article_id = data.get('article_id')
    if article_id:
        try:
            supabase.table('articles').update({'views': supabase.table('articles').select('views').eq('uuid', article_id).single().execute().data['views'] + 1}).eq('uuid', article_id).execute()
            result = supabase.table('articles').select('views').eq('uuid', article_id).single().execute()
            new_views = result.data['views'] if result.data else 0
            socketio.emit('article_update', {'id': article_id, 'views': new_views})
        except Exception as e:
            logger.error(f"Error tracking view for article {article_id}: {str(e)}")

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










