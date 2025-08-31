import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import markdown2
from dotenv import load_dotenv
from articles_data import articles_metadata
import uuid
from supabase import create_client, Client
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Flask app
app = Flask(__name__, template_folder='pages')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=int(os.getenv('SESSION_LIFETIME_MINUTES', 30)))
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
limiter.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    try:
        response = supabase.table('users').select('id, username').eq('id', user_id).execute()
        user_data = response.data[0] if response.data else None
        if user_data:
            logger.info(f"Loaded user: {user_data['username']} (ID: {user_data['id']})")
            return User(user_data['id'], user_data['username'])
        else:
            logger.warning(f"No user found for ID: {user_id}")
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}")
    return None

# Custom Jinja filters
def custom_markdown(text):
    return markdown2.markdown(text, extras=["fenced-code-blocks", "tables", "spoiler", "header-ids", "code-friendly", "pyshell", "markdown-in-html", "footnotes", "cuddled-lists"])

def datetimeformat(value, format='%d/%m/%Y %H:%M'):
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

# Model handling (only churn)
models = {
    'churn': None
}

MODEL_FILES = {
    'churn': 'models_trains/random_forest_model.pkl'
}

def load_model(model_type):
    if models[model_type] is None:
        try:
            local_file = MODEL_FILES[model_type]
            logger.info(f"Loading {model_type} model from {local_file}")
            if not os.path.exists(local_file):
                logger.error(f"Model file not found: {local_file}")
                return None
            models[model_type] = joblib.load(local_file)
            logger.info(f"{model_type.capitalize()} model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            models[model_type] = None
    return models[model_type]

# Initialize models at startup
try:
    load_model('churn')
except Exception as e:
    logger.error(f"Failed to initialize churn model at startup: {str(e)}")

churn_feature_names = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract']

# Initialize database
def init_db():
    try:
        # Create users table
        supabase.table('users').select('count(*)').execute()  # Test if exists, or create via dashboard
        # Create articles table
        supabase.table('articles').select('count(*)').execute()
        logger.info("Supabase tables initialized.")
    except Exception as e:
        logger.error(f"Error initializing Supabase tables: {str(e)}")
        # Optionally create tables via Supabase SQL editor

init_db()

# Sync articles metadata to Supabase
def sync_articles_to_db():
    try:
        for article_uuid, metadata in articles_metadata.items():
            tags_value = ','.join(metadata.get('tags', [])) if isinstance(metadata.get('tags', []), list) else ''
            timestamp = metadata.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')).isoformat())
            supabase.table('articles').upsert({
                'uuid': article_uuid,
                'title': metadata.get('title', 'Untitled Article'),
                'content': metadata.get('content', 'No content available.'),
                'category': metadata.get('category', 'uncategorized'),
                'description': metadata.get('description', 'No description available.'),
                'tags': tags_value,
                'image': metadata.get('image', 'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
                'read_time': metadata.get('read_time', 1),
                'timestamp': timestamp,
                'views': 0,
                'hidden': False
            }).eq('uuid', article_uuid).execute()
        logger.info("Articles metadata synced successfully to Supabase.")
        return True
    except Exception as e:
        logger.error(f"Error syncing articles_metadata to Supabase: {str(e)}")
        return False

sync_articles_to_db()

# Before request hook to check session timeout
@app.before_request
def check_session_timeout():
    if current_user.is_authenticated:
        last_activity = session.get('last_activity')
        if last_activity:
            last_activity_time = datetime.fromisoformat(last_activity)
            if datetime.now(pytz.timezone('Asia/Nicosia')) - last_activity_time > timedelta(minutes=1):
                logout_user()
                session.clear()
                flash('Session expired due to inactivity. Please log in again.', 'warning')
                logger.info("User logged out due to session timeout.")
                return redirect(url_for('login'))
        session['last_activity'] = datetime.now(pytz.timezone('Asia/Nicosia')).isoformat()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        logger.info("User already authenticated, redirecting to admin panel.")
        return redirect(url_for('admin_panel'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        logger.info(f"Login attempt for username: {username}")
        try:
            response = supabase.table('users').select('id, username, password_hash').eq('username', username).execute()
            user_data = response.data[0] if response.data else None
            if user_data and check_password_hash(user_data['password_hash'], password):
                user = User(user_data['id'], user_data['username'])
                login_user(user)
                session['last_activity'] = datetime.now(pytz.timezone('Asia/Nicosia')).isoformat()
                logger.info(f"User {username} logged in successfully.")
                next_page = request.args.get('next', url_for('admin_panel'))
                return redirect(next_page)
            else:
                flash('Invalid username or password', 'error')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash('Login error. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('You have been logged out.', 'info')
    logger.info("User logged out successfully.")
    return redirect(url_for('login'))

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin_panel():
    articles = []
    message = ""
    try:
        response = supabase.table('articles').select('*').execute()
        db_articles = response.data
        for article_data in db_articles:
            article_id = article_data['id']
            metadata = articles_metadata.get(article_id, {})
            timestamp = article_data['timestamp']
            if isinstance(timestamp, str):
                formatted_timestamp = datetime.fromisoformat(timestamp).astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")
            else:
                formatted_timestamp = timestamp.astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")
            articles.append({
                'id': article_data['id'],
                'title': article_data['title'] or metadata.get('title', ''),
                'category': article_data['category'] or metadata.get('category', ''),
                'description': article_data['description'] or metadata.get('description', ''),
                'tags': article_data['tags'].split(',') if article_data['tags'] else metadata.get('tags', []),
                'image': article_data['image'] or metadata.get('image', ''),
                'read_time': article_data['read_time'] or metadata.get('read_time', 0),
                'hidden': article_data['hidden'],
                'timestamp': formatted_timestamp,
                'views': article_data['views'] or 0,
                'content': metadata.get('content', '')
            })
    except Exception as e:
        logger.error(f"Error in admin panel: {str(e)}")
        message = f"Error: {str(e)}"
    return render_template('admin.html', articles=articles, message=message)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        monthly_charges = float(data['monthly_charges'])
        tenure = int(data['tenure'])
        total_charges = float(data['total_charges'])
        contract_type = data['contract_type']
        contract_map = {"month-to-month": 0, "one-year": 1, "two-year": 2}
        if contract_type not in contract_map:
            return jsonify({'error': 'Invalid contract type'}), 400
        input_data = pd.DataFrame([[monthly_charges, tenure, total_charges, contract_map[contract_type]]],
                                  columns=churn_feature_names)
        churn_model = load_model('churn')
        if churn_model is None:
            logger.warning("Churn model not loaded, using mock prediction")
            prediction = np.random.uniform(0.1, 0.9)
        else:
            prediction = churn_model.predict_proba(input_data)[0][1]
        recommendation = (
            "High churn risk. Consider retention campaign with discount offer or contract upgrade incentive."
            if prediction >= 0.7 else
            "Medium churn risk. Monitor engagement and consider proactive outreach."
            if prediction >= 0.5 else
            "Low churn risk. Customer is likely to remain loyal."
        )
        return jsonify({
            'probability': prediction,
            'recommendation': recommendation
        })
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/case_studies')
def case_studies():
    return render_template('case_studies.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/skills_tools')
def skills_tools():
    return render_template('skills_tools.html')

@app.route('/blog_insights')
def blog_insights():
    articles = []
    try:
        response = supabase.table('articles').select('*').eq('hidden', False).order('timestamp', desc=True).execute()
        db_articles = response.data
        for article_data in db_articles:
            article_id = article_data['id']
            metadata = articles_metadata.get(article_id, {})
            timestamp = article_data['timestamp']
            if isinstance(timestamp, str):
                formatted_timestamp = datetime.fromisoformat(timestamp).astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")
            else:
                formatted_timestamp = timestamp.astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")
            articles.append({
                'id': article_data['id'],
                'title': article_data['title'] or metadata.get('title', ''),
                'category': article_data['category'] or metadata.get('category', ''),
                'description': article_data['description'] or metadata.get('description', ''),
                'tags': article_data['tags'].split(',') if article_data['tags'] else metadata.get('tags', []),
                'image': article_data['image'] or metadata.get('image', ''),
                'read_time': article_data['read_time'] or metadata.get('read_time', 0),
                'timestamp': formatted_timestamp,
                'views': article_data['views'],
                'content': metadata.get('content', '')
            })
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}")
        flash('Unable to load articles at this time.', 'error')
    return render_template('blog_insights.html', articles=articles)

@app.route('/contact_collaboration')
def contact_collaboration():
    return render_template('contact_collaboration.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# Initialize database on startup
init_db()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)