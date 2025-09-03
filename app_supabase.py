import os
import logging
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from supabase import create_client, Client
import bcrypt
import uuid
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
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
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}

# Initialize Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://gtinadlpbreniysssjai.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize extensions
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

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id: str, username: str, **kwargs):
        self.id = id
        self.username = username
        self.__dict__.update(kwargs)

    @staticmethod
    def get(user_id: str) -> Optional['User']:
        """Get user by ID from Supabase"""
        try:
            response = supabase.table('users').select('*').eq('id', user_id).execute()
            if response.data:
                user_data = response.data[0]
                return User(
                    id=str(user_data['id']),
                    username=user_data['username'],
                    **{k: v for k, v in user_data.items() if k not in ['id', 'username']}
                )
            return None
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            return None

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return User.get(user_id)

# Article functions
def get_articles(page: int = 1, per_page: int = 10, status: str = 'all', search: str = None) -> Dict[str, Any]:
    """Get paginated articles from Supabase"""
    try:
        query = supabase.table('articles').select('*', count='exact')
        
        if status == 'published':
            query = query.eq('hidden', False)
        elif status == 'hidden':
            query = query.eq('hidden', True)
            
        if search:
            query = query.or_(f"title.ilike.%{search}%,description.ilike.%{search}%")
            
        count_response = query.execute()
        total = len(count_response.data)
        
        articles = query.order('created_at', desc=True).range((page-1)*per_page, page*per_page-1).execute()
        
        return {
            'articles': articles.data,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
    except Exception as e:
        logger.error(f"Error getting articles: {str(e)}")
        return {'articles': [], 'total': 0, 'pages': 0}

# Routes
@app.route('/')
def admin():
    """Show latest articles on homepage"""
    try:
        articles = supabase.table('articles')\
            .select('*')\
            .eq('hidden', False)\
            .order('created_at', desc=True)\
            .limit(3)\
            .execute()
            
        return render_template('admin.html', articles=articles.data)
    except Exception as e:
        logger.error(f"Error in admin: {str(e)}")
        return render_template('admin.html', articles=[])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('login.html')
            
        try:
            response = supabase.table('users').select('*').ilike('username', username).execute()
            
            if not response.data:
                flash('Invalid username or password', 'error')
                return render_template('login.html')
                
            user_data = response.data[0]
            
            if not bcrypt.checkpw(password.encode('utf-8'), user_data['password_hash'].encode('utf-8')):
                flash('Invalid username or password', 'error')
                return render_template('login.html')
                
            user = User(
                id=str(user_data['id']),
                username=user_data['username']
            )
            
            login_user(user)
            
            supabase.table('users').update({
                'last_login': datetime.utcnow().isoformat()
            }).eq('id', user_data['id']).execute()
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('admin'))
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('admin'))

if __name__ == '__main__':
    app.run(debug=True)
