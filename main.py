import eventlet
eventlet.monkey_patch()
import os
import time
import json
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np
from threading import Lock
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
import uuid
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from urllib.parse import urlparse
import pymysql
from google.generativeai import GenerativeModel
import random
from flask import Response
from datetime import timezone
import requests
from supabase import create_client, Client
import smtplib
from email.message import EmailMessage
import re
import secrets
from datetime import datetime, timedelta, timezone
import bcrypt
import logging
from functools import wraps
from flask import jsonify, request










# Chemin du fichier JSON pour les erreurs
ERROR_LOG_FILE = os.path.join(os.path.dirname(__file__), 'errors_log.json')
# Verrou pour éviter les conflits d'écriture
file_lock = Lock()

def track_error(error_name: str, error_location: str, resolution: str, error_details: str, ip_address: str, input_data: dict = None):
    """
    Enregistre une erreur dans un fichier JSON local et logge dans la console.
    
    Args:
        error_name: Nom de l'erreur (ex. ValueError, TypeError).
        error_location: Où l'erreur s'est produite (ex. nom de la route ou fonction).
        resolution: Conseil pour résoudre l'erreur.
        error_details: Message détaillé de l'erreur.
        ip_address: Adresse IP du client.
        input_data: Données d'entrée (optionnel, pour debugging).
    """
    try:
        tz, local_time = get_local_time_from_ip(ip_address)
        error_doc = {
            'error_name': error_name,
            'error_location': error_location,
            'resolution': resolution,
            'details': error_details,
            'input_data': input_data or {},
            'timestamp': local_time,
            'ip_address': ip_address,
            'timezone': tz
        }
        # Écrire dans le fichier JSON avec verrou
        with file_lock:
            # Si le fichier existe, lire son contenu
            if os.path.exists(ERROR_LOG_FILE):
                with open(ERROR_LOG_FILE, 'r', encoding='utf-8') as f:
                    try:
                        errors = json.load(f)
                    except json.JSONDecodeError:
                        errors = []
            else:
                errors = []
            # Ajouter la nouvelle erreur
            errors.append(error_doc)
            # Écrire dans le fichier
            with open(ERROR_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
        logger.error(f"Tracked error in {ERROR_LOG_FILE}: {error_doc}")
    except Exception as track_e:
        logger.exception(f"Failed to track error in JSON file: {track_e}")















def handle_errors(func):
    """
    Décorateur pour capturer et tracker les erreurs dans les routes Flask.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            error_name = type(ve).__name__
            error_location = f"Route: {func.__name__}"
            resolution = "Vérifiez les données d'entrée fournies (ex. format, type, valeurs manquantes)."
            error_details = str(ve)
            track_error(error_name, error_location, resolution, error_details, get_client_ip(request), request.get_json(silent=True))
            return jsonify({
                'error_name': error_name,
                'error_location': error_location,
                'resolution': resolution,
                'details': error_details
            }), 400
        except TimeoutError as te:
            error_name = type(te).__name__
            error_location = f"Route: {func.__name__}"
            resolution = "Le délai d'attente a été dépassé. Réessayez plus tard ou vérifiez votre connexion."
            error_details = str(te)
            track_error(error_name, error_location, resolution, error_details, get_client_ip(request), request.get_json(silent=True))
            return jsonify({
                'error_name': error_name,
                'error_location': error_location,
                'resolution': resolution,
                'details': error_details
            }), 504
        except KeyError as ke:
            error_name = type(ke).__name__
            error_location = f"Route: {func.__name__}"
            resolution = "Vérifiez que toutes les clés requises sont présentes dans les données envoyées."
            error_details = str(ke)
            track_error(error_name, error_location, resolution, error_details, get_client_ip(request), request.get_json(silent=True))
            return jsonify({
                'error_name': error_name,
                'error_location': error_location,
                'resolution': resolution,
                'details': error_details
            }), 400
        except Exception as e:
            error_name = type(e).__name__
            error_location = f"Route: {func.__name__}"
            resolution = "Erreur inattendue. Contactez le support technique ou vérifiez les logs pour plus de détails."
            error_details = str(e)
            track_error(error_name, error_location, resolution, error_details, get_client_ip(request), request.get_json(silent=True))
            return jsonify({
                'error_name': error_name,
                'error_location': error_location,
                'resolution': resolution,
                'details': error_details
            }), 500
    return wrapper







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



# Initialize Supabase with service role key
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # Clé de service
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)





@app.route("/robots.txt")
@handle_errors
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
app.jinja_env.filters['format_views'] = lambda v: f"{(v / 1000):.1f}K views" if v >= 1000 else f"{v} view{'s' if v != 1 else ''}"

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


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

 

def format_views(views):
    if views >= 1000:
        return f"{(views / 1000):.1f}K views"
    return f"{views} view{'s' if views != 1 else ''}"





# --- Identifier helpers ---
def is_valid_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(str(value))) == str(value)
    except Exception:
        return False

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
@handle_errors
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





# API Blueprint
api = Blueprint('api', __name__)# Python
# --- AJOUTS/MISES À JOUR POUR LE RESET PASSWORD ---

@app.errorhandler(Exception)
def handle_error(e):
    code = getattr(e, 'code', 500)
    error_name = type(e).__name__
    error_location = "Erreur globale (non capturée par une route spécifique)"
    resolution = "Erreur inattendue. Vérifiez les logs ou contactez le support technique."
    error_details = str(e)
    track_error(error_name, error_location, resolution, error_details, get_client_ip(request), request.get_json(silent=True))
    return jsonify({
        'error_name': error_name,
        'error_location': error_location,
        'resolution': resolution,
        'details': error_details
    }), code




# --- Email helper (compléter la classe) ---
class EmailService:
    def __init__(self):
        self.host = os.getenv('EMAIL_HOST')
        self.port = int(os.getenv('EMAIL_PORT', '587'))
        self.user = os.getenv('EMAIL_USER')
        self.password = os.getenv('EMAIL_PASSWORD')
        self.from_addr = os.getenv('EMAIL_FROM', self.user or 'no-reply@example.com')
        self.use_tls = os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'

    def send_reset_code(self, to_email: str, code: str):
        # Gabarit d'email simple
        subject = "Votre code de réinitialisation de mot de passe"
        text = (
            f"Bonjour,\n\n"
            f"Voici votre code de réinitialisation: {code}\n"
            f"Il expirera dans 15 minutes.\n\n"
            f"Si vous n'êtes pas à l'origine de cette demande, vous pouvez ignorer cet email.\n"
            f"Cordialement."
        )

        msg = EmailMessage()
        msg['From'] = self.from_addr
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.set_content(text)

        with smtplib.SMTP(self.host, self.port) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.user and self.password:
                smtp.login(self.user, self.password)
            smtp.send_message(msg)


email_service = EmailService()


# --- Helpers sécurité/validation ---

RESET_CODE_TTL_MINUTES = int(os.getenv('RESET_CODE_TTL_MINUTES', '15'))
RESET_CODE_LENGTH = int(os.getenv('RESET_CODE_LENGTH', '6'))

def normalize_email(email: str) -> str:
    return (email or '').strip().lower()

def is_valid_email(email: str) -> bool:
    # Regex simple, à ajuster selon ton besoin
    return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email or ""))

def generate_reset_code(length: int = RESET_CODE_LENGTH) -> str:
    if length < 4 or length > 10:
        length = 6
    # Code numérique, facile à saisir
    return f"{secrets.randbelow(10 ** length):0{length}d}"

def hash_code(code: str) -> str:
    # Hash du code pour ne pas stocker le code en clair
    return bcrypt.hashpw(code.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_code(code: str, code_hash: str) -> bool:
    try:
        return bcrypt.checkpw(code.encode('utf-8'), code_hash.encode('utf-8'))
    except Exception:
        return False

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def in_future_iso(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


# --- Supabase helpers ---

def get_user_by_email(email: str):
    # Adapte le nom de table/colonnes si nécessaire
    resp = supabase.table('users').select('id').eq('email', email).limit(1).execute()
    data = (resp.data or [])
    return data[0] if data else None

def delete_existing_reset_records(email: str):
    supabase.table('password_resets').delete().eq('email', email).eq('used', False).execute()

def insert_reset_record(email: str, code_hash: str, expires_at_iso: str):
    payload = {
        "email": email,
        "code_hash": code_hash,
        "expires_at": expires_at_iso,
        "used": False,
        "attempt_count": 0,
        "created_at": utcnow_iso(),
    }
    supabase.table('password_resets').insert(payload).execute()

def get_active_reset_record(email: str):
    # Récupère le plus récent non utilisé et non expiré
    now_iso = utcnow_iso()
    resp = (
        supabase.table('password_resets')
        .select('id,email,code_hash,expires_at,used,attempt_count,created_at')
        .eq('email', email)
        .eq('used', False)
        .gt('expires_at', now_iso)
        .order('created_at', desc=True)
        .limit(1)
        .execute()
    )
    data = (resp.data or [])
    return data[0] if data else None

def mark_reset_used(record_id: str):
    supabase.table('password_resets').update({"used": True, "used_at": utcnow_iso()}).eq('id', record_id).execute()

def increment_attempt(record_id: str):
    # Incrémente atomiquement attempt_count (façon simple)
    resp = supabase.table('password_resets').select('attempt_count').eq('id', record_id).limit(1).execute()
    cur = ((resp.data or [{}])[0].get('attempt_count') or 0) + 1
    supabase.table('password_resets').update({"attempt_count": cur}).eq('id', record_id).execute()

def update_user_password(user_id: str, new_password: str):
    pwd_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    # Adapte la colonne si besoin (ex: "password_hash" ou "password")
    supabase.table('users').update({"password_hash": pwd_hash, "updated_at": utcnow_iso()}).eq('id', user_id).execute()


# --- Rate limiting + routes ---

@app.post('/api/auth/request-reset')
@limiter.limit('5 per hour')
@handle_errors
def api_request_reset():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        email = normalize_email(payload.get('email'))
        if not is_valid_email(email):
            return jsonify({"error": "Email invalide"}), 400

        # Vérification utilisateur sans divulguer l'existence
        user = get_user_by_email(email)

        # Génère et stocke le code uniquement si l'utilisateur existe,
        # mais répond pareil dans tous les cas pour éviter l'énumération
        if user:
            try:
                delete_existing_reset_records(email)
                code = generate_reset_code()
                code_hash = hash_code(code)
                expires_at = in_future_iso(RESET_CODE_TTL_MINUTES)
                insert_reset_record(email, code_hash, expires_at)

                # Envoi email
                email_service.send_reset_code(email, code)
            except Exception as e:
                # On loggue l'erreur mais on renvoie quand même un message générique
                logger.warning(f"request-reset: échec traitement interne pour {email}: {e}")

        # Réponse générique
        return jsonify({"message": "Si un compte existe pour cet email, un code a été envoyé."})
    except Exception as e:
        logger.exception(f"request-reset: erreur: {e}")
        return jsonify({"error": "Erreur serveur"}), 500


@app.post('/api/auth/verify-reset')
@limiter.limit('30 per hour')
@handle_errors
def api_verify_reset():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        email = normalize_email(payload.get('email'))
        code = (payload.get('code') or '').strip()

        if not is_valid_email(email):
            return jsonify({"error": "Email invalide"}), 400
        if not re.match(r'^\d{6}$', code):
            return jsonify({"error": "Code invalide (6 chiffres)"}), 400

        record = get_active_reset_record(email)
        if not record:
            return jsonify({"error": "Code invalide ou expiré"}), 400

        if not verify_code(code, record['code_hash']):
            increment_attempt(record['id'])
            return jsonify({"error": "Code invalide"}), 400

        return jsonify({"valid": True})
    except Exception as e:
        logger.exception(f"verify-reset: erreur: {e}")
        return jsonify({"error": "Erreur serveur"}), 500


@app.post('/api/auth/reset-password')
@limiter.limit('10 per hour')
def api_reset_password():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        email = normalize_email(payload.get('email'))
        code = (payload.get('code') or '').strip()
        new_password = payload.get('new_password') or ''

        if not is_valid_email(email):
            return jsonify({"error": "Email invalide"}), 400
        if not re.match(r'^\d{6}$', code):
            return jsonify({"error": "Code invalide (6 chiffres)"}), 400
        if len(new_password) < 8:
            return jsonify({"error": "Mot de passe trop court (min 8 caractères)"}), 400

        user = get_user_by_email(email)
        if not user:
            # Réponse générique (ne pas divulguer)
            time.sleep(0.2)
            return jsonify({"error": "Code invalide ou expiré"}), 400

        record = get_active_reset_record(email)
        if not record:
            return jsonify({"error": "Code invalide ou expiré"}), 400

        if not verify_code(code, record['code_hash']):
            increment_attempt(record['id'])
            return jsonify({"error": "Code invalide"}), 400

        # Met à jour le mot de passe utilisateur
        update_user_password(user['id'], new_password)
        # Marque le code comme utilisé
        mark_reset_used(record['id'])

        return jsonify({"message": "Mot de passe réinitialisé avec succès"})
    except Exception as e:
        logger.exception(f"reset-password: erreur: {e}")
        return jsonify({"error": "Erreur serveur"}), 500
    


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





@app.route('/', defaults={'page': 'homepage'}, methods=['GET', 'POST'])
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

    valid_pages = {'homepage', 'case_studies', 'about', 'skills_tools', 'blog_insights', 'contact_collaboration'}
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





@app.route('/article/<id>')
def show_article(id):
    try:
        # Validate ID format
        if id.isdigit():
            article_id = int(id)
            response = supabase.table('articles').select('*').eq('id', article_id).single().execute()
        else:
            # Validate UUID format
            try:
                uuid.UUID(id)
                response = supabase.table('articles').select('*').eq('uuid', id).single().execute()
            except ValueError:
                logger.warning(f"Invalid UUID format for article ID: {id}")
                return render_template('404.html', message="Article not found"), 404

        article = response.data
        if not article or article.get('hidden'):
            logger.warning(f"Article with ID {id} not found or hidden")
            return render_template('404.html', message="Article not found"), 404

        # Check articles_metadata for consistency
        metadata_key = f"article_id_{article['id'] - 1}" if id.isdigit() else id
        if metadata_key not in articles_metadata:
            logger.warning(f"Article with ID {metadata_key} not found in articles_metadata")
            # Continue with database data, but log the issue

        # Apply metadata defaults
        article = {
            'id': article.get('id', ''),
            'uuid': article.get('uuid', id),
            'title': article.get('title', 'Untitled Article'),
            'description': article.get('description', 'No description available.'),
            'category': article.get('category', 'Uncategorized'),
            'tags': article.get('tags', '').split(',') if article.get('tags') else [],
            'image': article.get('image', 'https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg'),
            'read_time': int(article.get('read_time', 5)) if str(article.get('read_time', 5)).isdigit() else 5,
            'content': article.get('content', ''),
            'views': article.get('views', 0),
            'timestamp': article.get('timestamp', datetime.now(pytz.timezone('Asia/Nicosia')))
        }

        # Normalize timestamp
        timestamp = article['timestamp']
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid timestamp format for article {id}, fallback to now")
                timestamp = datetime.now(pytz.timezone('Asia/Nicosia'))
        article['timestamp'] = timestamp.astimezone(pytz.timezone('Asia/Nicosia')).strftime("%B %d, %Y %H:%M:%S")

        # Increment views
        try:
            updated_views = article['views'] + 1
            supabase.table('articles').update({'views': updated_views}).eq('uuid', article['uuid']).execute()
            article['views'] = updated_views
        except Exception as update_e:
            logger.error(f"Error updating views for article {id}: {update_e}")
            # Continue rendering with current views

        # Convert markdown to HTML
        content_md = article.get('content', '')
        try:
            content_html = markdown2.markdown(content_md, extras=["fenced-code-blocks", "tables", "strike", "footnotes", "code-friendly"])
        except Exception as md_e:
            logger.error(f"Markdown conversion error for article {id}: {md_e}")
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
            article['content'] = bleach.clean(content_html, tags=allowed_tags, attributes=allowed_attrs)
        except Exception as bleach_e:
            logger.error(f"Bleach sanitization error for article {id}: {bleach_e}")
            article['content'] = '<p>No content available.</p>'

        return render_template('article.html', article=article)
    except Exception as e:
        logger.error(f"Error fetching article {id}: {str(e)}", exc_info=True)
        return render_template('404.html', message="Article not found"), 404
    


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

        # --- SUPABASE IMPLEMENTATION ---
        query = supabase.table('articles').select('*')
        if status == 'visible':
            query = query.eq('hidden', False)
        elif status == 'hidden':
            query = query.eq('hidden', True)
        # else: no filter on hidden

        if category:
            query = query.eq('category', category)

        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.order('timestamp', desc=True).range(start, end)
        response = query.execute()
        articles = response.data or []

        # Get total count
        count_query = supabase.table('articles').select('id', count='exact')
        if category:
            count_query = count_query.eq('category', category)
        if status == 'visible':
            count_query = count_query.eq('hidden', False)
        elif status == 'hidden':
            count_query = count_query.eq('hidden', True)
        total = count_query.execute().count or 0

        return jsonify({
            'articles': [
                {
                    'id': article.get('uuid') or article.get('id'),
                    'title': article.get('title'),
                    'category': article.get('category'),
                    'hidden': article.get('hidden'),
                    'description': article.get('description'),
                    'tags': article.get('tags', '').split(',') if article.get('tags') else [],
                    'image': article.get('image'),
                    'read_time': article.get('read_time'),
                    'timestamp': article.get('timestamp').isoformat() if isinstance(article.get('timestamp'), (datetime,)) else article.get('timestamp'),
                    'views': article.get('views')
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
    """Get an article by its ID or UUID.

    Args:
        article_id (str): The numeric ID or UUID of the article to fetch.

    Returns:
        A JSON object containing the article's data, or a 404 error if the article is not found.
    """
    try:
        # First try to find by numeric ID if the input is a number
        if article_id.isdigit():
            response = supabase.table('articles').select(
                'id, uuid, title, category, hidden, description, tags, image, read_time, content'
            ).eq('id', int(article_id)).execute()
            
            if response.data and len(response.data) > 0:
                article = response.data[0]
            else:
                # If not found by numeric ID, try UUID
                response = supabase.table('articles').select(
                    'id, uuid, title, category, hidden, description, tags, image, read_time, content'
                ).eq('uuid', article_id).execute()
                article = response.data[0] if response.data and len(response.data) > 0 else None
        else:
            # Try UUID directly if input is not a number
            response = supabase.table('articles').select(
                'id, uuid, title, category, hidden, description, tags, image, read_time, content'
            ).eq('uuid', article_id).execute()
            article = response.data[0] if response.data and len(response.data) > 0 else None

        if not article:
            return jsonify({'error': 'Article not found'}), 404
            
        return jsonify({
            'id': article.get('uuid') or str(article.get('id')),
            'title': article.get('title'),
            'category': article.get('category'),
            'hidden': article.get('hidden'),
            'description': article.get('description'),
            'tags': article.get('tags', '').split(',') if article.get('tags') else [],
            'image': article.get('image'),
            'read_time': article.get('read_time'),
            'content': article.get('content')
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500



@api.route('/articles', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def create_article():
    try:
        data = request.get_json(silent=True) or {}
        title = (data.get('title') or '').strip()
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        article_id = str(uuid.uuid4())
        tags_value = data.get('tags', [])
        if isinstance(tags_value, list):
            tags = ','.join([str(tag).strip() for tag in tags_value if str(tag).strip()])
        else:
            tags = str(tags_value or '')
        payload = {
            'uuid': article_id,
            'title': title or 'Untitled Article',
            'category': (data.get('category') or 'uncategorized').strip(),
            'description': data.get('description', ''),
            'tags': tags,
            'image': data.get('image', ''),
            'read_time': int(data.get('read_time') or 5),
            'content': data.get('content', ''),
            'created_by': current_user.id,
            'hidden': bool(data.get('hidden', False)),
            'views': int(data.get('views') or 0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        result = supabase.table('articles').insert(payload).execute()
        created = (result.data or [{}])[0]
        socketio.emit('article_updated', {'article': {'id': article_id, 'uuid': article_id}})
        return jsonify({'article': created}), 201
    except Exception as e:
        logger.error(f"Error creating article: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500



@api.route('/articles/<article_id>', methods=['PUT'])
@limiter.limit("5 per minute")
@login_required
def update_article(article_id):
    try:
        data = request.get_json(silent=True) or {}
        updates = {'updated_by': current_user.id, 'updated_at': datetime.now(timezone.utc).isoformat()}
        if 'title' in data:
            title = (data.get('title') or '').strip()
            if not title:
                return jsonify({'error': 'Title cannot be empty'}), 400
            updates['title'] = title
        if 'category' in data:
            updates['category'] = (data.get('category') or 'uncategorized').strip()
        if 'description' in data:
            updates['description'] = data.get('description', '')
        if 'tags' in data:
            tv = data.get('tags', [])
            if isinstance(tv, list):
                updates['tags'] = ','.join([str(tag).strip() for tag in tv if str(tag).strip()])
            else:
                updates['tags'] = str(tv or '')
        if 'image' in data:
            updates['image'] = data.get('image', '')
        if 'read_time' in data:
            try:
                updates['read_time'] = int(data.get('read_time') or 5)
            except Exception:
                updates['read_time'] = 5
        if 'content' in data:
            updates['content'] = data.get('content', '')
        if 'hidden' in data:
            updates['hidden'] = bool(data.get('hidden'))
        if 'views' in data:
            try:
                updates['views'] = int(data.get('views'))
            except Exception:
                pass

        # Update selecting by numeric id first when applicable, else by valid uuid
        if str(article_id).isdigit():
            result = supabase.table('articles').update(updates).eq('id', int(article_id)).execute()
        elif is_valid_uuid(article_id):
            result = supabase.table('articles').update(updates).eq('uuid', article_id).execute()
        else:
            return jsonify({'error': 'Invalid article identifier'}), 400
        if not result.data:
            return jsonify({'error': 'Article not found'}), 404
        updated = (result.data or [{}])[0]
        socketio.emit('article_updated', {'article': {'id': article_id, 'uuid': article_id}})
        return jsonify({'article': updated}), 200
    except Exception as e:
        logger.error(f"Error updating article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    


@api.route('/articles/<article_id>', methods=['DELETE'])
@limiter.limit("5 per minute")
@login_required
def delete_article(article_id):
    try:
        if str(article_id).isdigit():
            # First get the article to ensure it exists and get its UUID
            article = supabase.table('articles').select('uuid').eq('id', int(article_id)).execute()
            if not article.data:
                return jsonify({'error': 'Article not found'}), 404
            article_id = article.data[0]['uuid']
        elif not is_valid_uuid(article_id):
            return jsonify({'error': 'Invalid article identifier'}), 400
            
        # Delete by UUID
        result = supabase.table('articles').delete().eq('uuid', article_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Article not found'}), 404
            
        socketio.emit('article_deleted', {'articleId': article_id, 'uuid': article_id})
        return jsonify({'message': 'Article deleted', 'id': article_id}), 200
    except Exception as e:
        logger.error(f"Error deleting article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/articles/<article_id>/toggle-visibility', methods=['POST'])
@limiter.limit("5 per minute")
@login_required
def toggle_article_visibility(article_id):
    try:
        data = request.get_json(silent=True) or {}
        hidden = data.get('hidden', False)
        if str(article_id).isdigit():
            result = supabase.table('articles').update({'hidden': hidden}).eq('id', int(article_id)).execute()
        elif is_valid_uuid(article_id):
            result = supabase.table('articles').update({'hidden': hidden}).eq('uuid', article_id).execute()
        else:
            return jsonify({'error': 'Invalid article identifier'}), 400
        if not result.data:
            return jsonify({'error': 'Article not found'}), 404

        socketio.emit('article_visibility_changed', {'articleId': article_id, 'uuid': article_id, 'hidden': hidden})
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
        remember = request.form.get('remember') == 'on'  # <-- Ajouté

        if not username or not password:
            flash('Username and password are required.', 'error')
            return redirect(url_for('login'))
        try:
            response = supabase.table('users').select('id, username, password_hash').eq('username', username).eq('is_active', True).single().execute()
            user = response.data
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                user_obj = User(id=user['id'], username=user['username'])
                login_user(user_obj, remember=remember)  # <-- Ajouté ici
                supabase.table('users').update({'last_login': datetime.now(timezone.utc).isoformat()}).eq('id', user['id']).execute()
                flash('Login successful!', 'success')
                return redirect(url_for('admin'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login failed.', 'error')
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Ici, tu dois vérifier si l'email existe, générer un token, envoyer un email, etc.
        # Pour la démo, on affiche juste un message
        flash('If this email exists, a reset link has been sent.', 'success')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))



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
        # Optionally, you can store file metadata in Supabase if you want to track uploads:
        # supabase.table('uploads').insert({
        #     'filename': filename,
        #     'user_id': current_user.id,
        #     'uploaded_at': datetime.now(timezone.utc).isoformat()
        # }).execute()
        return jsonify({'filename': filename, 'url': url_for('static', filename=f'Uploads/{filename}')})
    return jsonify({'error': 'File type not allowed'}),





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


@app.route('/api/track-view/<article_id>', methods=['POST'])
@limiter.limit("10 per minute")
def track_view(article_id):
    """
    Incrémente le nombre de vues d'un article.
    """
    try:
        response = supabase.table('articles').select('views, timestamp').eq('uuid', article_id).single().execute()
        if not response.data:
            return jsonify({'error': 'Article not found'}), 404
        current_views = response.data.get('views', 0)
        supabase.table('articles').update({'views': current_views + 1}).eq('uuid', article_id).execute()
        try:
            # Broadcast via Socket.IO as a fallback for clients
            socketio.emit('article_update', {
                'id': article_id,
                'uuid': article_id,
                'views': current_views + 1,
                'timestamp': response.data.get('timestamp')
            })
        except Exception:
            logger.debug('Socket.IO emit failed for article_update', exc_info=True)
        return jsonify({'views': current_views + 1}), 200
    except Exception as e:
        logger.exception(f"Error tracking view for article {article_id}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


# Custom rate limit exceeded response
@limiter.request_filter
def rate_limit_exceeded():
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


# Initialize models and datasets at startup

# Register API blueprint
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    try:
        socketio.run(
            app,
            debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
            use_reloader=True,
            host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', 5000))
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")


