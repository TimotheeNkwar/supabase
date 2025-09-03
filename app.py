import eventlet
eventlet.monkey_patch()
import os
import time
import json
import logging
from datetime import datetime, timedelta, timezone
import pytz
import numpy as np
import pandas as pd
import bleach
from typing import Dict, Union, Optional
from eventlet.semaphore import Semaphore
from pathlib import Path
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, session, send_from_directory, Blueprint, Response
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import bcrypt
import joblib
import html
import markdown2
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import google.generativeai as genai
import random
import requests
from zoneinfo import ZoneInfo
from supabase import create_client, Client
import uuid
from sklearn.preprocessing import StandardScaler
from google.generativeai import configure as genai_configure, embed_content, GenerativeModel

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully initialized Supabase client.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)
SENTIMENT_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "embedding-001"

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

# Initialize extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app, resources={r"/api/*": {"origins": "*"}})
limiter = Limiter(key_func=get_remote_address, default_limits=[os.environ.get('RATE_LIMIT_DAILY', '200 per day'), os.environ.get('RATE_LIMIT_HOURLY', '50 per hour')], storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
limiter.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    try:
        response = supabase.table('users').select('id, username').eq('id', user_id).execute()
        if response.data and len(response.data) > 0:
            return User(id=response.data[0]['id'], username=response.data[0]['username'])
        return None
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {str(e)}")
        return None

# Client IP and local time functions
def get_client_ip(request):
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"

def get_local_time_from_ip(ip):
    try:
        res = requests.get(f"http://ipinfo.io/{ip}/json", timeout=2).json()
        tz = res.get("timezone", "UTC")
        time_res = requests.get(f"http://worldtimeapi.org/api/timezone/{tz}", timeout=2).json()
        local_time = time_res.get("datetime", datetime.now(timezone.utc).isoformat())
        return tz, local_time
    except Exception:
        return "UTC", datetime.now(timezone.utc).isoformat()

# Routes
@app.route("/robots.txt")
def robots():
    return Response("User-agent: *\nDisallow:", mimetype="text/plain")

@app.route('/python-demo', methods=['GET'])
def python_demo():
    logger.info("Rendering Python Demo page")
    try:
        np.random.seed(42)
        num_customers = 1000
        recency = np.random.randint(2, 46, size=num_customers)
        frequency = np.random.randint(3, 21, size=num_customers)
        avg_order_value = np.full(num_customers, 100)
        monetary = frequency * avg_order_value
        sample_data = {
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'orders': frequency,
            'avg_order_value': avg_order_value
        }
        df = pd.DataFrame(sample_data)
        df['total_spent'] = df['orders'] * df['avg_order_value']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[['recency', 'frequency', 'monetary']])
        result = {
            'high_value': len(df[df['total_spent'] > df['total_spent'].quantile(0.8)]),
            'avg_clv': df['total_spent'].mean(),
            'retention_rate': (df['recency'] < 30).mean() * 100
        }
        return render_template('python_demo.html', sample_data=sample_data, result=result)
    except Exception as e:
        logger.error(f"Error rendering Python Demo page: {str(e)}")
        return render_template('500.html', message=str(e)), 500

# Model and dataset handling
models = {'churn': None}
model_lock = Semaphore()
MODEL_FILES = {'churn': 'models_trains/random_forest_model.pkl'}
DATA_FILES = {'retail_price': 'datasets/retail_price.csv', 'movies': 'datasets/movies.csv'}
datasets = {'retail_price': None, 'movies': None}

def load_model(model_type):
    with model_lock:
        if models[model_type] is not None:
            return models[model_type]
        try:
            models[model_type] = joblib.load(MODEL_FILES[model_type])
            return models[model_type]
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {str(e)}")
            models[model_type] = None
            return None

def load_dataset(dataset_name):
    if datasets[dataset_name] is not None:
        return datasets[dataset_name]
    try:
        df = pd.read_csv(DATA_FILES[dataset_name], quotechar='"', escapechar='\\')
        if dataset_name == 'retail_price':
            df['month_year'] = pd.to_datetime(df['month_year'], format='%m-%d-%Y')
            df = df.dropna()
        datasets[dataset_name] = df
        return df
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
        datasets[dataset_name] = pd.DataFrame()
        return datasets[dataset_name]

# Initialize models and datasets
for model_type in models:
    load_model(model_type)
load_dataset('retail_price')
load_dataset('movies')

# Initialize default admin and articles in Supabase
def init_db():
    # Initialize users
    try:
        response = supabase.table('users').select('count', count=True).execute()
        if response.data[0]['count'] == 0:
            username = os.getenv('DEFAULT_ADMIN_USERNAME', 'admin')
            password = os.getenv('DEFAULT_ADMIN_PASSWORD')
            if not password:
                raise ValueError("DEFAULT_ADMIN_PASSWORD not set")
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            email = os.getenv('DEFAULT_ADMIN_EMAIL', 'timotheenkwar16@gmail.com')
            supabase.table('users').insert({
                'id': str(uuid.uuid4()),
                'username': username,
                'password_hash': password_hash,
                'email': email,
                'is_active': True,
                'created_at': datetime.now(pytz.utc).isoformat()
            }).execute()
            logger.warning(f"Default admin created: {username}. Change password!")
    except Exception as e:
        logger.error(f"Error initializing users: {str(e)}")
        raise

    # Sync articles from JSON file
    try:
        with open('articles_metadata.json', 'r', encoding='utf-8') as file:
            articles_metadata = json.load(file)
        
        for key, meta in articles_metadata.items():
            existing = supabase.table('articles').select('id').eq('title', meta.get('title')).execute()
            if not existing.data:
                supabase.table('articles').insert({
                    'id': str(uuid.uuid4()),
                    'uuid': str(uuid.uuid4()),
                    'title': meta.get('title', ''),
                    'content': meta.get('content', ''),
                    'category': meta.get('category', ''),
                    'description': meta.get('description', ''),
                    'tags': meta.get('tags', []),
                    'image': meta.get('image', ''),
                    'read_time': meta.get('read_time', 0),
                    'hidden': False,
                    'views': 0,
                    'created_at': datetime.now(pytz.utc).isoformat()
                }).execute()
                logger.info(f"Inserted article: {meta.get('title')}")
    except Exception as e:
        logger.error(f"Error syncing articles: {str(e)}")

# Call init_db at startup
init_db()

# API Blueprint
api = Blueprint('api', __name__)

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

        # Logique d'optimisation
        optimal_price = competitor_price
        high_demand_multiplier = 1.05
        low_inventory_multiplier = 1.03
        low_demand_multiplier = 0.95

        # Ajustements spécifiques par catégorie
        category_multipliers = {
            "electronics": 1.02,
            "clothing": 0.98,
            "books": 1.0,
            "home": 1.01
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
            "ip_address": ip_address,
            "timezone": tz
        }
        supabase.table('pricing_analyses').insert(optimization_data).execute()
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
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

churn_feature_names = ['monthly_charges', 'tenure', 'total_charges', 'contract_type']

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    """Predict customer churn based on input data."""
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
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
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
        supabase.table('churn_predictions').insert(prediction_data).execute()
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

        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        sentiment_data = {
            'text': escaped_text,
            'language': language,
            'timestamp': local_time,
            'analysis': analysis,
            'ip_address': ip_address,
            'timezone': tz
        }
        supabase.table('sentiment_analyses').insert(sentiment_data).execute()
        logger.info(f"Stored sentiment analysis in sentiment_analyses: {sentiment_data}")

        logger.info(f"Sentiment analysis completed: {analysis}")
        return jsonify(analysis), 200

    except Exception as e:
        logger.exception("Error in sentiment analysis")
        return jsonify({'error': 'Internal server error'}), 500

# Supabase tables configuration
# Note: Make sure these tables exist in your Supabase database
# - users: for user authentication data
# - articles: for article data
# - movies: for movie data
# - user_searches: for storing user search queries
# - sentiment_analyses: for sentiment analysis results
# - churn_predictions: for churn prediction results
# - pricing_analyses: for pricing optimization results

def calculate_dynamic_budget_thresholds():
    """
    Calculate dynamic budget thresholds based on percentiles of budget values.
    """
    try:
        response = supabase.table('movies').select('Budget').not_.is_('Budget', 'null').gt('Budget', 0).execute()
        budgets = [movie['Budget'] for movie in response.data if movie.get('Budget')]
        
        if not budgets:
            logger.warning("No valid budget data found. Using default thresholds.")
            return {
                'budget': {'lte': 30_000_000},
                'mid-range': {'gt': 30_000_000, 'lte': 50_000_000},
                'premium': {'gt': 50_000_000}
            }

        budgets = sorted(budgets)
        p33, p66 = np.percentile(budgets, [33, 66])
        logger.info(f"Dynamic budget thresholds calculated: p33={p33:.2f}, p66={p66:.2f}")

        budget_filters = {
            'budget': {'lte': p33},
            'mid-range': {'gt': p33, 'lte': p66},
            'premium': {'gt': p66}
        }
        return budget_filters
    except Exception as e:
        logger.warning(f"Movies table not available or error occurred: {e}")
        logger.info("Using default budget thresholds")
        return {
            'budget': {'lte': 30_000_000},
            'mid-range': {'gt': 30_000_000, 'lte': 50_000_000},
            'premium': {'gt': 50_000_000}
        }

# Calculate thresholds at startup
budget_filters = calculate_dynamic_budget_thresholds()

@app.route('/api/recommend', methods=['POST'])
@limiter.limit("5 per minute")
def recommend_movies():
    """
    Handle a POST request to /api/recommend.
    """
    logger.info("Received movie recommendation request")
    try:
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        budget = data.get('budget', 'budget').lower()

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
        price_filter = budget_filters.get(budget, budget_filters['budget'])

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

        try:
            query = supabase.table('movies').select('title, genres, Budget')
            if 'lte' in price_filter:
                query = query.lte('Budget', price_filter['lte'])
            if 'gt' in price_filter:
                query = query.gt('Budget', price_filter['gt'])
            response = query.limit(10).execute()
            results = response.data
            logger.info(f"Found {len(results)} raw results from Supabase")
        except Exception as e:
            logger.warning(f"Movies table not available: {e}")
            logger.info("Returning mock recommendations")
            results = [
                {
                    'title': 'Sample Movie',
                    'genres': 'Action|Adventure',
                    'Budget': 50000000
                }
            ]

        def map_budget_to_price(budget_value):
            if not isinstance(budget_value, (int, float)) or budget_value <= 0:
                return 0
            if budget_value <= budget_filters['budget']['lte']:
                return 0
            elif budget_value <= budget_filters['mid-range']['lte']:
                return 50
            return 200

        recommendations = [
            {
                'name': doc.get('title', 'Unknown Title'),
                'price': map_budget_to_price(doc.get('Budget', 0)),
                'genres': doc.get('genres', '').split('|') if doc.get('genres') else [],
                'score': 85.0
            } for doc in results[:5]
        ]
        ip_address = get_remote_address()
        tz, local_time = get_local_time_from_ip(ip_address)
        search_data = {
            'prompt': sanitized_prompt,
            'budget': budget,
            'timestamp': local_time,
            'recommendations': recommendations,
            'ip_address': ip_address,
            'timezone': tz
        }
        supabase.table('user_searches').insert(search_data).execute()
        logger.info(f"Stored search in user_searches: {search_data}")

        if not results:
            logger.warning(f"No recommendations found for prompt: {sanitized_prompt[:50]}..., budget: {budget}")
            return jsonify({'recommendations': [], 'warning': 'No movies matched the prompt or budget'}), 200

        logger.info(f"Returning {len(recommendations)} recommendations for prompt: {sanitized_prompt[:50]}...")
        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        logger.error(f"Error in movie recommendation: {e}", exc_info=True)
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
        try:
            response = supabase.table('users').select('id, username, password_hash').eq('username', username).eq('is_active', True).execute()
            user = response.data[0] if response.data else None
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                user_obj = User(id=user['id'], username=user['username'])
                login_user(user_obj)
                supabase.table('users').update({'last_login': datetime.now(pytz.utc).isoformat()}).eq('id', user['id']).execute()
                flash('Login successful!', 'success')
                return redirect(url_for('admin'))
            else:
                flash('Invalid username or password.', 'error')
        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            flash('An error occurred during login.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin():
    try:
        response = supabase.table('articles').select('*').execute()
        articles = response.data
        categories = list(set(article['category'] for article in articles if article['category']))
        return render_template('admin.html', articles=articles, categories=categories)
    except Exception as e:
        logger.error(f"Error fetching articles: {str(e)}")
        flash('Error loading admin page.', 'error')
        return redirect(url_for('home'))

@socketio.on('track_view')
def handle_track_view(data):
    article_id = data.get('article_id')
    if article_id:
        try:
            current_views = supabase.table('articles').select('views').eq('uuid', article_id).eq('hidden', False).execute()
            if current_views.data:
                new_views = current_views.data[0]['views'] + 1
                supabase.table('articles').update({'views': new_views}).eq('uuid', article_id).execute()
                socketio.emit('article_update', {'id': article_id, 'views': new_views})
                logger.info(f"Updated views for article {article_id}: {new_views}")
        except Exception as e:
            logger.error(f"Error tracking view for article {article_id}: {str(e)}")

# Startup
if __name__ == '__main__':
    socketio.run(
        app,
        debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
        use_reloader=True,
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000))
    )
