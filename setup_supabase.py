import os
import json
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://gtinadlpbreniysssjai.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk')

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully initialized Supabase client.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

def setup_tables():
    """Create necessary tables in Supabase if they don't exist"""
    logger.info("Setting up Supabase tables...")
    
    # Note: Tables should be created manually in Supabase dashboard
    # This script will help populate them with initial data
    
    tables = [
        'users',
        'articles', 
        'movies',
        'user_searches',
        'sentiment_analyses',
        'churn_predictions',
        'pricing_analyses'
    ]
    
    for table in tables:
        try:
            # Test if table exists by trying to select from it
            response = supabase.table(table).select('*').limit(1).execute()
            logger.info(f"Table '{table}' exists and is accessible")
        except Exception as e:
            logger.warning(f"Table '{table}' may not exist or be accessible: {str(e)}")
            logger.info(f"Please create table '{table}' in your Supabase dashboard")

def load_movies_data():
    """Load movies data from CSV file into Supabase"""
    try:
        import pandas as pd
        
        # Load movies data from CSV
        movies_df = pd.read_csv('datasets/movies.csv')
        logger.info(f"Loaded {len(movies_df)} movies from CSV")
        
        # Convert DataFrame to list of dictionaries
        movies_data = movies_df.to_dict('records')
        
        # Insert data into Supabase (in batches to avoid limits)
        batch_size = 100
        for i in range(0, len(movies_data), batch_size):
            batch = movies_data[i:i + batch_size]
            try:
                response = supabase.table('movies').insert(batch).execute()
                logger.info(f"Inserted batch {i//batch_size + 1} ({len(batch)} movies)")
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error loading movies data: {str(e)}")

def load_articles_data():
    """Load articles data from JSON file into Supabase"""
    try:
        # Load articles data from JSON
        with open('articles_metadata.json', 'r', encoding='utf-8') as file:
            articles_data = json.load(file)
        
        logger.info(f"Loaded {len(articles_data)} articles from JSON")
        
        # Convert to list format for insertion
        articles_list = []
        for key, article in articles_data.items():
            article_data = {
                'title': article.get('title'),
                'category': article.get('category'),
                'description': article.get('description'),
                'tags': article.get('tags'),
                'image': article.get('image'),
                'read_time': article.get('read_time'),
                'content': article.get('content', ''),
                'views': 0,
                'hidden': False
            }
            articles_list.append(article_data)
        
        # Insert data into Supabase
        try:
            response = supabase.table('articles').insert(articles_list).execute()
            logger.info(f"Successfully inserted {len(articles_list)} articles")
        except Exception as e:
            logger.error(f"Error inserting articles: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error loading articles data: {str(e)}")

if __name__ == "__main__":
    try:
        setup_tables()
        load_movies_data()
        load_articles_data()
        logger.info("Supabase setup completed successfully!")
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
