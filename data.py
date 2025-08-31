
import json
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (optional)
load_dotenv()

# Supabase configuration
# Replace with your actual Supabase URL and anon key, or use environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Validate Supabase credentials
if SUPABASE_URL == 'https://your-project-id.supabase.co' or SUPABASE_KEY == 'your-anon-key':
    logger.error("Invalid Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY.")
    raise ValueError("Supabase URL and key must be provided via environment variables or directly in the script.")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully initialized Supabase client.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Load the articles metadata from a JSON file
# Replace with your actual file path or use the JSON string directly
json_file_path = 'articles_metadata.json'
try:
    with open(json_file_path, 'r') as file:
        articles_data = json.load(file)
    logger.info("Successfully loaded articles metadata from JSON file.")
except FileNotFoundError:
    logger.error(f"JSON file not found: {json_file_path}")
    raise
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON format: {str(e)}")
    raise

# Function to insert articles into Supabase
def insert_articles():
    for key, article in articles_data.items():
        try:
            # Prepare data for insertion
            data = {
                "title": article.get("title"),
                "category": article.get("category"),
                "description": article.get("description"),
                "tags": article.get("tags"),  # Stored as TEXT[] in Supabase
                "image": article.get("image"),
                "read_time": article.get("read_time"),
                "content": article.get("content")
            }
            
            # Insert into 'articles' table
            response = supabase.table('articles').insert(data).execute()
            
            # Check for errors
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error inserting article {key}: {response.error}")
            else:
                logger.info(f"Successfully inserted article {key}: {article['title']}")
        except Exception as e:
            logger.error(f"Exception while inserting article {key}: {str(e)}")
            continue

# Run the insertion
if __name__ == "__main__":
    try:
        insert_articles()
        logger.info("Article insertion process completed.")
    except Exception as e:
        logger.error(f"Article insertion process failed: {str(e)}")
