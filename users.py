
import os
import bcrypt
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (optional)
load_dotenv()


# Replace with your actual Supabase URL and anon key, or use environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL','https://gtinadlpbreniysssjai.supabase.co')
SUPABASE_KEY = os.getenv('SUPABASE_KEY','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd0aW5hZGxwYnJlbml5c3NzamFpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMTE3MjcsImV4cCI6MjA2OTg4NzcyN30.LLrCSXgAF30gFq5BrHZhc_KEiasF8LfyZTEExbfwjUk')


# Validate Supabase credentials
if SUPABASE_URL == 'https://your-project-id.supabase.co' or SUPABASE_KEY == 'your-anon-key':
    logger.error("Invalid Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY.")
    raise ValueError("Supabase URL and key must be provided via environment variables or directly in the script.")

# Debug: Print credentials (remove in production for security)
logger.info(f"SUPABASE_URL: {SUPABASE_URL}")
logger.info(f"SUPABASE_KEY: {SUPABASE_KEY[:10]}... (truncated for logging)")

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully initialized Supabase client.")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    raise

# Function to initialize default admin user
def initialize_default_admin():
    try:
        # Check if users table is empty
        response = supabase.table('users').select('count', count=True).execute()
        user_count = response.data[0]['count']
        logger.info(f"Users table contains {user_count} records.")

        if user_count == 0:
            # Get default admin credentials from environment variables
            default_admin = {
                'username': os.getenv('DEFAULT_ADMIN_USERNAME', 'timothee'),
                'password': os.getenv('DEFAULT_ADMIN_PASSWORD', 'secure_admin_password_2025'),
                'email': os.getenv('DEFAULT_ADMIN_EMAIL', 'timotheenkwar16@gmail.com')
            }
            
            # Debug: Log the email value
            logger.info(f"Email to validate: '{default_admin['email']}'")

            # Validate password
            if not default_admin['password']:
                logger.error("DEFAULT_ADMIN_PASSWORD environment variable is not set.")
                raise ValueError("DEFAULT_ADMIN_PASSWORD environment variable is not set")

            # Validate username length (≤50 characters)
            if len(default_admin['username']) > 50:
                logger.error("Username exceeds 50 characters.")
                raise ValueError("Username must be 50 characters or less.")

            # Validate email format (basic check)
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, default_admin['email']):
                logger.error(f"Invalid email format: {default_admin['email']}")
                raise ValueError(f"Invalid email format: {default_admin['email']}")

            # Hash the password
            password_hash = bcrypt.hashpw(
                default_admin['password'].encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')

            # Prepare data for insertion
            user_data = {
                'username': default_admin['username'],
                'password_hash': password_hash,
                'email': default_admin['email'],
                'is_active': True,
                # 'id', 'created_at', 'updated_at', and 'last_login' are handled by the database
            }

            # Insert default admin user
            response = supabase.table('users').insert(user_data).execute()

            # Check for errors
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error inserting default admin user: {response.error}")
                raise Exception(f"Failed to insert default admin user: {response.error}")
            else:
                logger.warning(f"Default admin user created with username: {default_admin['username']}. PLEASE CHANGE THE PASSWORD!")
        
        logger.info("Database initialization completed successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize default admin user: {str(e)}")
        raise

# Run the initialization
if __name__ == "__main__":
    try:
        initialize_default_admin()
    except Exception as e:
        logger.error(f"Initialization process failed: {str(e)}")
