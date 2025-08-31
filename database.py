import os
import time
import pymysql
import logging
from functools import wraps
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def retry_on_error(max_retries=3, delay=2):
    """Décorateur pour retenter une connexion DB en cas d'erreur."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except pymysql.MySQLError as e:
                    last_error = e
                    logger.warning(f"Tentative {attempt + 1} échouée pour {func.__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Échec total après {max_retries} tentatives : {str(e)}")
                        raise
                    time.sleep(delay)
            return last_error
        return wrapper
    return decorator

@retry_on_error()
def get_db_connection():
    """Retourne une connexion MySQL avec variables d'environnement."""
    load_dotenv()
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST').strip(),
            user=os.getenv('DB_USER').strip(),
            password=os.getenv('DB_PASSWORD').strip(),
            database=os.getenv('DB_NAME').strip(),
            port=int(os.getenv('DB_PORT', 3306)),  # 3306 par défaut
            cursorclass=pymysql.cursors.DictCursor,
            charset='utf8mb4',
            connect_timeout=10
        )
        return conn
    except pymysql.MySQLError as e:
        logger.error(f"Erreur de connexion à la base de données: {str(e)}")
        raise


def ensure_connection():
    """Teste la connexion à la base de données. Retourne True si OK, sinon False."""
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception:
        return False
