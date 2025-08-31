import uuid
import logging
from flask import Blueprint, request, jsonify, abort
from flask_login import login_required, current_user
from database import get_db_connection
from functools import wraps

api = Blueprint('api', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)


def with_db_connection(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        conn = get_db_connection()
        try:
            result = func(conn, *args, **kwargs)
            return result
        finally:
            conn.close()
    return wrapper


# --- LIST ARTICLES (pagination, filtre, recherche) ---
@api.route('/articles', methods=['GET'])
@login_required
@with_db_connection
def list_articles(conn):
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 5))
    category = request.args.get('category')
    status = request.args.get('status', 'all')
    search = request.args.get('search', '')

    where = []
    params = []
    if category:
        where.append("category=%s")
        params.append(category)
    if status == 'visible':
        where.append("hidden=0")
    elif status == 'hidden':
        where.append("hidden=1")
    if search:
        where.append("(title LIKE %s OR description LIKE %s)")
        params.extend([f"%{search}%", f"%{search}%"])

    where_clause = "WHERE " + " AND ".join(where) if where else ""
    offset = (page - 1) * per_page

    with conn.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) as total FROM articles {where_clause}", params)
        total = cursor.fetchone()['total']
        cursor.execute(
            f"SELECT * FROM articles {where_clause} ORDER BY updated_at DESC LIMIT %s OFFSET %s",
            params + [per_page, offset]
        )
        articles = cursor.fetchall()
        pages = (total + per_page - 1) // per_page
    return jsonify({'articles': articles, 'total': total, 'pages': pages})


# --- GET ARTICLE (pour édition) ---
@api.route('/articles/<int:article_id>', methods=['GET'])
@login_required
@with_db_connection
def get_article(conn, article_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
        article = cursor.fetchone()
    if not article:
        abort(404)
    return jsonify(article)


# --- CREATE ARTICLE ---
@api.route('/articles', methods=['POST'])
@login_required
@with_db_connection
def create_article(conn):
    data = request.get_json()
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO articles (title, content, category, description, tags, image, read_time, timestamp, views, hidden, created_at, created_by, updated_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0, NOW(), %s, %s)
        """, (
            data.get('title'),
            data.get('content'),
            data.get('category'),
            data.get('description'),
            ','.join(data.get('tags', [])),
            data.get('image'),
            data.get('read_time', 5),
            data.get('timestamp'),
            current_user.id,
            current_user.id
        ))
        conn.commit()
        article_id = cursor.lastrowid
        cursor.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
        article = cursor.fetchone()
    return jsonify({'success': True, 'article': article})


# --- UPDATE ARTICLE ---
@api.route('/articles/<int:article_id>', methods=['PUT'])
@login_required
@with_db_connection
def update_article(conn, article_id):
    data = request.get_json()
    with conn.cursor() as cursor:
        cursor.execute("""
            UPDATE articles SET
                title=%s, category=%s, description=%s, content=%s, tags=%s, image=%s, read_time=%s, updated_at=NOW(), updated_by=%s
            WHERE id=%s
        """, (
            data.get('title'),
            data.get('category'),
            data.get('description'),
            data.get('content'),
            ','.join(data.get('tags', [])),
            data.get('image'),
            data.get('read_time', 5),
            current_user.id,
            article_id
        ))
        conn.commit()
        cursor.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
        article = cursor.fetchone()
    return jsonify({'success': True, 'article': article})


# --- DELETE ARTICLE ---
@api.route('/articles/<int:article_id>', methods=['DELETE'])
@login_required
@with_db_connection
def delete_article(conn, article_id):
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM articles WHERE id=%s", (article_id,))
        conn.commit()
    return jsonify({'success': True})


# --- TOGGLE VISIBILITY ---
@api.route('/articles/<int:article_id>/toggle-visibility', methods=['POST'])
@login_required
@with_db_connection
def toggle_article_visibility(conn, article_id):
    data = request.get_json()
    hidden = bool(data.get('hidden', False))
    with conn.cursor() as cursor:
        cursor.execute("UPDATE articles SET hidden=%s, updated_at=NOW(), updated_by=%s WHERE id=%s", (hidden, current_user.id, article_id))
        conn.commit()
        cursor.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
        article = cursor.fetchone()
    return jsonify({'success': True, 'article': article})