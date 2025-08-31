# Article Management System

## Overview
This project is a web-based article management system designed for administrators to create, edit, delete, and manage articles. It features a responsive frontend built with HTML, Tailwind CSS, and JavaScript, a Flask-based backend with a PostgreSQL database, and real-time updates using WebSocket technology.

## Features
- **Article Management**: Add, edit, delete, and toggle visibility of articles.
- **Filtering and Pagination**: Filter articles by category and status, with paginated results.
- **Real-Time Updates**: WebSocket integration for live updates on article changes.
- **User Authentication**: Protected routes requiring login (currently bypassed for testing).
- **Responsive Design**: Mobile-friendly interface using Tailwind CSS.

## Prerequisites
- Python 3.8+
- PostgreSQL
- Node.js (for frontend dependencies, if needed)
- pip (Python package manager)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/article-management-system.git
cd article-management-system
```

### 2. Set Up the Backend
- Install Python dependencies:
  ```bash
  pip install flask flask-login psycopg2-binary flask-socketio
  ```
- Configure the database in `database.py` with your PostgreSQL credentials:
  ```python
  db_pool = psycopg2.pool.ThreadedConnectionPool(
      1, 20,
      host="localhost",
      database="your_db",
      user="your_user",
      password="your_password",
      port="5432"
  )
  ```
- Create the `articles` table:
  ```sql
  CREATE TABLE articles (
      id VARCHAR(36) PRIMARY KEY,
      title VARCHAR(255) NOT NULL,
      category VARCHAR(100) NOT NULL,
      description TEXT NOT NULL,
      content TEXT NOT NULL,
      tags VARCHAR(255),
      image VARCHAR(255),
      read_time INT NOT NULL,
      hidden BOOLEAN DEFAULT FALSE,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      created_by INT NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_by INT NOT NULL
  );
  ```
- Insert test data (optional):
  ```sql
  INSERT INTO articles (id, title, category, description, content, tags, image, read_time, hidden, created_by, updated_by)
  VALUES ('test-id-1', 'Test Article', 'Technologie', 'Test desc', 'Test content', 'test', 'http://example.com/img.jpg', 5, FALSE, 1, 1);
  ```

### 3. Set Up the Frontend
- Ensure static files (`css`, `js`) are in the `static` folder.
- No additional build steps are required; the HTML and JS files are served directly.

### 4. Run the Application
- Start the Flask app with WebSocket support:
  ```bash
  python app.py
  ```
- Access the admin interface at `http://localhost:5000/admin`.

## Usage
- **Login**: Currently bypassed for testing; implement `flask_login` for production.
- **Manage Articles**: Use the interface to filter, paginate, and modify articles.
- **Real-Time**: Changes are reflected live via WebSocket (requires backend integration).

## File Structure
- `app.py`: Main Flask application file.
- `api_routes.py`: API endpoints for article management.
- `admin.html`: Admin interface template.
- `static/css/admin.css`: Custom CSS (if any).
- `static/js/admin.js`: JavaScript for article management.
- `static/js/realtime-updates.js`: WebSocket handling (optional).
- `database.py`: Database connection configuration.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Troubleshooting
- **500 Error**: Check Flask logs for database errors (e.g., connection issues or schema mismatches).
- **WebSocket Issues**: Ensure Flask-SocketIO is installed and configured.
- **API Access**: Verify PostgreSQL is running and credentials are correct.

## Contact
For support or questions, contact [your-email@example.com](mailto:your-email@example.com).