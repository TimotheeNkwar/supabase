CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    uuid UUID UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    content TEXT,
    tags TEXT,
    image TEXT,
    read_time INT,
    views INT DEFAULT 0,
    clicks INT DEFAULT 0,
    created_by INTEGER REFERENCES users(id),
    updated_by INTEGER REFERENCES users(id),
    hidden BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Table for password reset codes with 5-minute expiry
CREATE TABLE IF NOT EXISTS password_resets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    code VARCHAR(6) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '5 minutes'),
    used BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast lookup by email+code and automatic expiry
CREATE INDEX IF NOT EXISTS idx_password_resets_email_code ON password_resets (email, code) WHERE used = FALSE;
CREATE INDEX IF NOT EXISTS idx_password_resets_expires_at ON password_resets (expires_at);


