# Configuration Supabase

Ce guide vous aide à configurer Supabase pour remplacer MongoDB dans votre application.

## 1. Configuration des Tables

Créez les tables suivantes dans votre dashboard Supabase :

### Table `users`
```sql
CREATE TABLE users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  last_login TIMESTAMP WITH TIME ZONE
);
```

### Table `articles`
```sql
CREATE TABLE articles (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  category VARCHAR(100),
  description TEXT,
  tags TEXT[],
  image VARCHAR(500),
  read_time INTEGER,
  content TEXT,
  views INTEGER DEFAULT 0,
  hidden BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Table `movies`
```sql
CREATE TABLE movies (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title VARCHAR(255),
  genres TEXT,
  Budget INTEGER,
  Revenue INTEGER,
  plot_embeddings VECTOR(768),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Table `user_searches`
```sql
CREATE TABLE user_searches (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  prompt TEXT NOT NULL,
  budget VARCHAR(50),
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  recommendations JSONB,
  ip_address INET,
  timezone VARCHAR(50)
);
```

### Table `sentiment_analyses`
```sql
CREATE TABLE sentiment_analyses (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  text TEXT NOT NULL,
  language VARCHAR(10),
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  analysis JSONB,
  ip_address INET,
  timezone VARCHAR(50)
);
```

### Table `churn_predictions`
```sql
CREATE TABLE churn_predictions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  input_data JSONB,
  probability DECIMAL(5,4),
  recommendation TEXT,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  ip_address INET,
  timezone VARCHAR(50)
);
```

### Table `pricing_analyses`
```sql
CREATE TABLE pricing_analyses (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  input_data JSONB,
  output JSONB,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  ip_address INET,
  timezone VARCHAR(50)
);
```

## 2. Configuration des Politiques RLS

Activez Row Level Security (RLS) et configurez les politiques :

### Politique pour `users`
```sql
-- Permettre la lecture pour les utilisateurs authentifiés
CREATE POLICY "Users can view their own data" ON users
  FOR SELECT USING (auth.uid() = id);

-- Permettre l'insertion pour les nouveaux utilisateurs
CREATE POLICY "Allow user registration" ON users
  FOR INSERT WITH CHECK (true);

-- Permettre la mise à jour pour les utilisateurs authentifiés
CREATE POLICY "Users can update their own data" ON users
  FOR UPDATE USING (auth.uid() = id);
```

### Politique pour `articles`
```sql
-- Permettre la lecture publique
CREATE POLICY "Public read access" ON articles
  FOR SELECT USING (hidden = false);

-- Permettre l'insertion/mise à jour pour les admins
CREATE POLICY "Admin write access" ON articles
  FOR ALL USING (auth.role() = 'authenticated');
```

### Politique pour les autres tables
```sql
-- Permettre l'insertion publique pour les analyses
CREATE POLICY "Public insert access" ON user_searches
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Public insert access" ON sentiment_analyses
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Public insert access" ON churn_predictions
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Public insert access" ON pricing_analyses
  FOR INSERT WITH CHECK (true);
```

## 3. Variables d'Environnement

Créez un fichier `.env` avec :

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key-here
GEMINI_API_KEY=your-gemini-api-key
FLASK_SECRET_KEY=your-secret-key
```

## 4. Exécution du Script de Configuration

```bash
python setup_supabase.py
```

Ce script va :
- Vérifier l'existence des tables
- Charger les données des films depuis `datasets/movies.csv`
- Charger les articles depuis `articles_metadata.json`

## 5. Test de l'Application

```bash
python app.py
```

## Notes Importantes

1. **Recherche Vectorielle** : Supabase ne supporte pas directement la recherche vectorielle comme MongoDB. L'application utilise maintenant une approche basée sur les filtres de budget.

2. **Authentification** : L'authentification utilise maintenant Supabase Auth au lieu de MongoDB.

3. **Performance** : Pour de meilleures performances, considérez l'utilisation de pgvector pour la recherche vectorielle dans Supabase.

4. **Sauvegarde** : Configurez des sauvegardes automatiques dans votre dashboard Supabase.
