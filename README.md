
# RAG Backend API with MongoDB Atlas Vector Search and Gemini AI

This is a Flask-based backend API implementing Retrieval-Augmented Generation (RAG). It uses precomputed embeddings stored in MongoDB Atlas with Vector Search for semantic retrieval, and Google Gemini AI for natural language answer generation based on retrieved content.

---

## Features

### 1. **Semantic Search with Vector Search**

- Uses sentence-transformers (`all-MiniLM-L6-v2`) to embed posts and user queries.
- Stores 384-dimensional embeddings inside MongoDB documents.
- Performs efficient vector similarity search via MongoDB Atlas `$vectorSearch` operator.
- Returns top relevant posts by semantic similarity.

### 2. **Fallback Text Search**

- Implements a weighted MongoDB full-text search on `title` and `content` fields.
- The text index assigns 5x weight to `content` over `title`.
- Acts as a fallback when vector search returns no results.

### 3. **Natural Language Answer Generation**

- Uses Google Gemini's `gemini-2.0-flash-lite` model to generate concise, contextual answers.
- Provides the retrieved posts’ content snippets as context to the AI prompt.
- Responds honestly if the answer is not in the context.

### 4. **Automatic Daily Embeddings Regeneration**

- Runs a background scheduled job every day at 3 AM IST using APScheduler.
- Updates embeddings for all posts in MongoDB to keep semantic search up to date.
- Ensures new or updated posts have fresh embeddings without manual intervention.

### 5. **Logging and Monitoring**

- Structured logging integrated using Python's `logging` module.
- Logs key events, errors, and scheduling job activity for easier debugging and monitoring.

### 6. **Health Check Endpoint**

- Provides a `/health` route returning HTTP 200 for uptime monitoring.

---

## API Endpoints

### POST `/search`

- **Description:** Receive a natural language query and return an AI-generated answer with source posts.
- **Request JSON:**

```

{
"query": "Your question here"
}

```

- **Response JSON:**

```

{
"query": "Your question here",
"answer": "Generated natural language answer...",
"sources": [
{ "title": "Post Title", "author": "Author Name", "score": 0.95 },
...
]
}

```

### GET `/health`

- **Description:** Returns status `ok` for health monitoring.
- **Response:** Plain text `"ok"`

---

## Setup and Deployment

### Requirements

- Python 3.9+
- MongoDB Atlas cluster with Vector Search support
- Google Gemini API access and key
- Installed dependencies from `requirements.txt`

### Environment Variables

Set these in a `.env` file or your deployment environment:

```

MONGODB_URI=<Your MongoDB connection string>
DATABASE_NAME=bloggite
COLLECTION_NAME=posts
GEMINI_API_KEY=<Your Gemini API key>

```

### Running Locally

1. Install dependencies:

```

pip install -r requirements.txt

```

2. Run Flask app (development):

```

python app.py

```

3. The scheduler starts automatically and regenerates embeddings daily at 3 AM IST.

### Production Deployment

- Use Gunicorn with the following `Procfile`:

```

web: gunicorn app:app --workers 2 --bind 0.0.0.0:\$PORT --timeout 120

```

- Deploy on services like [Render](https://render.com), configuring environment variables and ports as needed.

---

## How to Add New Posts

- Add or update posts in your MongoDB collection `posts`, including at least `title` and `content`.
- The daily embedding regeneration job will automatically update embeddings for these new posts by 3 AM IST.

---

## Notes

- Make sure the MongoDB Atlas Vector Search index on `embedding` exists and is properly configured.
- The text search index on `title` and `content` fields is created automatically by the API on startup.
- The answer generation depends on Google Gemini API and may incur usage costs.
- Logs are output on console for both search queries and scheduled job execution.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or support, please contact the maintainer.
```
