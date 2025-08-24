import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, TEXT
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from bson import ObjectId
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Create MongoClient once globally - connection reused across Lambda invocations
client = MongoClient(os.getenv("MONGODB_URI"), maxPoolSize=10, serverSelectionTimeoutMS=5000)
db = client[os.getenv("DATABASE_NAME")]
collection = db[os.getenv("COLLECTION_NAME")]

def ensure_text_index():
    indexes = collection.index_information()
    for idx_name, idx_info in indexes.items():
        if idx_info.get('key') and any(key[0] in ('title', 'content') for key in idx_info['key']):
            logger.info(f"Existing text index found: {idx_name}")
            return
    collection.create_index(
        [("title", TEXT), ("content", TEXT)],
        weights={"content": 5, "title": 1},
        name="content_weighted_text_index",
        default_language="english",
    )
    logger.info("Created text index 'content_weighted_text_index' with weights {'content': 5, 'title': 1}")

# Ensure text index at cold start
ensure_text_index()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class JSONEncoderCustom(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

app.json_encoder = JSONEncoderCustom

def prepare_text(post):
    parts = []
    if "title" in post:
        parts.append(post["title"])
    if "content" in post:
        parts.append(post["content"])
    if "tags" in post:
        parts.append(", ".join(post["tags"]))
    if "author" in post:
        parts.append(post["author"])
    return " | ".join(parts)

def embed_text(text):
    return embedding_model.encode(text).tolist()

def vector_search(query_embedding, top_k=10):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 200,
                "limit": top_k,
            }
        },
        {
            "$project": {
                "title": 1,
                "content": 1,
                "author": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    logger.info("Running vector search with pipeline")
    results = list(collection.aggregate(pipeline))
    logger.info(f"Vector search returned {len(results)} results")
    for i, doc in enumerate(results):
        logger.info(f"Result {i+1}: Title: {doc.get('title')}, Score: {doc.get('score')}")
    return results

def text_search_fallback(query, top_k=5):
    logger.info("Running fallback text search")
    results = list(
        collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "title": 1, "content": 1, "author": 1},
        )
        .sort([("score", {"$meta": "textScore"})])
        .limit(top_k)
    )
    logger.info(f"Text search returned {len(results)} results")
    return results

def generate_answer(query, context_posts):
    if not context_posts:
        return (
            "I am sorry, but I do not have the information to answer this question. "
            "Please try with a different query."
        )
    context = "\n\n".join(
        f"Title: {post.get('title', '')}\n"
        f"Content excerpt: {post.get('content', '')[:500]}..."
        for post in context_posts
    )
    prompt = f"""
You are a helpful AI assistant. Answer the question concisely and clearly based on the context below. 
If the answer is not available in the context, respond honestly that you do not know.

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        response = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
                top_p=0.9
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return "Sorry, I encountered an error while generating the response."

@app.route("/search", methods=["POST"])
def search_and_answer():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        logger.warning("Empty query received")
        return jsonify({"error": "Query cannot be empty"}), 400
    try:
        query_embedding = embed_text(user_query)
        logger.info(f"Query embedding generated with length {len(query_embedding)}")
    except Exception as ex:
        logger.error(f"Embedding generation failed: {ex}")
        return jsonify({"error": "Failed to generate embedding"}), 500
    retrieved_posts = vector_search(query_embedding, top_k=10)
    if not retrieved_posts:
        logger.info("No results from vector search, running fallback text search")
        retrieved_posts = text_search_fallback(user_query, top_k=5)
    answer = generate_answer(user_query, retrieved_posts)
    return jsonify(
        {
            "query": user_query,
            "answer": answer,
            "sources": [
                {
                    "title": post.get("title"),
                    "author": post.get("author", "Unknown"),
                    "score": post.get("score"),
                }
                for post in retrieved_posts
            ],
        }
    )

@app.route("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0")
