import os
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize(vec):
    """Normalize embedding for cosine similarity"""
    return (vec / np.linalg.norm(vec)).tolist()

def prepare_text(post):
    parts = []
    if 'title' in post:
        parts.append(post['title'])
    if 'content' in post:
        parts.append(post['content'])
    if 'tags' in post:
        parts.append(', '.join(post['tags']))
    if 'author' in post:
        parts.append(post['author'])
    return ' | '.join(parts)

def generate_and_save_embeddings():
    posts = collection.find({})
    for post in posts:
        if 'embedding' in post:  # already has vector
            continue  

        text = prepare_text(post)
        if not text.strip():  # skip empty docs
            print(f"Skipping empty post: {post.get('_id')}")
            continue

        embedding = normalize(embedding_model.encode(text))

        collection.update_one(
            {'_id': post['_id']},
            {'$set': {'embedding': embedding}}
        )

        print(f"✅ Updated embedding for: {post.get('title', post['_id'])}")

if __name__ == '__main__':
    generate_and_save_embeddings()
