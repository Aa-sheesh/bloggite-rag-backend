import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DATABASE_NAME")]
collection = db[os.getenv("COLLECTION_NAME")]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


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


def regenerate_embeddings():
    cursor = collection.find({})
    for post in cursor:
        text = prepare_text(post)
        embedding = embedding_model.encode(text).tolist()

        collection.update_one({"_id": post["_id"]}, {"$set": {"embedding": embedding}})

        print(f"Re-generated embedding for post: {post.get('title', post['_id'])}")


if __name__ == "__main__":
    regenerate_embeddings()
