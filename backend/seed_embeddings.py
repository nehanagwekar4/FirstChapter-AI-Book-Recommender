# seed_embeddings.py
"""
Compute & store sentence-transformer embeddings for books in Mongo.
Uses description + genres (NOT title) to avoid title-word bias.
Run once after populating books collection.
"""

import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
import math
import time

load_dotenv()
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
books_coll = db["books"]

model = SentenceTransformer(MODEL_NAME)

async def seed_all(batch_size=128):
    cursor = books_coll.find({})
    to_update = []
    i = 0
    batch = []
    async for b in cursor:
        # build text for embedding: prefer description + genres (avoid title to reduce exact-title matches)
        desc = (b.get("description") or "")
        genres = " ".join(b.get("genre") or b.get("genres") or [])
        text = (desc + " " + genres).strip()
        if not text:
            # fallback to title if no description/genres present
            text = (b.get("title") or "")
        batch.append((b["_id"], b["id"], text))
        i += 1
        if len(batch) >= batch_size:
            await _encode_and_store(batch)
            batch = []
    if batch:
        await _encode_and_store(batch)
    print(f"Done. Processed {i} books.")
    client.close()

async def _encode_and_store(batch):
    ids = [t[0] for t in batch]
    texts = [t[2] for t in batch]
    # compute embeddings (numpy)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # store back (as list of floats)
    ops = []
    for _id, numid, emb in zip(ids, [t[1] for t in batch], embs):
        emb_list = emb.astype(float).tolist()
        await books_coll.update_one({"_id": _id}, {"$set": {"embedding": emb_list}})
    print(f"Stored embeddings for batch size {len(batch)}")

if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(seed_all())
    print("Elapsed:", time.time() - t0)
