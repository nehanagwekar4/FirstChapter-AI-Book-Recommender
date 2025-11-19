# seed_openlibrary.py
"""
Seed Mongo DB with books fetched from OpenLibrary subjects.
Usage:
  python seed_openlibrary.py --subjects "fantasy,romance,mystery" --max 1000

This will fetch works for listed subjects, dedupe by work key, compute embeddings (sentence-transformers),
and insert documents into the `books` collection with normalized fields and `embedding` (list of floats).
"""

import os
import argparse
import time
import requests
from typing import List
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
import math

load_dotenv()
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# OpenLibrary endpoints
OL_SUBJECT_URL = "https://openlibrary.org/subjects/{subject}.json?limit=100"
OL_WORK_URL = "https://openlibrary.org{key}.json"  # key is like /works/OLxxxxxW

# Helper: clean text
def norm_text(s):
    if not s:
        return ""
    if isinstance(s, dict):
        # sometimes description is {'value': '...'}
        return s.get("value") or str(s)
    return str(s)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=str, default="fantasy,romance,mystery")
    p.add_argument("--max", type=int, default=500)
    p.add_argument("--clear", action="store_true", help="clear books collection before inserting")
    return p.parse_args()

async def insert_books(docs):
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    coll = db["books"]
    if args.clear:
        print("Clearing books collection...")
        await coll.delete_many({})
    if docs:
        print(f"Inserting {len(docs)} books...")
        # Use insert_many in batches to avoid memory overhead
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            await coll.insert_many(docs[i:i+batch_size])
            print(f"Inserted batch {i//batch_size+1}")
    client.close()

def fetch_subject_works(subject, limit=100):
    url = OL_SUBJECT_URL.format(subject=subject)
    works = []
    offset = 0
    while len(works) < limit:
        res = requests.get(url)
        if res.status_code != 200:
            print("Failed to fetch subject", subject, res.status_code)
            break
        data = res.json()
        works.extend(data.get("works", []))
        break  # the subject API returns up to 100 per request; for simplicity stop here
    return works[:limit]

def fetch_work_details(work_key):
    # work_key like '/works/OLxxxxW'
    url = OL_WORK_URL.format(key=work_key)
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def build_doc_from_work(work, details):
    # work is a summary from subject list, details is full /works/... JSON
    title = work.get("title") or details.get("title")
    authors = []
    if details.get("authors"):
        for a in details.get("authors", []):
            if isinstance(a, dict) and a.get("author") and a["author"].get("key"):
                # fetch author name? sometimes author object already has name
                # try to use a.get('name') else leave blank
                # author key like /authors/OLxxxA
                authors.append(a.get("name") or "Unknown")
    if not authors and work.get("authors"):
        authors = [a.get("name") for a in work.get("authors", []) if a.get("name")]
    description = norm_text(details.get("description") or work.get("description"))
    subjects = details.get("subjects") or work.get("subject") or work.get("subjects") or []
    # year: try to extract first_publish_date or created
    year = None
    if details.get("first_publish_date"):
        try:
            year = int(details.get("first_publish_date")[:4])
        except:
            year = None
    if not year:
        if details.get("created") and isinstance(details["created"], dict):
            # created: {"type": "/type/datetime", "value": "2009-10-..."}
            try:
                year = int(details["created"]["value"][:4])
            except:
                year = None

    cover_id = work.get("cover_id") or (details.get("covers") and details["covers"][0] if details.get("covers") else None)
    cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None

    doc = {
        "id": work.get("key").replace("/works/", "") if work.get("key") else work.get("edition_key", [None])[0],
        "title": title,
        "author": ", ".join(authors) if authors else "Unknown",
        "year": year or 0,
        "genre": subjects[:6],
        "subgenres": [],  # we will leave empty or try to infer later
        "major_genres": [],  # optional mapping logic
        "description": description,
        "avg_rating": work.get("rating") or 0.0,
        "num_ratings": work.get("ratings_count") or 0,
        "pacing": "moderate",
        "writing_style": "balanced",
        "moods": [],
        "cover_url": cover_url,
        # embedding to be filled later
    }
    return doc

if __name__ == "__main__":
    args = parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # collect works
    seen = set()
    docs = []
    for subj in subjects:
        print("Fetching subject:", subj)
        works = fetch_subject_works(subj, limit=args.max)
        print(f"  got {len(works)} works")
        for w in works:
            wk = w.get("key")
            if not wk or wk in seen:
                continue
            seen.add(wk)
            details = fetch_work_details(wk)
            if not details:
                continue
            doc = build_doc_from_work(w, details)
            # build small text for embedding: title + description + top subjects
            emb_text = " ".join([doc.get("title",""), doc.get("description",""), " ".join(doc.get("genre",[]))])
            try:
                emb = model.encode(emb_text, convert_to_numpy=True)
                doc["embedding"] = emb.tolist()
            except Exception as e:
                print("Embedding failed for", doc["title"], e)
                doc["embedding"] = None
            docs.append(doc)

    print("Total docs collected:", len(docs))
    # insert into mongo (async)
    asyncio.run(insert_books(docs))
    print("Done.")
