# server_embedding.py
"""
Embedding-backed recommender:
 - Loads sentence-transformers model at startup
 - Uses stored per-book embeddings (field "embedding") in Mongo
 - Genre-first candidate selection + semantic similarity ranking
 - Penalizes recently recommended items for the same user
 - Dedupe by title and diversify per major genre
"""

import os
import logging
import math
import time
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load env
load_dotenv()

# Config
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_embedding")

# Mongo client
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
books_coll = db["books"]
profiles_coll = db["profiles"]
feedback_coll = db["feedback"]

# FastAPI app
app = FastAPI(title="Embedding Recommender API")
# For development you can set allow_origins=["*"] if CORS blocks you.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model at startup
embedding_model: Optional[SentenceTransformer] = None

@app.on_event("startup")
def startup_event():
    global embedding_model
    try:
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("Loaded embedding model: %s", EMBED_MODEL_NAME)
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        embedding_model = None

# Pydantic models
class FictionBranch(BaseModel):
    major_genres: List[str] = []
    subgenres: Optional[Dict[str, List[str]]] = {}
    mood_preferences: List[str] = []
    story_pace: Optional[str] = None
    writing_style: Optional[str] = None

class NonFictionBranch(BaseModel):
    major_categories: List[str] = []
    mood_preferences: List[str] = []
    writing_style: Optional[str] = None

class RecommendationRequest(BaseModel):
    fiction_or_nonfiction: str
    fiction: Optional[FictionBranch] = None
    nonfiction: Optional[NonFictionBranch] = None
    user_id: Optional[str] = None

class FeedbackModel(BaseModel):
    user_id: Optional[str]
    book_id: str
    liked: bool

# Helpers
def normalize_genres(arr):
    if not arr:
        return []
    return [str(x).strip().lower() for x in arr if str(x).strip()]

def normalize_popularity(num_ratings:int, avg_rating:float, year:int) -> float:
    # returns popularity signal (non-negative). Tunable.
    pop = math.log1p(max(0, int(num_ratings))) * max(float(avg_rating or 0.0), 3.0)
    age = max(0, datetime.now().year - (int(year) if year else 0))
    boost = 1.0
    if age <= 1:
        boost = 1.12
    elif age <= 2:
        boost = 1.06
    return float(pop * boost)

async def fetch_candidates(req: RecommendationRequest, limit:int=3000):
    """
    If user selected major genres/categories, fetch only books that match any of those (strict).
    If this returns empty, fall back to full collection.
    """
    requested = set()
    if req.fiction_or_nonfiction == "fiction" and req.fiction and req.fiction.major_genres:
        requested = set(normalize_genres(req.fiction.major_genres))
    if req.fiction_or_nonfiction == "nonfiction" and req.nonfiction and req.nonfiction.major_categories:
        requested = set(normalize_genres(req.nonfiction.major_categories))

    candidates = []
    if requested:
        # attempt strict fetch by genre field (case-sensitive issues handled by storing normalized tags in DB ideally)
        cursor = books_coll.find({"genre": {"$in": list(requested)}}).limit(limit)
        async for b in cursor:
            candidates.append(b)
        if not candidates:
            # fallback to full set if strict produced nothing
            cursor = books_coll.find({}).limit(limit)
            async for b in cursor:
                candidates.append(b)
    else:
        cursor = books_coll.find({}).limit(limit)
        async for b in cursor:
            candidates.append(b)
    return candidates

def book_embedding_to_np(book_doc):
    e = book_doc.get("embedding")
    if not e:
        return None
    try:
        arr = np.array(e, dtype=float)
        return arr
    except Exception:
        # defensive: try converting element-wise
        try:
            arr = np.array([float(x) for x in e], dtype=float)
            return arr
        except Exception:
            return None

async def get_user_prev_recs(user_id: Optional[str], lookback:int=10):
    if not user_id:
        return []
    prev_ids = []
    cursor = profiles_coll.find({"request.user_id": user_id}).sort("created_at", -1).limit(lookback)
    async for p in cursor:
        prev_ids.extend(p.get("result_ids", []))
    return prev_ids

def diversify_and_dedup_items(scored_items, max_results=12):
    """
    scored_items: list of tuples (book_doc, final_score)
    returns list of tuples (book_doc, final_score) of length <= max_results
    """
    final = []
    seen_titles = set()
    genre_counts = {}
    for b, sc in scored_items:
        title = (b.get("title") or "").strip().lower()
        if title in seen_titles:
            continue
        majors = b.get("genre") or []
        key = (majors[0].lower() if majors else "other")
        if genre_counts.get(key, 0) >= 2:
            continue
        final.append((b, sc))
        seen_titles.add(title)
        genre_counts[key] = genre_counts.get(key, 0) + 1
        if len(final) >= max_results:
            break
    if len(final) < max_results:
        for b, sc in scored_items:
            title = (b.get("title") or "").strip().lower()
            if title in seen_titles:
                continue
            final.append((b, sc))
            seen_titles.add(title)
            if len(final) >= max_results:
                break
    return final

# Main recommendation endpoint
@app.post("/api/recommendations")
async def recommend(req: RecommendationRequest):
    # Validate
    if req.fiction_or_nonfiction not in ("fiction","nonfiction"):
        raise HTTPException(status_code=400, detail="fiction_or_nonfiction must be 'fiction' or 'nonfiction'")

    # 1) Candidate selection (genre-first)
    candidates = await fetch_candidates(req, limit=3000)
    if not candidates:
        raise HTTPException(status_code=500, detail="No candidate books found in DB")

    # 2) Build candidate embeddings array and keep doc list
    candidate_docs = []
    candidate_embs = []
    for b in candidates:
        emb = book_embedding_to_np(b)
        if emb is not None:
            candidate_docs.append(b)
            candidate_embs.append(emb)
    if not candidate_embs:
        raise HTTPException(status_code=500, detail="No book embeddings found. Run seed_embeddings.py")

    candidate_matrix = np.vstack(candidate_embs)  # shape (N, dim)

    # 3) Build query text from questionnaire (description+genres style — do NOT include title)
    qparts = []
    if req.fiction_or_nonfiction == "fiction" and req.fiction:
        f = req.fiction
        if f.mood_preferences: qparts.append(" ".join(f.mood_preferences))
        if f.story_pace: qparts.append(f.story_pace)
        if f.writing_style: qparts.append(f.writing_style or "")
        if f.subgenres:
            for k, arr in (f.subgenres or {}).items():
                if arr: qparts.append(" ".join(arr))
        if f.major_genres: qparts.append(" ".join(f.major_genres))
    elif req.fiction_or_nonfiction == "nonfiction" and req.nonfiction:
        n = req.nonfiction
        if n.mood_preferences: qparts.append(" ".join(n.mood_preferences))
        if n.writing_style: qparts.append(n.writing_style or "")
        if n.major_categories: qparts.append(" ".join(n.major_categories))
    qtext = " ".join([p for p in qparts if p]).strip()
    if not qtext:
        qtext = "books"

    # 4) Compute query embedding and ensure dtype compatibility
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded on server")

    q_emb = embedding_model.encode(qtext, convert_to_numpy=True)
    # q_emb dtype likely float32; candidate_matrix may be float64 -> convert candidate to q_emb dtype
    try:
        if candidate_matrix.dtype != q_emb.dtype:
            candidate_matrix = candidate_matrix.astype(q_emb.dtype)
    except Exception:
        # fallback to float32 for both
        candidate_matrix = candidate_matrix.astype("float32")
        q_emb = q_emb.astype("float32")

    logger.debug("q_emb dtype=%s shape=%s candidate_matrix dtype=%s shape=%s",
                 q_emb.dtype, q_emb.shape, candidate_matrix.dtype, candidate_matrix.shape)

    # 5) compute cosine similarities
    sims_tensor = util.cos_sim(q_emb, candidate_matrix)  # returns tensor-like object
    # pick first row (query) - shape (N,)
    sims = sims_tensor[0]

    # 6) compute popularity and other signals
    pops = [normalize_popularity(b.get("num_ratings",0), b.get("avg_rating",0), b.get("year",0)) for b in candidate_docs]
    max_pop = max(pops) if pops else 1.0

    # 7) previous recommendations for repeat penalty
    prev_ids = await get_user_prev_recs(req.user_id, lookback=10)
    prev_set = set(prev_ids)

    # determine requested genres for genre_score (normalized)
    requested_genres = set()
    if req.fiction_or_nonfiction == "fiction" and req.fiction and req.fiction.major_genres:
        requested_genres = set(normalize_genres(req.fiction.major_genres))
    if req.fiction_or_nonfiction == "nonfiction" and req.nonfiction and req.nonfiction.major_categories:
        requested_genres = set(normalize_genres(req.nonfiction.major_categories))

    def compute_genre_score(book):
        if not requested_genres:
            return 0.0
        bg = set(normalize_genres(book.get("genre") or []))
        if not bg:
            return 0.0
        return float(len(bg & requested_genres) / max(1, len(requested_genres)))

    # 8) Score candidates with hybrid formula
    scored = []
    for idx, b in enumerate(candidate_docs):
        # extract sim value as float
        try:
            sim_val = float(sims[idx].item()) if hasattr(sims[idx], "item") else float(sims[idx])
        except Exception:
            sim_val = float(sims[idx])

        pop_norm = (pops[idx] / max_pop) if max_pop > 0 else 0.0
        feedback_score = float(b.get("feedback_score", 0.0))
        gscore = compute_genre_score(b)

        # repeat penalty
        repeat_penalty = -0.25 if b.get("id") in prev_set else 0.0

        # hybrid weights: genre-first if request had genre selections
        if requested_genres:
            final_score = 0.60 * gscore + 0.25 * sim_val + 0.10 * pop_norm + 0.05 * (feedback_score / (abs(feedback_score) + 5))
        else:
            final_score = 0.55 * sim_val + 0.30 * pop_norm + 0.10 * (feedback_score / (abs(feedback_score) + 5))

        final_score = final_score + repeat_penalty
        scored.append((b, float(final_score), float(gscore), float(sim_val)))

    # 9) sort and diversify/dedupe
    scored.sort(key=lambda x: x[1], reverse=True)
    # convert to (book, score) pairs for diversify function
    scored_pairs = [(b, sc) for (b, sc, g, s) in scored]
    final_selected = diversify_and_dedup_items(scored_pairs, max_results=12)

    # 10) store profile for feedback/repeat tracking
    profile_doc = {
        "request": req.dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_ids": [item[0]["id"] for item in final_selected]
    }
    await profiles_coll.insert_one(profile_doc)

    # 11) prepare response (strip heavy fields)
    out_books = []
    for b, sc in final_selected:
        copy = dict(b)
        copy.pop("_id", None)
        copy.pop("embedding", None)
        # index_pos may not be present; remove for cleaner response
        copy.pop("index_pos", None)
        out_books.append(copy)

    return {"books": out_books, "reasoning": "genre-filter -> embedding similarity -> popularity+feedback (repeat penalty applied)"}

# feedback endpoint
@app.post("/api/feedback")
async def feedback(f: FeedbackModel):
    doc = {
        "user_id": f.user_id,
        "book_id": f.book_id,
        "liked": f.liked,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    await feedback_coll.insert_one(doc)
    # increment book feedback_score for quick signal in ranking
    delta = 1 if f.liked else -1
    await books_coll.update_one({"id": f.book_id}, {"$inc": {"feedback_score": delta}}, upsert=False)
    return {"status": "ok"}

# health endpoint
@app.get("/api/health")
async def health():
    count = await books_coll.count_documents({})
    model_ok = embedding_model is not None
    return {"ok": True, "books_in_db": count, "embedding_model_loaded": model_ok}
