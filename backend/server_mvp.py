# server_mvp.py (updated)
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from scipy import sparse
import json
import pickle

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

TFIDF_PATH = "tfidf.npz"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mvp")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
books_coll = db["books"]
feedback_coll = db["feedback"]
profiles_coll = db["profiles"]

app = FastAPI(title="Novel TF-IDF Recommender")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# load matrix & vectorizer
tfidf_matrix = None
vectorizer = None
if os.path.exists(TFIDF_PATH) and os.path.exists(VECTORIZER_PATH):
    tfidf_matrix = sparse.load_npz(TFIDF_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Loaded TF-IDF matrix (%s) and vectorizer.", tfidf_matrix.shape)
else:
    logger.warning("TF-IDF artifacts missing. Run ingestion script.")

class FictionBranch(BaseModel):
    major_genres: list = []
    subgenres: dict = {}
    mood_preferences: list = []
    story_pace: str = None
    writing_style: str = None

class NonFictionBranch(BaseModel):
    major_categories: list = []
    mood_preferences: list = []
    writing_style: str = None

class RecommendationRequest(BaseModel):
    fiction_or_nonfiction: str
    fiction: FictionBranch = None
    nonfiction: NonFictionBranch = None
    user_id: str = None

class FeedbackModel(BaseModel):
    user_id: str = None
    book_id: str = None
    liked: bool = True

def normalize_popularity(num_ratings: int, avg_rating: float, year:int):
    pop = np.log1p(max(0,num_ratings)) * max(avg_rating, 3.0)
    age = max(0, datetime.now().year - (year or 0))
    recency_boost = 1.0
    if age <= 1: recency_boost = 1.12
    elif age <= 2: recency_boost = 1.06
    return float(pop * recency_boost)

def build_query_text(req: RecommendationRequest):
    parts = []
    if req.fiction_or_nonfiction == "fiction" and req.fiction:
        f = req.fiction
        parts.append("Fiction")
        if f.major_genres: parts.append(" ".join(f.major_genres))
        if f.subgenres:
            for k,v in f.subgenres.items():
                parts.append(" ".join([k] + (v or [])))
        if f.mood_preferences: parts.append(" ".join(f.mood_preferences))
        if f.story_pace: parts.append(f.story_pace)
        if f.writing_style: parts.append(f.writing_style)
    elif req.fiction_or_nonfiction == "nonfiction" and req.nonfiction:
        n = req.nonfiction
        parts.append("Nonfiction")
        if n.major_categories: parts.append(" ".join(n.major_categories))
        if n.mood_preferences: parts.append(" ".join(n.mood_preferences))
        if n.writing_style: parts.append(n.writing_style or "")
    return " ".join([p for p in parts if p]) or "books"

@app.post("/api/recommendations")
async def recommend(req: RecommendationRequest):
    if req.fiction_or_nonfiction not in ("fiction","nonfiction"):
        raise HTTPException(status_code=400, detail="invalid fiction_or_nonfiction")

    # 1) Strict filtering by major genres / categories if provided
    apply_strict = False
    requested_genres = set()
    if req.fiction_or_nonfiction == "fiction" and req.fiction and req.fiction.major_genres:
        apply_strict = True
        requested_genres = set([g.strip().lower() for g in req.fiction.major_genres])
    if req.fiction_or_nonfiction == "nonfiction" and req.nonfiction and req.nonfiction.major_categories:
        apply_strict = True
        requested_genres = set([g.strip().lower() for g in req.nonfiction.major_categories])

    # Fetch candidate docs from Mongo - if strict, use query for genre match
    candidates = []
    if apply_strict:
        # query: any of genre array elements matches requested_genres
        q = {"genre": {"$in": list(requested_genres)}}
        cursor = books_coll.find(q).limit(2000)
        async for b in cursor:
            candidates.append(b)
        if not candidates:
            # fallback: fetch many docs if strict produced nothing
            cursor = books_coll.find({}).limit(2000)
            async for b in cursor:
                candidates.append(b)
    else:
        cursor = books_coll.find({}).limit(2000)
        async for b in cursor:
            candidates.append(b)

    if not candidates:
        raise HTTPException(status_code=500, detail="No books in DB")

    # 2) Build TF-IDF query vector with same vectorizer (exact)
    qtext = build_query_text(req)
    sim_scores = {}
    if tfidf_matrix is not None and vectorizer is not None:
        qvec = vectorizer.transform([qtext])  # EXACT transform
        # Compute similarity only for candidates (use their index_pos)
        # Convert tfidf_matrix to csr (should already be)
        for b in candidates:
            pos = b.get("index_pos")
            if pos is None:
                sim_scores[b["id"]] = 0.0
                continue
            # compute dot product: qvec * tfidf_matrix[pos].T
            candidate_vec = tfidf_matrix.getrow(int(pos))
            score = float(linear_kernel(qvec, candidate_vec)[0,0])
            sim_scores[b["id"]] = float(score)
    else:
        # fallback: zero similarity
        for b in candidates:
            sim_scores[b["id"]] = 0.0

    # 3) Hybrid scoring: content(sim) heavy + popularity + feedback
    pops = [normalize_popularity(b.get("num_ratings",0), b.get("avg_rating",0), b.get("year",0)) for b in candidates]
    max_pop = max(pops) if pops else 1.0

    scored = []
    for b in candidates:
        sim = sim_scores.get(b["id"], 0.0)
        pop_norm = normalize_popularity(b.get("num_ratings",0), b.get("avg_rating",0), b.get("year",0)) / max_pop if max_pop>0 else 0.0
        feedback_score = float(b.get("feedback_score",0.0))
        final = 0.72 * sim + 0.18 * pop_norm + 0.10 * (feedback_score / (abs(feedback_score) + 5))
        scored.append((b, final, sim, pop_norm))

    # 4) Filter by a minimum similarity threshold if user selected a genre (prevents unrelated pop books)
    if apply_strict:
        MIN_SIM = 0.03   # small positive threshold, tuneable
        scored = [s for s in scored if s[2] >= MIN_SIM]
        if not scored:
            # relax threshold if nothing left
            scored = [(b, final, sim, pop) for (b, final, sim, pop) in scored]  # keep as is; fallback below sorts all

    # 5) Sort + diversify (max 2 per genre)
    scored.sort(key=lambda x: x[1], reverse=True)
    final_out = []
    genre_count = {}
    for b, score, sim, pop in scored:
        majors = b.get("genre") or []
        key = (majors[0].lower() if majors else "other")
        if genre_count.get(key, 0) >= 2:
            continue
        final_out.append((b, score))
        genre_count[key] = genre_count.get(key, 0) + 1
        if len(final_out) >= 12:
            break
    if len(final_out) < 12:
        # append additional top scored items if needed
        for b, score, sim, pop in scored:
            if any(x[0]["id"] == b["id"] for x in final_out):
                continue
            final_out.append((b, score))
            if len(final_out) >= 12:
                break

    # write profile and return
    profile = {"request": req.dict(), "created_at": datetime.now(timezone.utc).isoformat(), "result_ids": [b["id"] for b,_ in final_out]}
    await profiles_coll.insert_one(profile)

    out = []
    for b, score in final_out:
        doc = dict(b)
        doc.pop("_id", None)
        doc.pop("index_pos", None)
        out.append(doc)
    return {"books": out, "reasoning": "TF-IDF strict-filter + popularity + feedback"}
    
@app.post("/api/feedback")
async def feedback(f: FeedbackModel):
    await feedback_coll.insert_one({"user_id": f.user_id, "book_id": f.book_id, "liked": f.liked, "ts": datetime.now(timezone.utc).isoformat()})
    delta = 1 if f.liked else -1
    await books_coll.update_one({"id": f.book_id}, {"$inc": {"feedback_score": delta}}, upsert=False)
    return {"status": "ok"}

@app.get("/api/health")
async def health():
    count = await books_coll.count_documents({})
    return {"ok": True, "books_in_db": count, "tfidf_loaded": tfidf_matrix is not None}
