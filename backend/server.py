# server.py (patched - stronger content filtering + richer prompt + tuned weights + logging)
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import numpy as np

from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(__file__)
load_dotenv(os.path.join(ROOT, ".env"))

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
books_coll = db["books"]
profiles_coll = db["profiles"]
feedback_coll = db["feedback"]

app = FastAPI(title="Novel Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
embedding_model = None

# in-memory
emb_matrix = None
book_index = []
book_id_to_pos = {}

@app.on_event("startup")
async def startup_event():
    global embedding_model
    try:
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        embedding_model = None

    try:
        await build_index()
    except Exception as e:
        logger.warning("Index build failed at startup: %s", e)

# ----- Pydantic models -----
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

# ----- Helpers -----
def json_safe(doc):
    d = dict(doc)
    d.pop("embedding", None)
    if "_id" in d:
        try:
            d["_id"] = str(d["_id"])
        except:
            d["_id"] = None
    return jsonable_encoder(d)

def normalize_popularity(num_ratings: int, avg_rating: float, year: int):
    pop = np.log1p(max(0, num_ratings)) * max(avg_rating, 3.0)
    age = max(0, datetime.now().year - (year or 0))
    recency_boost = 1.0
    if age <= 1:
        recency_boost = 1.12
    elif age <= 2:
        recency_boost = 1.06
    return float(pop * recency_boost)

def normalize_text_list(lst):
    return [str(x).strip().lower() for x in (lst or [])]

def book_matches_request(book: dict, req: RecommendationRequest):
    """
    Return True if the book matches at least one major requested genre/category or subgenre.
    This ensures the returned list responds to the user's initial selections.
    """
    # normalize book tags
    book_genres = set(normalize_text_list(book.get("major_genres") or book.get("genre") or []))
    book_subs = set(normalize_text_list(book.get("subgenres") or []))
    book_subjects = set(normalize_text_list(book.get("genre") or book.get("subjects") or []))

    if req.fiction_or_nonfiction == "fiction" and req.fiction:
        fg = req.fiction
        requested = set(normalize_text_list(fg.major_genres))
        requested_subs = set()
        if fg.subgenres:
            for v in fg.subgenres.values():
                requested_subs.update(normalize_text_list(v))
        # if user requested any major genre and this book has at least one -> True
        if requested and (requested & (book_genres | book_subjects)):
            return True
        # if subgenre selections exist, require overlap with book_subs or book_subjects
        if requested_subs and (requested_subs & (book_subs | book_subjects)):
            return True
        # else, no match
        return False

    elif req.fiction_or_nonfiction == "nonfiction" and req.nonfiction:
        nf = req.nonfiction
        requested = set(normalize_text_list(nf.major_categories))
        if requested and (requested & (book_genres | book_subjects)):
            return True
        return False

    # default: no restriction
    return False

# ----- Index utilities -----
async def build_index():
    global emb_matrix, book_index, book_id_to_pos
    book_index = []
    embeddings = []
    cursor = books_coll.find({"embedding": {"$ne": None}})
    async for d in cursor:
        book_index.append(d)
        emb = d.get("embedding")
        if emb:
            embeddings.append(np.array(emb, dtype=float))
        else:
            embeddings.append(np.zeros((384,), dtype=float))
    if embeddings:
        emb_matrix = np.vstack(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
    else:
        emb_matrix = None
    book_id_to_pos = {}
    for i, b in enumerate(book_index):
        bid = str(b.get("id") or b.get("_id"))
        book_id_to_pos[bid] = i
    logger.info(f"Built index with {len(book_index)} items.")

def semantic_top_k(req_embedding, top_k=200):
    if emb_matrix is None or req_embedding is None:
        return []
    v = req_embedding.astype(float)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return []
    v = v / v_norm
    scores = emb_matrix.dot(v)
    if len(scores) <= top_k:
        idxs = np.argsort(-scores)
    else:
        idxs = np.argpartition(-scores, top_k)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]
    return [(int(i), float(scores[i])) for i in idxs]

# ----- Recommendation endpoint (improved) -----
@app.post("/api/recommendations")
async def recommend(req: RecommendationRequest):
    if req.fiction_or_nonfiction not in ("fiction", "nonfiction"):
        raise HTTPException(status_code=400, detail="fiction_or_nonfiction must be 'fiction' or 'nonfiction'")

    # Build richer prompt text (explicitly include genres, subgenres, moods, pacing)
    prompt_parts = []
    if req.fiction_or_nonfiction == "fiction" and req.fiction:
        f = req.fiction
        prompt_parts.append("Fiction")
        if f.major_genres:
            prompt_parts.append("Genres: " + ", ".join(f.major_genres))
        if f.subgenres:
            for k, arr in (f.subgenres or {}).items():
                if arr:
                    prompt_parts.append(f"{k}: " + ", ".join(arr))
        if f.mood_preferences:
            prompt_parts.append("Mood: " + ", ".join(f.mood_preferences))
        if f.story_pace:
            prompt_parts.append("Pacing: " + f.story_pace)
        if f.writing_style:
            prompt_parts.append("Writing style: " + f.writing_style)
    elif req.fiction_or_nonfiction == "nonfiction" and req.nonfiction:
        n = req.nonfiction
        prompt_parts.append("Nonfiction")
        if n.major_categories:
            prompt_parts.append("Categories: " + ", ".join(n.major_categories))
        if n.mood_preferences:
            prompt_parts.append("Mood: " + ", ".join(n.mood_preferences))
        if n.writing_style:
            prompt_parts.append("Writing style: " + n.writing_style)

    prompt_text = ". ".join(prompt_parts).strip()[:1600]
    if not prompt_text:
        prompt_text = "books"

    # compute request embedding (if available)
    req_emb = None
    if embedding_model and prompt_text:
        try:
            req_emb = embedding_model.encode(prompt_text, convert_to_numpy=True)
        except Exception as e:
            logger.warning("Failed to encode request: %s", e)
            req_emb = None

    # Candidate selection:
    # 1) use semantic_top_k to get a broad candidate set
    semantic_candidates = semantic_top_k(req_emb, top_k=400) if req_emb is not None else []

    # Map to book docs with semscore
    candidates = []
    if semantic_candidates:
        for pos, semscore in semantic_candidates:
            book = book_index[pos]
            candidates.append((book, semscore))
    else:
        cursor = books_coll.find({}).limit(400)
        async for b in cursor:
            candidates.append((b, 0.0))

    # 2) filter candidates by strict matching of user's requested genres/categories if the request provided any
    # This prevents the same set of popular books from appearing irrespective of the questionnaire.
    strict_filter_applied = False
    filtered_candidates = []
    try:
        # only apply strict filter if user explicitly selected genres/categories
        apply_strict = False
        if req.fiction_or_nonfiction == "fiction" and req.fiction and req.fiction.major_genres:
            apply_strict = True
        if req.fiction_or_nonfiction == "nonfiction" and req.nonfiction and req.nonfiction.major_categories:
            apply_strict = True

        if apply_strict:
            strict_filter_applied = True
            for b, s in candidates:
                if book_matches_request(b, req):
                    filtered_candidates.append((b, s))
            # if filtered is empty, we will fallback to the semantic candidate list (no strict match found)
            if not filtered_candidates:
                logger.info("Strict filter found no matches; falling back to semantic candidates.")
                filtered_candidates = candidates.copy()
        else:
            filtered_candidates = candidates.copy()
    except Exception as e:
        logger.warning("Filtering error: %s", e)
        filtered_candidates = candidates.copy()

    # Scoring: hybrid mixture tuned to favor content matches
    scored = []
    pop_values = [normalize_popularity(b.get("num_ratings", 0), b.get("avg_rating", 0), b.get("year", 0)) for b, _ in filtered_candidates]
    max_pop = max(pop_values) if pop_values else 1.0

    for b, semscore in filtered_candidates:
        content_s = compute_content_score(b, req) if 'compute_content_score' in globals() else 0.0
        embed_s = 0.0
        if semscore is not None:
            # semscore is cosine in [-1,1], convert to 0..1
            embed_s = (semscore + 1.0) / 2.0
        pop = normalize_popularity(b.get("num_ratings", 0), b.get("avg_rating", 0), b.get("year", 0))
        pop_norm = pop / max_pop if max_pop > 0 else 0.0
        feedback_score = float(b.get("feedback_score", 0.0))

        # Tuned weights:
        # content (explicit genre/mood/pacing) gets most weight,
        # embeddings help find semantically related items,
        # popularity and feedback add small nudges.
        final = (0.60 * content_s) + (0.25 * embed_s) + (0.10 * pop_norm) + (0.05 * (feedback_score / (abs(feedback_score) + 5)))
        final += float(np.random.normal(0, 1e-5))
        scored.append({"book": b, "final_score": float(final), "content_score": float(content_s), "embed_score": float(embed_s), "pop_score": float(pop_norm), "feedback_score": float(feedback_score)})

    # Sort and diversify/deduplicate
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # Diversify: limit two per main genre where possible
    final_items = []
    genre_count = {}
    for item in scored:
        majors = item["book"].get("major_genres") or item["book"].get("genre") or ["other"]
        key = (majors[0] or "other").lower()
        if genre_count.get(key, 0) >= 2:
            continue
        final_items.append(item)
        genre_count[key] = genre_count.get(key, 0) + 1
        if len(final_items) >= 12:
            break
    if len(final_items) < 12:
        for item in scored:
            if item not in final_items:
                final_items.append(item)
            if len(final_items) >= 12:
                break

    # Save profile
    profile_doc = {
        "request": req.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_ids": [str(item["book"].get("id") or item["book"].get("_id")) for item in final_items],
    }
    await profiles_coll.insert_one(profile_doc)

    # Prepare output
    out_books = []
    for item in final_items:
        b = item["book"].copy()
        b.pop("embedding", None)
        out_books.append(json_safe(b))

    # Log top reasons for debugging
    logger.info("Recommendation request: strict_filter_applied=%s prompt='%s' returned=%d", strict_filter_applied, prompt_text[:200], len(out_books))

    return {"books": out_books, "reasoning": "Hybrid: content-heavy + semantics + popularity + feedback", "profile_id": str(profile_doc.get("_id") or profile_doc.get("created_at"))}

# feedback endpoint unchanged
@app.post("/api/feedback")
async def feedback(f: FeedbackModel):
    doc = {
        "user_id": f.user_id,
        "book_id": f.book_id,
        "liked": f.liked,
        "ts": datetime.now(timezone.utc).isoformat()
    }
    await feedback_coll.insert_one(doc)
    delta = 1 if f.liked else -1
    await books_coll.update_one({"id": f.book_id}, {"$inc": {"feedback_score": delta}}, upsert=False)
    return {"status": "ok"}

@app.post("/api/rebuild_index")
async def rebuild_index():
    try:
        await build_index()
        return {"status": "ok", "items": len(book_index)}
    except Exception as e:
        logger.exception("Rebuild failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/books")
async def list_books(limit: int = 50):
    docs = []
    cursor = books_coll.find({}).limit(limit)
    async for d in cursor:
        docs.append(json_safe(d))
    return docs

@app.get("/api/health")
async def health():
    model_ok = embedding_model is not None
    count = await books_coll.count_documents({})
    return {"ok": True, "embedding_model_loaded": model_ok, "books_in_db": count, "index_built": len(book_index)}

# helper compute_content_score kept out-of-line for clarity; reuse your previous implementation if present
def compute_content_score(book: dict, req: RecommendationRequest):
    # conservative scoring: matches for genres, subgenres, moods, pacing, writing style
    score = 0.0
    weight = 0.0
    if req.fiction_or_nonfiction == "fiction" and req.fiction:
        fg = req.fiction
        requested_genres = set([g.lower() for g in fg.major_genres])
        book_genres = set([g.lower() for g in (book.get("major_genres") or [])] + [g.lower() for g in (book.get("genre") or [])])
        if requested_genres:
            gscore = len(requested_genres & book_genres) / len(requested_genres)
            score += 0.30 * gscore; weight += 0.30
        chosen_subs = set()
        if fg.subgenres:
            for arr in fg.subgenres.values():
                chosen_subs.update([s.lower() for s in arr])
        if chosen_subs:
            subscore = len(chosen_subs & set([s.lower() for s in (book.get("subgenres") or [])])) / (len(chosen_subs) or 1)
            score += 0.20 * subscore; weight += 0.20
        moods_req = set([m.lower() for m in fg.mood_preferences])
        if moods_req:
            mood_common = len(moods_req & set([m.lower() for m in (book.get("moods") or [])])) / (len(moods_req) or 1)
            score += 0.30 * mood_common; weight += 0.30
        if fg.story_pace:
            score += 0.10 * (1.0 if book.get("pacing") == fg.story_pace else 0.0); weight += 0.10
        if fg.writing_style:
            score += 0.10 * (1.0 if book.get("writing_style") == fg.writing_style else 0.0); weight += 0.10
    elif req.fiction_or_nonfiction == "nonfiction" and req.nonfiction:
        nf = req.nonfiction
        requested = set([c.lower() for c in nf.major_categories])
        book_genres = set([g.lower() for g in (book.get("major_genres") or [])] + [g.lower() for g in (book.get("genre") or [])])
        if requested:
            gscore = len(requested & book_genres) / len(requested)
            score += 0.55 * gscore; weight += 0.55
        moods_req = set([m.lower() for m in nf.mood_preferences])
        if moods_req:
            mood_common = len(moods_req & set([m.lower() for m in (book.get("moods") or [])])) / (len(moods_req) or 1)
            score += 0.30 * mood_common; weight += 0.30
        if nf.writing_style:
            score += 0.15 * (1.0 if book.get("writing_style") == nf.writing_style else 0.0); weight += 0.15
    if weight == 0:
        return 0.0
    return float(score / weight)
