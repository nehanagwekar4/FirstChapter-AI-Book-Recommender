# seed_books.py
"""
Seed script: inserts sample book documents into MongoDB and computes embeddings.
Run: python seed_books.py
Make sure your backend .env points to your local Mongo (MONGO_URL).
"""

import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "novel_recommender_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Full sample dataset (10 books). This expands the small example so your recommender has variety.
BOOKS = [
    {
        "id": "1",
        "title": "The Midnight Library",
        "author": "Matt Haig",
        "year": 2020,
        "major_genres": ["Fantasy","Fiction"],
        "subgenres": ["contemporary", "magical realism"],
        "genre": ["Fantasy", "Fiction", "Philosophy"],
        "description": "Between life and death there is a library, and within that library, the shelves go on forever. Every book provides a chance to try another life you could have lived.",
        "avg_rating": 4.2,
        "num_ratings": 125000,
        "pacing": "moderate",
        "writing_style": "description-heavy",
        "moods": ["introspective", "hopeful"],
        "cover_url": "https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?w=400"
    },
    {
        "id": "2",
        "title": "Project Hail Mary",
        "author": "Andy Weir",
        "year": 2021,
        "major_genres": ["Science Fiction"],
        "subgenres": ["space opera"],
        "genre": ["Science Fiction", "Thriller", "Adventure"],
        "description": "A lone astronaut must save the earth from disaster in this incredible new science-based thriller from the author of The Martian.",
        "avg_rating": 4.6,
        "num_ratings": 98000,
        "pacing": "fast-paced",
        "writing_style": "dialogue-heavy",
        "moods": ["suspenseful", "humorous"],
        "cover_url": "https://images.unsplash.com/photo-1614732414444-096e5f1122d5?w=400"
    },
    {
        "id": "3",
        "title": "The Seven Husbands of Evelyn Hugo",
        "author": "Taylor Jenkins Reid",
        "year": 2017,
        "major_genres": ["Historical Fiction","Romance"],
        "subgenres": ["period drama"],
        "genre": ["Historical Fiction", "Romance", "LGBTQ"],
        "description": "Aging Hollywood icon Evelyn Hugo finally tells the story of her glamorous and scandalous life.",
        "avg_rating": 4.5,
        "num_ratings": 145000,
        "pacing": "moderate",
        "writing_style": "balanced",
        "moods": ["emotional", "dramatic"],
        "cover_url": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400"
    },
    {
        "id": "4",
        "title": "Circe",
        "author": "Madeline Miller",
        "year": 2018,
        "major_genres": ["Fantasy"],
        "subgenres": ["mythology","epic fantasy"],
        "genre": ["Fantasy", "Mythology", "Historical Fiction"],
        "description": "A bold and subversive retelling of the goddess's story, this tale of gods, monsters, and magic manages to be both epic and intimate in its scope.",
        "avg_rating": 4.3,
        "num_ratings": 110000,
        "pacing": "slow-burn",
        "writing_style": "description-heavy",
        "moods": ["atmospheric", "introspective"],
        "cover_url": "https://images.unsplash.com/photo-1532012197267-da84d127e765?w=400"
    },
    {
        "id": "5",
        "title": "The Thursday Murder Club",
        "author": "Richard Osman",
        "year": 2020,
        "major_genres": ["Mystery","Crime"],
        "subgenres": ["cozy mystery"],
        "genre": ["Mystery", "Crime", "Comedy"],
        "description": "In a peaceful retirement village, four unlikely friends meet weekly to investigate unsolved killings. When a local developer is found dead, they find themselves in the middle of their first live case.",
        "avg_rating": 4.3,
        "num_ratings": 87000,
        "pacing": "fast-paced",
        "writing_style": "dialogue-heavy",
        "moods": ["witty", "lighthearted"],
        "cover_url": "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=400"
    },
    {
        "id": "6",
        "title": "Klara and the Sun",
        "author": "Kazuo Ishiguro",
        "year": 2021,
        "major_genres": ["Science Fiction","Literary Fiction"],
        "subgenres": ["dystopian"],
        "genre": ["Science Fiction", "Literary Fiction", "Dystopian"],
        "description": "Klara and the Sun offers a look at our changing world through the eyes of an unforgettable narrator and explores the fundamental question: what does it mean to love?",
        "avg_rating": 3.9,
        "num_ratings": 76000,
        "pacing": "slow-burn",
        "writing_style": "description-heavy",
        "moods": ["melancholic", "philosophical"],
        "cover_url": "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=400"
    },
    {
        "id": "7",
        "title": "Mexican Gothic",
        "author": "Silvia Moreno-Garcia",
        "year": 2020,
        "major_genres": ["Horror","Gothic"],
        "subgenres": ["gothic horror"],
        "genre": ["Horror", "Gothic", "Mystery"],
        "description": "Noemí Taboada heads to High Place, a distant house in the Mexican countryside, after receiving a frantic letter from her newly-wed cousin begging for help.",
        "avg_rating": 4.0,
        "num_ratings": 92000,
        "pacing": "moderate",
        "writing_style": "description-heavy",
        "moods": ["eerie", "atmospheric"],
        "cover_url": "https://images.unsplash.com/photo-1509266272358-7701da638078?w=400"
    },
    {
        "id": "8",
        "title": "The Invisible Life of Addie LaRue",
        "author": "V.E. Schwab",
        "year": 2020,
        "major_genres": ["Fantasy","Romance"],
        "subgenres": ["dark fantasy", "historical fantasy"],
        "genre": ["Fantasy", "Historical Fiction", "Romance"],
        "description": "A life no one will remember. A story you will never forget. A young woman makes a Faustian bargain to live forever and is cursed to be forgotten by everyone she meets.",
        "avg_rating": 4.3,
        "num_ratings": 103000,
        "pacing": "slow-burn",
        "writing_style": "balanced",
        "moods": ["romantic", "melancholic"],
        "cover_url": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=400"
    },
    {
        "id": "9",
        "title": "Anxious People",
        "author": "Fredrik Backman",
        "year": 2020,
        "major_genres": ["Fiction","Contemporary"],
        "subgenres": ["literary comedy"],
        "genre": ["Fiction", "Comedy", "Contemporary"],
        "description": "A bank robbery gone wrong leads to a hostage situation where eight people are trapped in an apartment viewing. As events unfold, we learn about each person's secrets and struggles.",
        "avg_rating": 4.2,
        "num_ratings": 88000,
        "pacing": "moderate",
        "writing_style": "dialogue-heavy",
        "moods": ["heartwarming", "humorous"],
        "cover_url": "https://images.unsplash.com/photo-1495446815901-a7297e633e8d?w=400"
    },
    {
        "id": "10",
        "title": "The Song of Achilles",
        "author": "Madeline Miller",
        "year": 2011,
        "major_genres": ["Fantasy","Mythology"],
        "subgenres": ["epic fantasy", "myth retelling"],
        "genre": ["Fantasy", "Mythology", "Romance"],
        "description": "A tale of gods, kings, immortal fame and the human heart, The Song of Achilles brilliantly reimagines Homer's enduring work.",
        "avg_rating": 4.4,
        "num_ratings": 132000,
        "pacing": "moderate",
        "writing_style": "description-heavy",
        "moods": ["epic", "tragic"],
        "cover_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400"
    }
]

async def main():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    books_coll = db["books"]

    print("Using embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Clearing existing books collection (optional)...")
    await books_coll.delete_many({})

    for b in BOOKS:
        # Compose embedding text: title + description + genres
        text = " ".join([b.get("title",""), b.get("description",""), " ".join(b.get("genre",[]))])
        emb = model.encode(text, convert_to_numpy=True)
        doc = b.copy()
        doc["embedding"] = emb.tolist()
        await books_coll.insert_one(doc)
        print("Inserted:", b["title"])

    print("Seeding complete.")
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
