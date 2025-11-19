# Machine Learning Documentation – FirstChapter Book Recommender

This document describes the complete ML pipeline powering the FirstChapter recommendation engine. It explains dataset formats, preprocessing, model logic, embeddings, TF-IDF, hybrid scoring, and feedback incorporation.

---

## 1. Overview of the ML Approach

FirstChapter uses a **hybrid recommendation system** that combines three approaches:

1. Content-based filtering (TF-IDF keyword vectors)
2. Semantic similarity (Sentence-Transformer embeddings)
3. Popularity and user feedback weighting

This hybrid approach ensures recommendations are relevant even when the user selects abstract preferences such as moods, pacing, or writing style.

---

## 2. Dataset Sources

Two dataset ingestion scripts are provided:

### a. Kaggle dataset (Goodreads)
File format example:
title,authors,description,genres,avg_rating,num_ratings

shell
Copy code

### b. OpenLibrary API
Used via:
seed_openlibrary.py

shell
Copy code

### c. Local sample dataset
Used for initial testing via:
seed_books.py

cpp
Copy code

---

## 3. Data Normalization

Each book stored in MongoDB follows this unified schema:

```json
{
  "id": "string",
  "title": "string",
  "author": "string",
  "year": 2020,
  "description": "string",
  "genre": ["Fantasy", "Romance"],
  "subgenres": ["dark fantasy"],
  "moods": ["atmospheric", "romantic"],
  "pacing": "moderate",
  "writing_style": "dialogue-heavy",
  "avg_rating": 4.3,
  "num_ratings": 103000,
  "cover_url": "...",
  "embedding": [768-dimensional vector],
  "feedback_score": 0
}
Missing metadata is inferred or defaulted when possible.

4. TF-IDF Feature Generation
Script: data_ingest_kaggle.py and automatic generation inside servers.

Input text:

nginx
Copy code
title + description + genres + subgenres
Vectorizer:

scss
Copy code
TfidfVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1,2)
)
Vector dimensions: typically between 10k–30k features, depending on vocabulary.

TF-IDF matrix is saved as:

tfidf.npz (compressed sparse matrix)

tfidf_vectorizer.pkl

tfidf_vocab.json

5. Sentence-Transformer Embeddings
Model used:

css
Copy code
all-MiniLM-L6-v2
Embedding dimension: 384

Text used for embeddings:

nginx
Copy code
description + genres
Stored as a list of floats directly inside MongoDB.

Generated via:

nginx
Copy code
python seed_embeddings.py
6. Hybrid Scoring Algorithm
For a given user request, the scoring function computes:

1. TF-IDF similarity (S_tfidf)
Cosine similarity between:

scss
Copy code
tfidf(query_text) vs tfidf(book_text)
2. Embedding similarity (S_emb)
Cosine similarity between:

scss
Copy code
MiniLM(query_embedding) vs book_embedding
3. Popularity score (S_pop)
scss
Copy code
log(1 + num_ratings) * avg_rating
Normalized across all books.

4. Feedback adjustment (S_fb)
Weighted adjustment:

ini
Copy code
positive_feedback = +0.05
negative_feedback = -0.05
Final Hybrid Score
makefile
Copy code
score = 0.40 * S_tfidf
      + 0.40 * S_emb
      + 0.15 * S_pop
      + 0.05 * S_fb
7. Query Construction from Questionnaire
User input is transformed into a structured prompt:

Example:

yaml
Copy code
Fiction. Genre: Romance. Mood: heartwarming, emotional.
Pacing: moderate. Writing style: dialogue-heavy.
This text is used for:

TF-IDF query vector

Semantic embedding query

This ensures that recommendations depend on user intent, not on title-word overlap.

8. Evaluation
Manual Evaluation Criteria
Category relevance

Mood alignment

Pacing match

Writing style similarity

Semantic compatibility

Model Strengths
Handles abstract user preferences effectively.

Performs well on unseen titles due to embeddings.

Scales with additional metadata.

Limitations
Depends on quality and consistency of metadata.

TF-IDF tends to favor longer descriptions.

Embeddings require GPU for large-scale datasets.

9. Future ML Enhancements
Integrate collaborative filtering using user profiles.

Add LightFM or matrix factorization models.

Implement multi-modal embeddings using book covers.

Add evaluation metrics such as NDCG and MAP.

Improve metadata extraction via NLP (NER, entity linking).

10. Reproducibility
Run the following to reproduce ML assets:

css
Copy code
python data_ingest_kaggle.py --csv backend/data/goodreads_books.csv --max 5000
python seed_embeddings.py
Ensure MongoDB is running and backend/server_embedding.py loads the TF-IDF model on startup.

