# inside data_ingest_kaggle.py — replace build_tfidf(...) with this

def build_tfidf(rows, max_features=50000, genre_boost=6):
    """
    Create TF-IDF matrix. To make genres/tags influential, we repeat each
    genre token `genre_boost` times inside the combined text.
    """
    texts = []
    for r in rows:
        genres = r.get("genres") or r.get("genre") or []
        # build boosted genre tokens like: "genre_fantasy genre_fantasy ..."
        boosted_genre_tokens = []
        for g in genres:
            token = f"genre_{str(g).strip().lower().replace(' ', '_')}"
            boosted_genre_tokens.extend([token] * genre_boost)
        combined_parts = [
            r.get("title", ""),
            r.get("author", ""),
            " ".join(boosted_genre_tokens),
            " ".join(genres),
            r.get("description", "")
        ]
        combined = " ".join([p for p in combined_parts if p])
        texts.append(combined)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix
