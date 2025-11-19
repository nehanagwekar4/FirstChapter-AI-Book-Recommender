# FirstChapter – AI-Powered Book Recommendation System

FirstChapter is a full-stack, production-ready book recommendation system that combines a modern React frontend, a FastAPI backend, MongoDB storage, and a hybrid recommendation engine powered by NLP embeddings and TF-IDF.  
The application enables users to select reading preferences through a questionnaire and receive tailored book suggestions based on genres, moods, pacing, writing style, and semantic similarity.

This repository contains both the backend (FastAPI) and the frontend (React + TailwindCSS) along with all data ingestion and ML-pipeline scripts.

---

## 1. Features

### Frontend
- Modern UI built with React, TailwindCSS, and CRACO.
- Dynamic multi-step questionnaire for personalized recommendations.
- Recommendations page with book cards, metadata, and feedback buttons.
- Fully responsive layout with subtle gradients and a clean, minimal theme.
- Integration with backend APIs for recommendations and user feedback.

### Backend
- FastAPI server for all API endpoints.
- MongoDB database to store books, embeddings, and user feedback.
- TF-IDF based content filtering engine.
- Sentence-Transformer semantic embeddings for deeper matching.
- Hybrid scoring: TF-IDF + Embeddings + Popularity + Feedback weighting.
- Seed scripts for:
  - Kaggle dataset ingestion
  - OpenLibrary dataset ingestion
  - Embedding generation
  - Sample book population (fallback)

---

## 2. Tech Stack

### Frontend
- React (CRA via CRACO)
- JavaScript (ES6+)
- TailwindCSS
- Axios

### Backend
- FastAPI
- Python 3.10+
- MongoDB (local or Atlas)
- SentenceTransformers (MiniLM)
- NumPy
- Scikit-learn TF-IDF Vectorizer

---

## 3. Project Structure

Novel_Recommender/
│
├── backend/
│ ├── server.py
│ ├── server_embedding.py
│ ├── server_mvp.py
│ ├── seed_books.py
│ ├── seed_embeddings.py
│ ├── seed_openlibrary.py
│ ├── data_ingest_kaggle.py
│ ├── requirements.txt
│ ├── .env.example
│ └── data/
│ └── goodreads_books.csv
│
├── frontend/
│ ├── src/
│ ├── public/
│ ├── package.json
│ ├── .env.example
│ ├── tailwind.config.js
│ ├── craco.config.js
│ └── postcss.js
│
├── ML_DOCUMENTATION.md
├── README.md
└── .gitignore

yaml
Copy code

---

## 4. How to Run the Project

### Backend (FastAPI)

1. Navigate to backend:
cd backend

cpp
Copy code

2. Create and activate virtual environment:
python -m venv venv
venv\Scripts\activate # Windows

markdown
Copy code

3. Install dependencies:
pip install -r requirements.txt

javascript
Copy code

4. Create `.env` from `.env.example`:
MONGO_URL=mongodb://localhost:27017
DB_NAME=novel_recommender_db
EMBED_MODEL_NAME=all-MiniLM-L6-v2

csharp
Copy code

5. Start MongoDB service (ensure it is running on port 27017).

6. Run the backend:
uvicorn server_embedding:app --reload

yaml
Copy code

---

### Frontend (React)

1. Navigate to frontend:
cd frontend

markdown
Copy code

2. Install node modules:
npm install

javascript
Copy code

3. Create `.env` from `.env.example`:
REACT_APP_API_URL=http://localhost:8000

markdown
Copy code

4. Start frontend:
npm start

yaml
Copy code

Frontend runs at:
http://localhost:3000

yaml
Copy code

Backend runs at:
http://localhost:8000

yaml
Copy code

---

## 5. Data and ML Pipeline

- The backend includes multiple ingestion scripts for converting CSV datasets into MongoDB documents.
- Embeddings are generated using:
python seed_embeddings.py

yaml
Copy code
- TF-IDF features are generated automatically at server startup if not present.

Complete details are provided in **ML_DOCUMENTATION.md**.

---

## 6. API Endpoints

### POST /api/recommendations  
Returns a list of recommended books based on questionnaire input.

### POST /api/feedback  
Stores thumbs-up/thumbs-down user feedback.

### GET /api/books  
Lists stored books (limited).

### GET /api/health  
Basic system health check.

More details are documented in the backend section of ML_DOCUMENTATION.md.

---

## 7. Setup Notes

- Do not commit secrets; only commit `.env.example`.
- Large files and virtual environments are excluded via `.gitignore`.
- Book datasets should be placed inside `backend/data/`.

---

## 8. Future Enhancements

- Deploy backend on Render or Railway.
- Deploy frontend on Netlify or Vercel.
- Add collaborative filtering using user profiles.
- Improve evaluation metrics and add A/B testing.

---

## 9. License

This project is provided for academic and portfolio purposes.  
Add an open-source license if required.

---

## 10. Author

Neha Nagvekar  
GitHub: https://github.com/nehanagwekar4