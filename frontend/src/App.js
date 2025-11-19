// src/App.js
import React, { useEffect, useState } from "react";
import "./App.css";
import axios from "axios";
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Button = ({ children, className = "", ...rest }) => (
  <button className={`btn ${className}`} {...rest}>{children}</button>
);
const Tag = ({ children }) => <span className="tag">{children}</span>;

function Questionnaire() {
  const navigate = useNavigate();
  const [branch, setBranch] = useState(null);
  const [fictionState, setFictionState] = useState({
    major_genres: [],
    subgenres: {},
    mood_preferences: [],
    story_pace: "",
    writing_style: ""
  });
  const [nonfictionState, setNonfictionState] = useState({
    major_categories: [],
    mood_preferences: [],
    writing_style: ""
  });
  const [loading, setLoading] = useState(false);

  const defaultFictionGenres = {
    "Fantasy": ["Epic fantasy", "Dark fantasy", "Urban fantasy", "High fantasy"],
    "Science Fiction": ["Hard sci-fi", "Space opera", "Military sci-fi"],
    "Mystery": ["Cozy mystery", "Detective fiction", "Police procedural"],
    "Thriller": ["Psychological thriller", "Legal thriller", "Spy thriller"],
    "Romance": ["Contemporary romance", "Historical romance", "Romantic comedy"],
    "Historical Fiction": ["Period drama", "Historical romance"],
    "Horror": ["Gothic", "Weird fiction", "Paranormal"],
    "Action-Adventure": ["Swashbuckler", "Pulp adventure"]
  };

  const moodOptions = ["introspective","hopeful","suspenseful","humorous","emotional","dramatic","atmospheric","witty","lighthearted","melancholic","philosophical","eerie","romantic","heartwarming","epic","tragic"];
  const nonfictionCategories = ["Biography","Memoir","Science","History","Self-help","Journalism","Travel","True Crime","Business","Academic","Culinary"];

  const toggleArray = (arr, setter, val) => {
    setter(prev => {
      const exists = prev.includes(val);
      return exists ? prev.filter(x => x !== val) : [...prev, val];
    });
  };

  const handleFictionMajorToggle = (genre) => {
    setFictionState(prev => {
      const majors = prev.major_genres.includes(genre) ? prev.major_genres.filter(g => g !== genre) : [...prev.major_genres, genre];
      const sub = { ...prev.subgenres };
      if (!majors.includes(genre)) delete sub[genre];
      return { ...prev, major_genres: majors, subgenres: sub };
    });
  };

  const handleSubgenreToggle = (major, sub) => {
    setFictionState(prev => {
      const current = prev.subgenres[major] || [];
      const exists = current.includes(sub);
      const updated = exists ? current.filter(s => s !== sub) : [...current, sub];
      return { ...prev, subgenres: { ...prev.subgenres, [major]: updated } };
    });
  };

  // NEW: toggle behavior for story pace (click selected to unselect)
  const togglePace = (p) => {
    setFictionState(prev => ({ ...prev, story_pace: prev.story_pace === p ? "" : p }));
  };
  // NEW: toggle for writing style
  const toggleWritingStyleFiction = (s) => {
    setFictionState(prev => ({ ...prev, writing_style: prev.writing_style === s ? "" : s }));
  };
  const toggleWritingStyleNonfiction = (s) => {
    setNonfictionState(prev => ({ ...prev, writing_style: prev.writing_style === s ? "" : s }));
  };

  function handleSubmit(e) {
    e.preventDefault();
    const user_id = "anon_" + uuidv4();
    let payload = { fiction_or_nonfiction: branch, user_id };
    if (branch === "fiction") {
      payload.fiction = {
        major_genres: fictionState.major_genres,
        subgenres: fictionState.subgenres,
        mood_preferences: fictionState.mood_preferences,
        story_pace: fictionState.story_pace,
        writing_style: fictionState.writing_style
      };
    } else {
      payload.nonfiction = {
        major_categories: nonfictionState.major_categories,
        mood_preferences: nonfictionState.mood_preferences,
        writing_style: nonfictionState.writing_style
      };
    }

    if (!branch) return alert("Pick Fiction or Non-fiction first.");
    if (branch === "fiction" && payload.fiction.major_genres.length === 0) return alert("Choose at least one fiction genre.");
    if (branch === "nonfiction" && payload.nonfiction.major_categories.length === 0) return alert("Choose at least one nonfiction category.");
    if ((branch === "fiction" && payload.fiction.mood_preferences.length === 0) || (branch === "nonfiction" && payload.nonfiction.mood_preferences.length === 0)) return alert("Pick at least one mood.");

    setLoading(true);
    axios.post(`${API}/recommendations`, payload, { timeout: 20000 })
      .then(resp => {
        setLoading(false);
        navigate("/recommendations", { state: { data: resp.data, user_id: user_id } });
      })
      .catch(err => {
        setLoading(false);
        console.error(err);
        alert("Failed to fetch recommendations. Check backend & CORS (see console).");
      });
  }

  return (
    <div className="page-wrap">
      {/* Large centered site title - separated from the card below */}
      <header className="site-header">
        <h1 className="site-title">First Chapter</h1>
        <p className="subtitle">Discover your perfect next read with AI-powered recommendations</p>
      </header>

      {/* Separate question card */}
      <main className="question-card">
        <form onSubmit={handleSubmit} className="question-form">
          <div className="first-question">
            <div className="q-number">1)</div>
            <div className="q-block">
              <div className="q-label">Fiction or Non-fiction?</div>
              <div className="btn-row">
                <button type="button" className={`pill ${branch==="fiction" ? "active" : ""}`} onClick={() => setBranch("fiction")}>Fiction</button>
                <button type="button" className={`pill ${branch==="nonfiction" ? "active" : ""}`} onClick={() => setBranch("nonfiction")}>Non-fiction</button>
              </div>
            </div>
          </div>

          {branch === "fiction" && (
            <>
              <div className="section">
                <div className="q-label">2) Choose fiction genres (choose one or more)</div>
                <div className="grid-genres">
                  {Object.keys(defaultFictionGenres).map(g => (
                    <label key={g} className={`genre-card ${fictionState.major_genres.includes(g) ? "selected" : ""}`}>
                      <input type="checkbox" checked={fictionState.major_genres.includes(g)} onChange={() => handleFictionMajorToggle(g)} />
                      <span>{g}</span>
                    </label>
                  ))}
                </div>
              </div>

              {fictionState.major_genres.map(major => (
                <div key={major} className="section">
                  <div className="q-label">Subgenres for {major} (optional)</div>
                  <div className="grid-genres">
                    {(defaultFictionGenres[major] || []).map(sub => (
                      <label key={sub} className={`genre-card ${((fictionState.subgenres[major]||[]).includes(sub)) ? "selected" : ""}`}>
                        <input type="checkbox" checked={(fictionState.subgenres[major]||[]).includes(sub)} onChange={() => handleSubgenreToggle(major, sub)} />
                        <span>{sub}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}

              <div className="section">
                <div className="q-label">3) What are you in the mood for right now? (choose one or more)</div>
                <div className="grid-genres">
                  {moodOptions.map(m => (
                    <label key={m} className={`genre-card ${fictionState.mood_preferences.includes(m) ? "selected" : ""}`}>
                      <input type="checkbox" checked={fictionState.mood_preferences.includes(m)} onChange={() => {
                        setFictionState(prev => {
                          const exists = prev.mood_preferences.includes(m);
                          return { ...prev, mood_preferences: exists ? prev.mood_preferences.filter(x=>x!==m) : [...prev.mood_preferences, m] };
                        });
                      }} />
                      <span>{m}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="section">
                <div className="q-label">4) Story pace</div>
                <div className="btn-row">
                  {/* USE togglePace to allow unselect */}
                  {["fast-paced","moderate","slow-burn"].map(p => (
                    <button type="button" key={p} className={`pace-pill ${fictionState.story_pace===p ? "active" : ""}`} onClick={() => togglePace(p)}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              <div className="section">
                <div className="q-label">5) Optional: Writing style</div>
                <div className="btn-row">
                  {["dialogue-heavy","description-heavy","balanced"].map(s => (
                    <button type="button" key={s} className={`pace-pill ${fictionState.writing_style===s ? "active" : ""}`} onClick={() => toggleWritingStyleFiction(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {branch === "nonfiction" && (
            <>
              <div className="section">
                <div className="q-label">2) Pick non-fiction categories (choose one or more)</div>
                <div className="grid-genres">
                  {nonfictionCategories.map(c => (
                    <label key={c} className={`genre-card ${nonfictionState.major_categories.includes(c) ? "selected" : ""}`}>
                      <input type="checkbox" checked={nonfictionState.major_categories.includes(c)} onChange={() => {
                        setNonfictionState(prev => {
                          const exists = prev.major_categories.includes(c);
                          return { ...prev, major_categories: exists ? prev.major_categories.filter(x=>x!==c) : [...prev.major_categories, c] };
                        });
                      }} />
                      <span>{c}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="section">
                <div className="q-label">3) What are you in the mood for right now? (choose one or more)</div>
                <div className="grid-genres">
                  {moodOptions.map(m => (
                    <label key={m} className={`genre-card ${nonfictionState.mood_preferences.includes(m) ? "selected" : ""}`}>
                      <input type="checkbox" checked={nonfictionState.mood_preferences.includes(m)} onChange={() => {
                        setNonfictionState(prev => {
                          const exists = prev.mood_preferences.includes(m);
                          return { ...prev, mood_preferences: exists ? prev.mood_preferences.filter(x=>x!==m) : [...prev.mood_preferences, m] };
                        });
                      }} />
                      <span>{m}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="section">
                <div className="q-label">4) Optional: Writing style</div>
                <div className="btn-row">
                  {["dialogue-heavy","description-heavy","balanced"].map(s => (
                    <button type="button" key={s} className={`pace-pill ${nonfictionState.writing_style===s ? "active" : ""}`} onClick={() => toggleWritingStyleNonfiction(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="form-actions">
            <Button type="submit" disabled={loading}>{loading ? "Searching..." : "Get My Recommendations"}</Button>
          </div>
        </form>
      </main>
    </div>
  );
}

/* Recommendations view unchanged except minor class adjustments */
function Recommendations() {
  const loc = useLocation();
  const navigate = useNavigate();
  const data = loc.state?.data;
  const user_id = loc.state?.user_id || "anon_" + uuidv4();
  const [books, setBooks] = useState(data?.books || []);
  const [sending, setSending] = useState({});

  useEffect(() => {
    if (!data) navigate("/");
  }, [data, navigate]);

  const sendFeedback = (book_id, liked) => {
    setSending(prev => ({ ...prev, [book_id]: true }));
    if (!liked) {
      setBooks(prev => prev.filter(b => b.id !== book_id));
    } else {
      setBooks(prev => prev.map(b => b.id === book_id ? { ...b, _liked: true } : b));
    }

    axios.post(`${API}/feedback`, { user_id, book_id, liked })
      .then(() => {
        setSending(prev => ({ ...prev, [book_id]: false }));
      })
      .catch(err => {
        console.error("Feedback send failed", err);
        setSending(prev => ({ ...prev, [book_id]: false }));
        alert("Feedback failed to send — please check the backend.");
      });
  };

  if (!data) return null;

  return (
    <div className="recs-page">
      <div className="recs-header">
        <h1>Your Recommendations</h1>
        <p className="muted">{data.reasoning}</p>
        <div className="recs-actions">
          <Button onClick={() => navigate("/")}>Get new recommendations</Button>
        </div>
      </div>

      <div className="cards-grid">
        {books.map(book => (
          <div className="card" key={book.id}>
            <div className="cover">
              <img src={book.cover_url} alt={book.title} onError={(e)=>{e.target.style.display='none'}} />
            </div>
            <div className="card-body">
              <h3 className="title">{book.title}</h3>
              <div className="meta">by {book.author} ({book.year})</div>
              <div className="tags">
                {(book.genre || []).slice(0,3).map(g => <Tag key={g}>{g}</Tag>)}
              </div>
              <p className="desc">{book.description ? (book.description.length > 140 ? book.description.slice(0,140)+"..." : book.description) : ""}</p>
            </div>

            <div className="card-footer">
              <div className="feedback">
                <button className={`feedback-btn ${book._liked ? "liked" : ""}`} onClick={()=>sendFeedback(book.id, true)} disabled={sending[book.id]}>
                  👍
                </button>
                <button className="feedback-btn" onClick={()=>sendFeedback(book.id, false)} disabled={sending[book.id]}>
                  👎
                </button>
              </div>
              <div className="rating">⭐ {book.avg_rating}/5 · {Math.round((book.num_ratings||0)/1000)}k</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* App wrapper */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Questionnaire />} />
        <Route path="/recommendations" element={<Recommendations />} />
      </Routes>
    </BrowserRouter>
  );
}
