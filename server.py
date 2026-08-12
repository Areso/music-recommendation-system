import json
import joblib
from collections import Counter
import csv
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

def newline_tokenizer(text):
    return [line.strip() for line in text.splitlines() if line.strip()]

app = FastAPI(title="Artist Tag Search API")

# Allow cross-origin requests for AJAX frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables loaded once on startup
print("Loading static artifacts into memory...")
tfidf_matrix = load_npz("tfidf_matrix.npz")
vectorizer = joblib.load("vectorizer.joblib")

vectorizer = joblib.load("vectorizer.joblib")
vectorizer.input = "content"  # Switch from 'filename' to raw text 'content'

with open("artist_ids.json", "r", encoding="utf-8") as f:
    artist_ids = json.load(f)

# Cache feature names (tags)
feature_names = vectorizer.get_feature_names_out()

print(f"Loaded TF-IDF matrix: {tfidf_matrix.shape}")

artists_kv = {}

def load_artists():
    with open("clean/artist_id_mapping.csv", "r", encoding="utf-8") as csvfile:
        artists = csv.reader(csvfile, delimiter=',')
        # Skip the header row
        next(artists, None)
        # Populate the dictionary
        for row in artists:
            # Ensure the row has enough columns to avoid IndexError
            if len(row) > 2:
                artist_id   = row[0]
                artist_name = row[4]
                #print(f"artist_id is {artist_id} and artist_name is {artist_name}")
                artists_kv[artist_id] = artist_name
            #break
load_artists()

@app.get("/api/autocomplete")
def autocomplete(q: str = Query(..., min_length=1), limit: int = 10):
    """Returns matching tag suggestions for AJAX autocomplete."""
    q_lower = q.lower().strip()

    # Prefix match first, then substring match
    prefix_matches = [tag for tag in feature_names if tag.startswith(q_lower)]
    substring_matches = [
        tag
        for tag in feature_names
        if q_lower in tag and not tag.startswith(q_lower)
    ]

    results = (prefix_matches + substring_matches)[:limit]
    return {"query": q, "suggestions": results}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), top_k: int = 30):
    """Computes cosine similarity between input tag query and all artists."""
    # Transform query string into TF-IDF vector space
    query_vec = vectorizer.transform([q])

    # Compute cosine similarity against all artist rows
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top_k indices sorted by score descending
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score <= 0:
            break
        artist_name = artists_kv[artist_ids[idx]]
        results.append({"artist_id": artist_ids[idx], "artist_name": artist_name, "score": round(score, 4)})

    return {"query": q, "results": results}


@app.get("/api/fetch_tags_of_the_artist_old")
def fetch_artist_tags_old(q: int = Query(...)):
    results = []
    print(q)
    file_path = f"biblioteca/{q}.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as artist_tags:
            for line in artist_tags:
                results.append(line.strip())
    except FileNotFoundError:
        return {"query": q, "results": [], "error": "Artist tags not found"}
    return {"query": q, "results": results}

@app.get("/api/fetch_tags_of_the_artist")
def fetch_artist_tags(q: int = Query(...)):
  raw_results = []
  print(q)
  file_path = f"biblioteca/{q}.txt"
  try:
    with open(file_path, "r", encoding="utf-8") as artist_tags:
      for line in artist_tags:
        cleaned_line = line.strip()
        if cleaned_line:  # Skip empty lines
          raw_results.append(cleaned_line)
  except FileNotFoundError:
    return {"query": q, "results": [], "error": "Artist tags not found"}

  # Count occurrences and remove duplicates while preserving order
  tag_counts = Counter(raw_results)
  deduped_results = []
  seen = set()

  for tag in raw_results:
    if tag not in seen:
      seen.add(tag)
      count = tag_counts[tag]
      if count > 1:
        deduped_results.append(f"{tag} [{count}]")
      else:
        deduped_results.append(tag)

  return {"query": q, "results": deduped_results}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)