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

# vectorizer.joblib was pickled in a notebook, so it looks the tokenizer up as
# __main__.newline_tokenizer. Under an ASGI server this module isn't __main__,
# so the name has to be planted there before unpickling.
import __main__
__main__.newline_tokenizer = newline_tokenizer

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
vectorizer.input = "content"  # Switch from 'filename' to raw text 'content'

with open("artist_ids.json", "r", encoding="utf-8") as f:
    artist_ids = json.load(f)

# Cache feature names (tags)
feature_names = vectorizer.get_feature_names_out()

# Artist id -> row index in the TF-IDF matrix
row_of = {aid: i for i, aid in enumerate(artist_ids)}

# Non-zeros per row == number of distinct tags the artist has.
# Used to break similarity ties and to flag artists too thinly tagged to trust.
tag_counts = tfidf_matrix.getnnz(axis=1)

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


@app.get("/api/autocomplete_artist")
def autocomplete_artist(q: str = Query(..., min_length=1), limit: int = 10):
    """Returns matching artist name suggestions for AJAX autocomplete."""
    q_lower = q.lower().strip()

    # Only artists that have a row in the matrix can be used as a seed
    hits = [
        {"artist_id": aid, "artist_name": name}
        for aid, name in artists_kv.items()
        if aid in row_of and q_lower in name.lower()
    ]

    # Prefix matches first, then shortest names (closest to the typed query)
    hits.sort(
        key=lambda h: (
            not h["artist_name"].lower().startswith(q_lower),
            len(h["artist_name"]),
        )
    )

    return {"query": q, "suggestions": hits[:limit]}


def shared_tags(i, j, limit=5):
    """Tags both artists share, ranked by how much they drove the similarity."""
    row_i, row_j = tfidf_matrix[i], tfidf_matrix[j]
    weights_i = dict(zip(row_i.indices, row_i.data))
    weights_j = dict(zip(row_j.indices, row_j.data))

    common = set(weights_i) & set(weights_j)
    ranked = sorted(common, key=lambda c: -(weights_i[c] * weights_j[c]))

    return [feature_names[c] for c in ranked[:limit]]


@app.get("/api/similar_artists")
def similar_artists(q: str = Query(...), top_k: int = 20, min_tags: int = 1):
    """Finds artists whose tag profile points in the same direction as the seed."""
    if q not in row_of:
        return {"query": q, "results": [], "error": "Unknown artist id"}

    idx = row_of[q]
    similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    similarities[idx] = -1.0                      # never recommend the seed itself
    similarities[tag_counts < min_tags] = -1.0    # drop thinly tagged artists

    # Primary sort on descending score, ties broken by richer tag profiles.
    # np.lexsort treats the last key as primary.
    order = np.lexsort((-tag_counts, -similarities))

    results = []
    for other in order[:top_k]:
        score = float(similarities[other])
        if score <= 0:
            break
        other_id = artist_ids[other]
        results.append(
            {
                "artist_id": other_id,
                "artist_name": artists_kv.get(other_id, "Unknown"),
                "score": round(score, 4),
                "shared_tags": shared_tags(idx, other),
                "tag_count": int(tag_counts[other]),
            }
        )

    return {
        "query": q,
        "artist_name": artists_kv.get(q, "Unknown"),
        "tag_count": int(tag_counts[idx]),
        # A one- or two-tag seed carries too little signal to rank meaningfully
        "confident": bool(tag_counts[idx] >= 3),
        "results": results,
    }


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