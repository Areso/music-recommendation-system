import json
import joblib
from collections import Counter
import csv
import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity

def newline_tokenizer(text):
    return [line.strip() for line in text.splitlines() if line.strip()]

# vectorizer.joblib was pickled in a notebook, so it looks the tokenizer up as
# __main__.newline_tokenizer. Under an ASGI server this module isn't __main__,
# so the name has to be planted there before unpickling.
import __main__
__main__.newline_tokenizer = newline_tokenizer

app = FastAPI(title="Music Recommendation API")

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

# Module 4 collaborative-filtering artifacts. The JSON lists preserve the
# exact row/column ordering used when the sparse matrix was exported.
user_artist_matrix = load_npz("user_artist_matrix.npz").tocsr()
with open("cf_user_ids.json", "r", encoding="utf-8") as f:
    cf_user_ids = [int(user_id) for user_id in json.load(f)]
with open("cf_artist_ids.json", "r", encoding="utf-8") as f:
    cf_artist_ids = [int(artist_id) for artist_id in json.load(f)]

if user_artist_matrix.shape != (len(cf_user_ids), len(cf_artist_ids)):
    raise RuntimeError("CF matrix shape does not match its exported ID mappings")

cf_user_row_of = {user_id: row for row, user_id in enumerate(cf_user_ids)}
# Translate a CF matrix column directly to its Module 3 TF-IDF row. Artists
# without eligible tags are marked -1 and omitted from content profiles.
content_row_by_cf_col = np.fromiter(
    (row_of.get(str(artist_id), -1) for artist_id in cf_artist_ids),
    dtype=np.int64,
    count=len(cf_artist_ids),
)
print(f"Loaded user-artist matrix: {user_artist_matrix.shape}")

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

@app.get("/api/healthcheck")
def healthcheck():
    """Liveness probe. Artifacts load at import time, so a 200 here means the
    whole dataset is in memory and every endpoint can be served."""
    return {
        "status": "ok",
        "artists": len(artists_kv),
        "tags": len(feature_names),
        "tfidf_matrix": list(tfidf_matrix.shape),
        "user_artist_matrix": list(user_artist_matrix.shape),
    }


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


def top_features(row, limit=10):
    """Return the strongest tags in a sparse TF-IDF row."""
    if row.nnz == 0:
        return []
    order = np.argsort(row.data)[::-1][:limit]
    return feature_names[row.indices[order]].tolist()


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


def shared_listening_artists(user_i, user_j, limit=5):
    """Return shared artists ranked by joint implicit-feedback confidence."""
    row_i = user_artist_matrix.getrow(user_i)
    row_j = user_artist_matrix.getrow(user_j)
    weights_i = dict(zip(row_i.indices, row_i.data))
    weights_j = dict(zip(row_j.indices, row_j.data))

    common = set(weights_i) & set(weights_j)
    ranked = sorted(common, key=lambda col: -(weights_i[col] * weights_j[col]))

    return [
        {
            "artist_id": str(cf_artist_ids[col]),
            "artist_name": artists_kv.get(str(cf_artist_ids[col]), "Unknown"),
        }
        for col in ranked[:limit]
    ]


@app.get("/api/similar_users")
def similar_users(q: int = Query(...), top_k: int = Query(20, ge=1, le=100)):
    """Find users whose log-scaled implicit listening profiles are most similar."""
    if q not in cf_user_row_of:
        return {"query": q, "results": [], "error": "Unknown user id"}

    idx = cf_user_row_of[q]
    similarities = cosine_similarity(
        user_artist_matrix.getrow(idx), user_artist_matrix
    ).flatten()
    similarities[idx] = -1.0
    order = np.argsort(similarities)[::-1]

    results = []
    for other in order:
        score = float(similarities[other])
        if score <= 0 or len(results) >= top_k:
            break

        shared = shared_listening_artists(idx, other)
        shared_count = int(
            np.intersect1d(
                user_artist_matrix.getrow(idx).indices,
                user_artist_matrix.getrow(other).indices,
                assume_unique=True,
            ).size
        )
        results.append(
            {
                "user_id": int(cf_user_ids[other]),
                "score": round(score, 4),
                "artist_count": int(user_artist_matrix.getrow(other).nnz),
                "shared_artist_count": shared_count,
                "shared_artists": shared,
            }
        )

    artist_count = int(user_artist_matrix.getrow(idx).nnz)
    return {
        "query": q,
        "artist_count": artist_count,
        "confident": artist_count >= 5,
        "results": results,
    }


@app.get("/api/recommend_content")
def recommend_content(
    q: int = Query(...),
    top_k: int = Query(10, ge=1, le=100),
):
    """Recommend unlistened artists from a listen-weighted TF-IDF user profile."""
    if q not in cf_user_row_of:
        return {"query": q, "results": [], "error": "Unknown user id"}

    user_row = cf_user_row_of[q]
    history = user_artist_matrix.getrow(user_row)
    history_content_rows = content_row_by_cf_col[history.indices]
    covered_mask = history_content_rows >= 0
    covered_rows = history_content_rows[covered_mask]

    if len(covered_rows) == 0:
        return {
            "query": q,
            "artist_count": int(history.nnz),
            "covered_artist_count": 0,
            "content_coverage": 0.0,
            "profile_tags": [],
            "confident": False,
            "results": [],
            "error": "User has no listened artists with content vectors",
        }

    confidence = history.data[covered_mask]
    weighted_sum = (
        tfidf_matrix[covered_rows]
        .multiply(confidence[:, None])
        .sum(axis=0)
    )
    profile = csr_matrix(weighted_sum / confidence.sum())
    norm = np.sqrt(profile.multiply(profile).sum())
    if norm > 0:
        profile = profile / norm

    scores = cosine_similarity(profile, tfidf_matrix).ravel()
    scores[np.unique(covered_rows)] = -np.inf
    candidate_rows = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    ranked_rows = candidate_rows[
        np.argsort(scores[candidate_rows])[::-1][:top_k]
    ]

    results = []
    for content_row in ranked_rows:
        artist_id = artist_ids[content_row]
        results.append(
            {
                "artist_id": artist_id,
                "artist_name": artists_kv.get(artist_id, "Unknown"),
                "score": round(float(scores[content_row]), 4),
                "top_tags": top_features(tfidf_matrix.getrow(content_row), limit=5),
            }
        )

    covered_count = len(covered_rows)
    artist_count = int(history.nnz)
    return {
        "query": q,
        "artist_count": artist_count,
        "covered_artist_count": covered_count,
        "content_coverage": round(covered_count / artist_count, 4),
        "profile_tags": top_features(profile),
        "confident": covered_count >= 5,
        "results": results,
    }


@app.get("/api/recommend_cf")
def recommend_cf(
    q: int = Query(...),
    top_k: int = Query(10, ge=1, le=100),
    n_neighbors: int = Query(30, ge=1, le=200),
):
    """Recommend unlistened artists from similar users' listening confidence."""
    if q not in cf_user_row_of:
        return {"query": q, "results": [], "error": "Unknown user id"}

    user_row = cf_user_row_of[q]
    history = user_artist_matrix.getrow(user_row)
    similarities = cosine_similarity(history, user_artist_matrix).ravel()
    similarities[user_row] = 0.0

    ranked_users = np.argsort(similarities)[::-1]
    neighbor_rows = ranked_users[similarities[ranked_users] > 0][:n_neighbors]
    if len(neighbor_rows) == 0:
        return {
            "query": q,
            "artist_count": int(history.nnz),
            "neighbors_used": 0,
            "confident": False,
            "results": [],
        }

    neighbor_matrix = user_artist_matrix[neighbor_rows]
    neighbor_similarities = similarities[neighbor_rows]
    scores = np.asarray(
        neighbor_matrix.multiply(neighbor_similarities[:, None]).sum(axis=0)
    ).ravel() / neighbor_similarities.sum()
    support = np.asarray(neighbor_matrix.getnnz(axis=0)).ravel()

    scores[history.indices] = -np.inf
    candidate_cols = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    ranked_cols = candidate_cols[
        np.argsort(scores[candidate_cols])[::-1][:top_k]
    ]

    results = []
    for artist_col in ranked_cols:
        artist_id = str(cf_artist_ids[artist_col])
        results.append(
            {
                "artist_id": artist_id,
                "artist_name": artists_kv.get(artist_id, "Unknown"),
                "score": round(float(scores[artist_col]), 4),
                "neighbor_support": int(support[artist_col]),
            }
        )

    return {
        "query": q,
        "artist_count": int(history.nnz),
        "neighbors_used": len(neighbor_rows),
        "top_neighbor_similarity": round(
            float(similarities[neighbor_rows[0]]), 4
        ),
        "confident": history.nnz >= 5,
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