import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


class CollaborativeFilter:


    def __init__(self, n_components=30):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.scaler = MinMaxScaler()
        self.user_movie_matrix = None
        self.user_factors = None
        self.movie_factors = None
        self.movie_ids = None
        self.user_ids = None

    def fit(self, ratings_df):
        
        matrix = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating", fill_value=0
        )
        self.user_ids = matrix.index.tolist()
        self.movie_ids = matrix.columns.tolist()
        self.user_movie_matrix = matrix.values.astype(float)

        # Normalize
        normalized = self.scaler.fit_transform(self.user_movie_matrix)

        # SVD decomposition
        self.user_factors = self.svd.fit_transform(normalized)
        self.movie_factors = self.svd.components_.T  

        # Evaluate 
        reconstructed = self.user_factors @ self.svd.components_
        reconstructed = self.scaler.inverse_transform(reconstructed)
        mask = self.user_movie_matrix > 0
        rmse = np.sqrt(mean_squared_error(
            self.user_movie_matrix[mask],
            np.clip(reconstructed[mask], 0, 5)
        ))
        return rmse

    def recommend(self, user_id, top_n=10, exclude_rated=True):
        if user_id not in self.user_ids:
            return []

        user_idx = self.user_ids.index(user_id)
        user_vec = self.user_factors[user_idx]

        # Predict scores for all movies
        scores = self.movie_factors @ user_vec
        score_map = dict(zip(self.movie_ids, scores))

        if exclude_rated:
            rated = set(
                self.movie_ids[i]
                for i, v in enumerate(self.user_movie_matrix[user_idx])
                if v > 0
            )
            score_map = {k: v for k, v in score_map.items() if k not in rated}

        top = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(mid, float(score)) for mid, score in top]

    def explained_variance(self):
        return self.svd.explained_variance_ratio_.sum()


class ContentFilter:
    """TF-IDF + cosine similarity content-based filtering."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.movie_index = None
        self.movies_df = None

    def _build_soup(self, row):
        genres = row["genres"].replace("|", " ")
        director = row["director"].replace(" ", "_")
        cast = row["cast"].replace("|", " ").replace(" ", "_")
        year_tag = f"era_{(row['year'] // 10) * 10}s"
        return f"{genres} {genres} {director} {cast} {year_tag}"

    def fit(self, movies_df):
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_index = {
            row["title"].lower(): idx
            for idx, row in self.movies_df.iterrows()
        }
        soup = self.movies_df.apply(self._build_soup, axis=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(soup)

    def recommend(self, movie_title, top_n=10):
        key = movie_title.lower()
        
        if key not in self.movie_index:
            matches = [t for t in self.movie_index if key in t]
            if not matches:
                return []
            key = matches[0]

        idx = self.movie_index[key]
        movie_vec = self.tfidf_matrix[idx]
        sims = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()
        sims[idx] = 0  

        top_indices = sims.argsort()[::-1][:top_n]
        results = []
        for i in top_indices:
            row = self.movies_df.iloc[i]
            results.append((int(row["movie_id"]), float(sims[i])))
        return results

    def get_similarity_score(self, movie_id_a, movie_id_b):
        idx_a = self.movies_df[self.movies_df["movie_id"] == movie_id_a].index
        idx_b = self.movies_df[self.movies_df["movie_id"] == movie_id_b].index
        if len(idx_a) == 0 or len(idx_b) == 0:
            return 0.0
        sim = cosine_similarity(
            self.tfidf_matrix[idx_a[0]], self.tfidf_matrix[idx_b[0]]
        )
        return float(sim[0][0])


class HybridRecommender:
    """Combines collaborative and content signals."""

    def __init__(self, collab_weight=0.6, content_weight=0.4):
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.collab = CollaborativeFilter()
        self.content = ContentFilter()
        self.movies_df = None
        self.ratings_df = None
        self.is_trained = False

    def fit(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df

        print("  🔧 Training Collaborative Filter (SVD)...")
        rmse = self.collab.fit(ratings_df)
        print(f"     ✅ RMSE: {rmse:.4f}  |  Variance explained: {self.collab.explained_variance():.1%}")

        print("  🔧 Building Content-Based Filter (TF-IDF)...")
        self.content.fit(movies_df)
        print(f"     ✅ TF-IDF matrix: {self.content.tfidf_matrix.shape[0]} movies × {self.content.tfidf_matrix.shape[1]} features")

        self.is_trained = True

    def recommend_for_user(self, user_id, top_n=10, genre_filter=None):
        collab_recs = dict(self.collab.recommend(user_id, top_n=top_n * 3))

        # Content signal:from user's top-rated movies
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        top_rated = user_ratings.nlargest(3, "rating")["movie_id"].tolist()
        content_scores = {}
        for mid in top_rated:
            row = self.movies_df[self.movies_df["movie_id"] == mid]
            if row.empty:
                continue
            title = row.iloc[0]["title"]
            recs = self.content.recommend(title, top_n=top_n * 3)
            for rec_id, score in recs:
                content_scores[rec_id] = content_scores.get(rec_id, 0) + score / len(top_rated)

        # Normalize and blend
        all_ids = set(collab_recs) | set(content_scores)
        if not all_ids:
            return []

        blended = {}
        max_c = max(collab_recs.values()) if collab_recs else 1
        max_cnt = max(content_scores.values()) if content_scores else 1
        for mid in all_ids:
            c = collab_recs.get(mid, 0) / max_c
            cnt = content_scores.get(mid, 0) / max_cnt
            blended[mid] = self.collab_weight * c + self.content_weight * cnt

        results = sorted(blended.items(), key=lambda x: x[1], reverse=True)

        # Apply genre filter
        if genre_filter:
            results = [
                (mid, score) for mid, score in results
                if self._has_genre(mid, genre_filter)
            ]

        return self._enrich(results[:top_n])

    def recommend_similar(self, movie_title, top_n=10, genre_filter=None):
        recs = self.content.recommend(movie_title, top_n=top_n * 3)
        if genre_filter:
            recs = [(mid, s) for mid, s in recs if self._has_genre(mid, genre_filter)]
        return self._enrich(recs[:top_n])

    def _has_genre(self, movie_id, genre):
        row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if row.empty:
            return False
        genres = row.iloc[0]["genres"].lower().split("|")
        return genre.lower() in genres

    def _enrich(self, id_score_list):
        enriched = []
        for movie_id, score in id_score_list:
            row = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if row.empty:
                continue
            m = row.iloc[0]
            enriched.append({
                "movie_id": int(m["movie_id"]),
                "title": m["title"],
                "year": int(m["year"]),
                "genres": m["genres"],
                "director": m["director"],
                "avg_rating": float(m["avg_rating"]),
                "score": round(score * 100, 1),
            })
        return enriched


def save_model(model, path="models/recommender.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path="models/recommender.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
