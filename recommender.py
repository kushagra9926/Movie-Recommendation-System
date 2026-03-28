import os
import sys
import time

from data.dataset import load_or_generate
from models.engine import HybridRecommender, save_model, load_model
from utils.display import (
    print_section, print_success, print_error, print_info,
    print_movie_table, print_movie_card, CYAN, RESET, BOLD, DIM, GREEN, YELLOW, RED
)

MODEL_PATH = "models/recommender.pkl"
DATA_DIR   = "data"


class MovieRecommender:
    def __init__(self):
        self.model   = None
        self.movies  = None
        self.ratings = None
        self._load_data()
        self._load_model()

    #  Data & Model Loading 

    def _load_data(self):
        print_section("Loading Dataset", "📂")
        self.movies, self.ratings = load_or_generate(DATA_DIR)
        print_success(
            f"Loaded {len(self.movies):,} movies  |  "
            f"{len(self.ratings):,} ratings  |  "
            f"{self.ratings['user_id'].nunique()} users"
        )

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            print_section("Loading Trained Model", "🤖")
            self.model = load_model(MODEL_PATH)
            self.model.movies_df  = self.movies
            self.model.ratings_df = self.ratings
            print_success("Model loaded from cache.")
        else:
            print_info("No trained model found. Run: python main.py train")

    # Commands 

    def train(self, force=False):
        if os.path.exists(MODEL_PATH) and not force:
            print_info("Model already trained. Use --force to retrain.")
            return

        print_section("Training Models", "🏋️")
        start = time.time()

        self.model = HybridRecommender(collab_weight=0.6, content_weight=0.4)
        self.model.fit(self.movies, self.ratings)

        save_model(self.model, MODEL_PATH)
        elapsed = time.time() - start
        print_success(f"Training complete in {elapsed:.2f}s  →  model saved.")

    def recommend(self, user_id=None, movie_title=None, genre_filter=None, top_n=10, method="hybrid"):
        self._ensure_model()

        if method in ("collaborative", "hybrid") and user_id is not None:
            print_section(
                f"Recommendations for User #{user_id}  [{method.upper()}]", "🎯"
            )
            if genre_filter:
                print_info(f"Genre filter: {genre_filter}")

            if method == "hybrid":
                recs = self.model.recommend_for_user(user_id, top_n=top_n, genre_filter=genre_filter)
            else:
                raw = self.model.collab.recommend(user_id, top_n=top_n * 2)
                recs = self.model._enrich(raw)
                if genre_filter:
                    recs = [r for r in recs if genre_filter.lower() in r["genres"].lower()]
                recs = recs[:top_n]

            if not recs:
                print_error(f"No recommendations found for user {user_id}.")
                print_info(f"Valid user IDs: 1 – {self.ratings['user_id'].max()}")
                return

            print_movie_table(recs, show_score=(method == "hybrid"))
            self._show_user_profile(user_id)

        elif movie_title is not None:
            print_section(f"Movies Similar to  \"{movie_title}\"", "🎞️")
            if genre_filter:
                print_info(f"Genre filter: {genre_filter}")

            recs = self.model.recommend_similar(movie_title, top_n=top_n, genre_filter=genre_filter)

            if not recs:
                print_error(f"Movie \"{movie_title}\" not found or no similar movies.")
                print_info("Tip: use  python main.py search <query>  to find exact titles.")
                return

            print_movie_table(recs, show_score=True)

    def search(self, query):
        print_section(f"Search: \"{query}\"", "🔍")
        matches = self.movies[
            self.movies["title"].str.lower().str.contains(query.lower(), na=False)
        ]
        if matches.empty:
            print_error(f"No movies found matching \"{query}\".")
            return

        print_info(f"{len(matches)} result(s) found:\n")
        for _, row in matches.head(20).iterrows():
            genres = row["genres"].replace("|", ", ")
            print(
                f"  {BOLD}{CYAN}{row['title']}{RESET} "
                f"{DIM}({row['year']}){RESET}  —  {YELLOW}{genres}{RESET}  "
                f"  ⭐ {row['avg_rating']}"
            )
        print()

    def movie_info(self, title):
        print_section(f"Movie Info: \"{title}\"", "🎬")
        matches = self.movies[
            self.movies["title"].str.lower().str.contains(title.lower(), na=False)
        ]
        if matches.empty:
            print_error(f"Movie \"{title}\" not found.")
            return

        row = matches.iloc[0].to_dict()
        print_movie_card(row)

        # Show rating distribution from ratings
        movie_ratings = self.ratings[self.ratings["movie_id"] == row["movie_id"]]["rating"]
        if not movie_ratings.empty:
            print(f"  {BOLD}Rating Distribution:{RESET}")
            for star in [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]:
                count = (movie_ratings == star).sum()
                bar = "█" * count
                print(f"    {star:.1f}  {CYAN}{bar:<50}{RESET} {DIM}{count}{RESET}")
            print()

    def stats(self):
        print_section("Dataset & Model Statistics", "📊")

        m = self.movies
        r = self.ratings

        print(f"\n  {BOLD}📽️  Movies{RESET}")
        print(f"    Total             : {len(m):,}")
        print(f"    Year range        : {int(m['year'].min())} – {int(m['year'].max())}")
        print(f"    Avg rating        : {m['avg_rating'].mean():.2f}")
        print(f"    Genres            : {len(self._all_genres())}")

        print(f"\n  {BOLD}👥  Users & Ratings{RESET}")
        print(f"    Total ratings     : {len(r):,}")
        print(f"    Users             : {r['user_id'].nunique()}")
        print(f"    Avg ratings/user  : {len(r) / r['user_id'].nunique():.1f}")
        print(f"    Rating mean       : {r['rating'].mean():.2f}")
        print(f"    Rating std        : {r['rating'].std():.2f}")

        # Sparsity
        possible = r["user_id"].nunique() * len(m)
        sparsity = 1 - len(r) / possible
        print(f"    Matrix sparsity   : {sparsity:.1%}")

        if self.model and self.model.is_trained:
            print(f"\n  {BOLD}🤖  Model{RESET}")
            print(f"    SVD components    : {self.model.collab.n_components}")
            print(f"    Variance explained: {self.model.collab.explained_variance():.1%}")
            vocab = len(self.model.content.vectorizer.vocabulary_)
            print(f"    TF-IDF vocab size : {vocab:,}")
            print(f"    Hybrid weights    : Collab {self.model.collab_weight:.0%}  /  Content {self.model.content_weight:.0%}")

        print()

    def list_genres(self):
        print_section("Available Genres", "🎭")
        genres = self._all_genres()
        for g in sorted(genres):
            count = self.movies["genres"].str.contains(g).sum()
            bar = "▓" * (count // 5)
            print(f"  {CYAN}{g:<18}{RESET}  {bar}  {DIM}{count} movies{RESET}")
        print()

    def top_rated(self, genre=None, top_n=10):
        title = f"Top Rated Movies"
        if genre:
            title += f"  —  {genre}"
        print_section(title, "🏆")

        df = self.movies.copy()
        if genre:
            df = df[df["genres"].str.lower().str.contains(genre.lower(), na=False)]
            if df.empty:
                print_error(f"No movies found for genre: {genre}")
                return

        df = df.nlargest(top_n, "avg_rating")
        records = df.to_dict("records")
        for r in records:
            r["score"] = r["avg_rating"] * 20  # convert to 0-100 scale

        print_movie_table(records, show_score=False)

    # Helpers 

    def _ensure_model(self):
        if self.model is None or not self.model.is_trained:
            print_info("Model not trained yet. Training now...")
            self.train()

    def _show_user_profile(self, user_id):
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        if user_ratings.empty:
            return

        top = user_ratings.nlargest(3, "rating")
        movie_ids = top["movie_id"].tolist()
        movies = self.movies[self.movies["movie_id"].isin(movie_ids)]

        print(f"  {DIM}Based on your top-rated movies:{RESET}")
        for _, row in movies.iterrows():
            rating = user_ratings[user_ratings["movie_id"] == row["movie_id"]]["rating"].values[0]
            print(f"    ⭐ {rating}  {CYAN}{row['title']}{RESET}  {DIM}({row['year']}){RESET}")
        print()

    def _all_genres(self):
        genres = set()
        for g_str in self.movies["genres"].dropna():
            for g in g_str.split("|"):
                genres.add(g.strip())
        return genres
