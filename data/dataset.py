import numpy as np
import pandas as pd
import random

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
]

DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "Quentin Tarantino",
    "James Cameron", "Ridley Scott", "David Fincher", "Wes Anderson",
    "Guillermo del Toro", "Denis Villeneuve", "Sofia Coppola", "Greta Gerwig",
    "Jordan Peele", "Bong Joon-ho", "Alfonso Cuarón", "Darren Aronofsky",
    "Paul Thomas Anderson", "Coen Brothers", "Spike Lee", "Tim Burton",
]

ACTORS = [
    "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Cate Blanchett",
    "Brad Pitt", "Natalie Portman", "Morgan Freeman", "Scarlett Johansson",
    "Robert De Niro", "Viola Davis", "Denzel Washington", "Emma Stone",
    "Ryan Gosling", "Tilda Swinton", "Joaquin Phoenix", "Jennifer Lawrence",
    "Christian Bale", "Charlize Theron", "Matt Damon", "Lupita Nyong'o",
]

MOVIE_TEMPLATES = [
    ("{adjective} {noun}", ["Action", "Thriller"]),
    ("The {adjective} {noun}", ["Drama", "Mystery"]),
    ("{noun} of {place}", ["Adventure", "Fantasy"]),
    ("Return to {place}", ["Sci-Fi", "Adventure"]),
    ("The Last {noun}", ["Drama", "Western"]),
    ("{name}'s {noun}", ["Comedy", "Romance"]),
    ("Beyond the {noun}", ["Sci-Fi", "Drama"]),
    ("Dark {noun}", ["Horror", "Thriller"]),
    ("The {noun} Chronicles", ["Fantasy", "Adventure"]),
    ("Lost in {place}", ["Drama", "Romance"]),
    ("{place} Story", ["Romance", "Drama"]),
    ("Night of the {adjective} {noun}", ["Horror", "Mystery"]),
    ("The {noun} Code", ["Mystery", "Thriller"]),
    ("{adjective} Justice", ["Action", "Crime"]),
    ("City of {noun}s", ["Crime", "Drama"]),
]

ADJECTIVES = [
    "Dark", "Silent", "Golden", "Broken", "Hidden", "Lost", "Last",
    "Forgotten", "Crimson", "Eternal", "Infinite", "Hollow", "Iron",
    "Savage", "Phantom", "Twisted", "Burning", "Frozen", "Ancient", "Wild",
]

NOUNS = [
    "Shadow", "Storm", "Legend", "Warrior", "Dream", "World", "Heart",
    "Soul", "Kingdom", "Empire", "Dragon", "Knight", "Angel", "Ghost",
    "Machine", "Code", "Matrix", "Oracle", "Destiny", "Phoenix",
]

PLACES = [
    "Midnight", "Avalon", "Eden", "Zion", "Mars", "Brooklyn", "Tokyo",
    "Paris", "Atlantis", "Olympus", "Arcadia", "Valhalla", "Babylon", "Rome",
]

NAMES = [
    "Alex", "John", "Maria", "Sam", "Lisa", "Marcus", "Elena", "Victor",
]


def generate_title():
    template, _ = random.choice(MOVIE_TEMPLATES)
    title = template.format(
        adjective=random.choice(ADJECTIVES),
        noun=random.choice(NOUNS),
        place=random.choice(PLACES),
        name=random.choice(NAMES),
    )
    return title


def generate_movies(n=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    titles = set()
    movies = []

    movie_id = 1
    while len(movies) < n:
        title = generate_title()
        if title in titles:
            continue
        titles.add(title)

        year = random.randint(1970, 2024)
        num_genres = random.randint(1, 3)
        genres = random.sample(GENRES, num_genres)
        director = random.choice(DIRECTORS)
        cast = random.sample(ACTORS, random.randint(2, 4))
        runtime = random.randint(75, 210)
        budget_m = round(random.uniform(1, 300), 1)
        avg_rating = round(np.random.beta(5, 2) * 4 + 1, 1)  # skewed towards higher ratings
        num_ratings = random.randint(50, 50000)
        popularity = round(num_ratings * avg_rating / 1000, 2)

        movies.append({
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "genres": "|".join(genres),
            "director": director,
            "cast": "|".join(cast),
            "runtime_min": runtime,
            "budget_million": budget_m,
            "avg_rating": avg_rating,
            "num_ratings": num_ratings,
            "popularity_score": popularity,
        })
        movie_id += 1

    return pd.DataFrame(movies)


def generate_ratings(movies_df, n_users=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    movie_ids = movies_df["movie_id"].tolist()
    ratings = []

    for user_id in range(1, n_users + 1):
        # Each user rates between 10 and 80 movies
        n_rated = random.randint(10, 80)
        rated_movies = random.sample(movie_ids, min(n_rated, len(movie_ids)))

        # User has a personal bias
        bias = np.random.normal(0, 0.5)

        for movie_id in rated_movies:
            movie = movies_df[movies_df["movie_id"] == movie_id].iloc[0]
            base = movie["avg_rating"]
            rating = round(np.clip(base + bias + np.random.normal(0, 0.8), 1, 5) * 2) / 2
            ratings.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
            })

    return pd.DataFrame(ratings)


def load_or_generate(data_dir="data"):
    import os
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
    else:
        os.makedirs(data_dir, exist_ok=True)
        print("  📦 Generating synthetic movie dataset...")
        movies = generate_movies(500)
        ratings = generate_ratings(movies, n_users=200)
        movies.to_csv(movies_path, index=False)
        ratings.to_csv(ratings_path, index=False)
        print(f"  ✅ Dataset saved → {movies_path}, {ratings_path}")

    return movies, ratings
