import argparse
import sys
from utils.display import print_banner, print_section
from recommender import MovieRecommender


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        prog="movie-recommender",
        description="🎬 AI-powered Movie Recommendation System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- recommend ---
    rec_parser = subparsers.add_parser("recommend", help="Get movie recommendations")
    rec_parser.add_argument("--user", type=int, help="User ID for collaborative filtering")
    rec_parser.add_argument("--movie", type=str, help="Movie title for content-based filtering")
    rec_parser.add_argument("--genre", type=str, help="Filter recommendations by genre")
    rec_parser.add_argument("--top", type=int, default=10, help="Number of recommendations (default: 10)")
    rec_parser.add_argument(
        "--method",
        choices=["collaborative", "content", "hybrid"],
        default="hybrid",
        help="Recommendation method (default: hybrid)",
    )

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Search for a movie")
    search_parser.add_argument("query", type=str, help="Movie title to search")

    # --- info ---
    info_parser = subparsers.add_parser("info", help="Show movie details")
    info_parser.add_argument("title", type=str, help="Movie title")

    # --- stats ---
    subparsers.add_parser("stats", help="Show dataset & model statistics")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train / retrain the recommendation models")
    train_parser.add_argument("--force", action="store_true", help="Force retrain even if model exists")

    # --- list-genres ---
    subparsers.add_parser("list-genres", help="List all available genres")

    # --- top-rated ---
    top_parser = subparsers.add_parser("top-rated", help="Show top-rated movies")
    top_parser.add_argument("--genre", type=str, help="Filter by genre")
    top_parser.add_argument("--top", type=int, default=10, help="Number of movies (default: 10)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    recommender = MovieRecommender()

    if args.command == "train":
        recommender.train(force=args.force)

    elif args.command == "recommend":
        if not args.user and not args.movie:
            print("❌  Please provide --user <id> or --movie <title>")
            rec_parser.print_help()
            sys.exit(1)
        recommender.recommend(
            user_id=args.user,
            movie_title=args.movie,
            genre_filter=args.genre,
            top_n=args.top,
            method=args.method,
        )

    elif args.command == "search":
        recommender.search(args.query)

    elif args.command == "info":
        recommender.movie_info(args.title)

    elif args.command == "stats":
        recommender.stats()

    elif args.command == "list-genres":
        recommender.list_genres()

    elif args.command == "top-rated":
        recommender.top_rated(genre=args.genre, top_n=args.top)


if __name__ == "__main__":
    main()
