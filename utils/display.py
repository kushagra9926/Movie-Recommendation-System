import shutil

# color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
PURPLE = "\033[95m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

BG_BLUE   = "\033[44m"
BG_PURPLE = "\033[45m"


def _width():
    return shutil.get_terminal_size((80, 20)).columns


def print_banner():
    w = min(_width(), 72)
    print()
    print(f"{BLUE}{'═' * w}{RESET}")
    print(f"{BOLD}{BLUE}{'🎬  MOVIE RECOMMENDATION SYSTEM':^{w}}{RESET}")
    print(f"{DIM}{'Powered by scikit-learn  |  SVD + TF-IDF Hybrid Engine':^{w}}{RESET}")
    print(f"{BLUE}{'═' * w}{RESET}")
    print()


def print_section(title, emoji="▸"):
    w = min(_width(), 72)
    print(f"\n{CYAN}{BOLD}{emoji} {title}{RESET}")
    print(f"{CYAN}{'─' * min(len(title) + 4, w)}{RESET}")


def print_success(msg):
    print(f"{GREEN}✅  {msg}{RESET}")


def print_error(msg):
    print(f"{RED}❌  {msg}{RESET}")


def print_info(msg):
    print(f"{YELLOW}ℹ️   {msg}{RESET}")


def print_movie_table(movies, show_score=True):
    """Print a formatted table of movie recommendations."""
    if not movies:
        print_error("No movies to display.")
        return

    # Column widths
    w_rank   = 4
    w_title  = 34
    w_year   = 6
    w_genres = 22
    w_rating = 7
    w_score  = 8

    header_parts = [
        f"{'#':>{w_rank}}",
        f"{'Title':<{w_title}}",
        f"{'Year':>{w_year}}",
        f"{'Genres':<{w_genres}}",
        f"{'Rating':>{w_rating}}",
    ]
    if show_score:
        header_parts.append(f"{'Match%':>{w_score}}")

    header = "  ".join(header_parts)
    sep = "─" * len(header)

    print(f"\n{BOLD}{WHITE}{header}{RESET}")
    print(f"{DIM}{sep}{RESET}")

    for i, m in enumerate(movies, 1):
        title = m["title"][:w_title - 1] if len(m["title"]) > w_title else m["title"]
        genres = m["genres"].replace("|", ", ")
        genres = genres[:w_genres - 1] if len(genres) > w_genres else genres
        rating = m["avg_rating"]
        stars = _stars(rating)

        rank_str  = f"{BOLD}{i:>{w_rank}}{RESET}"
        title_str = f"{CYAN}{title:<{w_title}}{RESET}"
        year_str  = f"{DIM}{m['year']:>{w_year}}{RESET}"
        genre_str = f"{YELLOW}{genres:<{w_genres}}{RESET}"
        rat_str   = f"{stars} {BOLD}{rating:>3.1f}{RESET}"

        row = f"{rank_str}  {title_str}  {year_str}  {genre_str}  {rat_str}"

        if show_score and "score" in m:
            color = GREEN if m["score"] >= 70 else YELLOW if m["score"] >= 40 else DIM
            row += f"  {color}{m['score']:>{w_score}.1f}{RESET}"

        print(row)

    print(f"{DIM}{sep}{RESET}\n")


def print_movie_card(movie):
    """Print a detailed card for a single movie."""
    w = min(_width(), 64)
    print(f"\n{BOLD}{BG_BLUE}  🎬 {movie['title']} ({movie['year']})  {RESET}")
    print(f"{CYAN}{'─' * w}{RESET}")

    fields = [
        ("Genres",    movie["genres"].replace("|", ", ")),
        ("Director",  movie["director"]),
        ("Cast",      movie["cast"].replace("|", ", ")),
        ("Runtime",   f"{movie['runtime_min']} min"),
        ("Budget",    f"${movie['budget_million']}M"),
        ("Avg Rating",f"{movie['avg_rating']} / 5.0  {_stars(movie['avg_rating'])}"),
        ("# Ratings", f"{movie['num_ratings']:,}"),
        ("Popularity",f"{movie['popularity_score']:.1f}"),
    ]

    for label, value in fields:
        print(f"  {BOLD}{label:<12}{RESET}  {value}")

    print(f"{CYAN}{'─' * w}{RESET}\n")


def _stars(rating):
    full = int(rating // 1)
    half = 1 if (rating % 1) >= 0.5 else 0
    empty = 5 - full - half
    return (
        f"{YELLOW}{'★' * full}{'½' * half}{'☆' * empty}{RESET}"
    )


def spinner_start(msg):
    print(f"{DIM}{msg}...{RESET}", end="", flush=True)


def spinner_stop():
    print(f"\r{' ' * 60}\r", end="", flush=True)
