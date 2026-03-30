# Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![CLI](https://img.shields.io/badge/Interface-CLI-black)](#cli-command-reference)

A **command-line movie recommender** built with **scikit-learn**. It suggests films using **collaborative filtering** (how users rated movies), **content-based filtering** (similarity of genres, cast, and metadata), and a **hybrid** blend of both. Everything runs locally: **no web UI, no API keys, and no network** after you install dependencies.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [What you need](#what-you-need)
- [Get the code](#get-the-code)
- [Setup (step by step)](#setup-step-by-step)
- [Run it the first time](#run-it-the-first-time)
- [Daily usage](#daily-usage)
- [CLI command reference](#cli-command-reference)
- [How it works (short)](#how-it-works-short)
- [Dataset](#dataset)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## What this project does

| Capability | Description |
|------------|-------------|
| **Recommend by user** | Given a user ID (1–200), suggests movies that user has not rated, using hybrid scores by default. |
| **Recommend by movie** | Given a movie title, suggests similar movies (content-based). |
| **Search & info** | Search titles by keyword; show a detailed “card” for one movie. |
| **Browse** | List genres, show top-rated movies (optionally by genre), print dataset/model stats. |

The “movies” and “ratings” are **synthetic** (generated code + CSVs in `data/`), inspired by MovieLens-style data but **not** real box-office titles.

---

## What you need

- **Python 3.8 or newer** ([python.org](https://www.python.org/downloads/))
- **pip** (usually bundled with Python)
- A terminal (PowerShell or Command Prompt on Windows; any shell on macOS/Linux)
- About **50 MB** disk space for a virtual environment plus packages (rough estimate)

---

## Get the code

**Option A — Git clone (if you use Git)**

```bash
git clone https://github.com/kushagra9926/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

Replace `<your-repo-url>` with your GitHub repository URL. The folder you `cd` into must contain `main.py` and `requirements.txt`.

**Option B — ZIP download**

1. On GitHub, use **Code → Download ZIP**.
2. Unzip the archive.
3. Open a terminal and `cd` into the unzipped folder (the one that contains `main.py`).

**Important:** All commands below assume your **current working directory** is this project root (where `main.py` lives). If you see import or file-not-found errors, you are usually in the wrong folder.

---

## Setup (step by step)

### 1. Open a terminal in the project folder

```bash
cd path/to/movie_recommender
```

### 2. (Recommended) Create a virtual environment

This keeps dependencies isolated from other Python projects.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation fails with an execution policy error, run PowerShell as Administrator once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` or similar at the start of your prompt.

### 3. Install Python packages

```bash
pip install -r requirements.txt
```

This installs:

| Package | Role |
|---------|------|
| `scikit-learn` | SVD, TF-IDF, cosine similarity |
| `numpy` | Numerical arrays |
| `pandas` | Loading CSVs and tables |

---

## Run it the first time

### Step 1 — Generate data (automatic) and train models

**Recommended:** train once explicitly so later commands load the cached model quickly:

```bash
python main.py train
```

**What happens:**

1. If `data/movies.csv` or `data/ratings.csv` is missing, the app **generates** a synthetic dataset and saves those files.
2. It trains the recommendation models and saves them to **`models/recommender.pkl`**.

If a model file already exists and you did not pass `--force`, `train` skips work and tells you to use `--force` to retrain.

**Force a full retrain** (same data files, rebuild model):

```bash
python main.py train --force
```

> **Note:** The dataset is created the first time **any** command loads data (e.g. `stats`, `search`) if the CSVs are missing. If you run **`recommend`** before a model exists, the app **trains automatically** on first use—running `train` yourself is optional but avoids that wait on the first recommendation.

### Step 2 — Try a recommendation

```bash
python main.py recommend --user 5
```

```bash
python main.py recommend --movie "Dark Empire"
```

You must pass **either** `--user <id>` **or** `--movie "title"` (see [CLI reference](#cli-command-reference)).

### Step 3 — Explore other commands

```bash
python main.py stats
python main.py search "Storm"
python main.py --help
```

---

## Daily usage

Always activate the virtual environment first (if you use one), `cd` to the project root, then run:

```bash
python main.py <command> [options]
```

Examples:

```bash
python main.py recommend --user 12 --genre Action --top 8
python main.py top-rated --genre Sci-Fi --top 5
python main.py info "The Last Shadow"
```

---

## CLI command reference

Pattern:

```text
python main.py <command> [options]
```

Global help:

```bash
python main.py
python main.py --help
```

Per-command help:

```bash
python main.py recommend --help
python main.py train --help
```

### `train` — Build or refresh the saved model

```bash
python main.py train
python main.py train --force
```

### `recommend` — Main recommendation command

Requires **`--user`** or **`--movie`** (at least one).

```bash
# By user (default method: hybrid)
python main.py recommend --user 5
python main.py recommend --user 5 --top 15
python main.py recommend --user 5 --genre Action

# By movie (content-based similarity)
python main.py recommend --movie "Dark Empire"
python main.py recommend --movie "Lost Heart" --genre Comedy --top 5

# Explicit method (user-based flows)
python main.py recommend --user 5 --method collaborative
python main.py recommend --user 5 --method content
python main.py recommend --user 5 --method hybrid
```

| Option | Meaning |
|--------|---------|
| `--user` | Integer user ID (synthetic users 1–200). |
| `--movie` | Movie title (partial / fuzzy matching as implemented). |
| `--genre` | Optional filter on genre name. |
| `--top` | Number of results (default: 10). |
| `--method` | `collaborative`, `content`, or `hybrid` (default: hybrid). |

### `search` — Find movies by keyword in the title

```bash
python main.py search "Dark"
python main.py search "Storm"
```

### `info` — Show one movie’s details

```bash
python main.py info "Dark Empire"
```

### `top-rated` — Highest-rated movies

```bash
python main.py top-rated
python main.py top-rated --genre Drama --top 5
```

### `list-genres` — All genres and counts

```bash
python main.py list-genres
```

### `stats` — Dataset and model summary

```bash
python main.py stats
```

---

## How it works (short)

1. **Data** — Synthetic movies and ratings are generated or loaded from `data/*.csv`.
2. **Collaborative filtering** — User × movie ratings are factorized with **TruncatedSVD**; unseen movies are scored in latent space.
3. **Content-based** — Text built from genres, director, cast, and era is vectorized with **TF-IDF**; **cosine similarity** ranks “similar to this movie.”
4. **Hybrid** — For user recommendations, collaborative and content signals are combined (default blend: **60% collaborative / 40% content**).

For more detail, see comments in `models/engine.py` and `data/dataset.py`.

---

## Dataset

| Property | Typical value |
|----------|----------------|
| Movies | 500 |
| Users | 200 |
| Ratings | ~9,500 |
| Rating scale | 0.5–5.0 (0.5 steps) |
| Genres | 15 |

Generation uses a **fixed random seed** so runs are reproducible when CSVs are created from scratch.

---

## Project layout

```text
movie_recommender/
├── main.py              # CLI entry point
├── recommender.py       # Loads data/model and runs commands
├── requirements.txt
├── README.md
├── data/
│   ├── dataset.py       # Generate / load CSVs
│   ├── movies.csv       # Present or created on first load
│   └── ratings.csv
├── models/
│   ├── engine.py        # SVD, TF-IDF, hybrid logic
│   └── recommender.pkl  # Created after `train`
└── utils/
    └── display.py       # Terminal formatting and colors
```

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `ModuleNotFoundError` (e.g. `sklearn`, `pandas`) | From project root, with venv activated: `pip install -r requirements.txt` |
| “No trained model” / recommend fails | Run `python main.py train` |
| Wrong folder / imports fail | `cd` to the directory that contains `main.py`; run `python main.py stats` to verify |
| Movie not found for `--movie` | Use `python main.py search "<keyword>"` to find an exact title, then retry |
| Colors look wrong on Windows | Use [Windows Terminal](https://aka.ms/terminal) for best ANSI support |

**Reset data and model** (bash-like shell):

```bash
rm -f data/movies.csv data/ratings.csv models/recommender.pkl
python main.py train
```

**Windows (PowerShell)** — delete the files in Explorer or:

```powershell
Remove-Item -ErrorAction SilentlyContinue data\movies.csv, data\ratings.csv, models\recommender.pkl
python main.py train
```

---

## Acknowledgements

- Dataset **shape** inspired by [MovieLens](https://grouplens.org/datasets/movielens/); **all titles and ratings here are synthetic**.
- Built with [scikit-learn](https://scikit-learn.org/).
