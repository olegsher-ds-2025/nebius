"""
GitHub Repository Summarizer — Flask API
POST /summarize  →  { summary, technologies, structure }

LLM: Nebius Token Factory (OpenAI-compatible API)
     Configure via NEBIUS_API_KEY environment variable.
"""

import os
import re
import json
import logging
import requests
from flask import Flask, request, jsonify

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

GITHUB_API  = "https://api.github.com"
NEBIUS_URL  = "https://api.studio.nebius.com/v1/chat/completions"
NEBIUS_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # strong, fast, free-tier friendly

# How many bytes of a source file to include in the prompt
FILE_CONTENT_LIMIT = 3_000   # per file
# Rough token budget for all file contents combined (1 token ≈ 4 chars)
CONTEXT_CHAR_BUDGET = 28_000  # leaves room for prompt overhead in a 32k context

# Extensions we consider worth reading
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".r", ".sql", ".sh", ".bash", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".env.example",
    ".md", ".rst", ".txt",
    ".json",   # only small ones (filtered by size below)
    ".html", ".css",
    "Dockerfile", "Makefile",
}

# Paths/patterns to always skip
SKIP_PATTERNS = {
    # dependency dirs
    "node_modules/", ".venv/", "venv/", "env/", "__pycache__/",
    ".git/", ".github/", "dist/", "build/", ".next/", ".nuxt/",
    "target/", "vendor/", "site-packages/",
    # generated / lock files
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    "composer.lock", "Gemfile.lock", "cargo.lock",
    # binary / media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    # IDE / OS
    ".DS_Store", ".idea/", ".vscode/",
    # large data
    ".csv", ".parquet", ".feather", ".h5", ".hdf5", ".pkl", ".npy",
    ".ipynb",   # notebooks are huge; we skip them (README describes them)
    ".min.js", ".min.css",  # minified
}

# Files to prioritise first (evaluated in order)
PRIORITY_FILENAMES = [
    "readme", "readme.md", "readme.rst", "readme.txt",
    "main.py", "app.py", "server.py", "index.py", "run.py",
    "main.go", "main.rs", "main.js", "index.js", "index.ts",
    "setup.py", "setup.cfg", "pyproject.toml",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "makefile",
    "requirements.txt", "package.json", "cargo.toml", "go.mod",
]


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def parse_owner_repo(url: str) -> tuple[str, str] | None:
    url = url.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/?\s#]+)", url)
    if not match:
        return None
    owner = match.group(1)
    repo  = match.group(2).removesuffix(".git")
    return owner, repo


def gh_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def gh_get(path: str) -> requests.Response:
    return requests.get(f"{GITHUB_API}{path}", headers=gh_headers(), timeout=15)


def fetch_repo_context(owner: str, repo: str) -> dict:
    """
    Fetch everything useful about a repo and return a structured context dict.
    Raises ValueError for 404/private, RuntimeError for other API failures.
    """
    # 1. Metadata
    meta_r = gh_get(f"/repos/{owner}/{repo}")
    if meta_r.status_code == 404:
        raise ValueError(f"Repository '{owner}/{repo}' not found or is private.")
    if meta_r.status_code == 403:
        raise ValueError("GitHub API rate limit exceeded. Try again later or set GITHUB_TOKEN.")
    if meta_r.status_code != 200:
        raise RuntimeError(f"GitHub API error {meta_r.status_code}: {meta_r.text[:200]}")
    meta = meta_r.json()

    # 2. README (raw text)
    readme_r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/readme",
        headers={**gh_headers(), "Accept": "application/vnd.github.raw"},
        timeout=15,
    )
    readme = readme_r.text[:5_000] if readme_r.status_code == 200 else ""

    # 3. Full file tree
    tree_r = gh_get(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
    all_files: list[dict] = []
    if tree_r.status_code == 200:
        all_files = [
            item for item in tree_r.json().get("tree", [])
            if item["type"] == "blob"
        ]

    # 4. Language breakdown
    lang_r    = gh_get(f"/repos/{owner}/{repo}/languages")
    languages = lang_r.json() if lang_r.status_code == 200 else {}

    # 5. Select & fetch source files
    selected_files = select_files(all_files)
    file_contents  = fetch_file_contents(owner, repo, selected_files)

    return {
        "meta":          meta,
        "readme":        readme,
        "all_file_paths": [f["path"] for f in all_files],
        "languages":     languages,
        "file_contents": file_contents,
    }


# ── File selection strategy ────────────────────────────────────────────────────

def should_skip(path: str, size: int) -> bool:
    """Return True if this file should be excluded from context."""
    path_lower = path.lower()

    # Skip if any skip pattern matches path prefix or suffix
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("."):          # extension
            if path_lower.endswith(pattern):
                return True
        else:                                # directory or filename pattern
            if pattern in path_lower:
                return True

    # Skip files with no recognised extension (likely binary)
    _, ext = os.path.splitext(path_lower)
    basename = os.path.basename(path_lower)
    if ext not in SOURCE_EXTENSIONS and basename not in {
        "dockerfile", "makefile", "procfile", "gemfile", "rakefile"
    }:
        return True

    # Skip very large files (> 100 KB) — probably generated or data files
    if size > 100_000:
        return True

    return False


def priority_rank(path: str) -> int:
    """Lower number = higher priority."""
    basename = os.path.basename(path).lower()
    for i, name in enumerate(PRIORITY_FILENAMES):
        if basename == name:
            return i
    # Source files closer to root get priority
    depth = path.count("/")
    return len(PRIORITY_FILENAMES) + depth


def select_files(all_files: list[dict]) -> list[dict]:
    """Filter and rank files; return the list to actually fetch."""
    candidates = [
        f for f in all_files
        if not should_skip(f["path"], f.get("size", 0))
    ]
    candidates.sort(key=lambda f: priority_rank(f["path"]))
    return candidates


def fetch_file_contents(owner: str, repo: str, files: list[dict]) -> list[dict]:
    """
    Fetch raw content for selected files, stopping when we hit CONTEXT_CHAR_BUDGET.
    Returns list of { path, content } dicts.
    """
    results   = []
    total_chars = 0

    for f in files:
        if total_chars >= CONTEXT_CHAR_BUDGET:
            log.info("Context budget reached — stopping file fetch at %d chars", total_chars)
            break

        path = f["path"]
        url  = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue

            # Detect binary content quickly
            try:
                text = r.content.decode("utf-8")
            except UnicodeDecodeError:
                log.debug("Skipping binary file: %s", path)
                continue

            snippet = text[:FILE_CONTENT_LIMIT]
            results.append({"path": path, "content": snippet})
            total_chars += len(snippet)
            log.debug("Added %s (%d chars)", path, len(snippet))

        except requests.RequestException as e:
            log.warning("Could not fetch %s: %s", path, e)
            continue

    log.info("Fetched %d files, %d chars total", len(results), total_chars)
    return results


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(ctx: dict) -> str:
    meta      = ctx["meta"]
    languages = ctx["languages"]
    readme    = ctx["readme"]
    all_paths = ctx["all_file_paths"]
    files     = ctx["file_contents"]

    lang_str = ", ".join(languages.keys()) if languages else "Unknown"
    tree_str = "\n".join(f"  {p}" for p in all_paths[:150])
    if len(all_paths) > 150:
        tree_str += f"\n  ... and {len(all_paths) - 150} more files"

    files_section = ""
    for f in files:
        files_section += f"\n### {f['path']}\n```\n{f['content']}\n```\n"

    return f"""You are a senior software engineer analysing a GitHub repository.
Your job is to produce a structured JSON analysis. Respond with ONLY valid JSON — no markdown fences, no preamble.

## Repository Info
- Name: {meta['full_name']}
- Description: {meta.get('description') or 'None provided'}
- Primary language: {meta.get('language') or 'Unknown'}
- All languages detected: {lang_str}
- Stars: {meta.get('stargazers_count', 0)}
- License: {(meta.get('license') or {}).get('name', 'None')}

## File Tree (up to 150 entries)
{tree_str}

## README
{readme if readme else '(No README found)'}

## Key Source Files
{files_section if files_section else '(No source files fetched)'}

---

Return a JSON object with exactly these three fields:

{{
  "summary": "A 3–5 sentence human-readable description of what the project does, its purpose, and who it's for.",
  "technologies": ["list", "of", "main", "technologies", "languages", "frameworks", "libraries"],
  "structure": "1–3 sentences describing how the project is organised: key directories, entry points, test layout, config approach, etc."
}}

Be specific and accurate. Base everything on the actual files above, not general assumptions.
"""


# ── LLM call ───────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> dict:
    """
    Call Nebius Token Factory API (OpenAI-compatible).
    Returns parsed dict with summary, technologies, structure.
    Raises RuntimeError on failure.
    """
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY environment variable is not set. "
            "Set it before starting the server."
        )

    payload = {
        "model": NEBIUS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,   # low temp → consistent, factual output
        "max_tokens": 1024,
    }

    try:
        r = requests.post(
            NEBIUS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"LLM API request failed: {e}")

    if r.status_code == 401:
        raise RuntimeError("Invalid NEBIUS_API_KEY — check your credentials.")
    if r.status_code == 429:
        raise RuntimeError("LLM API rate limit exceeded. Try again shortly.")
    if r.status_code != 200:
        raise RuntimeError(f"LLM API error {r.status_code}: {r.text[:300]}")

    raw_text = r.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wrapped output anyway
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Last-resort: try to extract JSON object from the response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise RuntimeError(f"LLM returned non-JSON output: {raw_text[:300]}")

    # Validate expected fields
    for field in ("summary", "technologies", "structure"):
        if field not in result:
            raise RuntimeError(f"LLM response missing field '{field}': {result}")

    return result


# ── Route ───────────────────────────────────────────────────────────────────────

@app.post("/summarize")
def summarize():
    # ── Parse request ──────────────────────────────────────────────────────
    body = request.get_json(silent=True)
    if not body or "github_url" not in body:
        return jsonify({"status": "error", "message": "Request body must include 'github_url'."}), 400

    github_url = body["github_url"]
    parsed     = parse_owner_repo(github_url)
    if not parsed:
        return jsonify({"status": "error", "message": f"Invalid GitHub URL: '{github_url}'"}), 400

    owner, repo = parsed
    log.info("Request: %s/%s", owner, repo)

    # ── Fetch repo ─────────────────────────────────────────────────────────
    try:
        ctx = fetch_repo_context(owner, repo)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 502

    # ── Build prompt & call LLM ────────────────────────────────────────────
    try:
        prompt = build_prompt(ctx)
        result = call_llm(prompt)
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 502

    log.info("Done: %s/%s", owner, repo)
    return jsonify(result), 200


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": NEBIUS_MODEL}), 200


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
