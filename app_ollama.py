"""
GitHub Repository Summarizer — Flask API (Ollama / Local LLM edition)
POST /summarize  →  { summary, technologies, structure }

LLM: Ollama running locally at http://localhost:11434
     No API key needed. Configure model via OLLAMA_MODEL env var.
     Default model: phi3:mini  (best for technical/code content on CPU)

Start Ollama first:
    ollama serve
    ollama pull phi3:mini

Then run this server:
    python app_ollama.py
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
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# phi3:mini is recommended for Intel Xe / CPU-only machines:
#   - strong at code and technical content
#   - fast on CPU (~12 tok/s)
#   - only 2.2 GB download
# Alternatives: llama3.2, gemma2:2b, mistral

# Context limits — tuned for small models (phi3:mini has a 4k context window)
# For larger models like llama3.2 or mistral you can raise these.
FILE_CONTENT_LIMIT  = 1_500   # chars per file (keep prompt tight for small models)
CONTEXT_CHAR_BUDGET = 8_000   # total chars for file content

# Extensions we consider worth reading
SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".scala", ".r", ".sql", ".sh", ".bash", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".env.example",
    ".md", ".rst", ".txt",
    ".json",
    ".html", ".css",
    "Dockerfile", "Makefile",
}

# Paths / patterns to always skip
SKIP_PATTERNS = {
    "node_modules/", ".venv/", "venv/", "env/", "__pycache__/",
    ".git/", ".github/", "dist/", "build/", ".next/", ".nuxt/",
    "target/", "vendor/", "site-packages/",
    "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock",
    "composer.lock", "Gemfile.lock", "cargo.lock",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".whl", ".egg",
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".DS_Store", ".idea/", ".vscode/",
    ".csv", ".parquet", ".feather", ".h5", ".hdf5", ".pkl", ".npy",
    ".ipynb",
    ".min.js", ".min.css",
}

# Files to fetch first (highest signal, lowest token cost)
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
    meta_r = gh_get(f"/repos/{owner}/{repo}")
    if meta_r.status_code == 404:
        raise ValueError(f"Repository '{owner}/{repo}' not found or is private.")
    if meta_r.status_code == 403:
        raise ValueError("GitHub API rate limit exceeded. Try again later or set GITHUB_TOKEN.")
    if meta_r.status_code != 200:
        raise RuntimeError(f"GitHub API error {meta_r.status_code}: {meta_r.text[:200]}")
    meta = meta_r.json()

    readme_r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/readme",
        headers={**gh_headers(), "Accept": "application/vnd.github.raw"},
        timeout=15,
    )
    readme = readme_r.text[:3_000] if readme_r.status_code == 200 else ""

    tree_r = gh_get(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
    all_files: list[dict] = []
    if tree_r.status_code == 200:
        all_files = [
            item for item in tree_r.json().get("tree", [])
            if item["type"] == "blob"
        ]

    lang_r    = gh_get(f"/repos/{owner}/{repo}/languages")
    languages = lang_r.json() if lang_r.status_code == 200 else {}

    selected_files = select_files(all_files)
    file_contents  = fetch_file_contents(owner, repo, selected_files)

    return {
        "meta":           meta,
        "readme":         readme,
        "all_file_paths": [f["path"] for f in all_files],
        "languages":      languages,
        "file_contents":  file_contents,
    }


# ── File selection ─────────────────────────────────────────────────────────────

def should_skip(path: str, size: int) -> bool:
    path_lower = path.lower()
    for pattern in SKIP_PATTERNS:
        if pattern.startswith("."):
            if path_lower.endswith(pattern):
                return True
        else:
            if pattern in path_lower:
                return True
    _, ext = os.path.splitext(path_lower)
    basename = os.path.basename(path_lower)
    if ext not in SOURCE_EXTENSIONS and basename not in {
        "dockerfile", "makefile", "procfile", "gemfile", "rakefile"
    }:
        return True
    if size > 100_000:
        return True
    return False


def priority_rank(path: str) -> int:
    basename = os.path.basename(path).lower()
    for i, name in enumerate(PRIORITY_FILENAMES):
        if basename == name:
            return i
    return len(PRIORITY_FILENAMES) + path.count("/")


def select_files(all_files: list[dict]) -> list[dict]:
    candidates = [f for f in all_files if not should_skip(f["path"], f.get("size", 0))]
    candidates.sort(key=lambda f: priority_rank(f["path"]))
    return candidates


def fetch_file_contents(owner: str, repo: str, files: list[dict]) -> list[dict]:
    results     = []
    total_chars = 0
    for f in files:
        if total_chars >= CONTEXT_CHAR_BUDGET:
            log.info("Context budget reached at %d chars", total_chars)
            break
        path = f["path"]
        url  = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            try:
                text = r.content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            snippet = text[:FILE_CONTENT_LIMIT]
            results.append({"path": path, "content": snippet})
            total_chars += len(snippet)
        except requests.RequestException as e:
            log.warning("Could not fetch %s: %s", path, e)
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

    # Trim tree to 80 entries for small-context models
    tree_str = "\n".join(f"  {p}" for p in all_paths[:80])
    if len(all_paths) > 80:
        tree_str += f"\n  ... and {len(all_paths) - 80} more files"

    files_section = ""
    for f in files:
        files_section += f"\n### {f['path']}\n```\n{f['content']}\n```\n"

    return f"""You are a senior software engineer. Analyse this GitHub repository and return ONLY a valid JSON object — no explanation, no markdown fences.

Repository: {meta['full_name']}
Description: {meta.get('description') or 'None'}
Language: {meta.get('language') or 'Unknown'} | All: {lang_str}
Stars: {meta.get('stargazers_count', 0)} | License: {(meta.get('license') or {}).get('name', 'None')}

FILE TREE:
{tree_str}

README:
{readme if readme else '(none)'}

KEY FILES:
{files_section if files_section else '(none)'}

Return this exact JSON structure:
{{"summary":"3-5 sentence description of what the project does and who it is for","technologies":["tech1","tech2"],"structure":"1-2 sentences on project layout and entry points"}}"""


# ── Ollama LLM call ────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> dict:
    """
    Call the local Ollama API (non-streaming).
    Returns parsed dict with summary, technologies, structure.
    Raises RuntimeError on connection failure or bad response.
    """
    # Check Ollama is reachable before sending the full prompt
    try:
        ping = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure it is running: ollama serve"
        )

    # Check the model is actually pulled
    available = [m["name"] for m in ping.json().get("models", [])]
    # Normalise: "phi3:mini" matches "phi3:mini" or just check prefix
    model_available = any(
        m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL.split(":")[0])
        for m in available
    )
    if not model_available:
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' is not pulled. "
            f"Run: ollama pull {OLLAMA_MODEL}  "
            f"(available: {', '.join(available) or 'none'})"
        )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temp → consistent JSON output
            "num_predict": 512,   # max tokens to generate
        },
    }

    log.info("Calling Ollama model '%s' …", OLLAMA_MODEL)
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=180,   # local models can be slow on CPU — give them time
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama timed out. The model may be too slow for this machine. "
            f"Try a smaller model: OLLAMA_MODEL=gemma2:2b python app_ollama.py"
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    if r.status_code != 200:
        raise RuntimeError(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")

    raw_text = r.json().get("response", "").strip()
    log.info("Ollama response length: %d chars", len(raw_text))

    # Strip markdown fences if the model added them
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text.strip())

    # Parse JSON
    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract a JSON object from noisy output
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                raise RuntimeError(f"Could not parse JSON from model output: {raw_text[:300]}")
        else:
            raise RuntimeError(f"Model returned non-JSON output: {raw_text[:300]}")

    # Validate required fields
    for field in ("summary", "technologies", "structure"):
        if field not in result:
            raise RuntimeError(f"Model response missing field '{field}'. Got: {result}")

    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/summarize")
def summarize():
    body = request.get_json(silent=True)
    if not body or "github_url" not in body:
        return jsonify({"status": "error", "message": "Request body must include 'github_url'."}), 400

    github_url = body["github_url"]
    parsed     = parse_owner_repo(github_url)
    if not parsed:
        return jsonify({"status": "error", "message": f"Invalid GitHub URL: '{github_url}'"}), 400

    owner, repo = parsed
    log.info("Request: %s/%s", owner, repo)

    try:
        ctx = fetch_repo_context(owner, repo)
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 502

    try:
        prompt = build_prompt(ctx)
        result = call_llm(prompt)
    except RuntimeError as e:
        return jsonify({"status": "error", "message": str(e)}), 502

    log.info("Done: %s/%s", owner, repo)
    return jsonify(result), 200


@app.get("/health")
def health():
    """Check both server and Ollama status."""
    status = {"status": "ok", "model": OLLAMA_MODEL, "ollama_host": OLLAMA_HOST}
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        status["ollama"] = "connected"
        status["available_models"] = models
    except Exception:
        status["ollama"] = "unreachable"
        status["status"] = "degraded"
    return jsonify(status), 200


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting server on http://localhost:8000")
    log.info("Using Ollama model: %s  (override with OLLAMA_MODEL env var)", OLLAMA_MODEL)
    log.info("Ollama host: %s  (override with OLLAMA_HOST env var)", OLLAMA_HOST)
    app.run(host="0.0.0.0", port=8000, debug=False)
