# GitHub Repository Summarizer API

A Flask API that takes a GitHub repository URL, intelligently extracts the most relevant content, and returns a structured summary powered by an LLM.

```
POST /summarize
```

```json
{
  "summary": "What the project does...",
  "technologies": ["Python", "Flask", "..."],
  "structure": "How the project is organised..."
}
```

---

## Setup & Run

**Requirements:** Python 3.10+

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/github-repo-summarizer
cd github-repo-summarizer
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Windows (Command Prompt)
set NEBIUS_API_KEY=your_key_here

# Windows (PowerShell)
$env:NEBIUS_API_KEY = "your_key_here"

# macOS / Linux
export NEBIUS_API_KEY=your_key_here
```

> **Optional:** Set `GITHUB_TOKEN` the same way to raise the GitHub API rate limit from 60 to 5,000 req/hr. Not required, but useful for heavy testing.

### 3. Start the server

```bash
python app.py
```

The server starts on `http://localhost:8000`.

### 4. Test it

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

Expected response:

```json
{
  "summary": "Requests is a popular Python HTTP library...",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer"],
  "structure": "The main source code lives in src/requests/. Tests are in tests/ and documentation in docs/."
}
```

---

## Design Decisions

### Model choice

**`meta-llama/Meta-Llama-3.1-70B-Instruct`** via Nebius Token Factory.

This model produces accurate, structured JSON output reliably, has strong code comprehension, and fits comfortably within the free credit allocation. Its 128k context window also gives us room to include meaningful file content without aggressive truncation.

### Repository content strategy

Sending an entire repository to an LLM is impractical — large repos can be hundreds of MB. The strategy used here prioritises signal over volume:

**What gets included:**

| Content | Limit | Rationale |
|---------|-------|-----------|
| Repository metadata | full | Stars, language, license, description |
| README | 5,000 chars | Usually the best single-source description of a project |
| File tree | up to 150 paths | Gives the LLM a structural overview without content cost |
| Language breakdown | full | From GitHub's API — reliable, cheap |
| Source files | up to 28,000 chars total, 3,000 per file | Actual code for deeper understanding |

**What gets skipped:**

- **Dependency directories** — `node_modules/`, `.venv/`, `vendor/`, etc. (never relevant to understanding the project)
- **Lock files** — `package-lock.json`, `poetry.lock`, etc. (machine-generated, zero signal)
- **Binary and media files** — images, executables, compiled artifacts
- **Large data files** — `.csv`, `.parquet`, `.pkl`, notebooks (too large, not structural)
- **Minified files** — `.min.js`, `.min.css` (unreadable)
- **Files > 100 KB** — almost certainly generated or data files

**Prioritisation order within the budget:**

1. README (always first)
2. Entry points: `main.py`, `app.py`, `server.py`, `index.js`, etc.
3. Project config: `pyproject.toml`, `package.json`, `Dockerfile`, `Makefile`
4. Dependency manifests: `requirements.txt`, `go.mod`, `Cargo.toml`
5. All other source files, ordered by directory depth (root files first)

This means even if the budget fills up early, the LLM always sees the most informative files first.

**Context budget:** 28,000 characters (~7,000 tokens) reserved for file content, leaving headroom for the prompt template and the model's 1,024-token response within a 32k context window.

---

## Error Responses

| Scenario | HTTP | Response |
|----------|------|----------|
| Missing `github_url` field | 400 | `{"status": "error", "message": "..."}` |
| Invalid GitHub URL format | 400 | `{"status": "error", "message": "..."}` |
| Repo not found / private | 404 | `{"status": "error", "message": "..."}` |
| GitHub API rate limit | 502 | `{"status": "error", "message": "..."}` |
| Missing `NEBIUS_API_KEY` | 502 | `{"status": "error", "message": "..."}` |
| LLM API failure | 502 | `{"status": "error", "message": "..."}` |

---

## Project Structure

```
github-repo-summarizer/
├── app.py            # Flask application — all logic in one readable file
├── requirements.txt  # Two dependencies: flask, requests
└── README.md
```

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"}
```
