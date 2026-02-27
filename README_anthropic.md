# GitHub Repository Summarizer API — Anthropic Edition

Uses the **Anthropic Claude API** (`claude-haiku-4-5`) to generate summaries.
Fast, accurate, and great at returning structured JSON.

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

**Requirements:** Python 3.10+, an Anthropic API key

Get your key at: **https://console.anthropic.com**

### 1. Clone and install

```bash
git clone https://github.com/your-username/github-repo-summarizer
cd github-repo-summarizer
pip install -r requirements.txt
```

### 2. Set your API key

```cmd
:: Windows Command Prompt
set ANTHROPIC_API_KEY=sk-ant-...

:: Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

:: macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start the server

```bash
python app_anthropic.py
```

### 4. Test it

```cmd
curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" -d "{\"github_url\": \"https://github.com/psf/requests\"}"
```

Expected response (~2–3 seconds):

```json
{
  "summary": "Requests is a simple, elegant Python HTTP library...",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer"],
  "structure": "Main source lives in src/requests/. Tests are in tests/, documentation in docs/."
}
```

---

## Model choice

**`claude-haiku-4-5-20251001`** — Anthropic's fastest and most cost-efficient model.
Ideal for this task: it has a 200k token context window, follows JSON formatting
instructions reliably, and responses typically arrive in 2–4 seconds.
Cost is roughly $0.001–$0.003 per summarize request.

To use a more powerful model, change `ANTHROPIC_MODEL` in `app_anthropic.py`:
- `claude-sonnet-4-6` — better reasoning, ~5–10× more expensive
- `claude-opus-4-6` — highest quality, most expensive

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `GITHUB_TOKEN` | *(none)* | Optional — raises GitHub rate limit to 5,000/hr |

---

## Error Responses

| Scenario | HTTP | Notes |
|----------|------|-------|
| Missing `github_url` | 400 | |
| Invalid GitHub URL | 400 | |
| Repo not found / private | 404 | |
| Missing `ANTHROPIC_API_KEY` | 502 | Set env var before starting |
| Invalid API key | 502 | Check key at console.anthropic.com |
| Rate limit hit | 502 | Wait and retry |

---

## Project Structure

```
github-repo-summarizer/
├── app.py              # Nebius / cloud LLM version
├── app_ollama.py       # Local Ollama version
├── app_anthropic.py    # This file — Anthropic Claude version
├── requirements.txt    # flask, requests
├── README.md
├── README_ollama.md
└── README_anthropic.md
```
