# GitHub Repository Summarizer API — Local LLM Edition

Same API as the Nebius version, but runs **100% locally** using [Ollama](https://ollama.com).  
No API key, no internet connection needed for the LLM, no cost per request.

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

**Requirements:** Python 3.10+, [Ollama](https://ollama.com/download/windows)

### 1. Install Ollama and pull a model

Download Ollama from **https://ollama.com/download/windows** and install it.

Then pull a model. `phi3:mini` is recommended — it's fast on CPU and strong on technical content:

```cmd
ollama pull phi3:mini
```

Verify Ollama is running:
```cmd
ollama list
```

### 2. Clone and install Python dependencies

```bash
git clone https://github.com/your-username/github-repo-summarizer
cd github-repo-summarizer
pip install -r requirements.txt
```

### 3. Start the server

```cmd
python app_ollama.py
```

You'll see:
```
10:00:00  INFO     Starting server on http://localhost:8000
10:00:00  INFO     Using Ollama model: phi3:mini
10:00:00  INFO     Ollama host: http://localhost:11434
```

### 4. Test it

```cmd
curl -X POST http://localhost:8000/summarize -H "Content-Type: application/json" -d "{\"github_url\": \"https://github.com/psf/requests\"}"
```

Expected response (may take 15–60 seconds on CPU):

```json
{
  "summary": "Requests is a simple, elegant Python HTTP library...",
  "technologies": ["Python", "urllib3", "certifi", "charset-normalizer"],
  "structure": "Main source lives in src/requests/. Tests are in tests/, documentation in docs/."
}
```

---

## Configuration

All settings are controlled via environment variables — no code changes needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `phi3:mini` | Model to use for summarization |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `GITHUB_TOKEN` | *(none)* | Optional — raises GitHub rate limit to 5,000/hr |

### Switch models on the fly

```cmd
set OLLAMA_MODEL=llama3.2
python app_ollama.py
```

---

## Model Recommendations

Tested and working with this API. Pull any of these with `ollama pull <model>`:

| Model | Size | RAM | Speed (CPU) | Best for |
|-------|------|-----|-------------|----------|
| `phi3:mini` ⭐ | 2.2 GB | 4 GB | ~12 tok/s | Code & technical repos |
| `gemma2:2b` | 1.6 GB | 4 GB | ~14 tok/s | Fastest — good for quick summaries |
| `llama3.2` | 2.0 GB | 4 GB | ~8 tok/s | Most detailed output |
| `mistral` | 4.1 GB | 8 GB | ~5 tok/s | Highest quality, needs more RAM |

> **Intel Xe / integrated graphics:** Ollama runs on CPU automatically. `phi3:mini` is the recommended choice.

---

## Design Decisions

### Model choice

**`phi3:mini`** (Microsoft) — optimised specifically for code and technical reasoning, runs well on CPU-only machines, 2k–4k context window, and produces reliable JSON output. Its smaller context window (vs cloud models) is why the file content limits are tighter than the Nebius version.

### Context limits vs cloud version

Small local models have smaller context windows (typically 4k–8k tokens vs 32k–128k for cloud models). This version therefore uses:

- **3,000 char README limit** (vs 5,000)
- **1,500 chars per source file** (vs 3,000)
- **8,000 char total file budget** (vs 28,000)
- **80 file tree entries** (vs 150)

The prioritisation strategy is identical: README → entry points → config files → source files by directory depth.

### Why no streaming?

The endpoint returns a complete JSON object, so streaming the LLM output would require buffering it anyway. Using `stream: false` simplifies error handling and JSON parsing.

---

## Health Check

The `/health` endpoint also reports Ollama connectivity and available models:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "phi3:mini",
  "ollama_host": "http://localhost:11434",
  "ollama": "connected",
  "available_models": ["phi3:mini", "llama3.2"]
}
```

---

## Error Responses

| Scenario | HTTP | Notes |
|----------|------|-------|
| Missing `github_url` | 400 | |
| Invalid GitHub URL | 400 | |
| Repo not found / private | 404 | |
| Ollama not running | 502 | Start with `ollama serve` |
| Model not pulled | 502 | Run `ollama pull phi3:mini` |
| Ollama timeout | 502 | Try a smaller/faster model |

---

## Project Structure

```
github-repo-summarizer/
├── app.py             # Nebius / cloud LLM version
├── app_ollama.py      # This file — local Ollama version
├── requirements.txt   # flask, requests
└── README_ollama.md   # This file
```
