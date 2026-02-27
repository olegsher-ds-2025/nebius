# The goal

Welcome, and thank you for your interest in the AI Performance Engineering course, starting on March 17 in Tel Aviv.

We’re inviting you to complete the admission assignment designed to help you understand whether the course is a good fit for your background, while also giving us a clearer sense of your experience. Also, please keep in mind that the course will be held entirely in English.

Working on our practical task, you'll build a simple API service that takes a GitHub repository URL and returns a human-readable summary of the project.

Use this personal ID number assigned to you when submitting your solution: .

Please plan for 4 to 7 hours approximately to complete the admission assignment (depending on the tools and your experience). Make sure to complete it by February 28.



# 🔍 GitHub Repo Summarizer

A lightweight command-line tool that fetches a GitHub repository's metadata, README, and file structure — then generates a human-readable summary using a **local LLM via [Ollama](https://ollama.com)**. Supports running **multiple models in one pass** with a side-by-side performance and quality comparison. No cloud APIs, no API keys, no cost per run.

---

## How It Works

1. Calls the **GitHub REST API** (using Python `requests`) to collect repo metadata, README content, language breakdown, and file tree
2. Builds a structured prompt from all that context
3. Runs one or more **locally installed Ollama models** and streams each response
4. Prints a **comparison report** with timing, token throughput, and output depth metrics

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download/windows) installed and running
- At least one pulled Ollama model

---

## Installation

```bash
git clone https://github.com/your-username/github-repo-summarizer
cd github-repo-summarizer
pip install -r requirements.txt
```

Install Ollama, then pull one or more models:

```bash
ollama pull phi3:mini
ollama pull llama3.2
ollama pull gemma2:2b
```

---

## Usage

```bash
python summarize.py <github-repo-url> [options]
```

### Examples

```bash
# Single model (default: llama3.2)
python summarize.py https://github.com/olegsher-ds-2025/langchain

# Single specific model
python summarize.py https://github.com/olegsher-ds-2025/langchain --model phi3:mini

# Compare specific models side by side
python summarize.py https://github.com/olegsher-ds-2025/langchain --models phi3:mini llama3.2 gemma2:2b

# Auto-detect and compare ALL models installed in Ollama
python summarize.py https://github.com/olegsher-ds-2025/langchain --compare

# With a GitHub token (avoids 60 req/hr rate limit)
python summarize.py https://github.com/olegsher-ds-2025/langchain --compare --token ghp_yourtoken

# Print the prompt without calling any LLM (debugging)
python summarize.py https://github.com/olegsher-ds-2025/langchain --prompt-only
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `url` | *(required)* | GitHub repository URL |
| `--model NAME` | `llama3.2` | Single model to use |
| `--models A B C` | — | Compare specific models |
| `--compare` | — | Auto-detect and compare all installed Ollama models |
| `--token TOKEN` | None | GitHub personal access token |
| `--prompt-only` | False | Print prompt and exit, no LLM call |

`--model`, `--models`, and `--compare` are mutually exclusive.

---

## Example Comparison Output

```
════════════════════════════════════════════════════════════════════════
  📊  MODEL COMPARISON REPORT
════════════════════════════════════════════════════════════════════════

  PERFORMANCE
  Model                   Time (s)   Tokens   Tok/sec  Speed
  ──────────────────────  ────────   ──────   ───────  ────────────────────
  gemma2:2b                   8.2s      112    13.7/s  ████████████░░░░░░░░
  phi3:mini                  11.4s      138    12.1/s  ██████████░░░░░░░░░░
  llama3.2                   18.7s      151     8.1/s  ██████░░░░░░░░░░░░░░

  OUTPUT QUALITY INDICATORS
  Model                  Chars  Sentences  Depth
  ──────────────────────  ─────  ─────────  ────────────────────
  gemma2:2b                 612          5  ████████████░░░░░░░░
  phi3:mini                 789          6  ████████████████░░░░
  llama3.2                  834          6  ████████████████████

  ⚡ Fastest       : gemma2:2b (13.7 tok/s)
  📝 Most detailed : llama3.2 (834 chars)

════════════════════════════════════════════════════════════════════════
  📄  SUMMARIES SIDE BY SIDE
════════════════════════════════════════════════════════════════════════

  ┌─ gemma2:2b (8.2s · 13.7 tok/s · 612 chars) ──────────────────┐
  │  This repository is a hands-on LangChain learning project ...
  └────────────────────────────────────────────────────────────────┘

  ┌─ phi3:mini (11.4s · 12.1 tok/s · 789 chars) ─────────────────┐
  │  This project serves as a personal study collection ...
  └────────────────────────────────────────────────────────────────┘

  ┌─ llama3.2 (18.7s · 8.1 tok/s · 834 chars) ───────────────────┐
  │  The olegsher-ds-2025/langchain repository appears to be ...
  └────────────────────────────────────────────────────────────────┘
```

---

## Choosing a Model

The tool runs fully on CPU — no GPU required. For **Intel Xe / integrated graphics**, Ollama uses CPU automatically.

| Model | Size | RAM | Tok/sec (CPU) | Best for |
|-------|------|-----|---------------|----------|
| `gemma2:2b` | 1.6 GB | 4 GB | ~14/s | Fastest responses |
| `phi3:mini` | 2.2 GB | 4 GB | ~12/s | Technical/code content ⭐ |
| `llama3.2` | 2.0 GB | 4 GB | ~8/s | Most detailed output |
| `mistral` | 4.1 GB | 8 GB | ~5/s | Highest quality, slower |

> Token/sec figures are approximate CPU estimates. Your actual speed depends on your CPU and available RAM.

---

## GitHub API Rate Limits

Without a token: **60 requests/hour**. Each run makes 4 API calls (repo, README, tree, languages), so you'd hit the limit after ~15 runs.

To raise the limit to **5,000 requests/hour**:
1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. No scopes needed for public repos — just generate and copy
4. Use: `--token ghp_yourtoken`

---

## Project Structure

```
github-repo-summarizer/
├── summarize.py       # Main script
├── requirements.txt   # Python dependencies (just requests)
└── README.md
```

---

## Dependencies

```
requests>=2.31.0
```

Ollama is a standalone application, not a Python package.

---

## License

MIT
