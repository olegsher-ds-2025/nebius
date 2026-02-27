"""
GitHub Repository Summarizer — Local Edition with Model Comparison
Fetches repo data via GitHub API (requests) and summarizes with local Ollama models.

Usage:
    # Single model (default: llama3.2)
    python summarize.py https://github.com/olegsher-ds-2025/langchain

    # Specific model
    python summarize.py https://github.com/olegsher-ds-2025/langchain --model phi3:mini

    # Compare ALL models installed in Ollama
    python summarize.py https://github.com/olegsher-ds-2025/langchain --compare

    # Compare specific models
    python summarize.py https://github.com/olegsher-ds-2025/langchain --models phi3:mini llama3.2 gemma2:2b

    # With GitHub token (avoids 60 req/hr rate limit)
    python summarize.py https://github.com/olegsher-ds-2025/langchain --compare --token ghp_yourtoken
"""

import argparse
import json
import re
import sys
import time
import textwrap
import requests


# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_API    = "https://api.github.com"
OLLAMA_API    = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
README_LIMIT  = 3000   # chars sent to the model
MAX_FILES     = 60     # files listed from the tree
WRAP_WIDTH    = 70     # text wrap width for summary display


# ── GitHub helpers ────────────────────────────────────────────────────────────

def parse_owner_repo(url: str) -> tuple[str, str]:
    """Extract owner and repo name from a GitHub URL."""
    url = url.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/?\s]+)", url)
    if not match:
        sys.exit(f"[ERROR] Cannot parse GitHub URL: {url}")
    owner = match.group(1)
    repo  = match.group(2).removesuffix(".git")
    return owner, repo


def gh_get(path: str, token: str | None) -> requests.Response:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(f"{GITHUB_API}{path}", headers=headers, timeout=15)


def fetch_repo_data(owner: str, repo: str, token: str | None) -> dict:
    """Fetch repo metadata, README, file tree, and language stats from GitHub."""
    print(f"\n[GitHub] Fetching metadata for {owner}/{repo} …")
    meta_r = gh_get(f"/repos/{owner}/{repo}", token)
    if meta_r.status_code == 404:
        sys.exit(f"[ERROR] Repo not found: {owner}/{repo}")
    if meta_r.status_code != 200:
        sys.exit(f"[ERROR] GitHub API returned {meta_r.status_code}: {meta_r.text}")
    meta = meta_r.json()

    print("[GitHub] Fetching README …")
    readme_headers = {"Accept": "application/vnd.github.raw"}
    if token:
        readme_headers["Authorization"] = f"Bearer {token}"
    readme_r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/readme",
        headers=readme_headers, timeout=15,
    )
    readme = readme_r.text if readme_r.status_code == 200 else ""

    print("[GitHub] Fetching file tree …")
    tree_r = gh_get(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1", token)
    files: list[str] = []
    if tree_r.status_code == 200:
        files = [
            item["path"] for item in tree_r.json().get("tree", [])
            if item["type"] == "blob"
        ]

    print("[GitHub] Fetching languages …")
    lang_r  = gh_get(f"/repos/{owner}/{repo}/languages", token)
    languages: dict = lang_r.json() if lang_r.status_code == 200 else {}

    return {"meta": meta, "readme": readme, "files": files, "languages": languages}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(data: dict) -> str:
    meta         = data["meta"]
    readme       = data["readme"][:README_LIMIT]
    files        = data["files"][:MAX_FILES]
    languages    = data["languages"]
    topics       = ", ".join(meta.get("topics", [])) or "none"
    license_name = (meta.get("license") or {}).get("name", "None")
    lang_summary = ", ".join(f"{k} ({v:,} bytes)" for k, v in languages.items()) if languages else "Unknown"
    file_list    = "\n".join(f"  - {f}" for f in files) if files else "  (no files found)"

    return f"""You are a helpful technical writer. Write a clear, human-friendly summary of this GitHub repository for a developer who has never seen it before.

## Repository Metadata
- **Full name**: {meta['full_name']}
- **Description**: {meta.get('description') or 'No description provided'}
- **Primary language**: {meta.get('language') or 'Unknown'}
- **All languages**: {lang_summary}
- **Topics/tags**: {topics}
- **Stars**: {meta.get('stargazers_count', 0):,}
- **Forks**: {meta.get('forks_count', 0):,}
- **Open issues**: {meta.get('open_issues_count', 0):,}
- **License**: {license_name}
- **Default branch**: {meta.get('default_branch', 'main')}
- **Last updated**: {meta.get('updated_at', 'Unknown')}

## File Structure (first {MAX_FILES} files)
{file_list}

## README (first {README_LIMIT} characters)
{readme if readme else 'No README available.'}

---

Please write a concise summary (4–6 sentences) covering:
1. What this project is and what problem it solves
2. Who it's intended for and key use cases
3. The main technologies or frameworks used
4. Any notable structure or design choices you can infer from the files

Write in plain, friendly English. Avoid bullet points — use flowing prose.
"""


# ── Ollama helpers ────────────────────────────────────────────────────────────

def get_installed_models() -> list[str]:
    """Return list of model names currently pulled in Ollama."""
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=10)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        pass
    return []


def check_ollama_running(model: str) -> None:
    """Exit with a helpful message if Ollama isn't reachable."""
    try:
        requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
    except requests.exceptions.ConnectionError:
        sys.exit(
            "\n[ERROR] Cannot connect to Ollama at http://localhost:11434\n"
            "        Start it with:  ollama serve\n"
            f"        Then pull:     ollama pull {model}\n"
        )


# ── Core: run one model ───────────────────────────────────────────────────────

def run_model(prompt: str, model: str) -> dict:
    """
    Run a single model and return a result dict with:
      summary, elapsed_sec, tokens_generated, tokens_per_sec, error
    """
    print(f"\n{'─' * 60}")
    print(f"  Model: {model}")
    print(f"{'─' * 60}")

    payload = {"model": model, "prompt": prompt, "stream": True}

    try:
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json=payload, stream=True, timeout=300,
        )
    except requests.exceptions.ConnectionError as e:
        return {"model": model, "summary": "", "elapsed_sec": 0,
                "tokens_generated": 0, "tokens_per_sec": 0.0, "error": str(e)}

    if response.status_code == 404:
        msg = f"Model '{model}' not found. Pull it with: ollama pull {model}"
        print(f"  [SKIP] {msg}")
        return {"model": model, "summary": "", "elapsed_sec": 0,
                "tokens_generated": 0, "tokens_per_sec": 0.0, "error": msg}

    if response.status_code != 200:
        msg = f"Ollama returned HTTP {response.status_code}"
        print(f"  [ERROR] {msg}")
        return {"model": model, "summary": "", "elapsed_sec": 0,
                "tokens_generated": 0, "tokens_per_sec": 0.0, "error": msg}

    tokens: list[str] = []
    tokens_generated  = 0
    eval_duration_ns  = 0   # Ollama reports this natively
    t_start           = time.perf_counter()

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        token = chunk.get("response", "")
        print(token, end="", flush=True)
        tokens.append(token)

        if chunk.get("done"):
            # Ollama final chunk carries timing info
            tokens_generated = chunk.get("eval_count", len(tokens))
            eval_duration_ns = chunk.get("eval_duration", 0)
            break

    elapsed_sec   = time.perf_counter() - t_start
    # Prefer Ollama's own eval_duration for tokens/sec accuracy
    if eval_duration_ns > 0:
        tokens_per_sec = tokens_generated / (eval_duration_ns / 1e9)
    elif elapsed_sec > 0:
        tokens_per_sec = tokens_generated / elapsed_sec
    else:
        tokens_per_sec = 0.0

    summary = "".join(tokens).strip()
    print()  # newline after streamed output

    return {
        "model":            model,
        "summary":          summary,
        "elapsed_sec":      elapsed_sec,
        "tokens_generated": tokens_generated,
        "tokens_per_sec":   tokens_per_sec,
        "error":            None,
    }


# ── Comparison report ─────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if max_value == 0:
        filled = 0
    else:
        filled = round((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


def print_comparison(results: list[dict]) -> None:
    """Print a side-by-side performance and quality comparison table."""
    ok = [r for r in results if not r["error"]]
    if not ok:
        print("\n[Comparison] No successful results to compare.")
        return

    max_speed  = max(r["tokens_per_sec"]   for r in ok) or 1
    max_tokens = max(r["tokens_generated"] for r in ok) or 1
    max_time   = max(r["elapsed_sec"]      for r in ok) or 1
    max_chars  = max(len(r["summary"])     for r in ok) or 1

    W = 72
    print(f"\n\n{'═' * W}")
    print(f"  📊  MODEL COMPARISON REPORT")
    print(f"{'═' * W}")

    # ── Performance table ──────────────────────────────────────────────────
    print(f"\n  {'PERFORMANCE':}")
    print(f"  {'Model':<22} {'Time (s)':>8}  {'Tokens':>7}  {'Tok/sec':>8}  {'Speed':}")
    print(f"  {'─' * 22} {'─' * 8}  {'─' * 7}  {'─' * 8}  {'─' * 20}")

    for r in ok:
        bar = _bar(r["tokens_per_sec"], max_speed)
        print(
            f"  {r['model']:<22} {r['elapsed_sec']:>7.1f}s"
            f"  {r['tokens_generated']:>7}  {r['tokens_per_sec']:>7.1f}/s"
            f"  {bar}"
        )

    # ── Content / quality table ────────────────────────────────────────────
    print(f"\n  {'OUTPUT QUALITY INDICATORS':}")
    print(f"  {'Model':<22} {'Chars':>6}  {'Sentences':>9}  {'Depth':}")
    print(f"  {'─' * 22} {'─' * 6}  {'─' * 9}  {'─' * 20}")

    for r in ok:
        char_count = len(r["summary"])
        sentences  = r["summary"].count(".") + r["summary"].count("!") + r["summary"].count("?")
        depth_bar  = _bar(char_count, max_chars)
        print(
            f"  {r['model']:<22} {char_count:>6}"
            f"  {sentences:>9}  {depth_bar}"
        )

    # ── Fastest / most detailed ────────────────────────────────────────────
    fastest   = max(ok, key=lambda r: r["tokens_per_sec"])
    most_det  = max(ok, key=lambda r: len(r["summary"]))
    print(f"\n  ⚡ Fastest       : {fastest['model']} ({fastest['tokens_per_sec']:.1f} tok/s)")
    print(f"  📝 Most detailed : {most_det['model']} ({len(most_det['summary'])} chars)")

    # ── Full summaries side by side ────────────────────────────────────────
    print(f"\n{'═' * W}")
    print(f"  📄  SUMMARIES SIDE BY SIDE")
    print(f"{'═' * W}")

    for r in ok:
        print(f"\n  ┌─ {r['model']} "
              f"({r['elapsed_sec']:.1f}s · {r['tokens_per_sec']:.1f} tok/s · "
              f"{len(r['summary'])} chars) {'─' * max(0, W - 60)}┐")
        wrapped = textwrap.fill(r["summary"], width=WRAP_WIDTH,
                                initial_indent="  │  ", subsequent_indent="  │  ")
        print(wrapped)
        print(f"  └{'─' * (W - 2)}┘")

    # ── Skipped models ─────────────────────────────────────────────────────
    failed = [r for r in results if r["error"]]
    if failed:
        print(f"\n  ⚠  Skipped models:")
        for r in failed:
            print(f"     • {r['model']}: {r['error']}")

    print(f"\n{'═' * W}\n")


# ── Single-model report ───────────────────────────────────────────────────────

def print_report(data: dict, result: dict) -> None:
    meta      = data["meta"]
    languages = data["languages"]
    files     = data["files"]
    r         = result

    print(f"\n{'═' * 60}")
    print(f"  REPO SUMMARY: {meta['full_name']}")
    print(f"{'═' * 60}")
    print(f"  URL      : {meta['html_url']}")
    print(f"  Stars    : {meta.get('stargazers_count', 0):,}")
    print(f"  Forks    : {meta.get('forks_count', 0):,}")
    print(f"  Language : {meta.get('language') or 'Unknown'}")
    if languages:
        print(f"  All langs: {', '.join(languages.keys())}")
    print(f"  Topics   : {', '.join(meta.get('topics', [])) or 'none'}")
    print(f"  Files    : {len(files)} found")
    print(f"  Updated  : {meta.get('updated_at', 'Unknown')}")
    print(f"{'─' * 60}")
    print(f"  Model    : {r['model']}")
    print(f"  Time     : {r['elapsed_sec']:.1f}s")
    print(f"  Tokens   : {r['tokens_generated']} ({r['tokens_per_sec']:.1f} tok/s)")
    print(f"{'─' * 60}")
    print("\n📝 SUMMARY\n")
    print(textwrap.fill(r["summary"], width=WRAP_WIDTH))
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a GitHub repo using local Ollama models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="GitHub repository URL")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Single Ollama model to use (default: {DEFAULT_MODEL})",
    )
    model_group.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="List of models to compare, e.g. --models phi3:mini llama3.2 gemma2:2b",
    )
    model_group.add_argument(
        "--compare", action="store_true",
        help="Auto-detect and compare ALL models currently installed in Ollama",
    )

    parser.add_argument("--token", default=None,
                        help="GitHub personal access token (avoids rate limits)")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Print the generated prompt and exit")
    args = parser.parse_args()

    # ── Resolve model list ─────────────────────────────────────────────────
    if args.compare:
        check_ollama_running(DEFAULT_MODEL)
        model_list = get_installed_models()
        if not model_list:
            sys.exit("[ERROR] No models found in Ollama. Pull one with: ollama pull phi3:mini")
        print(f"\n[Ollama] Found {len(model_list)} installed model(s): {', '.join(model_list)}")
    elif args.models:
        model_list = args.models
    else:
        model_list = [args.model]

    check_ollama_running(model_list[0])

    # ── Fetch repo (once, shared across all models) ────────────────────────
    owner, repo = parse_owner_repo(args.url)
    data        = fetch_repo_data(owner, repo, args.token)
    prompt      = build_prompt(data)

    if args.prompt_only:
        print(prompt)
        return

    # ── Run models ─────────────────────────────────────────────────────────
    multi_mode = len(model_list) > 1
    results: list[dict] = []

    if multi_mode:
        print(f"\n[Run] Comparing {len(model_list)} model(s): {', '.join(model_list)}")

    for model in model_list:
        result = run_model(prompt, model)
        results.append(result)

    # ── Output ─────────────────────────────────────────────────────────────
    if multi_mode:
        print_comparison(results)
    else:
        print_report(data, results[0])


if __name__ == "__main__":
    main()
