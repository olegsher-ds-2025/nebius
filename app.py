"""
GitHub Repo Summarizer — Streamlit UI
Run with: streamlit run app.py
"""

import json
import re
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_API   = "https://api.github.com"
OLLAMA_API   = "http://localhost:11434"
README_LIMIT = 3000
MAX_FILES    = 60

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GitHub Repo Summarizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #313244;
    }
    .summary-box {
        background: #181825;
        border-left: 4px solid #89b4fa;
        border-radius: 4px;
        padding: 16px;
        margin: 8px 0;
        line-height: 1.7;
    }
    .model-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        margin-bottom: 4px;
    }
    .stat-label { color: #a6adc8; font-size: 0.8rem; }
    .winner-badge {
        background: #a6e3a1;
        color: #1e1e2e;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def parse_owner_repo(url: str) -> tuple[str, str] | None:
    url = url.strip().rstrip("/")
    match = re.search(r"github\.com/([^/]+)/([^/?\s]+)", url)
    if not match:
        return None
    owner = match.group(1)
    repo  = match.group(2).removesuffix(".git")
    return owner, repo


def gh_get(path: str, token: str | None) -> requests.Response:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(f"{GITHUB_API}{path}", headers=headers, timeout=15)


@st.cache_data(show_spinner=False)
def fetch_repo_data(owner: str, repo: str, token: str | None) -> dict | str:
    """Fetch repo data; returns dict on success, error string on failure."""
    meta_r = gh_get(f"/repos/{owner}/{repo}", token)
    if meta_r.status_code == 404:
        return f"Repository '{owner}/{repo}' not found."
    if meta_r.status_code != 200:
        return f"GitHub API error: {meta_r.status_code}"
    meta = meta_r.json()

    readme_headers = {"Accept": "application/vnd.github.raw"}
    if token:
        readme_headers["Authorization"] = f"Bearer {token}"
    readme_r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{repo}/readme",
        headers=readme_headers, timeout=15,
    )
    readme = readme_r.text if readme_r.status_code == 200 else ""

    tree_r = gh_get(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1", token)
    files: list[str] = []
    if tree_r.status_code == 200:
        files = [
            item["path"] for item in tree_r.json().get("tree", [])
            if item["type"] == "blob"
        ]

    lang_r    = gh_get(f"/repos/{owner}/{repo}/languages", token)
    languages = lang_r.json() if lang_r.status_code == 200 else {}

    return {"meta": meta, "readme": readme, "files": files, "languages": languages}


# ── Ollama helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def get_installed_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def ollama_running() -> bool:
    try:
        requests.get(f"{OLLAMA_API}/api/tags", timeout=3)
        return True
    except Exception:
        return False


# ── Prompt builder ─────────────────────────────────────────────────────────────

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
- **License**: {license_name}
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


def stream_model(prompt: str, model: str):
    """Generator: yields (token_str, final_stats_dict_or_None) tuples."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json=payload, stream=True, timeout=300,
        )
    except requests.exceptions.ConnectionError as e:
        yield "", {"error": str(e)}
        return

    if response.status_code != 200:
        yield "", {"error": f"HTTP {response.status_code}"}
        return

    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        token = chunk.get("response", "")
        if chunk.get("done"):
            tokens_generated = chunk.get("eval_count", 0)
            eval_ns          = chunk.get("eval_duration", 0)
            tokens_per_sec   = (tokens_generated / (eval_ns / 1e9)) if eval_ns > 0 else 0
            yield token, {
                "tokens_generated": tokens_generated,
                "tokens_per_sec":   tokens_per_sec,
                "error":            None,
            }
        else:
            yield token, None


# ── Charts ─────────────────────────────────────────────────────────────────────

def make_speed_chart(results: list[dict]) -> go.Figure:
    ok = [r for r in results if not r.get("error")]
    df = pd.DataFrame({
        "Model":   [r["model"] for r in ok],
        "tok/sec": [round(r["tokens_per_sec"], 1) for r in ok],
    }).sort_values("tok/sec", ascending=True)

    fig = px.bar(
        df, x="tok/sec", y="Model", orientation="h",
        title="⚡ Speed — Tokens per Second",
        color="tok/sec",
        color_continuous_scale="Blues",
        text="tok/sec",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=40, b=10), height=300,
    )
    fig.update_xaxes(showgrid=False, color="#a6adc8")
    fig.update_yaxes(showgrid=False, color="#a6adc8")
    return fig


def make_time_chart(results: list[dict]) -> go.Figure:
    ok = [r for r in results if not r.get("error")]
    df = pd.DataFrame({
        "Model":    [r["model"] for r in ok],
        "Time (s)": [round(r["elapsed_sec"], 1) for r in ok],
    }).sort_values("Time (s)", ascending=True)

    fig = px.bar(
        df, x="Time (s)", y="Model", orientation="h",
        title="🕒 Total Response Time (seconds)",
        color="Time (s)",
        color_continuous_scale="Reds_r",
        text="Time (s)",
    )
    fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig.update_layout(
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4", coloraxis_showscale=False,
        margin=dict(l=10, r=40, t=40, b=10), height=300,
    )
    fig.update_xaxes(showgrid=False, color="#a6adc8")
    fig.update_yaxes(showgrid=False, color="#a6adc8")
    return fig


def make_quality_chart(results: list[dict]) -> go.Figure:
    ok = [r for r in results if not r.get("error")]
    df = pd.DataFrame({
        "Model":     [r["model"] for r in ok],
        "Chars":     [len(r["summary"]) for r in ok],
        "Sentences": [
            r["summary"].count(".") + r["summary"].count("!") + r["summary"].count("?")
            for r in ok
        ],
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Characters", x=df["Model"], y=df["Chars"],
        marker_color="#89b4fa", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Sentences", x=df["Model"], y=df["Sentences"],
        mode="lines+markers+text", text=df["Sentences"],
        textposition="top center",
        marker=dict(color="#a6e3a1", size=10),
        line=dict(color="#a6e3a1", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        title="📝 Output Depth — Characters & Sentences",
        plot_bgcolor="#1e1e2e", paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        yaxis=dict(title="Characters", color="#89b4fa", showgrid=False),
        yaxis2=dict(title="Sentences", overlaying="y", side="right",
                    color="#a6e3a1", showgrid=False),
        legend=dict(bgcolor="#313244", bordercolor="#45475a"),
        margin=dict(l=10, r=60, t=40, b=10), height=300,
        barmode="group",
    )
    fig.update_xaxes(showgrid=False, color="#a6adc8")
    return fig


def make_radar_chart(results: list[dict]) -> go.Figure:
    """Normalised radar: speed, detail, efficiency (tokens/char ratio inverted)."""
    ok = [r for r in results if not r.get("error")]
    if len(ok) < 2:
        return None

    max_speed  = max(r["tokens_per_sec"]   for r in ok) or 1
    max_chars  = max(len(r["summary"])      for r in ok) or 1
    max_sents  = max(
        r["summary"].count(".") + r["summary"].count("?") + r["summary"].count("!")
        for r in ok
    ) or 1
    min_time   = min(r["elapsed_sec"] for r in ok) or 1

    fig = go.Figure()
    categories = ["Speed", "Detail (chars)", "Sentences", "Responsiveness"]

    colors = ["#89b4fa", "#a6e3a1", "#fab387", "#f38ba8", "#cba6f7"]
    for i, r in enumerate(ok):
        sents = r["summary"].count(".") + r["summary"].count("?") + r["summary"].count("!")
        vals  = [
            r["tokens_per_sec"]  / max_speed  * 10,
            len(r["summary"])    / max_chars  * 10,
            sents                / max_sents  * 10,
            (min_time / r["elapsed_sec"]) * 10,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself", name=r["model"],
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.25,
        ))

    fig.update_layout(
        title="🕸 Overall Profile (normalised 0–10)",
        polar=dict(
            bgcolor="#1e1e2e",
            radialaxis=dict(visible=True, range=[0, 10], color="#a6adc8", gridcolor="#313244"),
            angularaxis=dict(color="#cdd6f4", gridcolor="#313244"),
        ),
        paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
        legend=dict(bgcolor="#313244", bordercolor="#45475a"),
        margin=dict(l=20, r=20, t=60, b=20), height=380,
    )
    return fig


def make_language_pie(languages: dict) -> go.Figure | None:
    if not languages:
        return None
    df = pd.DataFrame({
        "Language": list(languages.keys()),
        "Bytes":    list(languages.values()),
    })
    fig = px.pie(
        df, names="Language", values="Bytes",
        title="📊 Language Breakdown",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig.update_layout(
        paper_bgcolor="#1e1e2e", font_color="#cdd6f4",
        legend=dict(bgcolor="#313244"),
        margin=dict(l=10, r=10, t=40, b=10), height=300,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Repo Summarizer")
    st.caption("Powered by Ollama · Local LLM")
    st.divider()

    repo_url = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/olegsher-ds-2025/langchain",
        placeholder="https://github.com/owner/repo",
    )
    github_token = st.text_input(
        "GitHub Token (optional)",
        type="password",
        help="Raises rate limit from 60 to 5,000 req/hr. No scopes needed for public repos.",
    )

    st.divider()
    st.subheader("🤖 Model Selection")

    if not ollama_running():
        st.error("❌ Ollama not running\n\nStart it with:\n```\nollama serve\n```")
        installed = []
    else:
        st.success("✅ Ollama connected")
        installed = get_installed_models()

    if installed:
        selected_models = st.multiselect(
            "Models to run",
            options=installed,
            default=installed[:min(3, len(installed))],
            help="Select one or more models to compare",
        )
    else:
        st.warning("No models found.\n\nPull one:\n```\nollama pull phi3:mini\n```")
        selected_models = []

    st.divider()
    run_btn = st.button(
        "▶ Run Analysis",
        type="primary",
        disabled=not (repo_url and selected_models),
        use_container_width=True,
    )

    st.divider()
    st.caption("**Recommended models for Intel Xe / CPU:**")
    st.caption("• `phi3:mini` — best for code/tech")
    st.caption("• `gemma2:2b` — fastest")
    st.caption("• `llama3.2` — most detailed")


# ── Main area ──────────────────────────────────────────────────────────────────

st.title("GitHub Repository Summarizer")
st.caption("Fetch · Analyse · Compare local LLM models")

if not run_btn:
    st.info("👈 Enter a GitHub URL, select models, and click **Run Analysis**")
    st.stop()

# ── Validate URL ───────────────────────────────────────────────────────────────

parsed = parse_owner_repo(repo_url)
if not parsed:
    st.error(f"Could not parse GitHub URL: `{repo_url}`")
    st.stop()
owner, repo_name = parsed

# ── Fetch repo data ────────────────────────────────────────────────────────────

with st.status(f"Fetching `{owner}/{repo_name}` from GitHub …", expanded=True) as status:
    st.write("📡 Fetching metadata, README, file tree, languages …")
    token_val = github_token.strip() or None
    data = fetch_repo_data(owner, repo_name, token_val)

    if isinstance(data, str):
        status.update(label="GitHub fetch failed", state="error")
        st.error(data)
        st.stop()
    status.update(label="✅ Repository data fetched", state="complete", expanded=False)

meta      = data["meta"]
languages = data["languages"]
files     = data["files"]

# ── Repo info header ───────────────────────────────────────────────────────────

st.subheader(f"📦 {meta['full_name']}")
if meta.get("description"):
    st.markdown(f"*{meta['description']}*")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("⭐ Stars",      f"{meta.get('stargazers_count', 0):,}")
c2.metric("🍴 Forks",      f"{meta.get('forks_count', 0):,}")
c3.metric("🐛 Issues",     f"{meta.get('open_issues_count', 0):,}")
c4.metric("📁 Files",      len(files))
c5.metric("🗣 Language",   meta.get("language") or "—")

col_lang, col_topics = st.columns(2)
with col_lang:
    if languages:
        fig_pie = make_language_pie(languages)
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
with col_topics:
    topics = meta.get("topics", [])
    if topics:
        st.markdown("**Topics**")
        st.markdown(" ".join(f"`{t}`" for t in topics))
    st.markdown(f"**License:** {(meta.get('license') or {}).get('name', 'None')}")
    st.markdown(f"**Branch:** `{meta.get('default_branch', 'main')}`")
    st.markdown(f"**Updated:** {meta.get('updated_at', '')[:10]}")
    st.markdown(f"[Open on GitHub ↗]({meta['html_url']})")

st.divider()

# ── Run models ─────────────────────────────────────────────────────────────────

prompt  = build_prompt(data)
results = []

st.subheader("🤖 Model Outputs")

if len(selected_models) == 1:
    # ── Single model: plain streaming ──────────────────────────────────────
    model = selected_models[0]
    st.markdown(f"**Model:** `{model}`")

    t_start      = time.perf_counter()
    placeholder  = st.empty()
    stats_ph     = st.empty()
    tokens_text  = ""
    final_stats  = {}

    for token, stats in stream_model(prompt, model):
        tokens_text += token
        placeholder.markdown(
            f'<div class="summary-box">{tokens_text}▌</div>',
            unsafe_allow_html=True,
        )
        if stats:
            final_stats = stats

    elapsed = time.perf_counter() - t_start
    placeholder.markdown(
        f'<div class="summary-box">{tokens_text}</div>',
        unsafe_allow_html=True,
    )

    if final_stats.get("error"):
        st.error(f"Error: {final_stats['error']}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("⏱ Time",       f"{elapsed:.1f}s")
        m2.metric("🔢 Tokens",    final_stats.get("tokens_generated", "—"))
        m3.metric("⚡ Tok/sec",   f"{final_stats.get('tokens_per_sec', 0):.1f}")

    results.append({
        "model":            model,
        "summary":          tokens_text,
        "elapsed_sec":      elapsed,
        "tokens_generated": final_stats.get("tokens_generated", 0),
        "tokens_per_sec":   final_stats.get("tokens_per_sec", 0),
        "error":            final_stats.get("error"),
    })

else:
    # ── Multi-model: tab per model, stream sequentially ────────────────────
    tabs = st.tabs([f"🤖 {m}" for m in selected_models])

    for i, model in enumerate(selected_models):
        with tabs[i]:
            st.markdown(f'<div class="model-header">{model}</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            time_ph  = m1.empty()
            tok_ph   = m2.empty()
            speed_ph = m3.empty()
            time_ph.metric("⏱ Time",    "running…")
            tok_ph.metric("🔢 Tokens",  "—")
            speed_ph.metric("⚡ Tok/sec", "—")

            stream_ph = st.empty()
            stream_ph.markdown(
                '<div class="summary-box"><em>Waiting for model…</em></div>',
                unsafe_allow_html=True,
            )

            t_start     = time.perf_counter()
            tokens_text = ""
            final_stats = {}

            for token, stats in stream_model(prompt, model):
                tokens_text += token
                elapsed_now  = time.perf_counter() - t_start
                stream_ph.markdown(
                    f'<div class="summary-box">{tokens_text}▌</div>',
                    unsafe_allow_html=True,
                )
                time_ph.metric("⏱ Time", f"{elapsed_now:.1f}s")
                if stats:
                    final_stats = stats

            elapsed = time.perf_counter() - t_start

            if final_stats.get("error"):
                stream_ph.error(f"Error: {final_stats['error']}")
                tok_ph.metric("🔢 Tokens",   "error")
                speed_ph.metric("⚡ Tok/sec", "error")
                time_ph.metric("⏱ Time",     f"{elapsed:.1f}s")
            else:
                stream_ph.markdown(
                    f'<div class="summary-box">{tokens_text}</div>',
                    unsafe_allow_html=True,
                )
                tok_ph.metric("🔢 Tokens",   final_stats.get("tokens_generated", "—"))
                speed_ph.metric("⚡ Tok/sec", f"{final_stats.get('tokens_per_sec', 0):.1f}")
                time_ph.metric("⏱ Time",     f"{elapsed:.1f}s")

            results.append({
                "model":            model,
                "summary":          tokens_text,
                "elapsed_sec":      elapsed,
                "tokens_generated": final_stats.get("tokens_generated", 0),
                "tokens_per_sec":   final_stats.get("tokens_per_sec", 0),
                "error":            final_stats.get("error"),
            })

# ── Comparison charts (only when 2+ models) ────────────────────────────────────

ok_results = [r for r in results if not r.get("error")]

if len(ok_results) >= 2:
    st.divider()
    st.subheader("📊 Model Comparison")

    # Winner badges
    fastest   = max(ok_results, key=lambda r: r["tokens_per_sec"])
    most_det  = max(ok_results, key=lambda r: len(r["summary"]))
    col_a, col_b = st.columns(2)
    col_a.success(f"⚡ **Fastest:** `{fastest['model']}` — {fastest['tokens_per_sec']:.1f} tok/s")
    col_b.info(   f"📝 **Most detailed:** `{most_det['model']}` — {len(most_det['summary'])} chars")

    # Row 1: speed + time
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(make_speed_chart(results), use_container_width=True)
    with ch2:
        st.plotly_chart(make_time_chart(results), use_container_width=True)

    # Row 2: quality + radar
    ch3, ch4 = st.columns(2)
    with ch3:
        st.plotly_chart(make_quality_chart(results), use_container_width=True)
    with ch4:
        radar = make_radar_chart(results)
        if radar:
            st.plotly_chart(radar, use_container_width=True)

    # Summary table
    st.subheader("📋 Results Table")
    df_table = pd.DataFrame([
        {
            "Model":       r["model"],
            "Time (s)":    round(r["elapsed_sec"], 1),
            "Tokens":      r["tokens_generated"],
            "Tok/sec":     round(r["tokens_per_sec"], 1),
            "Chars":       len(r["summary"]),
            "Sentences":   r["summary"].count(".") + r["summary"].count("?") + r["summary"].count("!"),
            "Status":      "✅" if not r.get("error") else "❌",
        }
        for r in results
    ])
    st.dataframe(df_table, use_container_width=True, hide_index=True)

st.divider()
st.caption("GitHub Repo Summarizer · Local LLM via Ollama · No data sent to the cloud")
