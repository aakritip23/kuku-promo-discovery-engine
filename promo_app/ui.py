"""ui.py - Streamlit widgets for the Promo Discovery Engine (layout only; all logic lives in runner.py / results.py)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from promo_app import results as R
from promo_app.runner import DEFAULT_REPLAY, ENGINE_DIR, LLMChoice, RunResult

def _default_folder() -> str:
    """Folder pre-filled in 'Folder on this computer': $PROMO_SAMPLE_DIR, else data/screenplay_input/ if it holds files, else data/demo_input/."""
    env = os.environ.get("PROMO_SAMPLE_DIR")
    if env:
        return env
    for name in ("screenplay_input", "demo_input"):
        d = os.path.join(ENGINE_DIR, "data", name)
        if os.path.isdir(d) and any(f.endswith((".txt", ".zip")) for f in os.listdir(d)):
            return d
    return os.path.join(ENGINE_DIR, "data", "demo_input")


SAMPLE_DIR = _default_folder()
SCORE_COL = st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0.1rem;}
.hero p {color: #5b6470; font-size: 1.05rem; margin-top: 0;}
.quote {border-left: 4px solid #c7ced9; padding: 0.25rem 0.8rem; margin: 0.3rem 0; background: #f6f8fb; border-radius: 0 6px 6px 0; color:#1c2430;}
.quote small {color: #6b7683;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; background:#e8eef7; color:#243b5a; margin-right:6px;}
.pill.warn {background:#fff1d6; color:#7a4b00;} .pill.ok {background:#dff3e4; color:#165c2a;}
</style>
"""


def header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Promo Discovery Engine</h1>'
                '<p>Upload microdrama screenplays and get the best 20-60 second promo clips, ranked, with exact timestamps '
                'and the reason each one works.</p></div>', unsafe_allow_html=True)


def sidebar() -> Tuple[LLMChoice, int]:
    st.sidebar.header("Settings")
    has_replay = os.path.exists(DEFAULT_REPLAY)
    options = ["Rules only (no model)"]
    if has_replay:
        options.append("Saved model verdicts (demo pack only)")
    options += ["Anthropic API", "OpenAI-compatible API (e.g. Gemini)"]
    mode_label = st.sidebar.radio("How should hook / cliffhanger / context be judged?", options,
                                  index=1 if has_replay else 0, key="judge_mode")
    llm = LLMChoice()
    if mode_label.startswith("Saved"):
        llm.mode = "replay"
    elif mode_label.startswith("Anthropic"):
        llm.mode = "anthropic"
        llm.api_key = st.sidebar.text_input("Anthropic API key", type="password")
        llm.model = st.sidebar.text_input("Model (optional)", value=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"))
    elif mode_label.startswith("OpenAI"):
        llm.mode = "openai-compat"
        llm.api_key = st.sidebar.text_input("API key", type="password")
        llm.base_url = st.sidebar.text_input("Base URL", value=os.environ.get("LLM_BASE_URL", ""))
        llm.model = st.sidebar.text_input("Model", value=os.environ.get("LLM_MODEL", ""))
    if llm.mode in ("anthropic", "openai-compat"):
        llm.top_k = st.sidebar.slider("Candidates sent to the model", 10, 120, 40, 5,
                                      help="Each one is one model call. Higher = slower and costlier, but fewer missed clips.")
    st.sidebar.caption({
        "none": "Rules only: fast, free, but the rule-based cliffhanger and context scores are weak. Results are labelled UNVERIFIED.",
        "replay": "Replays model judgements saved for the demo screenplay pack. They only match that exact pack; any other "
                  "screenplay falls back to rules only.",
        "anthropic": "Your key stays in this session's memory and is never written to disk.",
        "openai-compat": "Your key stays in this session's memory and is never written to disk."}[llm.mode])
    top_n = st.sidebar.slider("Promos to show", 3, 15, 10)
    return llm, top_n


def input_panel() -> Tuple[str, Any]:
    """Returns (kind, payload): ('upload', [files]) or ('folder', path)."""
    st.subheader("1. Choose screenplay files")
    kind = st.radio("Source", ["Upload files", "Folder on this computer"], horizontal=True, label_visibility="collapsed", key="src_kind")
    if kind == "Upload files":
        files = st.file_uploader("Episode .txt files, or one .zip of them", type=["txt", "zip"], accept_multiple_files=True)
        if files:
            st.caption(f"{len(files)} file(s) selected")
        return "upload", files
    path = st.text_input("Folder containing the episode .txt files", value=SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else "", key="folder_path")
    ok = bool(path) and os.path.isdir(path)
    if path and not ok:
        st.warning("That folder does not exist.")
    elif ok:
        n = sum(1 for _, _, fs in os.walk(path) for f in fs if f.lower().endswith(".txt"))
        st.caption(f"{n} .txt file(s) found")
    return "folder", path if ok else ""


# ------------------------------------------------------------------------------------------ results
def quote_block(lines: List[Dict[str, Any]]) -> None:
    if not lines:
        st.caption("(none)")
    for l in lines:
        who = l["speaker"] if l.get("speaker") else l["kind"].upper()
        d = f" <small>({l['delivery']})</small>" if l.get("delivery") else ""
        st.markdown(f'<div class="quote"><small>{l["start_ts"]} · {who}{d}</small><br>{l["text"]}</div>', unsafe_allow_html=True)


def summary(run: RunResult) -> None:
    pl = run.final["pipeline"]
    n_rej = int((R.rejection_table(run)["Status"] == "Rejected").sum()) if len(run.candidates["candidates"]) else 0
    c = st.columns(5)
    c[0].metric("Episodes parsed", pl["episodes"])
    c[1].metric("Candidates found", pl["candidates"])
    c[2].metric("Promos selected", len(run.final["top_promos"]))
    c[3].metric("Rejected", n_rej)
    c[4].metric("Model-verified", pl["verified_candidates"])
    bad = [e["source_file"] for e in run.extracted["episodes"] if e.get("parse_status") == "failed" or not e.get("n_items")]
    if bad:
        st.warning(f"{len(bad)} file(s) could not be read as screenplay episodes and were skipped: " + ", ".join(bad[:5])
                   + (" ..." if len(bad) > 5 else ""))
    issues = run.extracted["summary"].get("issues_by_severity", {})
    if any(issues.values()):
        st.caption("Parser notes: " + ", ".join(f"{v} {k}" for k, v in issues.items() if v))


def downloads(run: RunResult) -> None:
    a, b, _ = st.columns([1, 1, 2])
    a.download_button("Download ranked promos (CSV)", R.to_csv(run), "ranked_promos.csv", "text/csv", width="stretch")
    b.download_button("Download detailed results (JSON)", R.to_detail_json(run), "promo_results.json", "application/json", width="stretch")


def ranked(run: RunResult) -> None:
    df = R.promo_table(run)
    if df.empty:
        st.warning("No candidate met the final quality bar. See the 'Rejected' tab for the reasons, or connect a model in the sidebar.")
        return
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Hook": SCORE_COL, "Cliffhanger": SCORE_COL, "Standalone": SCORE_COL, "Overall": SCORE_COL,
        "Duration (s)": st.column_config.NumberColumn(format="%.1f"),
        "Post order": st.column_config.NumberColumn(help="Suggested publishing order (differs from quality rank)", format="%d")})
    st.caption("Rank = quality (overall score). Post order = suggested publishing sequence, which favours clips that work "
               "on first-time viewers and avoids spoilers.")
    with st.expander("Which to post first, and why"):
        by_id = {p["promo_id"]: p for p in run.final["top_promos"]}
        for x in run.final["posting_plan"]:
            p = by_id[x["promo_id"]]
            st.markdown(f"**{x['post_order']}. {x['promo_id']}** (Episode {', '.join(map(str, x['episodes']))}, wave {x['wave']}): {x['rationale']}")


def inspector(run: RunResult) -> None:
    promos = run.final["top_promos"]
    if not promos:
        st.info("Nothing to inspect yet.")
        return
    labels = {f"{p['promo_id']} · Episode {', '.join(map(str, p['episodes']))} · "
              f"{p['segments'][0]['start_ts']}-{p['segments'][-1]['end_ts']} · {p['overall_ranking_score']}": p for p in promos}
    p = labels[st.selectbox("Choose a promo to inspect", list(labels))]
    st.markdown(f"### {p['promo_id']}  ·  overall {p['overall_ranking_score']}")
    pills = [f'<span class="pill">Episode {", ".join(map(str, p["episodes"]))}</span>',
             f'<span class="pill">{R.segments_text(p)}</span>', f'<span class="pill">{p["total_duration_sec"]} s</span>',
             f'<span class="pill {"ok" if p["verification"] == "model-verified" else "warn"}">{p["verification"]}</span>']
    st.markdown(" ".join(pills), unsafe_allow_html=True)
    for v in R.validate_promo(p):
        (st.error if v["level"] == "error" else st.warning)(v["msg"])

    sc = p["scores"]
    c = st.columns(6)
    for col, (k, label) in zip(c, [("hook", "Hook"), ("cliffhanger", "Cliffhanger"), ("standalone", "Standalone"),
                                   ("intensity", "Intensity"), ("coherence", "Coherence"), ("character_clarity", "Characters")]):
        col.metric(label, sc[k])
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Opening hook (first 3 seconds)")
        quote_block(p["opening_hook"]["first_3_seconds"])
        st.write(p["why_it_works"]["hook_explanation"])
        st.markdown("#### Ending cliffhanger")
        quote_block(p["ending_cliffhanger"]["last_spoken_lines"])
        fb = p["ending_cliffhanger"]["final_beat"]
        if fb and fb["kind"] != "dialogue":
            quote_block([fb])
        st.write(p["why_it_works"]["cliffhanger_explanation"])
        st.markdown("#### Standalone context")
        st.write(p["why_it_works"]["context_explanation"])
        if p["caveats"]:
            st.info("A new viewer would be missing: " + "; ".join(p["caveats"]))
    with right:
        st.markdown("#### Why it works")
        st.write(p["why_it_works"]["text"])
        st.caption(f"Source: {p['why_it_works']['source']}")
        st.markdown("#### Score profile")
        st.bar_chart(pd.DataFrame({"score": sc}), horizontal=True, height=240)
        st.caption("Moment types: " + ", ".join(p["moment_types"]) + "  |  Characters: " + ", ".join(p["main_characters"]))
    with st.expander("Full clip script (from the screenplay)", expanded=False):
        st.dataframe(R.clip_script(run, p), hide_index=True, width="stretch")


def rejected(run: RunResult) -> None:
    df = R.rejection_table(run)
    if df.empty:
        st.success("Every candidate was selected.")
        return
    st.write("Candidates that did not make the final list, and the specific gate or threshold each one failed.")
    statuses = sorted(df["Status"].unique())
    pick = st.multiselect("Show", statuses, default=statuses)
    view = df[df["Status"].isin(pick)]
    st.dataframe(view.drop(columns=["Opening line", "Closing line"]), hide_index=True, width="stretch", column_config={
        "Hook": SCORE_COL, "Cliffhanger": SCORE_COL, "Standalone": SCORE_COL, "Overall": SCORE_COL,
        "Why it failed": st.column_config.TextColumn(width="large")})
    st.caption({"Rejected": "failed a hard requirement of the brief", "Needs LLM review": "rules-only score is borderline; a model should decide",
                "Not selected": "valid, but below the final bar or a duplicate episode"}.get(statuses[0], ""))
    if not view.empty:
        cid = st.selectbox("Inspect a candidate", view["Candidate"])
        e = next(x for x in run.evaluated["evaluations"] if x["candidate_id"] == cid)
        st.markdown(f"**{e['episode_label']}  {e['start_ts']}-{e['end_ts']}  ({e['duration_sec']}s)**  ·  overall {e['overall_score']}")
        for k in ("hook", "cliffhanger", "standalone"):
            st.markdown(f"**{k.title()} ({e['scores'][k]})**: {e['explanations'][k]}")


def method(run: RunResult) -> None:
    ev = run.evaluated
    st.markdown("#### Scoring weights and why")
    st.dataframe(pd.DataFrame([{"Dimension": k.replace("_", " ").title(), "Weight": w, "Why": ev["weight_rationale"][k]}
                               for k, w in ev["weights"].items()]), hide_index=True, width="stretch")
    st.caption(ev["balance_rule"])
    st.markdown("#### Hard gates (any failure rejects a candidate)")
    st.json(ev["gates"], expanded=False)
    st.markdown("#### Known limitations")
    for l in run.final.get("limitations", []):
        st.markdown(f"- {l}")
    st.markdown("- Cliffhanger and context scores from rules alone are weak (rank correlation with a model: 0.34 and 0.04 on the demo pack). Connect a model for reliable results.")
    with st.expander("Run log"):
        st.code(run.log or "(empty)")
        st.caption("Seconds per stage: " + ", ".join(f"{k} {v}" for k, v in run.seconds.items()))
