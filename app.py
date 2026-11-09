"""
Promo Discovery Engine - Streamlit app.

Run:   streamlit run app.py
The app calls the real pipeline (screenplay_parser -> candidate_generator -> promo_evaluator -> promo_ranker);
nothing on screen is a mock-up.
"""
import os
import shutil
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from promo_app import runner, ui  # noqa: E402

st.set_page_config(page_title="Promo Discovery Engine", page_icon="🎬", layout="wide")
ui.header()
llm, top_n = ui.sidebar()
kind, payload = ui.input_panel()

st.subheader("2. Process")
ready = bool(payload)
clicked = st.button("Process screenplay", type="primary", disabled=not ready,
                    help=None if ready else "Choose files or a folder first")

if clicked:
    bar = st.progress(0.0, text="Starting")
    work = tempfile.mkdtemp(prefix="promo_in_")
    try:
        src = os.path.join(work, "screenplays")
        if kind == "upload":
            names = runner.ingest_uploads(payload, src)
        else:
            names = runner.copy_folder(payload, src)

        def on_stage(key, message, frac):
            bar.progress(min(frac, 1.0), text=message + ("" if key == "done" else "..."))

        with st.spinner("Running the pipeline"):
            st.session_state["run"] = runner.run_pipeline(src, llm, on_stage, top_n=top_n)
        st.session_state["meta"] = {"files": len(names), "llm": llm.mode}
        bar.progress(1.0, text="Done")
    except runner.PipelineError as exc:
        bar.empty()
        st.session_state.pop("run", None)
        st.error(str(exc))
    except Exception as exc:  # last resort: never show a raw traceback to a demo audience
        bar.empty()
        st.session_state.pop("run", None)
        st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

run = st.session_state.get("run")
if run is None:
    st.info("Choose the screenplay files, then press **Process screenplay**. A full 50-episode run takes a few seconds with rules "
            "or saved verdicts, longer if you connect a live model.")
else:
    st.divider()
    st.subheader("3. Results")
    ui.summary(run)
    ui.downloads(run)
    t1, t2, t3, t4 = st.tabs(["Ranked promos", "Inspect a promo", "Rejected candidates", "Method and limits"])
    with t1:
        ui.ranked(run)
    with t2:
        ui.inspector(run)
    with t3:
        ui.rejected(run)
    with t4:
        ui.method(run)
