import datetime
import tempfile
import os
import uuid

import streamlit as st

# Streamlit Cloud stores secrets in st.secrets, not as real environment
# variables — but our config.py (pydantic-settings) reads from os.environ.
# Bridge them here, BEFORE importing anything from app/, so config.py
# picks them up correctly both locally (via .env) and on Streamlit Cloud.
if hasattr(st, "secrets"):
    for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                "JUDGE_MODEL", "TARGET_MODEL", "DATABASE_URL"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db import Base
from app.runner import run_suite
from app.target_app import call_target
from app.judge import score_answer
from app.models import RunResult

st.set_page_config(page_title="EvalOps Dashboard", layout="wide")
st.title("EvalOps — LLM evaluation dashboard")

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)


def load_results() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM run_results ORDER BY created_at DESC", engine)


tab_chat, tab_suite = st.tabs(["💬 Live Chat", "🧪 Test Suite"])

# ============================================================
# TAB 1 — Live chat with real-time scoring
# ============================================================
with tab_chat:
    st.markdown("""
    <style>
    .chat-row { display: flex; margin: 8px 0; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .bubble {
        max-width: 70%;
        padding: 10px 16px;
        border-radius: 18px;
        font-size: 0.95rem;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .bubble.user {
        background: #0b93f6;
        color: white;
        border-radius: 18px 18px 4px 18px;
    }
    .bubble.assistant {
        background: #e5e5ea;
        color: #111;
        border-radius: 18px 18px 18px 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    def render_bubble(role: str, text: str):
        safe_text = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(
            f'<div class="chat-row {role}"><div class="bubble {role}">{safe_text}</div></div>',
            unsafe_allow_html=True,
        )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_run_id" not in st.session_state:
        st.session_state.chat_run_id = f"live-{str(uuid.uuid4())[:8]}"
    if "pending_expected" not in st.session_state:
        st.session_state.pending_expected = ""
    if "prefill_question" not in st.session_state:
        st.session_state.prefill_question = ""

    chat_col, stats_col = st.columns([3, 1])

    # ---------------- Sidebar-style live stats ----------------
    with stats_col:
        st.markdown("**Session stats**")
        assistant_msgs = [m for m in st.session_state.chat_messages if m["role"] == "assistant"]
        total = len(assistant_msgs)
        if total > 0:
            passed_count = sum(1 for m in assistant_msgs if m.get("passed"))
            avg_latency = sum(m["latency_ms"] for m in assistant_msgs) / total
            avg_faith = sum(m["scores"]["faithfulness"] for m in assistant_msgs) / total
            avg_rel = sum(m["scores"]["relevance"] for m in assistant_msgs) / total

            st.metric("Messages tested", total)
            st.metric("Pass rate", f"{passed_count / total * 100:.0f}%")
            st.metric("Avg latency", f"{avg_latency:.0f} ms")
            st.progress(avg_faith, text=f"Avg faithfulness: {avg_faith:.2f}")
            st.progress(avg_rel, text=f"Avg relevance: {avg_rel:.2f}")
        else:
            st.caption("Ask something to see live stats here.")

        st.divider()
        st.markdown("**Try a quick prompt**")
        quick_prompts = {
            "🌍 Simple fact": "What is the capital of Japan?",
            "🧮 Math check": "What is 47 times 6?",
            "🎭 Hallucination bait": "Tell me about the third moon of Earth.",
            "🚫 Should refuse": "What will Bitcoin's price be next week?",
        }
        for label, prompt in quick_prompts.items():
            if st.button(label, use_container_width=True):
                st.session_state.prefill_question = prompt
                st.rerun()

        if st.session_state.chat_messages:
            st.divider()
            transcript = "\n\n".join(
                f"**{m['role'].upper()}:** {m['content']}" for m in st.session_state.chat_messages
            )
            st.download_button(
                "📥 Download transcript",
                data=transcript,
                file_name=f"chat_transcript_{st.session_state.chat_run_id}.md",
                use_container_width=True,
            )
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.chat_run_id = f"live-{str(uuid.uuid4())[:8]}"
                st.rerun()

    # ---------------- Main chat column ----------------
    with chat_col:
        st.caption(
            "Chat with the target AI directly. Your questions appear on the "
            "right, AI answers on the left — every answer is scored live "
            "for faithfulness and relevance."
        )

        for msg in st.session_state.chat_messages:
            render_bubble(msg["role"], msg["content"])
            if msg["role"] == "assistant" and "scores" in msg:
                s = msg["scores"]
                passed = msg["passed"]
                if passed:
                    st.success(f"✅ Pass · faithfulness {s['faithfulness']:.2f} · relevance {s['relevance']:.2f}"
                               + (f" · correctness {s['correctness']:.2f}" if msg.get("has_expected") else ""))
                else:
                    st.error(f"❌ Fail · faithfulness {s['faithfulness']:.2f} · relevance {s['relevance']:.2f}"
                              + (f" · correctness {s['correctness']:.2f}" if msg.get("has_expected") else ""))
                with st.expander("Why this score? · latency & judge reasoning"):
                    st.write(f"**Latency:** {msg['latency_ms']:.0f} ms")
                    st.write(f"**Judge reasoning:** {s.get('reasoning', 'n/a')}")

        with st.expander("➕ Add an expected answer (optional, checks correctness too)"):
            st.session_state.pending_expected = st.text_input(
                "Expected/reference answer for the next question",
                value=st.session_state.pending_expected,
                placeholder="e.g. Tokyo",
            )

        user_input = st.chat_input("Ask the AI anything...")

        if st.session_state.prefill_question:
            st.info(f"Quick prompt ready: *{st.session_state.prefill_question}*")
            col_send, col_cancel = st.columns([1, 1])
            with col_send:
                if st.button("Send quick prompt ▶️", use_container_width=True):
                    user_input = st.session_state.prefill_question
                    st.session_state.prefill_question = ""
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.prefill_question = ""
                    st.rerun()

        if user_input:
            expected = st.session_state.pending_expected.strip() or None
            st.session_state.pending_expected = ""

            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            render_bubble("user", user_input)

            with st.spinner("Generating and scoring response..."):
                try:
                    target_output = call_target(user_input)
                    answer = target_output["answer"]
                    scores = score_answer(question=user_input, answer=answer, expected=expected)

                    has_expected = expected is not None
                    passed = (
                        scores["faithfulness"] >= 0.7
                        and scores["relevance"] >= 0.7
                        and (scores.get("correctness", 1.0) >= 0.7 if has_expected else True)
                    )

                    render_bubble("assistant", answer)
                    if passed:
                        st.success(f"✅ Pass · faithfulness {scores['faithfulness']:.2f} · relevance {scores['relevance']:.2f}"
                                   + (f" · correctness {scores['correctness']:.2f}" if has_expected else ""))
                    else:
                        st.error(f"❌ Fail · faithfulness {scores['faithfulness']:.2f} · relevance {scores['relevance']:.2f}"
                                  + (f" · correctness {scores['correctness']:.2f}" if has_expected else ""))
                    with st.expander("Why this score? · latency & judge reasoning"):
                        st.write(f"**Latency:** {target_output['latency_ms']:.0f} ms")
                        st.write(f"**Judge reasoning:** {scores.get('reasoning', 'n/a')}")

                    db = SessionLocal()
                    db.add(RunResult(
                        run_id=st.session_state.chat_run_id,
                        test_case_id=f"chat-{len(st.session_state.chat_messages)}",
                        question=user_input,
                        answer=answer,
                        expected=expected,
                        faithfulness_score=scores["faithfulness"],
                        relevance_score=scores["relevance"],
                        correctness_score=scores.get("correctness", 1.0),
                        passed=int(passed),
                        latency_ms=target_output["latency_ms"],
                        cost_usd=target_output["cost_usd"],
                        model_used=settings.target_model,
                    ))
                    db.commit()
                    db.close()

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "scores": scores,
                        "latency_ms": target_output["latency_ms"],
                        "passed": passed,
                        "has_expected": has_expected,
                    })
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ============================================================
# TAB 2 — Batch test suite (existing functionality)
# ============================================================
with tab_suite:
    st.subheader("Run a test suite")
    uploaded_file = st.file_uploader("Upload a test cases YAML file", type=["yaml", "yml"])

    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_clicked = st.button("Execute", type="primary", disabled=uploaded_file is None)

    if run_clicked and uploaded_file is not None:
        with st.spinner("Running test suite — calling target app and judge for each case..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                db = SessionLocal()
                summary = run_suite(db, path=tmp_path)
                db.close()
                st.success(
                    f"Run complete — {summary['passed']}/{summary['total']} passed "
                    f"({summary['pass_rate'] * 100:.0f}% pass rate). Run ID: {summary['run_id']}"
                )
            except Exception as e:
                st.error(f"Run failed: {e}")
            finally:
                os.remove(tmp_path)

    st.divider()

    df = load_results()

    if df.empty:
        st.info("No runs yet. Upload a YAML file above and click Execute, or try the Live Chat tab.")
    else:
        df["created_at"] = pd.to_datetime(df["created_at"])

        latest_run_id = df.iloc[0]["run_id"]
        latest = df[df["run_id"] == latest_run_id]

        st.subheader("Latest run summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total tests", len(latest))
        col2.metric("Pass rate", f"{latest['passed'].mean() * 100:.0f}%")
        col3.metric("Avg latency", f"{latest['latency_ms'].mean():.0f} ms")
        col4.metric("Avg cost/query", f"${latest['cost_usd'].mean():.4f}")

        st.subheader("Latest run — test case results")
        st.dataframe(
            latest[["test_case_id", "question", "faithfulness_score", "relevance_score", "correctness_score", "passed", "latency_ms"]],
            use_container_width=True,
        )

        st.subheader("Pass rate trend across runs")
        trend = df.groupby("run_id", sort=False)["passed"].mean().reset_index()
        trend = trend.iloc[::-1]  # oldest first
        fig = px.line(trend, x="run_id", y="passed", markers=True, labels={"passed": "pass rate"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Today's test history")
        today = datetime.datetime.now().date()
        today_df = df[df["created_at"].dt.date == today]

        if today_df.empty:
            st.info("No runs executed today yet.")
        else:
            today_summary = (
                today_df.groupby("run_id", sort=False)
                .agg(
                    time=("created_at", "min"),
                    total=("test_case_id", "count"),
                    passed=("passed", "sum"),
                    pass_rate=("passed", "mean"),
                    avg_latency_ms=("latency_ms", "mean"),
                    avg_cost_usd=("cost_usd", "mean"),
                )
                .reset_index()
                .sort_values("time", ascending=False)
            )
            today_summary["time"] = today_summary["time"].dt.strftime("%H:%M:%S")
            today_summary["pass_rate"] = (today_summary["pass_rate"] * 100).round(0).astype(int).astype(str) + "%"
            today_summary["avg_cost_usd"] = today_summary["avg_cost_usd"].round(4)
            today_summary["avg_latency_ms"] = today_summary["avg_latency_ms"].round(0)

            st.dataframe(today_summary, use_container_width=True)

            with st.expander("View individual test case results from today"):
                st.dataframe(
                    today_df[["run_id", "test_case_id", "question", "faithfulness_score", "relevance_score", "correctness_score", "passed"]],
                    use_container_width=True,
                )
