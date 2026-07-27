import datetime
import tempfile
import os

import streamlit as st

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

st.set_page_config(page_title="EvalOps Dashboard", layout="wide")
st.title("EvalOps — LLM evaluation dashboard")

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)


def load_results() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM run_results ORDER BY created_at DESC", engine)


# ---------- Upload & run section ----------
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

# ---------- Results ----------
df = load_results()

if df.empty:
    st.info("No runs yet. Upload a YAML file above and click Execute.")
    st.stop()

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
    latest[["test_case_id", "question", "faithfulness_score", "relevance_score", "passed", "latency_ms"]],
    use_container_width=True,
)

st.subheader("Pass rate trend across runs")
trend = df.groupby("run_id", sort=False)["passed"].mean().reset_index()
trend = trend.iloc[::-1]  # oldest first
fig = px.line(trend, x="run_id", y="passed", markers=True, labels={"passed": "pass rate"})
st.plotly_chart(fig, use_container_width=True)

# ---------- Today's history ----------
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
            today_df[["run_id", "test_case_id", "question", "faithfulness_score", "relevance_score", "passed"]],
            use_container_width=True,
        )
