# EvalOps — AI Evaluation & Regression Testing Framework

A QA-style testing framework for LLM/RAG applications. Runs a suite of test
cases against a target app, scores each answer for faithfulness and
relevance using an LLM-as-judge, and flags regressions across runs.

## Folder structure

```
eval-framework/
├── app/
│   ├── main.py            FastAPI entrypoint — exposes /run-evals
│   ├── config.py          Env vars, API keys, model settings
│   ├── target_app.py      Wrapper around the RAG/LLM system under test
│   ├── judge.py           LLM-as-judge scoring logic (faithfulness, relevance)
│   ├── runner.py          Orchestrates: load cases -> call target -> score -> save
│   ├── models.py          SQLAlchemy models (TestCase, RunResult)
│   └── db.py               DB session/engine setup
├── test_cases/
│   └── sample_cases.yaml  Your test question/expected-answer pairs
├── tests/
│   └── test_runner.py     Unit tests for the runner itself (pytest)
├── dashboard.py            Streamlit dashboard (separate deploy)
├── requirements.txt
├── .env.example
└── .github/workflows/eval.yml   Runs the suite on every push
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API keys
uvicorn app.main:app --reload
# in another terminal
streamlit run dashboard.py
```

## Next steps once this runs

1. Add 10-15 real test cases in test_cases/sample_cases.yaml
2. Point target_app.py at your actual RAG pipeline
3. Push to GitHub, add OPENAI_API_KEY / ANTHROPIC_API_KEY as repo secrets
4. Watch the GitHub Action run your suite automatically and comment results on PRs
