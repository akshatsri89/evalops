from fastapi import FastAPI, Depends, UploadFile, File
from sqlalchemy.orm import Session
import shutil
import tempfile
import os

from app.db import engine, Base, get_db
from app.runner import run_suite

Base.metadata.create_all(bind=engine)

app = FastAPI(title="EvalOps")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-evals")
def run_evals(path: str = "test_cases/sample_cases.yaml", db: Session = Depends(get_db)):
    """Run the suite from a YAML file already on disk. Pass a custom path
    via query param, e.g. POST /run-evals?path=test_cases/my_cases.yaml"""
    summary = run_suite(db, path=path)
    return summary


@app.post("/run-evals/upload")
async def run_evals_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a YAML file directly and run it — no need to place it in
    test_cases/ manually first. Use this from the /docs page's file picker."""
    suffix = os.path.splitext(file.filename)[1] or ".yaml"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        summary = run_suite(db, path=tmp_path)
    finally:
        os.remove(tmp_path)

    return summary
