import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime

from app.db import Base


class RunResult(Base):
    __tablename__ = "run_results"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, index=True)          # groups results from one suite run
    test_case_id = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    expected = Column(Text, nullable=True)
    faithfulness_score = Column(Float)
    relevance_score = Column(Float)
    passed = Column(Integer)                      # 1 = pass, 0 = fail
    latency_ms = Column(Float)
    cost_usd = Column(Float)
    model_used = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
