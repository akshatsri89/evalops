import uuid
import yaml
from sqlalchemy.orm import Session

from app.target_app import call_target
from app.judge import score_answer
from app.models import RunResult
from app.config import settings

PASS_THRESHOLD = 0.7


def load_test_cases(path: str = "test_cases/sample_cases.yaml") -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["test_cases"]


def run_suite(db: Session, path: str = "test_cases/sample_cases.yaml") -> dict:
    run_id = str(uuid.uuid4())[:8]
    test_cases = load_test_cases(path)
    results = []

    for case in test_cases:
        target_output = call_target(case["question"])
        scores = score_answer(
            question=case["question"],
            answer=target_output["answer"],
            expected=case.get("expected"),
        )

        avg_score = (scores["faithfulness"] + scores["relevance"] + scores["correctness"]) / 3
        passed = avg_score >= PASS_THRESHOLD

        result = RunResult(
            run_id=run_id,
            test_case_id=case["id"],
            question=case["question"],
            answer=target_output["answer"],
            expected=case.get("expected"),
            faithfulness_score=scores["faithfulness"],
            relevance_score=scores["relevance"],
            correctness_score=scores["correctness"],
            passed=int(passed),
            latency_ms=target_output["latency_ms"],
            cost_usd=target_output["cost_usd"],
            model_used=settings.target_model,
        )
        db.add(result)
        results.append(result)

    db.commit()

    pass_count = sum(r.passed for r in results)
    return {
        "run_id": run_id,
        "total": len(results),
        "passed": pass_count,
        "pass_rate": pass_count / len(results) if results else 0,
    }