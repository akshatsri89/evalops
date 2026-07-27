"""
LLM-as-judge: asks a strong model to score an answer for faithfulness
(does it avoid making things up?) and relevance (does it actually answer
the question?). Returns floats 0-1 so you can threshold pass/fail.

Uses Google Gemini's free tier — deliberately a DIFFERENT provider than
the Groq-based target app, so the judge isn't grading a sibling model's
output (self-grading bias). Get a free key: https://aistudio.google.com/apikey
"""
import json
import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel(settings.judge_model)

JUDGE_PROMPT = """You are grading an AI system's answer to a question.

Question: {question}
Answer: {answer}
{expected_block}

Score the answer from 0.0 to 1.0 on two dimensions:
- faithfulness: does the answer avoid unsupported claims or fabrication?
- relevance: does the answer directly address the question asked?

Respond ONLY with JSON, no other text, in this exact format:
{{"faithfulness": 0.0, "relevance": 0.0, "reasoning": "one sentence"}}
"""


def score_answer(question: str, answer: str, expected: str | None = None) -> dict:
    expected_block = f"Expected/reference answer: {expected}" if expected else ""
    prompt = JUDGE_PROMPT.format(
        question=question, answer=answer, expected_block=expected_block
    )

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Gemini sometimes wraps JSON in markdown code fences — strip if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Judge didn't return clean JSON — fail safe rather than crash the run
        result = {"faithfulness": 0.0, "relevance": 0.0, "reasoning": "parse_error"}

    return result