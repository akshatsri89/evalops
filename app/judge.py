"""
LLM-as-judge: asks a strong model to score an answer for faithfulness
(does it avoid making things up?), relevance (does it actually answer
the question?), and correctness (does it match the expected reference
answer, if one was provided?). Returns floats 0-1 so you can threshold
pass/fail.

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

Score the answer from 0.0 to 1.0 on these dimensions:
- faithfulness: does the answer avoid unsupported claims or fabrication?
- relevance: does the answer directly address the question asked?
{correctness_instruction}

Respond ONLY with JSON, no other text, in this exact format:
{{"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0, "reasoning": "one sentence"}}
"""

CORRECTNESS_INSTRUCTION = (
    "- correctness: does the answer factually MATCH the expected/reference "
    "answer above? Score 0.0 if it contradicts or differs from the reference "
    "on the core fact being asked, even if the answer sounds confident and "
    "well-written. Score 1.0 only if it agrees with the reference."
)

# If there's no expected answer to check against, correctness isn't
# applicable — treat it as a full score so it doesn't drag down the average.
NO_CORRECTNESS_INSTRUCTION = (
    '- correctness: no reference answer was provided, so set this to 1.0.'
)


def score_answer(question: str, answer: str, expected: str | None = None) -> dict:
    if expected:
        expected_block = f"Expected/reference answer: {expected}"
        correctness_instruction = CORRECTNESS_INSTRUCTION
    else:
        expected_block = ""
        correctness_instruction = NO_CORRECTNESS_INSTRUCTION

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        expected_block=expected_block,
        correctness_instruction=correctness_instruction,
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
        result = {"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0, "reasoning": "parse_error"}

    # Older cached responses or edge cases might omit correctness — default safe
    result.setdefault("correctness", 1.0 if not expected else 0.0)

    return result