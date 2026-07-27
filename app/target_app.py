"""
This is the system UNDER TEST. Replace call_target() with a call into your
real RAG pipeline / agent / LLM app. For now it calls OpenAI directly so you
have something runnable immediately.
"""
import time
from openai import OpenAI

from app.config import settings

# Groq's API is OpenAI-compatible — same client, just a different base_url.
# Get a free key at https://console.groq.com
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url="https://api.groq.com/openai/v1",
)


def call_target(question: str) -> dict:
    """Returns the answer plus latency and (rough) cost for one question."""
    start = time.time()

    response = client.chat.completions.create(
        model=settings.target_model,  # e.g. "llama-3.1-8b-instant"
        messages=[{"role": "user", "content": question}],
    )

    latency_ms = (time.time() - start) * 1000
    answer = response.choices[0].message.content

    # Groq's free tier has no per-token cost — record 0 unless you're on a paid plan
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_usd = 0.0

    return {
        "answer": answer,
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }