"""Agent workflows for explain/review/improve tasks with RAG context."""

import re

from src.config import MAX_NEW_TOKENS, TOP_K

VALID_TASKS = {"Explain", "Review", "Improve"}


def _format_context(chunks):
    if not chunks:
        return "No retrieved context."
    return "\n\n".join(f"[{i}] ({chunk.source})\n{chunk.text}" for i, chunk in enumerate(chunks, start=1))


def build_prompt(task: str, code: str, user_query: str, context: str) -> str:
    return f"""<|im_start|>system
You are an expert Python assistant. Use the provided context to help the user.<|im_end|>
<|im_start|>user
Task: {task}
Context: {context}
Code:
{code}

Question: {user_query}<|im_end|>
<|im_start|>assistant
"""


def run_agent(generator, indexer, task: str, code: str, user_query: str) -> str:
    from src.models import generate_text

    retrieved = indexer.search(user_query or code, top_k=TOP_K)
    context = _format_context(retrieved)
    prompt = build_prompt(task=task, code=code, user_query=user_query, context=context)
    response = generate_text(generator, prompt, max_new_tokens=MAX_NEW_TOKENS)


    return response
