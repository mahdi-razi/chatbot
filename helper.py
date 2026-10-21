from typing import Literal
from pydantic import create_model


def build_prompt(system, query, chunks=None, history=None):
    prompt = [{"role": "system", "content": system}]
    if history:
        prompt.extend(history)

    content = ""
    if chunks:
        context = "\n\n".join(f'<doc id="{i}">\n{c}\n</doc>' for i, c in enumerate(chunks, 1))
        content += f"<context>\n{context}\n</context>\n\n"

    content += f"<question>\n{query}\n</question>"
    prompt.append({"role": "user", "content": content})

    return prompt


def make_answer_model(n):
    ids = Literal[tuple(range(1, n + 1))]
    return create_model("RagAnswer", answer=(str, ...), used_chunks=(list[ids], ...))