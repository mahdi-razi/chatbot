from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, ValidationError
from helper import build_prompt
from prompts import ROUTER_SYSTEM_PROMPT

import logging
logger = logging.getLogger(__name__)


class RouterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[0, 1, 2, 3]
    queries: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_queries(self):
        if self.type in (0, 1, 2) and len(self.queries) != 1:
            raise ValueError(
                "Types 0, 1, and 2 require exactly one query."
            )

        if self.type == 3 and len(self.queries) < 2:
            raise ValueError(
                "Type 3 requires at least two queries."
            )

        if any(not q.strip() for q in self.queries):
            raise ValueError("Queries cannot be empty.")

        return self


def router(gen, query, history=None):
    prompt = build_prompt(ROUTER_SYSTEM_PROMPT, query, history=history)
    raw = gen(prompt, schema=RouterOutput.model_json_schema(), name="router")
    try:
        return RouterOutput.model_validate_json(raw), False
    except ValidationError:
        logger.warning("Invalid router output, falling back to the original query.", exc_info=True)
        return RouterOutput(type=1, queries=[query]), True