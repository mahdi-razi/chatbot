from sentence_transformers import CrossEncoder
import torch
from errors import RankError
from langfuse import observe

class Reranker:
    def __init__(self, name="Qwen/Qwen3-Reranker-0.6B", max_length=2048):
        self.model = CrossEncoder(
            name,
            max_length=max_length,
            prompts={
                "rag": (
                    "Given a user question, determine whether "
                    "the passage contains information that answers "
                    "the question."
                )
            },
            default_prompt_name="rag",
            model_kwargs={"dtype": torch.float16}
        )

    @observe(name="rerank", capture_input=False, capture_output=False)
    def rerank(self, query, results, top_k=5, min_score=float("-inf")):
        if not results:
            return []

        pairs = [(query, result["text"]) for result in results]

        try:
            scores = self.model.predict(pairs)
            
        except Exception as e:
            raise RankError("Reranker model failed") from e


        ranked = [
            {**result, "rerank_score": float(score)}
            for result, score in zip(results, scores)
        ]

        ranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        ranked = [r for r in ranked if r["rerank_score"] >= min_score]

        return ranked[:top_k]
