from sentence_transformers import CrossEncoder
import torch


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


    def rerank(self, query, results, top_k=5):
        if not results:
            return []

        pairs = [(query, result["text"]) for result in results]

        scores = self.model.predict(pairs)

        ranked = [
            {**result, "rerank_score": float(score)}
            for result, score in zip(results, scores)
        ]

        ranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]
