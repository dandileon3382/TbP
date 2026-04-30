from embedding import TextEmbedder
from vector_store import VectorStore
from typing import List


class Retriever:
    def __init__(self, embedder: TextEmbedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve_context(
        self,
        exercise: str,
        mistake: str,
        top_k: int = 6
    ) -> List[str]:
        """
        Builds a rich biomechanical query string, embeds it, and retrieves
        the most relevant chunks — prioritising the exact mistake explanation
        followed by general coaching context.
        """
        # Build a query that mimics how a trainer would think about the mistake
        readable_mistake = mistake.replace("_", " ") if mistake else ""
        if readable_mistake:
            query = (
                f"{exercise} exercise {readable_mistake} mistake "
                f"cause consequence correction coaching cue fix biomechanics"
            )
        else:
            query = f"{exercise} exercise proper form coaching cues biomechanics"

        query_embedding = self.embedder.embed_text(query)

        chunks = self.store.query(
            query_embedding=query_embedding,
            exercise=exercise,
            mistake=mistake,
            top_k=top_k
        )

        return chunks
