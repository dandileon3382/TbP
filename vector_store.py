import chromadb
import uuid


class VectorStore:
    def __init__(
        self,
        collection_name: str = "exercises_v2",
        persist_directory: str = "./chroma_db"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Stores chunks in ChromaDB.
        Each chunk must have: text, type, exercise, mistake (can be empty string).
        """
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "exercise": c.get("exercise", ""),
                "type": c.get("type", ""),
                "mistake": c.get("mistake", "")   # empty string if not mistake-specific
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Stored {len(chunks)} chunks in vector store.")

    def query(
        self,
        query_embedding: list[float],
        exercise: str = None,
        mistake: str = None,
        top_k: int = 6
    ) -> list[str]:
        """
        Retrieves top_k relevant chunks.

        Strategy:
          1. If a specific mistake is provided, first try to pull the exact
             mistake_explanation chunk for that mistake (highest precision).
          2. Then fill remaining slots with general exercise context chunks.
        """
        results_texts = []

        # ── Step 1: Targeted mistake chunk ─────────────────────────────────────
        if exercise and mistake:
            try:
                exact = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=2,
                    where={"$and": [{"exercise": exercise}, {"mistake": mistake}]}
                )
                if exact and exact["documents"] and exact["documents"][0]:
                    results_texts.extend(exact["documents"][0])
            except Exception:
                pass  # filter may fail if no matching docs — fall through

        # ── Step 2: Broad exercise context (rules, cues, description) ───────────
        remaining = top_k - len(results_texts)
        if remaining > 0:
            where_clause = {"exercise": exercise} if exercise else None
            try:
                broad = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=remaining + len(results_texts),  # over-fetch, dedup below
                    where=where_clause
                )
                if broad and broad["documents"] and broad["documents"][0]:
                    for doc in broad["documents"][0]:
                        if doc not in results_texts:
                            results_texts.append(doc)
                            if len(results_texts) >= top_k:
                                break
            except Exception:
                pass

        return results_texts[:top_k]
