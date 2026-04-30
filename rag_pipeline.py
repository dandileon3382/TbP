from data_loader import load_exercise_data
from chunker import process_all_exercises
from embedding import TextEmbedder
from vector_store import VectorStore
from retriever import Retriever
from ollama_client import OllamaClient


class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.embedder = TextEmbedder()
        self.store = VectorStore()   # uses collection "exercises_v2"
        self.retriever = Retriever(self.embedder, self.store)
        self.llm_client = OllamaClient()
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        if self.store.collection.count() == 0:
            print("Vector store empty — building knowledge base...")
            raw_data = load_exercise_data("./data")
            chunks = process_all_exercises(raw_data)

            if not chunks:
                print("No chunks generated. Check the data directory.")
                return

            texts = [c["text"] for c in chunks]
            print(f"Embedding {len(texts)} chunks...")
            embeddings = self.embedder.embed_batch(texts)

            self.store.add_chunks(chunks, embeddings)
            print("Knowledge base ready.")
        else:
            print(f"Knowledge base loaded ({self.store.collection.count()} chunks).")

    def retrieve(self, exercise: str, mistake: str) -> list[str]:
        return self.retriever.retrieve_context(exercise, mistake)

    def analyze(
        self,
        exercise: str,
        mistake: str,
        timestamps: list[float],
        occurrence_count: int = 1
    ) -> dict:
        """
        Full RAG cycle. Returns a structured feedback dict:
        { headline, why, fix, encouragement }
        """
        retrieved_chunks = self.retriever.retrieve_context(exercise, mistake)

        feedback = self.llm_client.generate_feedback(
            context_chunks=retrieved_chunks,
            exercise=exercise,
            mistake=mistake,
            timestamps=timestamps,
            occurrence_count=occurrence_count
        )
        return feedback
