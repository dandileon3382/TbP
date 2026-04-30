import os
import sys

# Ensure data gets loaded
from rag_pipeline import RAGPipeline

def run_tests():
    print("Testing Pipeline Initialization...")
    pipeline = RAGPipeline()
    
    print("\n\n--- Testing Retrieval ---")
    chunks = pipeline.retrieve("squats", "knees caving in")
    if chunks:
        print(f"Retrieved {len(chunks)} chunks.")
        for i, c in enumerate(chunks):
            print(f"Chunk {i+1}: {c[:100]}...")
    else:
        print("Retrieval failed -> No chunks.")
        
    print("\n\n--- Testing Analysis ---")
    feedback = pipeline.analyze("lunges", "using momentum", [1.0, 1.5])
    print("Feedback received:")
    print(feedback)

if __name__ == "__main__":
    run_tests()
