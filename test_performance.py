#!/usr/bin/env python
"""Test RAG pipeline performance."""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.config import settings
from src.retrieval.engine import RAGQueryEngine

print("=== RAG Pipeline Performance Test ===")
print(f"Using LLM: {settings.llm_model}")
print()

engine = RAGQueryEngine()

test_query = "What is machine learning and how does it work?"

print(f'Test query: "{test_query}"')
print()

# Test full pipeline
start = time.time()
result = engine.query(test_query, use_rerank=True)
total_time = time.time() - start

print(f"Full pipeline: {total_time:.2f}s")
print(f"Answer length: {len(result['answer'])} chars")
print(f"Citations: {len(result['citations'])}")
print()
print("First 200 chars of answer:")
print(result["answer"][:200] + "...")
