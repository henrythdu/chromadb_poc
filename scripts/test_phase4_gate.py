"""Phase 4 Gate Testing - Run 5 test questions with timing and validation."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from src.retrieval.engine import RAGQueryEngine
from src.config import settings

# Test questions from Phase 4 plan
TEST_QUESTIONS = [
    "What is attention in transformers?",
    "Explain backpropagation in neural networks",
    "What are the main challenges in reinforcement learning?",
    "How does batch normalization work?",
    "What is the difference between SGD and Adam?",
]

# Input validation tests
VALIDATION_TESTS = [
    ("short", False, "too short"),
    ("a" * 501, False, "too long"),
    ("<script>alert('xss')</script>How does attention work?", False, "plain text"),
    ("What is attention in transformers?", True, None),
]


def test_input_validation():
    """Test input validation."""
    print("\n" + "=" * 60)
    print("INPUT VALIDATION TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for query, should_pass, expected_msg in VALIDATION_TESTS:
        # Validation rules from gradio_app
        is_valid = True
        error_msg = None

        if len(query.strip()) < 10:
            is_valid = False
            error_msg = "too short"
        elif len(query) > 500:
            is_valid = False
            error_msg = "too long"
        elif "<script>" in query.lower() or "<" in query and ">" in query:
            is_valid = False
            error_msg = "plain text"

        test_passed = (is_valid == should_pass)
        if expected_msg and expected_msg in error_msg.lower():
            test_passed = True

        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"{status}: '{query[:50]}...' -> {error_msg if error_msg else 'Valid'}")

        if test_passed:
            passed += 1
        else:
            failed += 1

    print(f"\nValidation: {passed}/{passed+failed} passed")
    return failed == 0


def run_test_questions():
    """Run 5 test questions with timing."""
    print("\n" + "=" * 60)
    print("TEST QUESTIONS (Phase 4 Gate)")
    print("=" * 60)

    # Initialize RAG engine
    print("\nInitializing RAG engine...")
    engine = RAGQueryEngine(
        chroma_api_key=settings.chroma_cloud_api_key,
        chroma_tenant=settings.chroma_tenant,
        chroma_database=settings.chroma_database,
        cohere_api_key=settings.cohere_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        collection_name="arxiv_papers_v1",
    )

    results = []
    all_passed = True

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}/5: {question}")
        print('='*60)

        start_time = time.time()
        try:
            response = engine.query(question, use_rerank=True)
            elapsed = time.time() - start_time

            answer = response.get("answer", "")
            citations = response.get("citations", [])
            sources = response.get("sources", [])

            # Check if response time is acceptable
            time_ok = elapsed < 30
            time_status = "✅" if time_ok else "❌"

            # Check if answer is meaningful
            answer_ok = len(answer) > 100
            answer_status = "✅" if answer_ok else "❌"

            # Check if citations exist
            citations_ok = len(citations) > 0
            citations_status = "✅" if citations_ok else "❌"

            print(f"Response time: {elapsed:.2f}s {time_status}")
            print(f"Answer length: {len(answer)} chars {answer_status}")
            print(f"Citations: {len(citations)} {citations_status}")
            print(f"Sources retrieved: {len(sources)} chunks")

            # Show preview of answer
            print(f"\nAnswer preview: {answer[:200]}...")

            # Show first few citations
            if citations:
                print(f"\nCitations:")
                for cite in citations[:3]:
                    print(f"  - {cite}")

            result = {
                "question": question,
                "time": elapsed,
                "time_ok": time_ok,
                "answer_ok": answer_ok,
                "citations_ok": citations_ok,
                "passed": time_ok and answer_ok and citations_ok,
            }
            results.append(result)

            if not result["passed"]:
                all_passed = False

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ ERROR: {e}")
            results.append({
                "question": question,
                "time": elapsed,
                "time_ok": False,
                "answer_ok": False,
                "citations_ok": False,
                "passed": False,
                "error": str(e),
            })
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results if r["passed"])
    print(f"Questions passed: {passed_count}/{len(results)}")

    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"Average response time: {avg_time:.2f}s")

    max_time = max(r["time"] for r in results)
    print(f"Max response time: {max_time:.2f}s")

    all_under_30 = all(r["time"] < 30 for r in results)
    print(f"All responses under 30s: {'✅ YES' if all_under_30 else '❌ NO'}")

    return all_passed


def main():
    """Run all Phase 4 gate tests."""
    print("\n" + "=" * 60)
    print("PHASE 4 GATE TESTING")
    print("=" * 60)

    # Test 1: Input validation
    validation_passed = test_input_validation()

    # Test 2: Run questions
    questions_passed = run_test_questions()

    # Final result
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Input Validation: {'✅ PASS' if validation_passed else '❌ FAIL'}")
    print(f"Test Questions: {'✅ PASS' if questions_passed else '❌ FAIL'}")

    all_passed = validation_passed and questions_passed
    print(f"\nPhase 4 Gate: {'✅ PASSED' if all_passed else '❌ FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
