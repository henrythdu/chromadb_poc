"""Test LLM integration."""


def test_llm_imports():
    """Test that llm module can be imported."""
    from src.generation.llm import OpenRouterLLM

    assert OpenRouterLLM is not None


def test_prompts_imports():
    """Test that prompts module can be imported."""
    from src.generation.prompts import RAG_PROMPT, build_rag_prompt

    assert build_rag_prompt is not None
    assert RAG_PROMPT is not None


def test_build_rag_prompt():
    """Test RAG prompt building."""
    from src.generation.prompts import build_rag_prompt

    chunks = [
        {
            "text": "Attention is a mechanism...",
            "metadata": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762",
                "page_number": 1,
            },
        }
    ]

    prompt = build_rag_prompt("What is attention?", chunks)

    assert "What is attention?" in prompt
    assert "Attention is a mechanism..." in prompt
    assert "arxiv:1706.03762" in prompt


def test_llm_initialization():
    """Test OpenRouterLLM initialization."""
    from src.generation.llm import OpenRouterLLM

    llm = OpenRouterLLM(api_key="test_key", model="anthropic/claude-3.5-sonnet")

    assert llm.model == "anthropic/claude-3.5-sonnet"
