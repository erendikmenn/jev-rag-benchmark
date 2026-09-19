from jev_rag_benchmark.generation import build_prompt
from jev_rag_benchmark.models import Candidate, Document


def test_prompt_enforces_shared_context_budget():
    contexts = [
        Candidate(Document("one", "a" * 10), 1.0, 1),
        Candidate(Document("two", "b" * 10), 0.5, 2),
    ]
    prompt = build_prompt("question", contexts, "abstain", context_char_budget=12)
    assert "a" * 10 in prompt
    assert "b" * 2 in prompt
    assert "b" * 3 not in prompt
