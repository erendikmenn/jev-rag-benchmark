import httpx

from jev_rag_benchmark.generation import OpenRouterGenerator


def test_openrouter_generation_records_usage_and_cost():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "qwen/qwen3.7-flash"
        assert payload["reasoning"] == {"enabled": False}
        return httpx.Response(
            200,
            json={
                "model": "qwen/qwen3.7-flash",
                "choices": [{"message": {"content": "Ankara [d1]"}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 5, "cost": 0.000002},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = OpenRouterGenerator(
        "qwen/qwen3.7-flash", api_key="test", max_budget_usd=1, client=client
    ).generate("prompt")
    assert result.answer == "Ankara [d1]"
    assert result.prompt_tokens == 40
    assert result.completion_tokens == 5
    assert result.cost_usd == 0.000002


def test_openrouter_generation_sends_reasoning_effort():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        assert payload["reasoning"] == {"effort": "medium"}
        return httpx.Response(
            200,
            json={
                "model": "google/gemini-3.8-flash",
                "choices": [{"message": {"content": "Ankara [d1]"}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 5, "cost": 0.0001},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = OpenRouterGenerator(
        "google/gemini-3.8-flash",
        reasoning_effort="medium",
        api_key="test",
        max_budget_usd=1,
        client=client,
    ).generate("prompt")
    assert result.answer == "Ankara [d1]"


def test_openrouter_generation_retries_empty_content_and_counts_usage(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request):
        nonlocal attempts
        attempts += 1
        content = "" if attempts == 1 else "Ankara [d1]"
        return httpx.Response(
            200,
            json={
                "model": "google/gemini-3.8-flash",
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "error" if not content else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 5, "cost": 0.0001},
            },
        )

    monkeypatch.setattr("jev_rag_benchmark.generation.time.sleep", lambda _: None)
    result = OpenRouterGenerator(
        "google/gemini-3.8-flash",
        api_key="test",
        max_budget_usd=1,
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ).generate("prompt")

    assert attempts == 2
    assert result.answer == "Ankara [d1]"
    assert result.prompt_tokens == 80
    assert result.completion_tokens == 10
    assert result.cost_usd == 0.0002
    assert result.error is None
