import httpx

from jev_rag_benchmark.generation import OllamaGenerator


def test_ollama_generation_has_hard_output_cap(monkeypatch):
    captured = {}

    def fake_post(url, *, json, timeout):
        captured.update(json)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"response": "short", "model": "fixture", "eval_count": 1},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OllamaGenerator("model", max_output_tokens=37).generate("prompt")
    assert result.answer == "short"
    assert captured["options"]["num_predict"] == 37
