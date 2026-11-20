from copy import deepcopy

import pytest

from app.core.config import settings
from app.services.rag import local_rerank
from app.services.rag.rerank import rerank_documents


@pytest.fixture(autouse=True)
def prevent_real_model_loading(monkeypatch):
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("Unit tests must not download or load a real reranker")

    monkeypatch.setattr(local_rerank, "_local_cross_encoder", forbidden_load)


def _candidate(identifier, content, *, score=0.5):
    return {
        "id": identifier,
        "source_id": "source-" + identifier,
        "title": identifier + ".md",
        "snippet": "short pre-rerank snippet",
        "_raw_content": content,
        "score": score,
        "metadata": {"path": "docs/" + identifier + ".md"},
        "retrieval": {"sources": ["vector", "keyword"]},
    }


def test_local_model_uses_raw_content_and_sigmoid_scores_without_mutating_candidates(monkeypatch):
    torch = pytest.importorskip("torch")
    candidates = [
        _candidate("unrelated", "The unrelated raw document.", score=0.9),
        _candidate("relevant", "The answer exists only in this raw document.", score=0.2),
        _candidate("outside-budget", "Not sent to the model.", score=0.99),
    ]
    before = deepcopy(candidates)
    monkeypatch.setattr(settings, "rag_rerank_top_n", 2)
    monkeypatch.setattr(settings, "local_rerank_batch_size", 2)

    class FakeModel:
        def predict(self, pairs, **kwargs):
            assert len(pairs) == 2
            assert all(pair[0] == "Where is the answer?" for pair in pairs)
            assert "The answer exists only in this raw document." in pairs[1][1]
            assert "short pre-rerank snippet" not in pairs[1][1]
            assert "docs/relevant.md" in pairs[1][1]
            assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
            assert kwargs["batch_size"] == 2
            assert kwargs["show_progress_bar"] is False
            return kwargs["activation_fn"](torch.tensor([-3.0, 3.0])).numpy()

    monkeypatch.setattr(local_rerank, "_local_cross_encoder", lambda *_args, **_kwargs: FakeModel())

    ranked = local_rerank.rerank_local_documents("Where is the answer?", candidates, 2)

    assert [row["id"] for row in ranked] == ["relevant", "unrelated"]
    assert ranked[0]["score"] == pytest.approx(0.9526, abs=0.0001)
    assert ranked[1]["score"] == pytest.approx(0.0474, abs=0.0001)
    assert ranked[0]["rerank_score"] == ranked[0]["score"]
    assert ranked[0]["retrieval"]["rerank_mode"] == "qwen_local"
    assert ranked[0]["retrieval"]["sources"] == ["vector", "keyword"]
    assert ranked[0]["source_id"] == "source-relevant"
    assert candidates == before


def test_default_dispatcher_uses_local_model_and_records_its_model_name(monkeypatch):
    monkeypatch.setattr(settings, "rerank_provider", "qwen_local")
    monkeypatch.setattr(settings, "local_rerank_model", "Qwen/Qwen3-Reranker-0.6B")

    class FakeModel:
        def predict(self, _pairs, **_kwargs):
            return [0.9]

    calls = []

    def load(model_name, *args, **kwargs):
        calls.append(model_name)
        return FakeModel()

    monkeypatch.setattr(local_rerank, "_local_cross_encoder", load)
    ranked = rerank_documents("answer", [_candidate("answer", "The answer.")], 1)

    assert calls == ["Qwen/Qwen3-Reranker-0.6B"]
    assert ranked[0]["retrieval"]["rerank_mode"] == "qwen_local"
    assert ranked[0]["retrieval"]["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert not ranked[0]["retrieval"].get("rerank_fallback", False)


def test_local_load_failure_has_explicit_heuristic_fallback(monkeypatch):
    def fail_load(*_args, **_kwargs):
        raise OSError("model cache is unavailable")

    monkeypatch.setattr(local_rerank, "_local_cross_encoder", fail_load)
    ranked = rerank_documents("answer", [_candidate("answer", "The answer.")], 1, provider="qwen_local")

    assert ranked[0]["retrieval"]["rerank_mode"] == "heuristic"
    assert ranked[0]["retrieval"]["rerank_requested"] == "qwen_local"
    assert ranked[0]["retrieval"]["rerank_fallback"] is True
    assert ranked[0]["retrieval"]["rerank_fallback_reason"] == "OSError"


@pytest.mark.parametrize("scores", [[], [float("nan")], [float("inf")], [-0.1], [1.1]])
def test_invalid_model_scores_are_rejected(monkeypatch, scores):
    class FakeModel:
        def predict(self, _pairs, **_kwargs):
            return scores

    monkeypatch.setattr(local_rerank, "_local_cross_encoder", lambda *_args, **_kwargs: FakeModel())
    with pytest.raises(ValueError):
        local_rerank.rerank_local_documents("answer", [_candidate("answer", "The answer.")], 1)


def test_empty_input_and_explicit_heuristic_do_not_load_the_model():
    assert local_rerank.rerank_local_documents("answer", [], 3) == []
    assert rerank_documents("answer", [], 3, provider="qwen_local") == []
    ranked = rerank_documents("answer", [_candidate("answer", "The answer.")], 1, provider="heuristic")
    assert ranked[0]["retrieval"]["rerank_mode"] == "heuristic"
    assert not ranked[0]["retrieval"].get("rerank_fallback", False)
