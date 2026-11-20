import logging
import math
from functools import lru_cache
from time import perf_counter
from typing import Any

from app.core.config import settings
from app.services.rag.local_inference import LOCAL_INFERENCE_LOCK


logger = logging.getLogger(__name__)
_LOCAL_RERANK_RUNTIME: dict[str, Any] = {"loaded": False, "device": None}


def local_rerank_runtime_status() -> dict[str, Any]:
    return dict(_LOCAL_RERANK_RUNTIME)


@lru_cache(maxsize=1)
def _local_cross_encoder(
    model_name: str, device: str | None, local_files_only: bool, max_length: int
) -> Any:
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Local reranking requires backend/requirements-local-embedding.txt"
        ) from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    # The official Qwen checkpoint provides the yes/no logit scoring module and
    # chat template. No generation or remote model code is needed.
    model = CrossEncoder(
        model_name,
        device=resolved_device,
        max_length=max(128, max_length),
        model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"},
        config_kwargs={"use_cache": False},
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model.eval()
    _LOCAL_RERANK_RUNTIME.update(
        loaded=True,
        model=model_name,
        device=str(model.device),
        dtype=str(next(model.parameters()).dtype),
        gpu=torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
        max_length=max(128, max_length),
    )
    logger.info("Local reranker loaded: model=%s device=%s dtype=%s", model_name, resolved_device, dtype)
    return model


def _document_text(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return "\n".join(
        part
        for part in (
            str(candidate.get("title") or ""),
            str(metadata.get("path") or ""),
            str(candidate.get("_raw_content") or candidate.get("snippet") or ""),
        )
        if part
    )


def rerank_local_documents(
    query: str,
    candidates: list[dict[str, Any]],
    limit: int,
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates or limit <= 0:
        return []
    import torch

    resolved_model = model_name or settings.local_rerank_model
    top_n = min(max(settings.rag_rerank_top_n, limit), len(candidates))
    head, tail = candidates[:top_n], candidates[top_n:]
    pairs = [(query, _document_text(candidate)) for candidate in head]
    started = perf_counter()
    with LOCAL_INFERENCE_LOCK:
        model = _local_cross_encoder(
            resolved_model,
            settings.local_rerank_device,
            settings.local_rerank_local_files_only,
            settings.local_rerank_max_length,
        )
        with torch.inference_mode():
            scores = model.predict(
                pairs,
                batch_size=max(1, settings.local_rerank_batch_size),
                show_progress_bar=False,
                # Qwen returns logit differences by default; the rest of this
                # pipeline, including its score threshold, expects [0, 1].
                activation_fn=torch.nn.Sigmoid(),
                convert_to_numpy=True,
                prompt=settings.qwen_retrieval_instruction,
            )
    values = [float(score) for score in scores]
    if len(values) != len(head):
        raise ValueError("Local reranker returned an unexpected number of scores")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in values):
        raise ValueError("Local reranker returned invalid relevance scores")

    ranked = []
    for candidate, score in zip(head, values):
        rounded = round(score, 4)
        ranked.append(
            {
                **candidate,
                "score": rounded,
                "rerank_score": rounded,
                "rank_reason": "Qwen3 本地模型相关性评分",
                "retrieval": {
                    **(candidate.get("retrieval") or {}),
                    "rerank_mode": "qwen_local",
                    "rerank_model": resolved_model,
                    "rerank_score": rounded,
                    "rerank_fallback": False,
                    "rerank_requested": "qwen_local",
                },
            }
        )
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    _LOCAL_RERANK_RUNTIME.update(last_candidates=top_n, last_seconds=round(perf_counter() - started, 3))
    logger.info("Local rerank finished: candidates=%d seconds=%.2f", top_n, perf_counter() - started)
    return [*ranked, *tail][:limit]


def preload_local_reranker() -> dict[str, Any]:
    rerank_local_documents(
        "What does DevFlow do?",
        [{"title": "DevFlow", "snippet": "DevFlow helps developers search project knowledge."}],
        1,
    )
    return local_rerank_runtime_status()
