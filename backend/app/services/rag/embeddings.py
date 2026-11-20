import hashlib
import math
import logging
import os
import re
from functools import lru_cache
from time import perf_counter
from typing import Any

import httpx

from app.core.config import settings
from app.services.rag.local_inference import LOCAL_INFERENCE_LOCK

logger = logging.getLogger(__name__)
_LOCAL_EMBEDDING_LOCK = LOCAL_INFERENCE_LOCK
_LOCAL_EMBEDDING_RUNTIME: dict[str, Any] = {"loaded": False, "device": None}


def local_embedding_runtime_status() -> dict[str, Any]:
    return dict(_LOCAL_EMBEDDING_RUNTIME)

def _deterministic_tokens(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def deterministic_embedding(text: str, dimensions: int = 128) -> list[float]:
    # 本地演示和 Eval 的兜底实现；生产环境请替换为 LLM embedding API。
    vector = [0.0] * dimensions
    for token in _deterministic_tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _validate_vector_dimensions(vector: list[float], dimensions: int, provider: str) -> list[float]:
    if len(vector) != dimensions:
        raise ValueError(
            f"Embedding provider '{provider}' returned {len(vector)} dimensions; expected {dimensions}. "
            "Refusing to truncate or pad vectors."
        )
    return vector


def normalize_embedding_provider(provider: str) -> str:
    provider = (provider or "").strip().lower().replace("-", "_")
    aliases = {
        "api": "openai_compatible",
        "openai": "openai_compatible",
        "remote": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "sentence_transformer": "local",
        "sentence_transformers": "local",
        "local_sentence_transformers": "local",
        "local": "local",
        "qwen_local": "qwen_local",
        "dashscope": "dashscope",
        "hash": "deterministic",
        "deterministic": "deterministic",
    }
    resolved = aliases.get(provider)
    if resolved is None:
        raise ValueError(f"Unsupported embedding provider: {provider or '<empty>'}")
    return resolved


def validate_embedding_contract(provider: str, model: str, dimensions: int) -> None:
    configured = (
        normalize_embedding_provider(settings.embedding_provider),
        settings.embedding_model.strip(),
        settings.embedding_dimensions,
    )
    requested = (normalize_embedding_provider(provider), model.strip(), dimensions)
    if requested != configured:
        raise ValueError(
            "Embedding contract mismatch: "
            f"requested provider={requested[0]}, model={requested[1]}, dimensions={requested[2]}; "
            f"configured provider={configured[0]}, model={configured[1]}, dimensions={configured[2]}. "
            "All repositories sharing the Milvus collection must use one vector space."
        )


def embedding_contract() -> dict[str, Any]:
    return {
        "provider": normalize_embedding_provider(settings.embedding_provider),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


def embedding_provider_name() -> str:
    return normalize_embedding_provider(settings.embedding_provider)


def _embedding_payload(texts: list[str], dimensions: int, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
    }
    if dimensions:
        payload["dimensions"] = dimensions
    return payload


def _embed_openai_compatible(
    texts: list[str],
    dimensions: int,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    api_key = api_key or settings.embedding_api_key or os.getenv("OPENAI_API_KEY") or settings.llm_api_key
    base_url = (base_url or settings.embedding_base_url or settings.llm_base_url).rstrip("/")
    model = model or settings.embedding_model
    if not api_key:
        raise RuntimeError("embedding API key is not configured")

    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{base_url}/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_embedding_payload(texts, dimensions, model),
        )
        response.raise_for_status()
        rows = sorted(response.json().get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in rows]
        if len(vectors) != len(texts):
            raise ValueError("embedding response length mismatch")
        return [
            _validate_vector_dimensions([float(value) for value in vector], dimensions, "openai_compatible")
            for vector in vectors
        ]


@lru_cache(maxsize=4)
def _local_sentence_transformer(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise RuntimeError(
            "sentence-transformers is not installed; install backend/requirements-local-embedding.txt"
        ) from exc

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Qwen's saved dtype is BF16, which is very slow on many laptop CPUs.
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        model_name,
        device=resolved_device,
        model_kwargs={"dtype": dtype},
        local_files_only=settings.local_embedding_local_files_only,
    )
    _LOCAL_EMBEDDING_RUNTIME.update(
        loaded=True,
        model=model_name,
        device=str(model.device),
        dtype=str(next(model.parameters()).dtype),
        gpu=torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
    )
    logger.info("Local embedding loaded: model=%s device=%s dtype=%s", model_name, resolved_device, dtype)
    return model


def _embed_local(
    texts: list[str],
    dimensions: int,
    *,
    model_name: str | None = None,
    is_query: bool = False,
) -> list[list[float]]:
    resolved_model = model_name or settings.local_embedding_model
    encode_kwargs: dict[str, Any] = {}
    if is_query and "qwen3-embedding" in resolved_model.lower():
        encode_kwargs["prompt_name"] = "query"
    # Serialize local inference to avoid duplicate cold loads and VRAM spikes
    # from simultaneous uploads or searches.
    started = perf_counter()
    with _LOCAL_EMBEDDING_LOCK:
        model = _local_sentence_transformer(resolved_model, settings.local_embedding_device)
        vectors = model.encode(
            texts,
            batch_size=max(1, settings.local_embedding_batch_size),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **encode_kwargs,
        )
    logger.info("Local embedding finished: texts=%d seconds=%.2f", len(texts), perf_counter() - started)
    output: list[list[float]] = []
    for vector in vectors:
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        output.append(_validate_vector_dimensions([float(value) for value in values], dimensions, "local"))
    return output


def embed_texts(texts: list[str], dimensions: int | None = None) -> list[list[float]]:
    dimensions = dimensions or settings.embedding_dimensions
    return embed_knowledge_texts(
        texts,
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        dimensions=dimensions,
    )


def embed_text(text: str, dimensions: int | None = None) -> list[float]:
    return embed_texts([text], dimensions=dimensions)[0]


def embed_knowledge_texts(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
    is_query: bool = False,
) -> list[list[float]]:
    clean_texts = [text or "" for text in texts]
    if not clean_texts:
        return []
    validate_embedding_contract(provider, model, dimensions)
    resolved = normalize_embedding_provider(provider)
    if resolved in {"qwen_local", "local"}:
        return _embed_local(clean_texts, dimensions, model_name=model, is_query=is_query)
    if resolved == "dashscope":
        return _embed_openai_compatible(
            clean_texts,
            dimensions,
            model=model,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_embedding_base_url,
        )
    if resolved == "openai_compatible":
        return _embed_openai_compatible(clean_texts, dimensions, model=model)
    if resolved == "deterministic":
        return [deterministic_embedding(text, dimensions) for text in clean_texts]
    raise ValueError(f"Unsupported embedding provider: {resolved}")


def embed_knowledge_query(text: str, *, provider: str, model: str, dimensions: int) -> list[float]:
    return embed_knowledge_texts(
        [text],
        provider=provider,
        model=model,
        dimensions=dimensions,
        is_query=True,
    )[0]
