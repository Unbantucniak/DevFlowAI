from threading import RLock


# Embedding and reranking share the GPU. Serialize their batches and cold loads
# to avoid simultaneous activation allocations on laptop GPUs.
LOCAL_INFERENCE_LOCK = RLock()
