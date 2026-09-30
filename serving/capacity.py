"""Dimensionnement GPU d'un LLM servi par vLLM.

Répond à la question qu'on se pose avant d'acheter ou de réserver des GPU :
« combien de requêtes simultanées ce modèle tiendra-t-il sur ces cartes ? »

Mémoire GPU = poids du modèle + cache KV + surcoût (activations, graphes CUDA).
Le cache KV est ce qui limite la concurrence : chaque token en cours de génération
occupe 2 (K et V) x couches x têtes KV x dimension de tête x octets par valeur.
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_PARAM = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "awq": 0.56, "gptq": 0.56}  # 4 bits + échelles + embeddings non quantifiés
KV_BYTES = {"auto": 2.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0}
GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float          # milliards de paramètres
    num_layers: int
    num_kv_heads: int        # < num_heads avec le Grouped Query Attention
    head_dim: int


@dataclass(frozen=True)
class Capacity:
    weights_gib_per_gpu: float
    kv_bytes_per_token: int
    kv_cache_gib_total: float
    max_tokens_in_cache: int
    max_concurrent_full_context: int

    def summary(self, max_model_len: int) -> str:
        return (
            f"poids/GPU {self.weights_gib_per_gpu:.1f} GiB · "
            f"KV {self.kv_bytes_per_token / 1024:.0f} KiB/token · "
            f"cache KV {self.kv_cache_gib_total:.1f} GiB = {self.max_tokens_in_cache:,} tokens · "
            f"{self.max_concurrent_full_context} requêtes simultanées à {max_model_len:,} tokens"
        ).replace(",", " ")


def kv_bytes_per_token(m: ModelSpec, kv_dtype: str = "auto") -> int:
    return int(2 * m.num_layers * m.num_kv_heads * m.head_dim * KV_BYTES[kv_dtype])


def estimate(
    m: ModelSpec,
    gpu_mem_gib: float,
    num_gpus: int = 1,
    quantization: str = "bf16",
    kv_dtype: str = "auto",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    overhead_gib_per_gpu: float = 1.5,
) -> Capacity:
    """Estimation prudente, à confirmer par le log « KV cache » de vLLM au démarrage."""
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization doit être dans ]0, 1]")
    weights_gib = m.params_b * 1e9 * BYTES_PER_PARAM[quantization] / GIB
    weights_per_gpu = weights_gib / num_gpus          # tensor parallel : poids répartis
    budget_per_gpu = gpu_mem_gib * gpu_memory_utilization
    kv_per_gpu = budget_per_gpu - weights_per_gpu - overhead_gib_per_gpu
    if kv_per_gpu <= 0:
        raise ValueError(
            f"{m.name} ne tient pas : {weights_per_gpu:.1f} GiB de poids par GPU "
            f"pour un budget de {budget_per_gpu:.1f} GiB. Augmentez le tensor parallel ou quantifiez."
        )
    kv_total = kv_per_gpu * num_gpus
    per_token = kv_bytes_per_token(m, kv_dtype)
    max_tokens = int(kv_total * GIB // per_token)
    return Capacity(
        weights_gib_per_gpu=round(weights_per_gpu, 2),
        kv_bytes_per_token=per_token,
        kv_cache_gib_total=round(kv_total, 2),
        max_tokens_in_cache=max_tokens,
        max_concurrent_full_context=max_tokens // max_model_len,
    )


# Architectures de référence (valeurs des config.json publiés sur Hugging Face).
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec("Qwen2.5-7B-Instruct", 7.6, 28, 4, 128),
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec("Llama-3.1-8B-Instruct", 8.0, 32, 8, 128),
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": ModelSpec("Llama-3.1-70B-AWQ", 70.6, 80, 8, 128),
}
