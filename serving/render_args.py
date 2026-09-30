"""Transforme un profil YAML en commande `vllm serve`, après validation.

    python -m serving.render_args profiles/qwen2.5-7b-l40s.yaml

Objectif : qu'aucun paramètre incohérent n'atteigne un GPU loué à l'heure.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml

from serving.capacity import MODELS, estimate

ALLOWED_QUANT = {None, "awq", "gptq", "fp8"}


class ProfileError(ValueError):
    pass


def load(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate(p: dict) -> None:
    hw, srv = p["hardware"], p["serving"]
    tp = srv.get("tensor_parallel_size", 1)
    if hw["num_gpus"] % tp:
        raise ProfileError(f"tensor_parallel_size={tp} doit diviser num_gpus={hw['num_gpus']}")
    if srv.get("quantization") not in ALLOWED_QUANT:
        raise ProfileError(f"quantization inconnue : {srv.get('quantization')}")
    if not 0.5 <= srv.get("gpu_memory_utilization", 0.9) <= 0.95:
        raise ProfileError("gpu_memory_utilization hors de [0.5, 0.95] : risque d'OOM ou de gaspillage")
    if srv["max_num_seqs"] < 1 or srv["max_model_len"] < 512:
        raise ProfileError("max_num_seqs >= 1 et max_model_len >= 512 attendus")

    spec = MODELS.get(p["model"])
    if spec:  # contrôle de capacité quand l'architecture est connue
        quant = srv.get("quantization") or "bf16"
        cap = estimate(spec, hw["gpu_mem_gib"], tp, quant, srv.get("kv_cache_dtype", "auto"),
                       srv.get("gpu_memory_utilization", 0.9), srv["max_model_len"])
        if cap.max_concurrent_full_context < 1:
            raise ProfileError("le cache KV ne contient même pas une requête à max_model_len")


def to_args(p: dict) -> list[str]:
    srv = p["serving"]
    args = [
        "vllm", "serve", p["model"],
        "--served-model-name", p["served_name"],
        "--host", "0.0.0.0", "--port", str(srv.get("port", 8000)),
        "--tensor-parallel-size", str(srv.get("tensor_parallel_size", 1)),
        "--gpu-memory-utilization", str(srv.get("gpu_memory_utilization", 0.9)),
        "--max-model-len", str(srv["max_model_len"]),
        "--max-num-seqs", str(srv["max_num_seqs"]),
    ]
    if srv.get("quantization"):
        args += ["--quantization", srv["quantization"]]
    if srv.get("kv_cache_dtype", "auto") != "auto":
        args += ["--kv-cache-dtype", srv["kv_cache_dtype"]]
    if srv.get("enable_prefix_caching", True):
        args.append("--enable-prefix-caching")       # prompts système partagés : TTFT réduit
    if srv.get("enable_chunked_prefill", True):
        args.append("--enable-chunked-prefill")      # longs prompts sans bloquer le décodage
    # Authentification : vLLM lit la variable d'environnement VLLM_API_KEY.
    # On ne passe jamais --api-key, qui exposerait la clé dans `ps` et les logs.
    return args


def main(path: str) -> None:
    p = load(path)
    validate(p)
    spec = MODELS.get(p["model"])
    if spec:
        srv, hw = p["serving"], p["hardware"]
        cap = estimate(spec, hw["gpu_mem_gib"], srv.get("tensor_parallel_size", 1),
                       srv.get("quantization") or "bf16", srv.get("kv_cache_dtype", "auto"),
                       srv.get("gpu_memory_utilization", 0.9), srv["max_model_len"])
        print("# " + cap.summary(srv["max_model_len"]), file=sys.stderr)
    if p["serving"].get("api_key_env", "VLLM_API_KEY") != "VLLM_API_KEY":
        print("# attention : vLLM lit la clé dans VLLM_API_KEY", file=sys.stderr)
    print(shlex.join(to_args(p)))


if __name__ == "__main__":
    main(sys.argv[1])
