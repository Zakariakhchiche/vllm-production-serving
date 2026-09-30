import pytest

from serving.capacity import MODELS, estimate, kv_bytes_per_token
from serving.render_args import ProfileError, load, to_args, validate


def test_kv_bytes_per_token_matches_architecture():
    # Qwen2.5-7B : 2 x 28 couches x 4 têtes KV x 128 x 2 octets = 56 KiB/token
    assert kv_bytes_per_token(MODELS["Qwen/Qwen2.5-7B-Instruct"]) == 57_344
    # Llama-3.1-8B : 2 x 32 x 8 x 128 x 2 = 128 KiB/token
    assert kv_bytes_per_token(MODELS["meta-llama/Llama-3.1-8B-Instruct"]) == 131_072


def test_fp8_kv_cache_halves_memory():
    m = MODELS["meta-llama/Llama-3.1-8B-Instruct"]
    assert kv_bytes_per_token(m, "fp8") * 2 == kv_bytes_per_token(m)


def test_qwen7b_on_l40s_capacity():
    cap = estimate(MODELS["Qwen/Qwen2.5-7B-Instruct"], gpu_mem_gib=44.4, max_model_len=8192)
    assert 14 < cap.weights_gib_per_gpu < 14.5
    assert cap.max_concurrent_full_context >= 50     # ~24 GiB de cache / 448 MiB par requête


def test_70b_bf16_does_not_fit_on_one_a100():
    with pytest.raises(ValueError, match="ne tient pas"):
        estimate(MODELS["hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"], 79.2, 1, "bf16")


def test_70b_awq_fits_on_two_a100():
    cap = estimate(MODELS["hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"], 79.2, 2, "awq", "fp8", 0.92, 16384)
    assert cap.max_concurrent_full_context >= 20


@pytest.mark.parametrize("profile", ["profiles/qwen2.5-7b-l40s.yaml", "profiles/llama3.1-70b-awq-2xa100.yaml"])
def test_profiles_are_valid_and_render(profile):
    p = load(profile)
    validate(p)
    args = to_args(p)
    assert args[:3] == ["vllm", "serve", p["model"]]
    assert "--enable-prefix-caching" in args
    assert "--api-key" not in args                    # la clé passe par VLLM_API_KEY, jamais par la ligne de commande


def test_invalid_tensor_parallel_is_rejected():
    p = load("profiles/llama3.1-70b-awq-2xa100.yaml")
    p["serving"]["tensor_parallel_size"] = 3
    with pytest.raises(ProfileError, match="doit diviser"):
        validate(p)
