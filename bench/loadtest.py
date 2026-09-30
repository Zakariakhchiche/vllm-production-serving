"""Banc de charge en streaming contre une API compatible OpenAI (vLLM ou routeur).

Mesure ce qui compte pour l'utilisateur et pour la facture :
- TTFT  : délai avant le premier token (réactivité perçue) ;
- ITL   : délai entre deux tokens (fluidité) ;
- débit : tokens générés par seconde sur l'ensemble du test ;
- goodput : part des requêtes qui respectent le SLO (TTFT et ITL p95).

    python -m bench.loadtest --url http://localhost:8000 --model assistant-7b \
        --concurrency 32 --requests 256 --ttft-slo 1.0 --itl-slo 0.05
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx

PROMPTS = [
    "Résume en trois points les risques d'un projet de migration vers le cloud.",
    "Écris une requête SQL qui calcule le chiffre d'affaires mensuel par région.",
    "Explique la différence entre un data lake et un data warehouse.",
    "Propose un plan de test pour une API de paiement.",
]
SYSTEM = "Tu es un assistant technique précis et concis. " * 20   # préfixe partagé : exerce le prefix caching


@dataclass
class Result:
    ok: bool
    ttft: float = 0.0
    itls: list[float] = field(default_factory=list)
    tokens: int = 0
    latency: float = 0.0


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


async def one(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int, key: str | None) -> Result:
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    start = time.perf_counter()
    last = None
    r = Result(ok=False)
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body, headers=headers) as resp:
            if resp.status_code != 200:
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                now = time.perf_counter()
                if last is None:
                    r.ttft = now - start
                else:
                    r.itls.append(now - last)
                last = now
                r.tokens += 1
        r.latency = time.perf_counter() - start
        r.ok = r.tokens > 0
    except httpx.HTTPError:
        pass
    return r


async def run(url: str, model: str, concurrency: int, n: int, max_tokens: int, key: str | None) -> tuple[list[Result], float]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
        async def task(i: int) -> Result:
            async with sem:
                return await one(client, url, model, PROMPTS[i % len(PROMPTS)], max_tokens, key)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(task(i) for i in range(n)))
        return list(results), time.perf_counter() - t0


def report(results: list[Result], wall: float, ttft_slo: float, itl_slo: float) -> dict:
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft for r in ok]
    itls = [x for r in ok for x in r.itls]
    good = [r for r in ok if r.ttft <= ttft_slo and pct(r.itls, 95) <= itl_slo] if ok else []
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "ttft_p50_s": round(pct(ttfts, 50), 3), "ttft_p95_s": round(pct(ttfts, 95), 3),
        "itl_p50_ms": round(pct(itls, 50) * 1000, 1), "itl_p95_ms": round(pct(itls, 95) * 1000, 1),
        "output_tokens_per_s": round(sum(r.tokens for r in ok) / wall, 1) if wall else 0,
        "e2e_p95_s": round(pct([r.latency for r in ok], 95), 3),
        "goodput_pct": round(100 * len(good) / len(results), 1) if results else 0,
        "mean_output_tokens": round(statistics.mean(r.tokens for r in ok), 1) if ok else 0,
    }


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://localhost:8000")
    a.add_argument("--model", required=True)
    a.add_argument("--concurrency", type=int, default=16)
    a.add_argument("--requests", type=int, default=128)
    a.add_argument("--max-tokens", type=int, default=256)
    a.add_argument("--ttft-slo", type=float, default=1.0)
    a.add_argument("--itl-slo", type=float, default=0.05)
    a.add_argument("--api-key", default=None)
    p = a.parse_args()
    results, wall = asyncio.run(run(p.url, p.model, p.concurrency, p.requests, p.max_tokens, p.api_key))
    print(json.dumps(report(results, wall, p.ttft_slo, p.itl_slo), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
