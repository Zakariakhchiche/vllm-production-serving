"""Routeur léger devant plusieurs répliques vLLM (API compatible OpenAI).

Stratégie inspirée de vLLM production-stack et llm-d :
1. Affinité de préfixe : les requêtes qui partagent le même début de prompt
   (prompt système, document RAG, historique de conversation) vont sur la même
   réplique, pour réutiliser son cache de préfixe et réduire le TTFT.
   -> hachage « rendezvous » : stable quand on ajoute ou retire une réplique.
2. Garde-fou de charge : si la réplique choisie a trop de requêtes en attente
   (vllm:num_requests_waiting), on bascule sur la moins chargée.
3. Santé : les répliques qui ne répondent pas à /health sont écartées.

Lancer : BACKENDS=http://vllm-0:8000,http://vllm-1:8000 uvicorn router.router:app
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

PREFIX_CHARS = int(os.getenv("PREFIX_CHARS", "2048"))       # taille du préfixe haché
MAX_WAITING = float(os.getenv("MAX_WAITING", "8"))           # seuil de bascule
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))

_WAITING_RE = re.compile(r"^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)", re.M)
_RUNNING_RE = re.compile(r"^vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)", re.M)
_KV_RE = re.compile(r"^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)", re.M)


@dataclass
class Backend:
    url: str
    healthy: bool = True
    waiting: float = 0.0
    running: float = 0.0
    kv_usage: float = 0.0

    @property
    def load(self) -> float:
        return self.waiting * 10 + self.running + self.kv_usage


def parse_metrics(text: str) -> dict[str, float]:
    """Somme par métrique (une ligne par model_name)."""
    def total(rx: re.Pattern) -> float:
        return sum(float(v) for v in rx.findall(text))
    return {"waiting": total(_WAITING_RE), "running": total(_RUNNING_RE), "kv": total(_KV_RE)}


def prefix_key(body: dict) -> str:
    """Début du prompt : system + premiers messages (chat) ou prompt (completions)."""
    if "messages" in body:
        text = "".join(f"{m.get('role')}:{m.get('content')}" for m in body["messages"] if isinstance(m.get("content"), str))
    else:
        text = str(body.get("prompt", ""))
    session = body.get("user") or ""          # affinité de session si le client la fournit
    return session or text[:PREFIX_CHARS]


def rendezvous(key: str, backends: list[Backend]) -> Backend:
    def score(b: Backend) -> int:
        return int.from_bytes(hashlib.blake2b(f"{key}|{b.url}".encode(), digest_size=8).digest(), "big")
    return max(backends, key=score)


@dataclass
class Pool:
    backends: list[Backend] = field(default_factory=list)

    def healthy(self) -> list[Backend]:
        return [b for b in self.backends if b.healthy]

    def choose(self, body: dict) -> tuple[Backend, str]:
        alive = self.healthy()
        if not alive:
            raise HTTPException(503, "aucune réplique vLLM disponible")
        preferred = rendezvous(prefix_key(body), alive)
        if preferred.waiting <= MAX_WAITING:
            return preferred, "prefix-affinity"
        least = min(alive, key=lambda b: b.load)
        return least, "least-loaded"

    async def refresh(self, client: httpx.AsyncClient) -> None:
        async def probe(b: Backend) -> None:
            try:
                h = await client.get(f"{b.url}/health", timeout=2)
                b.healthy = h.status_code == 200
                if b.healthy:
                    m = parse_metrics((await client.get(f"{b.url}/metrics", timeout=2)).text)
                    b.waiting, b.running, b.kv_usage = m["waiting"], m["running"], m["kv"]
            except httpx.HTTPError:
                b.healthy = False
        await asyncio.gather(*(probe(b) for b in self.backends))


pool = Pool([Backend(u.strip()) for u in os.getenv("BACKENDS", "").split(",") if u.strip()])
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5))


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def loop() -> None:
        while True:
            await pool.refresh(client)
            await asyncio.sleep(POLL_SECONDS)
    task = asyncio.create_task(loop()) if pool.backends else None
    yield
    if task:
        task.cancel()
    await client.aclose()


app = FastAPI(title="vLLM prefix-aware router", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"healthy_backends": len(pool.healthy()), "total": len(pool.backends)}


@app.get("/router/state")
async def state() -> list[dict]:
    return [b.__dict__ for b in pool.backends]


@app.post("/v1/{path:path}")
async def proxy(path: str, request: Request):
    body = await request.json()
    backend, reason = pool.choose(body)
    headers = {k: v for k, v in request.headers.items() if k.lower() in {"authorization", "content-type"}}
    url = f"{backend.url}/v1/{path}"
    extra = {"x-routed-to": backend.url, "x-routing-reason": reason}
    if body.get("stream"):
        req = client.build_request("POST", url, json=body, headers=headers)
        resp = await client.send(req, stream=True)
        return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code,
                                 media_type="text/event-stream", headers=extra,
                                 background=BackgroundTask(resp.aclose))
    resp = await client.post(url, content=json.dumps(body), headers=headers)
    return JSONResponse(resp.json(), status_code=resp.status_code, headers=extra)
