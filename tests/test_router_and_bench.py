import asyncio

import httpx

from bench import loadtest
from router import router as R
from tests.fake_vllm import make_app


def test_parse_metrics_sums_model_series():
    text = ('vllm:num_requests_waiting{model_name="a"} 3.0\n'
            'vllm:num_requests_waiting{model_name="b"} 2.0\n'
            'vllm:kv_cache_usage_perc{model_name="a"} 0.5\n')
    m = R.parse_metrics(text)
    assert m["waiting"] == 5.0 and m["kv"] == 0.5 and m["running"] == 0


def test_same_prefix_goes_to_same_backend_and_is_stable():
    backends = [R.Backend(f"http://vllm-{i}") for i in range(4)]
    body = {"messages": [{"role": "system", "content": "Politique RH interne ..."}, {"role": "user", "content": "Q1"}]}
    first = R.rendezvous(R.prefix_key(body), backends)
    assert all(R.rendezvous(R.prefix_key(body), backends) is first for _ in range(10))
    # Retirer une autre réplique ne déplace pas ce préfixe (propriété du hachage rendezvous).
    others = [b for b in backends if b is not first][:2] + [first]
    assert R.rendezvous(R.prefix_key(body), others) is first


def test_session_id_overrides_prefix():
    a = {"user": "session-42", "messages": [{"role": "user", "content": "x"}]}
    b = {"user": "session-42", "messages": [{"role": "user", "content": "tout autre chose"}]}
    assert R.prefix_key(a) == R.prefix_key(b) == "session-42"


def test_overloaded_backend_falls_back_to_least_loaded():
    backends = [R.Backend(f"http://vllm-{i}") for i in range(3)]
    pool = R.Pool(backends)
    body = {"prompt": "même préfixe"}
    preferred = R.rendezvous(R.prefix_key(body), backends)
    preferred.waiting = 50
    chosen, reason = pool.choose(body)
    assert reason == "least-loaded" and chosen is not preferred


def test_unhealthy_backends_are_skipped():
    pool = R.Pool([R.Backend("http://a", healthy=False), R.Backend("http://b")])
    chosen, _ = pool.choose({"prompt": "x"})
    assert chosen.url == "http://b"


def test_refresh_reads_health_and_metrics():
    apps = {"http://a": make_app("a", waiting=7), "http://b": make_app("b", waiting=0)}

    async def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}"
        transport = httpx.ASGITransport(app=apps[base])
        async with httpx.AsyncClient(transport=transport, base_url=base) as c:
            r = await c.request(request.method, request.url.path)
            return httpx.Response(r.status_code, content=r.content)

    pool = R.Pool([R.Backend("http://a"), R.Backend("http://b")])
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await pool.refresh(c)
    asyncio.run(go())
    assert pool.backends[0].waiting == 7 and pool.backends[1].waiting == 0
    assert all(b.healthy for b in pool.backends)


def test_loadtest_measures_ttft_itl_and_goodput(monkeypatch):
    app = make_app("x", tokens=6, delay=0.002)
    real_client = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.ASGITransport(app=app), base_url="http://test")
    monkeypatch.setattr(loadtest.httpx, "AsyncClient", patched)

    results, wall = asyncio.run(loadtest.run("http://test", "m", concurrency=4, n=12, max_tokens=6, key=None))
    rep = loadtest.report(results, wall, ttft_slo=5.0, itl_slo=1.0)
    assert rep["errors"] == 0 and rep["requests"] == 12
    assert rep["mean_output_tokens"] == 6
    assert rep["goodput_pct"] == 100.0
    # Le transport ASGI en mémoire regroupe les morceaux : on vérifie la structure, pas la valeur.
    assert rep["ttft_p95_s"] >= 0 and rep["itl_p50_ms"] >= 0


def test_pct():
    assert loadtest.pct([1, 2, 3, 4, 5], 50) == 3
    assert loadtest.pct([], 95) != loadtest.pct([], 95)   # NaN
