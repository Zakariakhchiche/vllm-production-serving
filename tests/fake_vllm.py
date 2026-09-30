"""Faux serveur vLLM pour les tests : mêmes routes et même format de flux SSE."""
import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse


def make_app(name: str, waiting: float = 0, tokens: int = 5, delay: float = 0.001) -> FastAPI:
    app = FastAPI()
    app.state.waiting = waiting
    app.state.hits = 0

    @app.get("/health")
    async def health():
        return {}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return (f'vllm:num_requests_waiting{{model_name="m"}} {app.state.waiting}\n'
                f'vllm:num_requests_running{{model_name="m"}} 2.0\n'
                f'vllm:kv_cache_usage_perc{{model_name="m"}} 0.4\n')

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.hits += 1
        if not body.get("stream"):
            return {"choices": [{"message": {"content": name}}]}

        async def gen():
            for i in range(tokens):
                await asyncio.sleep(delay)
                yield f"data: {json.dumps({'choices': [{'delta': {'content': f'{name}-{i} '}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
