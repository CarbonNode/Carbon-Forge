"""Carbon Forge MCP service — assembly + entry point."""
import asyncio
import contextlib
import hmac
import os
import types

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from forge_mcp import engine, storage
from forge_mcp.config import load_config
from forge_mcp.jobs import JobStore
from forge_mcp.characters import CharacterStore
from forge_mcp.tools import register_all

cfg = load_config()

mcp = FastMCP(
    "carbon-forge",
    host=cfg.host,
    port=cfg.port,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
    # /mcp is reached via LAN IP, docker hostnames, and forge.carbonrouting.dev —
    # the SDK's Host allow-list rejects those (spike: 400). Bearer auth on /mcp
    # makes DNS-rebinding moot (browsers can't forge the Authorization header).
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

ctx = types.SimpleNamespace(
    cfg=cfg,
    jobs=JobStore(os.path.join(cfg.results_root, "jobs.json")),
    characters=CharacterStore(cfg.results_root),
    http=None,             # created inside lifespan
    poll_and_finish=None,  # set by tools.gen.register
)
register_all(mcp, ctx)


class BearerAuthMiddleware:
    """401 anything under /mcp without the right bearer token. /health and /files stay open."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            expected = f"Bearer {self.token}"
            supplied = headers.get("authorization", "")
            if not self.token or not hmac.compare_digest(supplied, expected):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def health(_request):
    return JSONResponse({"status": "ok", "service": "carbon-forge"})


async def serve_file(request):
    file_id = request.path_params["file_id"]
    name = request.path_params["name"]
    # ids are token_urlsafe — reject anything that could traverse
    if "/" in file_id or "\\" in file_id or ".." in file_id or ".." in name or "/" in name or "\\" in name:
        return JSONResponse({"error": "bad path"}, status_code=400)
    path = os.path.join(cfg.results_root, "files", file_id, name)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


async def _resume_job(job: dict):
    """Resume polling a Veo job that survived a restart; failures become readable status."""
    try:
        await ctx.poll_and_finish(job["id"], job["operation_name"])
    except Exception as e:
        ctx.jobs.update(job["id"], status="failed", error=str(e))


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/files/{file_id}/{name}", serve_file, methods=["GET"]))

    original_lifespan = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(app_):
        async with original_lifespan(app_):
            ctx.http = httpx.AsyncClient(timeout=120)
            background = [
                asyncio.create_task(storage.janitor_loop(cfg)),
                asyncio.create_task(engine.preload_default_model()),
            ]
            for job in ctx.jobs.mark_interrupted():  # resume Veo polls that survived restart
                background.append(asyncio.create_task(_resume_job(job)))
            try:
                yield
            finally:
                for t in background:
                    t.cancel()
                await ctx.http.aclose()

    app.router.lifespan_context = lifespan
    return BearerAuthMiddleware(app, cfg.token)


def main():
    import uvicorn
    uvicorn.run(build_app(), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
