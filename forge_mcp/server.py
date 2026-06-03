"""Carbon Forge MCP service — assembly + entry point."""
import hmac
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route

from forge_mcp.config import load_config

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


def build_app():
    app = mcp.streamable_http_app()
    app.router.routes.append(Route("/health", health, methods=["GET"]))
    app.router.routes.append(Route("/files/{file_id}/{name}", serve_file, methods=["GET"]))
    return BearerAuthMiddleware(app, cfg.token)


def main():
    import uvicorn
    uvicorn.run(build_app(), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
