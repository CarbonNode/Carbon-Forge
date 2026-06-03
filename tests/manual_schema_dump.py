"""Dump generate_image inputSchema as seen direct vs through the gateway."""
import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


async def dump_direct():
    headers = {"Authorization": f"Bearer {os.environ['FORGE_TOKEN']}"}
    async with streamablehttp_client("http://192.168.0.177:5125/mcp", headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            t = next(t for t in tools.tools if t.name == "generate_image")
            print("DIRECT count schema:", json.dumps(t.inputSchema.get("properties", {}).get("count")))


async def dump_gateway():
    headers = {"Authorization": os.environ["GW_AUTH"]}
    async with sse_client("https://gateway.carbonrouting.dev/sse", headers=headers) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            t = next(t for t in tools.tools if t.name == "forge__generate_image")
            print("GATEWAY count schema:", json.dumps(t.inputSchema.get("properties", {}).get("count")))


asyncio.run(dump_direct())
asyncio.run(dump_gateway())
