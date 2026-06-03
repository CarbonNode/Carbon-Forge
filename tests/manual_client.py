"""Manual smoke client: python tests/manual_client.py [image-url-or-path] [base-url]"""
import asyncio
import json
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "https://picsum.photos/400"
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5125/mcp"
    headers = {"Authorization": f"Bearer {os.environ.get('FORGE_TOKEN', 'testtok')}"}
    async with streamablehttp_client(base, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", sorted(t.name for t in tools.tools))
            res = await s.call_tool("forge_status", {})
            print("STATUS:", res.content[0].text)
            res = await s.call_tool("list_projects", {})
            print("PROJECTS:", res.content[0].text)
            res = await s.call_tool("remove_background",
                                    {"image": image, "project": "TestProj"})
            print("REMOVE_BG:", res.content[0].text)
            out = json.loads(res.content[0].text)
            res = await s.call_tool("trim_image",
                                    {"image": out["url"], "project": "TestProj"})
            print("TRIM:", res.content[0].text)


asyncio.run(main())
