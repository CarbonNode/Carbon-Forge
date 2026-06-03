"""Status check: python tests/manual_status.py [base-url] — uses FORGE_TOKEN env."""
import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5125/mcp"
    headers = {"Authorization": f"Bearer {os.environ.get('FORGE_TOKEN', 'testtok')}"}
    async with streamablehttp_client(base, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("forge_status", {})
            print("STATUS:", res.content[0].text)
            res = await s.call_tool("list_projects", {})
            print("PROJECTS:", res.content[0].text[:500])


asyncio.run(main())
