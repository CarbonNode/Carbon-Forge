"""Full-chain E2E through the Carbon Cortex gateway: forge__ tools end to end.
Usage: python tests/manual_gateway_e2e.py  (reads GW_AUTH env: 'Bearer mgw-...')"""
import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    headers = {"Authorization": os.environ["GW_AUTH"]}
    async with sse_client("https://gateway.carbonrouting.dev/sse", headers=headers) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            forge_tools = sorted(t.name for t in tools.tools if t.name.startswith("forge__"))
            print(f"FORGE TOOLS ({len(forge_tools)}):", forge_tools)

            res = await s.call_tool("forge__forge_status", {})
            print("STATUS:", res.content[0].text)

            res = await s.call_tool("forge__remove_background",
                                    {"image": "https://picsum.photos/seed/forge/500",
                                     "project": "Carbon Forge",
                                     "filename": "e2e-test"})
            print("REMOVE_BG:", res.content[0].text)
            out = json.loads(res.content[0].text)

            res = await s.call_tool("forge__trim_image",
                                    {"image": out["url"], "project": "Carbon Forge",
                                     "filename": "e2e-trimmed"})
            print("TRIM:", res.content[0].text)


asyncio.run(main())
