"""One-off: purge carboncortex.dev zone via the gateway's cloudflare connector.
Usage: python tests/manual_purge.py  (reads GW_AUTH env: 'Bearer mgw-...')"""
import asyncio
import os

from mcp import ClientSession
from mcp.client.sse import sse_client

ZONE = "cd3f1e2ac516d67f6c638610bcc45928"


async def main():
    headers = {"Authorization": os.environ["GW_AUTH"]}
    async with sse_client("https://gateway.carbonrouting.dev/sse", headers=headers) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            cf = [t.name for t in tools.tools if "cloudflare" in t.name and "purge" in t.name]
            print("PURGE TOOLS:", cf)
            if not cf:
                print("No purge tool found; cloudflare tools:",
                      [t.name for t in tools.tools if "cloudflare" in t.name][:10])
                return
            res = await s.call_tool(cf[0], {"zone_id": ZONE})
            print("PURGE:", res.content[0].text[:300])


asyncio.run(main())
