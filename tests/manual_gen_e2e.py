"""Live generation E2E through the gateway: generate_image + edit_image w/ reference.
Usage: python tests/manual_gen_e2e.py  (reads GW_AUTH env)"""
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

            res = await s.call_tool("forge__generate_image", {
                "prompt": "A glowing orange blacksmith anvil forging a pixel-art game sprite, dark studio background, dramatic lighting",
                "project": "Carbon Forge",
                "model": "imagen-4-fast",
                "count": 1,
                "aspect_ratio": "1:1",
                "filename": "gen-test-anvil",
            })
            print("GENERATE:", res.content[0].text)
            out = json.loads(res.content[0].text)
            ref_url = out["images"][0]["url"]

            res = await s.call_tool("forge__edit_image", {
                "prompt": "Make this image look like a blueprint sketch — blue background, white technical linework",
                "reference_images": [ref_url],
                "project": "Carbon Forge",
                "filename": "gen-test-blueprint",
            })
            print("EDIT:", res.content[0].text)


asyncio.run(main())
