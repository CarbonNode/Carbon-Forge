"""Manual video smoke: python tests/manual_video_client.py <workspace-rel-video> [base-url]"""
import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    vid = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5125/mcp"
    headers = {"Authorization": f"Bearer {os.environ.get('FORGE_TOKEN', 'testtok')}"}
    async with streamablehttp_client(base, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool("video_trim", {"video_input": vid, "start": "0", "end": "2", "project": "TestProj"})
            print("VIDEO_TRIM:", res.content[0].text)
            res = await s.call_tool("video_extract_frames", {"video_input": vid, "timestamps": ["1"], "project": "TestProj"})
            print("FRAMES:", res.content[0].text)


asyncio.run(main())
