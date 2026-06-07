from forge_mcp.tools import audio, gen, meta, proc, vid


def register_all(mcp, ctx):
    proc.register(mcp, ctx)
    gen.register(mcp, ctx)
    vid.register(mcp, ctx)
    audio.register(mcp, ctx)
    meta.register(mcp, ctx)
