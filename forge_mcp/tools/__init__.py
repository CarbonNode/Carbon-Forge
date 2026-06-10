from forge_mcp.tools import audio, extract, gen, meta, proc, util, vid


def register_all(mcp, ctx):
    proc.register(mcp, ctx)
    gen.register(mcp, ctx)
    vid.register(mcp, ctx)
    audio.register(mcp, ctx)
    extract.register(mcp, ctx)
    util.register(mcp, ctx)
    meta.register(mcp, ctx)
