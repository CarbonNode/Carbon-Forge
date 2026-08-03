from forge_mcp.tools import audio, extract, gen, meta, proc, prompts, replicate, util, vid, world


def register_all(mcp, ctx):
    proc.register(mcp, ctx)
    gen.register(mcp, ctx); prompts.register(mcp, ctx)
    vid.register(mcp, ctx)
    audio.register(mcp, ctx)
    extract.register(mcp, ctx)
    replicate.register(mcp, ctx)
    util.register(mcp, ctx)
    world.register(mcp, ctx)
    meta.register(mcp, ctx)
