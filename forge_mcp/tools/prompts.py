"""Saved image-prompt favorites — MCP tools to store/recall reusable prompts."""


def register(mcp, ctx):
    prompts = ctx.prompts

    @mcp.tool()
    async def save_prompt(name: str, prompt: str, kind: str = "gemini",
                          refs: list[str] | None = None, model: str | None = None,
                          notes: str | None = None) -> dict:
        """Save a named, reusable image PROMPT (a favorite) so you can recall it later instead of
        retyping it. The prompt may contain a {subject} placeholder so one saved STYLE serves many
        subjects — fill it in when you fetch the prompt with get_prompt(subject=...).
          kind:  which generator the prompt targets — 'gemini' (edit_image), 'image'
                 (generate_image / Imagen) or 'icon' (generate_icon).
          refs:  optional default reference-image URLs (for gemini/edit-style prompts).
          model: optional preferred model.   notes: optional human note.
        Stored on the persistent /results volume, so favorites survive restarts/rebuilds.
        See list_prompts and get_prompt."""
        return prompts.save(name=name, prompt=prompt, kind=kind, refs=refs, model=model, notes=notes)

    @mcp.tool()
    async def list_prompts() -> list:
        """List every saved image-prompt favorite (name, kind, has_subject flag, model, notes).
        Use get_prompt to retrieve a full prompt, then hand it to generate_image / edit_image /
        generate_icon."""
        return prompts.list()

    @mcp.tool()
    async def get_prompt(name: str, subject: str | None = None) -> dict:
        """Fetch one saved prompt by name. If it has a {subject} placeholder and you pass `subject`,
        the returned `prompt` comes back with it filled in, ready to paste into a generate/edit tool.
        `needs_subject` is true when the prompt still contains an unfilled {subject}."""
        return prompts.render(name, subject=subject)

    @mcp.tool()
    async def delete_prompt(name: str) -> dict:
        """Delete a saved image-prompt favorite by name."""
        return prompts.delete(name)
