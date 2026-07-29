"""Replicate tools — the entire Replicate model catalog callable through forge."""
import asyncio
import json

from forge_mcp import replicate_api as R
from forge_mcp import storage
from forge_mcp.generation import GenerationError
from forge_mcp.storage import sniff_mime


def register(mcp, ctx):
    cfg, jobs = ctx.cfg, ctx.jobs

    def _token():
        tok = cfg.replicate_api_token
        if not tok:
            raise GenerationError("REPLICATE_API_TOKEN is not configured on the forge service")
        return tok

    async def _save_outputs(pred: dict, project, subpath, filename) -> tuple[list, str | None]:
        base = storage.safe_filename(filename or (pred.get("model") or "replicate").split("/")[-1])
        urls = R.collect_file_urls(pred.get("output"))
        results = []
        for i, url in enumerate(urls, 1):
            data = await R.download_output(ctx.http, _token(), url,
                                           cfg.max_video_mb * 1024 * 1024)
            ext = R.ext_for(url, data, sniff_mime)
            name = base if len(urls) == 1 else f"{base}-{i}"
            results.append(await storage.save_result(data, project=project, subpath=subpath,
                                                     filename=name, ext=ext, cfg=cfg))
        return results, R.output_text(pred.get("output"))

    def _result(pred: dict, files: list, text: str | None, model: str) -> dict:
        out = {"status": pred.get("status"), "model": pred.get("model") or model,
               "prediction_id": pred.get("id"), "count": len(files), "files": files}
        if text is not None:
            out["output_text"] = text[:6000]
        elif not files and pred.get("output") is not None:
            dump = json.dumps(pred["output"])
            out["output"] = pred["output"] if len(dump) <= 4000 else dump[:4000] + "…"
        metrics = pred.get("metrics") or {}
        if metrics.get("predict_time"):
            out["predict_time_s"] = round(metrics["predict_time"], 2)
        return out

    async def _run_replicate_job(job_id: str, pred: dict):
        try:
            jobs.update(job_id, operation_name=pred.get("id"), message=pred.get("status"))
            elapsed = 0.0
            while pred.get("status") in ("starting", "processing") and elapsed < 3600:
                await asyncio.sleep(5)
                elapsed += 5
                pred = await R.get_prediction(ctx.http, _token(), pred)
                jobs.update(job_id, message=pred.get("status"))
            if pred.get("status") != "succeeded":
                raise GenerationError(pred.get("error") or
                                      f"prediction ended {pred.get('status') or 'unresolved'}")
            job = jobs.get(job_id)
            files, text = await _save_outputs(pred, job["project"], job["subpath"], job["filename"])
            jobs.update(job_id, status="done", message="complete", results=files,
                        **({"output_text": text[:6000]} if text else {}))
        except Exception as e:  # job boundary: everything becomes a readable failed status
            jobs.update(job_id, status="failed", error=str(e))

    @mcp.tool()
    async def replicate_search(query: str, limit: int = 8) -> dict:
        """Search Replicate's public catalog of thousands of hosted models — image/video/
        audio generation and editing, upscalers, background removal, face restore, OCR,
        speech, music, LLMs, 3D. Returns owner/name + blurb + run_count (popularity).
        Feed a result's `model` into replicate_model (input schema) or replicate_run."""
        return {"query": query,
                "results": await R.search_models(ctx.http, _token(), query, limit=limit)}

    @mcp.tool()
    async def replicate_model(model: str) -> dict:
        """Inspect one Replicate model before running it: description and its INPUT SCHEMA
        (every input's name/type/required/default/enum, in order) so replicate_run gets the
        arguments right. model: 'owner/name', e.g. 'black-forest-labs/flux-schnell'."""
        m = await R.get_model(ctx.http, _token(), model)
        version = m.get("latest_version") or {}
        desc = (m.get("description") or "").strip()
        return {"model": f"{m.get('owner')}/{m.get('name')}",
                "description": desc[:500],
                "run_count": m.get("run_count"),
                "latest_version": version.get("id"),
                "inputs": R.summarize_inputs(version)}

    @mcp.tool()
    async def replicate_run(model: str, input: dict, project: str,
                            files: dict | None = None, wait_seconds: int = 60,
                            subpath: str | None = None, filename: str | None = None) -> dict:
        """Run ANY Replicate model. model: 'owner/name' (latest version) or
        'owner/name:versionhash' (pinned). input: the model's input dict — check
        replicate_model first for the schema (replicate_search to find models).

        files: {input_field: 'https url' or '<Project>/<path>'} — each is uploaded to
        Replicate and injected into input, for image/audio/video inputs (img2img, upscale,
        restore, transcribe…). Outputs: every file the model returns is saved into the
        project workspace (subpath default assets/forge) and returned with url +
        workspace_path; text output (LLMs) comes back as output_text.

        Fast models finish inline (wait_seconds, default 60, max 300). Longer runs return
        {job_id} immediately — poll job_status. Billed per run to the org Replicate account;
        typical image models are fractions of a cent, video models can be $0.10-0.50+."""
        _token()
        wait = max(1, min(int(wait_seconds or 60), 300))
        payload = dict(input or {})
        for field, ref in (files or {}).items():
            resolved = await storage.resolve_input(ref, cfg=cfg, kind="any")
            fname = str(ref).replace("\\", "/").rsplit("/", 1)[-1] or "input"
            payload[field] = await R.upload_file(ctx.http, _token(), resolved.data, fname)

        pred = await R.create_prediction(ctx.http, _token(), model, payload, wait=wait)
        pred = await R.wait_prediction(ctx.http, _token(), pred, budget_s=wait)

        status = pred.get("status")
        if status in ("starting", "processing"):
            job = jobs.create(kind="replicate", model=model,
                              prompt=json.dumps(payload)[:200], project=project,
                              subpath=subpath, filename=filename)
            asyncio.create_task(_run_replicate_job(job["id"], pred))
            return {"job_id": job["id"], "status": "running",
                    "prediction_id": pred.get("id"),
                    "note": f"still {status} after {wait}s — poll job_status(job_id)"}
        if status != "succeeded":
            raise GenerationError(
                f"Replicate prediction {status or 'failed'}: {pred.get('error') or 'no error detail'}")
        results, text = await _save_outputs(pred, project, subpath, filename)
        return _result(pred, results, text, model)
