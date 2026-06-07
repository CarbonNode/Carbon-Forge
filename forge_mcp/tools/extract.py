"""Content extraction tools — turn any URL into clean text.

Wraps mature libraries (no bespoke scraping): trafilatura (web page → markdown), yt-dlp (video/
audio sites → media + captions, 1000+ sites), faster-whisper (transcribe when there are no
captions). Output saves to the project workspace + returns the text, so it can feed knowledge.
Heavy imports are lazy (inside the tools) so a missing dep degrades to a clear error, not a crash.
"""
import asyncio
import os
import re
import tempfile

from forge_mcp import storage

_WHISPER_CACHE: dict = {}  # model_name -> loaded WhisperModel (load once, reuse)


def _strip_vtt(vtt: str) -> str:
    """VTT/SRT caption file → plain deduped text (drop headers, cue numbers, timestamp lines, tags)."""
    out, prev = [], None
    for line in vtt.splitlines():
        s = line.strip()
        if not s or s == "WEBVTT" or "-->" in s or s.isdigit() or s.startswith(("Kind:", "Language:", "NOTE")):
            continue
        s = re.sub(r"<[^>]+>", "", s)  # inline timing tags
        if s and s != prev:
            out.append(s)
            prev = s
    return " ".join(out)


def register(mcp, ctx):
    cfg = ctx.cfg

    @mcp.tool()
    async def extract_page(url: str, project: str, subpath: str | None = None,
                           filename: str | None = None, max_chars: int = 20000) -> dict:
        """Scrape a web page/article to CLEAN markdown (trafilatura) — strips nav/ads/boilerplate.
        Saves the .md to the project workspace and returns the text (truncated to max_chars in the
        response). Great for feeding articles/docs into knowledge. Best on static/SSR pages; a
        purely JS-rendered app may come back thin (use the media tool for video, or a browser tool)."""
        def _work():
            import trafilatura
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise storage.StorageError(f"Could not fetch {url}")
            md = trafilatura.extract(downloaded, output_format="markdown",
                                     include_links=True, include_comments=False) or ""
            meta = trafilatura.extract_metadata(downloaded)
            title = (getattr(meta, "title", None) if meta else None) or url
            return title, md
        title, md = await asyncio.to_thread(_work)
        if not md.strip():
            raise storage.StorageError(f"No extractable text at {url} (likely a JS-rendered page).")
        body = f"# {title}\n\nSource: {url}\n\n{md}"
        res = await storage.save_result(body.encode("utf-8"), project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or title[:60]), ext="md", cfg=cfg)
        return {"title": title, "url": url, "chars": len(md), "file": res, "text": body[:max_chars]}

    @mcp.tool()
    async def extract_media(url: str, project: str, transcribe: str = "auto", whisper_model: str = "base",
                            language: str | None = None, subpath: str | None = None,
                            filename: str | None = None, max_chars: int = 20000) -> dict:
        """Extract a TRANSCRIPT + metadata from a video/audio URL (YouTube, podcasts, X/TikTok, and
        1000+ sites via yt-dlp), then make it text you can search/ask.
          transcribe: 'auto' (use the site's captions if present, else Whisper) | 'captions' (captions
            only) | 'whisper' (always transcribe the audio locally) | 'off' (metadata only).
          whisper_model: tiny|base|small|medium (faster-whisper, local CPU). language: ISO code or auto.
        Saves the transcript .md to the workspace and returns it (truncated to max_chars)."""
        def _work():
            import yt_dlp
            tmp = tempfile.mkdtemp(prefix="forge_extract_")
            source = "none"
            # 1) metadata + (optionally) captions
            sub_opts = {
                "skip_download": True, "quiet": True, "no_warnings": True,
                "writesubtitles": transcribe in ("auto", "captions"),
                "writeautomaticsub": transcribe in ("auto", "captions"),
                "subtitleslangs": [language or "en", "en"], "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            }
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                info = ydl.extract_info(url, download=transcribe in ("auto", "captions"))
            meta = {
                "title": info.get("title"), "uploader": info.get("uploader") or info.get("channel"),
                "duration_sec": info.get("duration"), "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"), "webpage_url": info.get("webpage_url") or url,
            }
            transcript = ""
            vtts = [f for f in os.listdir(tmp) if f.endswith(".vtt")]
            if vtts and transcribe in ("auto", "captions"):
                with open(os.path.join(tmp, sorted(vtts)[0]), encoding="utf-8", errors="replace") as f:
                    transcript = _strip_vtt(f.read())
                source = "captions"
            # 2) Whisper fallback (no captions, and allowed)
            if not transcript and transcribe in ("auto", "whisper"):
                aud_opts = {
                    "format": "bestaudio/best", "quiet": True, "no_warnings": True,
                    "outtmpl": os.path.join(tmp, "audio.%(ext)s"),
                    "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
                }
                with yt_dlp.YoutubeDL(aud_opts) as ydl:
                    ydl.download([url])
                audio = next((os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith("audio.")), None)
                if audio:
                    from faster_whisper import WhisperModel
                    model = _WHISPER_CACHE.get(whisper_model)
                    if model is None:
                        model = WhisperModel(whisper_model, device="cpu", compute_type="int8",
                                             download_root=os.path.join(cfg.results_root, "whisper-models"))
                        _WHISPER_CACHE[whisper_model] = model
                    segments, _ = model.transcribe(audio, language=language)
                    transcript = " ".join(s.text.strip() for s in segments).strip()
                    source = f"whisper:{whisper_model}"
            return meta, transcript, source
        meta, transcript, source = await asyncio.to_thread(_work)
        title = meta.get("title") or url
        if not transcript and transcribe != "off":
            return {"title": title, "url": url, "metadata": meta, "transcript_source": "none",
                    "warning": "No captions found and transcription produced nothing.", "text": ""}
        header = (f"# {title}\n\nSource: {meta.get('webpage_url')}\nUploader: {meta.get('uploader')}\n"
                  f"Duration: {meta.get('duration_sec')}s | Transcript via: {source}\n\n")
        body = header + transcript
        res = await storage.save_result(body.encode("utf-8"), project=project, subpath=subpath,
                                        filename=storage.safe_filename(filename or title[:60]), ext="md", cfg=cfg)
        return {"title": title, "url": url, "metadata": meta, "transcript_source": source,
                "chars": len(transcript), "file": res, "text": body[:max_chars]}
