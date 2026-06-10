"""Pillow-based image format conversion — plain convert/resize/compress, no AI."""
from io import BytesIO

from PIL import Image, ImageOps


class ImagingError(Exception):
    """Readable, user-facing image-conversion failure."""


FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP",
           "gif": "GIF", "bmp": "BMP", "ico": "ICO", "avif": "AVIF"}
DEFAULT_QUALITY = {"JPEG": 90, "WEBP": 90, "AVIF": 75}


def convert_image(data: bytes, fmt: str, *, quality=None, max_dimension=None,
                  background="#ffffff") -> bytes:
    pil_fmt = FORMATS.get((fmt or "").lower())
    if not pil_fmt:
        raise ImagingError(f"Unsupported format '{fmt}' ({', '.join(sorted(set(FORMATS)))})")
    try:
        im = Image.open(BytesIO(data))
        im.load()
    except Exception as e:
        raise ImagingError(f"Could not open input image: {e}") from e
    im = ImageOps.exif_transpose(im)

    if pil_fmt == "ICO":
        max_dimension = min(max_dimension or 256, 256)  # ICO caps at 256px
    if max_dimension and max(im.size) > max_dimension:
        im.thumbnail((int(max_dimension), int(max_dimension)), Image.Resampling.LANCZOS)

    if pil_fmt == "JPEG" and im.mode not in ("RGB", "L"):
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, background)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        im = flat
    elif pil_fmt == "GIF" and im.mode not in ("P", "L"):
        im = im.convert("P", palette=Image.Palette.ADAPTIVE)
    elif pil_fmt == "BMP" and im.mode not in ("RGB", "L", "P"):
        im = im.convert("RGB")
    elif pil_fmt in ("WEBP", "AVIF", "PNG", "ICO") and im.mode not in ("RGB", "RGBA", "L", "LA", "P"):
        im = im.convert("RGBA")

    kwargs = {}
    q = quality or DEFAULT_QUALITY.get(pil_fmt)
    if q and pil_fmt in ("JPEG", "WEBP", "AVIF"):
        kwargs["quality"] = max(1, min(100, int(q)))
    if pil_fmt in ("PNG", "JPEG"):
        kwargs["optimize"] = True

    out = BytesIO()
    try:
        im.save(out, format=pil_fmt, **kwargs)
    except (KeyError, OSError, ValueError) as e:
        if pil_fmt == "AVIF":
            raise ImagingError("AVIF is not supported by this Pillow build — use webp instead") from e
        raise ImagingError(f"Could not save as {fmt}: {e}") from e
    blob = out.getvalue()
    if not blob:
        raise ImagingError("Conversion produced an empty file")
    return blob


def image_info(data: bytes) -> dict:
    try:
        with Image.open(BytesIO(data)) as im:
            return {"format": im.format, "width": im.width, "height": im.height,
                    "mode": im.mode, "frames": getattr(im, "n_frames", 1)}
    except Exception as e:
        raise ImagingError(f"Could not read image: {e}") from e
