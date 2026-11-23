"""Image upload validation: size cap, real image check (not just the extension), server-side names."""

from __future__ import annotations

import io
import uuid

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.core.config import get_settings

_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}


async def save_image(upload: UploadFile) -> str:
    settings = get_settings()
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {settings.max_upload_mb:g} MB")
    if not data:
        raise HTTPException(400, "Empty image upload")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(415, "Unsupported or corrupt image") from exc
    if fmt not in _FORMATS:
        raise HTTPException(415, f"Unsupported image format: {fmt}")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{uuid.uuid4().hex}{_FORMATS[fmt]}"
    path.write_bytes(data)
    return str(path)
