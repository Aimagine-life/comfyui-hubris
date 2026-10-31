"""Conversions between ComfyUI tensors and files the API sends and receives."""

from __future__ import annotations

import base64
import binascii
import io

import numpy as np
import torch
from PIL import Image

from . import client
from .client import HubrisError

MAX_REFERENCE_SIDE = 2048


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """One [H, W, C] float image -> PIL."""
    array = (image.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array[..., :3] if array.shape[-1] >= 3 else array[..., 0])


def tensor_to_data_urls(images: torch.Tensor | None, limit: int | None = None) -> list[str]:
    """IMAGE batch [B, H, W, C] -> data URLs (JPEG, longest side up to 2048 px)."""
    if images is None:
        return []
    urls = []
    for image in images[:limit] if limit else images:
        pil = tensor_to_pil(image).convert("RGB")
        pil.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE), Image.LANCZOS)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=92)
        urls.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    return urls


def decode_image(data: str | bytes) -> torch.Tensor:
    """base64 / data URL / https URL / bytes -> [H, W, 3] float tensor."""
    if isinstance(data, str):
        data = _image_bytes(data)
    head = data[:512].lstrip().lower()
    if head.startswith(b"<svg") or head.startswith(b"<?xml"):
        raise HubrisError("This model returns SVG vector images, which ComfyUI cannot load. Pick a raster model.")
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception as exc:
        raise HubrisError("Hubris API returned an image that cannot be decoded") from exc
    array = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)


def _image_bytes(data: str) -> bytes:
    if data.startswith(("http://", "https://")):
        try:
            response = client.requests.get(data, timeout=120, headers={"User-Agent": client.USER_AGENT})
            response.raise_for_status()
        except client.requests.RequestException as exc:
            raise HubrisError(f"Cannot download the generated image: {exc.__class__.__name__}") from exc
        return response.content
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        return base64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise HubrisError("Hubris API returned an image that cannot be decoded") from exc


def stack_images(images: list[torch.Tensor]) -> torch.Tensor:
    """List of [H, W, C] -> [B, H, W, C]; images of other sizes are resized to the first."""
    height, width = images[0].shape[:2]
    resized = []
    for image in images:
        if image.shape[:2] != (height, width):
            image = torch.nn.functional.interpolate(
                image.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0)
        resized.append(image)
    return torch.stack(resized)


def decode_audio(data: bytes) -> dict:
    """mp3 bytes -> ComfyUI AUDIO {"waveform": [1, C, S], "sample_rate"}."""
    try:
        import av
    except ImportError as exc:
        raise HubrisError("This ComfyUI has no PyAV (av) package; update ComfyUI to decode audio") from exc
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            sample_rate = stream.rate
            resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
            chunks = []
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray())
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray())
    except Exception as exc:
        raise HubrisError("Hubris API returned audio that cannot be decoded") from exc
    if not chunks:
        raise HubrisError("Hubris API returned empty audio")
    samples = np.concatenate(chunks, axis=1).astype(np.float32)
    return {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": sample_rate}
