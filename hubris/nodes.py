"""ComfyUI nodes: Hubris Chat, Image, Video and Speech."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import catalog, client, convert
from .client import HubrisError

CATEGORY = "Hubris"
ASPECT_RATIOS = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
VIDEO_RESOLUTIONS = ["auto", "480p", "720p", "1080p"]
VIDEO_ASPECT_RATIOS = ["auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
SEED = ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "control_after_generate": "fixed",
                "tooltip": "Same inputs are not re-run (and not paid again); change the seed or set it to randomize to get a new result. Sent to the model only if it supports seed."})


def _progress(node_id: str | None, text: str) -> None:
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        pass


def _check_interrupted() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _user_content(prompt: str, image_urls: list[str]) -> Any:
    if not image_urls:
        return prompt
    parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    parts += [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
    return parts


class HubrisChat:
    """Text from any catalog model: prompt enhancement, image captioning, JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.model_ids("text"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 200000,
                                       "tooltip": "0 = model default"}),
                "seed": SEED,
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Images for vision models (each image in the batch is sent)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Chat completion through Hubris: one API key for every text model in the catalog."

    def run(self, model, prompt, system_prompt, temperature, max_tokens, seed, images=None):
        if not prompt.strip():
            raise HubrisError("Prompt is empty")
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _user_content(prompt, convert.tensor_to_data_urls(images))})
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens > 0:
            body["max_tokens"] = max_tokens
        if catalog.supports(model, "seed"):
            body["seed"] = seed
        # Reasoning models can think for many minutes; a short read timeout would bill without an answer.
        response = client.request("POST", "/chat/completions", json=body, timeout=(15, 1800))
        choice = (response.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text:
            raise HubrisError(f"Model returned no text (finish_reason: {choice.get('finish_reason')})")
        return (text,)


class HubrisImage:
    """Image generation and editing with any image model of the catalog."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.model_ids("image"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto",
                                                 "tooltip": "auto = model default. Not every model accepts every ratio."}),
                "count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "seed": SEED,
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Images to edit or use as references"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generate or edit images: Nano Banana, GPT Image, FLUX, Seedream, Recraft and more."

    def run(self, model, prompt, aspect_ratio, count, seed, reference_images=None, unique_id=None):
        if not prompt.strip():
            raise HubrisError("Prompt is empty")
        references = convert.tensor_to_data_urls(reference_images)
        if catalog.get(model) is None:
            raise HubrisError(f"{model} is not in the Hubris catalog, or the catalog could not be loaded; "
                              "check the connection and try again in a minute")
        chat = catalog.is_chat_image_model(model)
        if not chat:
            check_image_options(model, aspect_ratio, references)
        generate = self._chat if chat else self._images_api
        # One request per image: many models reject n > 1, and the price is per image anyway.
        with ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
            futures = [pool.submit(generate, model, prompt, aspect_ratio, (seed + i) % 0x80000000, references)
                       for i in range(count)]
        images, errors = [], []
        for future in futures:
            try:
                images.extend(future.result())
            except HubrisError as error:
                errors.append(error)
        if not images:
            raise errors[0]
        if errors:
            # The images that did arrive are paid for: return them instead of failing the whole node.
            _progress(unique_id, f"{len(errors)} of {count} images failed: {errors[0]}")
        images = images[:count]
        return (convert.stack_images(images),)

    @staticmethod
    def _chat(model, prompt, aspect_ratio, seed, references):
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": _user_content(prompt, references)}],
            "modalities": ["image", "text"],
        }
        if aspect_ratio != "auto":
            body["image_config"] = {"aspect_ratio": aspect_ratio}
        if catalog.supports(model, "seed"):
            body["seed"] = seed
        response = client.request("POST", "/chat/completions", json=body)
        message = ((response.get("choices") or [{}])[0]).get("message") or {}
        urls = [(img.get("image_url") or {}).get("url") for img in message.get("images") or []]
        urls = [url for url in urls if url]
        if not urls:
            reply = (message.get("content") or "").strip()
            raise HubrisError("Model returned no image" + (f": {reply[:300]}" if reply else ""))
        return [convert.decode_image(url) for url in urls]

    @staticmethod
    def _images_api(model, prompt, aspect_ratio, seed, references):
        body: dict[str, Any] = {"model": model, "prompt": prompt, "response_format": "b64_json"}
        if aspect_ratio != "auto":
            body["aspect_ratio"] = aspect_ratio
        if catalog.supports(model, "seed"):
            body["seed"] = seed
        if references:
            body["input_references"] = [{"type": "image_url", "image_url": {"url": url}} for url in references]
        response = client.request("POST", "/images/generations", json=body)
        data = [item.get("b64_json") for item in response.get("data") or []]
        data = [item for item in data if item]
        if not data:
            raise HubrisError("Model returned no images")
        return [convert.decode_image(item) for item in data]


def check_image_options(model, aspect_ratio, references):
    """Per-image models: refuse settings the model does not take before any money is spent."""
    if aspect_ratio != "auto" and not catalog.supports(model, "aspect_ratio"):
        raise HubrisError(f"{model} sets the image size itself; use aspect_ratio auto")
    if references and not catalog.supports(model, "input_references"):
        raise HubrisError(f"{model} does not take reference images; disconnect reference_images or pick another model")


class HubrisVideo:
    """Video generation: text-to-video and image-to-video (first / last frame)."""

    POLL_SECONDS = 5
    TIMEOUT_SECONDS = 60 * 60

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.model_ids("video"),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 5, "min": 0, "max": 60,
                                     "tooltip": "Seconds. 0 = model default."}),
                "resolution": (VIDEO_RESOLUTIONS, {"default": "auto"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": False,
                                               "tooltip": "Only for models with sound; costs more"}),
                "seed": SEED,
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "Start frame (image-to-video) or the photo for avatar models"}),
                "last_frame": ("IMAGE", {"tooltip": "End frame, for models that support it"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "file_path")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generate video with Kling, Veo, Seedance, Wan, Hailuo and other models. Waits until the video is ready."

    def run(self, model, prompt, duration, resolution, aspect_ratio, generate_audio, seed,
            first_frame=None, last_frame=None, unique_id=None):
        try:  # checked before any money is spent
            from comfy_api.input_impl import VideoFromFile
        except ImportError as exc:
            raise HubrisError("This ComfyUI has no VIDEO type; update ComfyUI to use Hubris Video") from exc
        info = catalog.video_model(model)
        if info is None:
            raise HubrisError(f"Video model {model} is not available")
        body = build_video_body(info, prompt, duration, resolution, aspect_ratio, generate_audio, seed,
                                convert.tensor_to_data_urls(first_frame, 1), convert.tensor_to_data_urls(last_frame, 1))
        job = client.request("POST", "/videos", json=body, headers={"Idempotency-Key": str(uuid.uuid4())}, timeout=120)
        video_id = job.get("id")
        if not video_id:
            raise HubrisError("Hubris API did not return a video id; check https://hubris.pw/logs before retrying")
        try:
            path = self._wait_and_download(video_id, job, unique_id)
        except HubrisError as error:
            raise HubrisError(f"{error} (video {video_id}, see https://hubris.pw/logs)", error.status) from error
        return (VideoFromFile(str(path)), str(path))

    def _wait_and_download(self, video_id, job, unique_id):
        """Poll until the job ends; network hiccups and 5xx are retried, the job is already paid for."""
        status = job.get("status")
        started = time.monotonic()
        deadline = started + self.TIMEOUT_SECONDS

        def elapsed():
            return int(time.monotonic() - started)

        while status not in ("completed", "failed"):
            if time.monotonic() > deadline:
                raise HubrisError(f"Video is still not ready after {elapsed() // 60} min")
            _progress(unique_id, f"{status} · {elapsed()} s")
            for _ in range(self.POLL_SECONDS * 4):
                _check_interrupted()
                time.sleep(0.25)
            job = client.retry_until(deadline, lambda: client.request("GET", f"/videos/{video_id}", timeout=60),
                                     _check_interrupted)
            status = job.get("status")
        if status == "failed":
            error = job.get("error") or {}
            raise HubrisError(f"Video generation failed: {error.get('message') if isinstance(error, dict) else error}")
        _progress(unique_id, "downloading")
        response = client.retry_until(
            time.monotonic() + 15 * 60,
            lambda: client.request("GET", f"/videos/{video_id}/content", params={"index": 0}, timeout=600, raw=True),
            _check_interrupted,
        )
        path = _save_video(video_id, response.content)
        _progress(unique_id, f"done · {elapsed()} s")
        return path


def build_video_body(info, prompt, duration, resolution, aspect_ratio, generate_audio, seed, first, last):
    """Request body for POST /videos; rejects settings the model does not take before any money is spent."""
    model = info["id"]
    body: dict[str, Any] = {"model": model}
    if prompt.strip():
        body["prompt"] = prompt
    durations = info.get("supported_durations") or []
    if duration > 0 and durations:
        if duration not in durations:
            raise HubrisError(f"{model} supports durations {durations} s, not {duration}")
        body["duration"] = duration
    _pick(body, "resolution", resolution, info.get("supported_resolutions"), model)
    _pick(body, "aspect_ratio", aspect_ratio, info.get("supported_aspect_ratios"), model)
    if generate_audio:
        if not info.get("generate_audio"):
            raise HubrisError(f"{model} cannot generate audio; turn generate_audio off")
        body["generate_audio"] = True
    if info.get("seed"):
        body["seed"] = seed
    frame_types = info.get("supported_frame_images") or []
    required = info.get("required_input")
    if required == "video":
        raise HubrisError(f"{model} edits an existing video by URL; use the Hubris API directly for it")
    if required == "image":
        if not first:
            raise HubrisError(f"{model} needs a photo: connect it to first_frame")
        body["input_references"] = [{"type": "image_url", "image_url": {"url": first[0]}}]
    else:
        frames = []
        for urls, kind in ((first, "first_frame"), (last, "last_frame")):
            if not urls:
                continue
            if kind not in frame_types:
                raise HubrisError(f"{model} does not accept {kind}" + (f"; it takes {frame_types}" if frame_types else ""))
            frames.append({"type": "image_url", "image_url": {"url": urls[0]}, "frame_type": kind})
        if frames:
            body["frame_images"] = frames
    if "prompt" not in body and required != "image":
        raise HubrisError("Prompt is empty")
    return body


def _pick(body, field, value, allowed, model):
    if value == "auto":
        return
    if allowed and value not in allowed:
        raise HubrisError(f"{model} supports {field} {allowed}, not {value}")
    if allowed:
        body[field] = value


def _save_video(video_id: str, content: bytes) -> Path:
    try:
        import folder_paths

        directory = Path(folder_paths.get_temp_directory())
    except ImportError:
        directory = client.PACKAGE_DIR / "temp"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"hubris-{video_id}.mp4"
    path.write_bytes(content)
    return path


class HubrisSpeech:
    """Text to speech with ElevenLabs, Gemini TTS, MiniMax and other voices."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.model_ids("speech"),),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice": ("STRING", {"default": "",
                                     "tooltip": "Empty = first voice of the model. The error lists available voices."}),
                "seed": SEED,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Text to speech through Hubris."

    def run(self, model, text, voice, seed):
        if not text.strip():
            raise HubrisError("Text is empty")
        body: dict[str, Any] = {"model": model, "input": text, "response_format": "mp3"}
        voice = voice.strip() or next(iter(catalog.voices(model)), "")
        if voice:
            body["voice"] = voice
        response = client.request("POST", "/audio/speech", json=body, raw=True)
        return (convert.decode_audio(response.content),)


NODE_CLASS_MAPPINGS = {
    "HubrisChat": HubrisChat,
    "HubrisImage": HubrisImage,
    "HubrisVideo": HubrisVideo,
    "HubrisSpeech": HubrisSpeech,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HubrisChat": "Hubris Chat",
    "HubrisImage": "Hubris Image",
    "HubrisVideo": "Hubris Video",
    "HubrisSpeech": "Hubris Speech",
}
