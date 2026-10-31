import base64
import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hubris import catalog, client, convert  # noqa: E402
from hubris.client import HubrisError  # noqa: E402
from hubris.nodes import build_video_body, check_image_options  # noqa: E402

KLING = {
    "id": "kwaivgi/kling-v3.0-std",
    "supported_durations": [5, 10],
    "supported_resolutions": ["720p"],
    "supported_aspect_ratios": ["16:9", "9:16"],
    "supported_frame_images": ["first_frame", "last_frame"],
    "generate_audio": True,
    "seed": False,
    "required_input": None,
}
AVATAR = {**KLING, "id": "heygen/avatar-iv", "supported_durations": [], "supported_frame_images": [],
          "required_input": "image"}


@pytest.fixture(autouse=True)
def fake_catalog(monkeypatch):
    monkeypatch.setattr(catalog, "_models", [
        {"id": "a/text", "output_modalities": ["text"], "supported_parameters": ["seed"]},
        {"id": "a/tts", "output_modalities": ["speech"], "supported_voices": ["v1", "v2"]},
        {"id": "a/chat-image", "output_modalities": ["image", "text"]},
        {"id": "a/flux", "output_modalities": ["image"], "supported_parameters": ["aspect_ratio", "input_references"]},
        {"id": "a/gpt-image", "output_modalities": ["image"], "supported_parameters": []},
        {"id": "a/vector", "output_modalities": ["image"], "description": "the vector (SVG) variant"},
        {"id": "a/video", "output_modalities": ["video"]},
    ])


def test_text_list_excludes_speech_models():
    assert "a/tts" not in catalog.model_ids("text")


def test_image_list_excludes_svg_models():
    assert catalog.model_ids("image") == ["a/chat-image", "a/flux", "a/gpt-image"]


def test_text_and_image_models_go_through_chat():
    assert catalog.is_chat_image_model("a/chat-image")
    assert not catalog.is_chat_image_model("a/flux")


def test_aspect_ratio_rejected_before_request_when_model_sets_size():
    with pytest.raises(HubrisError, match="aspect_ratio auto"):
        check_image_options("a/gpt-image", "16:9", [])


def test_references_rejected_when_model_takes_none():
    with pytest.raises(HubrisError, match="reference images"):
        check_image_options("a/gpt-image", "auto", ["data:image/jpeg;base64,xx"])


def test_supported_image_options_pass():
    check_image_options("a/flux", "16:9", ["data:image/jpeg;base64,xx"])


def test_video_body_keeps_only_supported_settings():
    body = build_video_body(KLING, "fox", 5, "720p", "16:9", True, 42, ["first"], [])
    assert body == {
        "model": KLING["id"], "prompt": "fox", "duration": 5, "resolution": "720p", "aspect_ratio": "16:9",
        "generate_audio": True,
        "frame_images": [{"type": "image_url", "image_url": {"url": "first"}, "frame_type": "first_frame"}],
    }


def test_video_auto_values_are_not_sent():
    body = build_video_body(KLING, "fox", 0, "auto", "auto", False, 1, [], [])
    assert body == {"model": KLING["id"], "prompt": "fox"}


@pytest.mark.parametrize("args, message", [
    ((7, "auto", "auto", False), "durations"),
    ((5, "1080p", "auto", False), "resolution"),
    ((5, "auto", "1:1", False), "aspect_ratio"),
])
def test_video_unsupported_setting_fails_before_request(args, message):
    with pytest.raises(HubrisError, match=message):
        build_video_body(KLING, "fox", *args, 1, [], [])


def test_video_audio_rejected_for_silent_model():
    with pytest.raises(HubrisError, match="audio"):
        build_video_body({**KLING, "generate_audio": False}, "fox", 5, "auto", "auto", True, 1, [], [])


def test_video_last_frame_rejected_when_unsupported():
    with pytest.raises(HubrisError, match="last_frame"):
        build_video_body({**KLING, "supported_frame_images": ["first_frame"]}, "fox", 5, "auto", "auto", False, 1,
                         [], ["last"])


def test_avatar_photo_goes_to_input_references():
    body = build_video_body(AVATAR, "", 0, "auto", "auto", False, 1, ["photo"], [])
    assert body["input_references"] == [{"type": "image_url", "image_url": {"url": "photo"}}]
    assert "prompt" not in body


def test_avatar_without_photo_fails():
    with pytest.raises(HubrisError, match="first_frame"):
        build_video_body(AVATAR, "hi", 0, "auto", "auto", False, 1, [], [])


def test_empty_prompt_fails_for_text_to_video():
    with pytest.raises(HubrisError, match="Prompt is empty"):
        build_video_body(KLING, "  ", 5, "auto", "auto", False, 1, [], [])


def test_decode_image_from_data_url():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (255, 0, 0)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    image = convert.decode_image(url)
    assert image.shape == (3, 4, 3)
    assert image[0, 0].tolist() == [1.0, 0.0, 0.0]


def test_decode_svg_gives_readable_error():
    with pytest.raises(HubrisError, match="SVG"):
        convert.decode_image(base64.b64encode(b"<svg xmlns='http://www.w3.org/2000/svg'/>").decode())


def test_stack_resizes_to_first_image():
    batch = convert.stack_images([torch.zeros(8, 8, 3), torch.ones(4, 6, 3)])
    assert batch.shape == (2, 8, 8, 3)


def test_tensor_to_data_urls_limits_and_encodes_jpeg():
    urls = convert.tensor_to_data_urls(torch.rand(3, 16, 16, 3), limit=1)
    assert len(urls) == 1 and urls[0].startswith("data:image/jpeg;base64,")


def test_missing_key_explains_where_to_put_it(monkeypatch, tmp_path):
    monkeypatch.delenv("HUBRIS_API_KEY", raising=False)
    monkeypatch.setattr(client, "KEY_FILE", tmp_path / "hubris_api_key.txt")
    with pytest.raises(HubrisError, match="hubris.pw/keys"):
        client.get_api_key()


def test_key_file_is_read(monkeypatch, tmp_path):
    monkeypatch.delenv("HUBRIS_API_KEY", raising=False)
    key_file = tmp_path / "hubris_api_key.txt"
    key_file.write_text("sk-gw-abc\n", encoding="utf-8")
    monkeypatch.setattr(client, "KEY_FILE", key_file)
    assert client.get_api_key() == "sk-gw-abc"


def test_bad_base64_gives_readable_error():
    with pytest.raises(HubrisError, match="cannot be decoded"):
        convert.decode_image("data:image/png;base64,@@@")


def test_retry_until_repeats_transient_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise HubrisError("502", 502)
        return "ok"

    assert client.retry_until(float("inf"), flaky) == "ok"
    assert len(calls) == 3


def test_retry_until_does_not_repeat_client_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(HubrisError):
        client.retry_until(float("inf"), lambda: (_ for _ in ()).throw(HubrisError("404", 404)))


def test_failed_catalog_fetch_is_retried_later(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "_models", None)
    monkeypatch.setattr(catalog, "_failed_at", 0.0)
    monkeypatch.setattr(catalog, "CACHE_FILE", tmp_path / "cache.json")
    answers = [HubrisError("down"), {"data": [{"id": "a/text", "output_modalities": ["text"]}]}]

    def fake_request(*args, **kwargs):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, "request", fake_request)
    assert catalog._load() == []
    monkeypatch.setattr(catalog, "_failed_at", -1000.0)
    assert [m["id"] for m in catalog._load()] == ["a/text"]


def test_empty_catalog_does_not_overwrite_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text('[{"id": "a/old"}]', encoding="utf-8")
    monkeypatch.setattr(catalog, "_models", None)
    monkeypatch.setattr(catalog, "_failed_at", 0.0)
    monkeypatch.setattr(catalog, "CACHE_FILE", cache)
    monkeypatch.setattr(client, "request", lambda *a, **k: {"data": []})
    assert [m["id"] for m in catalog._load()] == ["a/old"]
    assert "a/old" in cache.read_text(encoding="utf-8")


def test_image_node_returns_paid_images_when_some_fail(monkeypatch):
    from hubris.nodes import HubrisImage

    calls = []

    def fake_images_api(model, prompt, aspect_ratio, seed, references):
        calls.append(seed)
        if len(calls) == 2:
            raise HubrisError("moderation", 400)
        return [torch.zeros(4, 4, 3)]

    monkeypatch.setattr(HubrisImage, "_images_api", staticmethod(fake_images_api))
    (batch,) = HubrisImage().run("a/flux", "fox", "auto", 3, 0)
    assert batch.shape[0] == 2


def test_image_node_fails_when_all_fail(monkeypatch):
    from hubris.nodes import HubrisImage

    def failing(*args):
        raise HubrisError("down", 503)

    monkeypatch.setattr(HubrisImage, "_images_api", staticmethod(failing))
    with pytest.raises(HubrisError, match="down"):
        HubrisImage().run("a/flux", "fox", "auto", 2, 0)


def test_image_node_refuses_unknown_model():
    from hubris.nodes import HubrisImage

    with pytest.raises(HubrisError, match="not in the Hubris catalog"):
        HubrisImage().run("a/missing", "fox", "auto", 1, 0)


def test_video_poll_survives_a_server_error(monkeypatch, tmp_path):
    from hubris import nodes
    from hubris.nodes import HubrisVideo

    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(nodes, "_save_video", lambda video_id, content: tmp_path / f"{video_id}.mp4")
    answers = [HubrisError("502", 502), {"status": "in_progress"}, None, {"status": "completed"}]

    class Raw:
        content = b"mp4"

    def fake_request(method, path, **kwargs):
        if path.endswith("/content"):
            return Raw()
        answer = answers.pop(0)
        if answer is None:
            raise HubrisError("Cannot reach Hubris API: ConnectionError")
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(client, "request", fake_request)
    path = HubrisVideo()._wait_and_download("vid-1", {"status": "pending"}, None)
    assert path.name == "vid-1.mp4"
    assert answers == []


def test_video_without_video_type_fails_before_paying(monkeypatch):
    from hubris.nodes import HubrisVideo

    monkeypatch.setitem(sys.modules, "comfy_api", None)
    monkeypatch.setattr(client, "request", lambda *a, **k: pytest.fail("no request expected"))
    with pytest.raises(HubrisError, match="update ComfyUI"):
        HubrisVideo().run("a/video", "fox", 5, "auto", "auto", False, 0)
