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
