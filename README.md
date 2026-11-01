# Hubris nodes for ComfyUI

Text, image, video and speech models from [Hubris](https://hubris.pw) inside ComfyUI, with one API key:

- **Hubris Chat**: hundreds of text models (Gemini, GPT, Claude, DeepSeek, Qwen and more). Takes images too, so it can write prompts, caption frames or describe a picture.
- **Hubris Image**: generate and edit images with Nano Banana, GPT Image, FLUX, Seedream, Recraft, Grok Imagine, Qwen Image and others. Reference images go in through `reference_images`.
- **Hubris Video**: Kling, Veo, Seedance, Wan, Hailuo, Grok Imagine Video and others. Text-to-video and image-to-video with a first and last frame. The node waits for the result and returns a `VIDEO`.
- **Hubris Speech**: text to speech with ElevenLabs, Gemini TTS, MiniMax, Kokoro and other voices. Returns `AUDIO`.

The model lists come from the live Hubris catalog when ComfyUI starts.

## Install

**ComfyUI Manager:** open *Manage Extensions*, search for **Hubris**, press **Install** and restart ComfyUI.

**comfy-cli:**

```
comfy node registry-install comfyui-hubris
```

**By hand:** clone the package into `custom_nodes` and restart ComfyUI:

```
cd ComfyUI/custom_nodes
git clone https://github.com/Aimagine-life/comfyui-hubris
```

No extra Python packages are needed on a current ComfyUI (the video and speech nodes rely on its `VIDEO` type and PyAV).

## API key

Create a key at [hubris.pw/keys](https://hubris.pw/keys), then either:

- paste it into `ComfyUI/user/hubris_api_key.txt` (one line, works without a restart, survives node updates), or
- set the `HUBRIS_API_KEY` environment variable and restart ComfyUI.

The key is never stored in workflows, so you can share them safely.

## Usage notes

- **Seed** defaults to `fixed`: running the same graph again reuses the previous result and does not spend money. Change the seed, or switch it to `randomize`, for a new result.
- **Image**, `aspect_ratio`: `auto` lets the model choose. Models that pick the size themselves reject other values before any request is made.
- **Image**, `count`: one request per image, so every model can return several. If some of them fail, the node returns the ones that arrived and shows how many failed.
- **Video**: duration, resolution, aspect ratio, audio and frame inputs are checked against the model before the job starts. The node shows the job status while it waits and rides out short network or server hiccups. If you cancel the run, the job still finishes on Hubris and is billed. The `file_path` output is the mp4 in ComfyUI's temp folder. Use **Save Video** to keep it.
- **Speech**: leave `voice` empty to use the model's first voice. A wrong voice name returns an error that lists the available voices.
- Vector (SVG) image models are left out of the list, because ComfyUI images are raster.

Prices are per request in rubles and are listed in the [catalog](https://hubris.pw/models). Every request shows up in your [request log](https://hubris.pw/logs) as **ComfyUI**.

## На русском

Узлы Hubris для ComfyUI: текст, картинки, видео и озвучка через один ключ, оплата в рублях. Ключ создаётся на [hubris.pw/keys](https://hubris.pw/keys). Вставьте его в файл `hubris_api_key.txt` в папке узла или задайте переменную `HUBRIS_API_KEY`. Подробная инструкция: [hubris.pw/docs/integrations/comfyui](https://hubris.pw/docs/integrations/comfyui).

## License

MIT
