"""Hubris nodes for ComfyUI: hundreds of LLM, image, video and speech models with one API key."""

# ComfyUI imports this folder as a package; pytest imports this file on its own while collecting tests.
if __package__:
    from .hubris.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
