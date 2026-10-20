# Notice

**Code licence: GPL-3.0** (full text in `LICENSE`).
The FLUX 3 model code in `flux3/` is a port of ComfyUI's `comfy/ldm/flux3/model.py`
and `comfy/text_encoders/flux.py` (GPL-3.0, Comfy Org). The port is modified: it uses Forge Neo's own
layer helpers, attention and sampler hooks. Same math, measured equal to ComfyUI.

**No model weights are in this folder.** The FLUX 3 Image (Klein) weights come from Black Forest Labs
and have their own licence (FLUX Kommunity licence: non-commercial use). Read and follow that licence
before you use or share the model or anything you make from it.

This extension is not made, checked or endorsed by Black Forest Labs.

Config files in `hf_config/Flux3Image/` (text encoder, tokenizer, VAE, scheduler
configs) are copied from the public Qwen3-VL / FLUX.2 [klein] 4B repos (Apache-2.0).
