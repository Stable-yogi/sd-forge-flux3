# sd-forge-flux3 - FLUX 3 Image [klein] for Forge Neo

Adds text-to-image support for **FLUX 3 Image [klein]** (Black Forest Labs) to
[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).
It runs inside Forge: the normal checkpoint list, Generate button and samplers.

## Files you need

From the official ComfyUI package [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
(accept the model licence on Hugging Face first):

| File | Size | Put it in |
|---|---|---|
| `flux3-image-klein-int8-convrot.safetensors` (or `-bf16`) | 5.6 GB / 10.6 GB | `models/Stable-diffusion` |
| `qwen3vl_4b_bf16.safetensors` (text encoder) | 8.5 GB | `models/text_encoder` |
| `flux2-vae.safetensors` | 0.3 GB | `models/VAE` |

You only pick the checkpoint. The extension finds the text encoder (a file with `qwen3vl_4b` in its name)
and the VAE (a file with `flux2` and `vae` in its name) by itself.

## Use

1. Copy this folder into `extensions/` (or *Extensions > Install from URL*). Restart Forge fully.
2. Pick the UI preset **flux3**.
3. Pick the FLUX 3 checkpoint.
4. The preset sets the official settings: **Euler, Simple, 16 steps, CFG 1, 1024 x 1024**.

FLUX 3 likes long, full scene descriptions. Short prompts give plain results.

## Notes

- Text-to-image only. Shift is fixed at the official value (2.77259).
- The step-distilled model is a "flow map" model: it needs to know the next step, so keep CFG at 1.
- The `-base` model (not distilled) loads too, but needs more steps and a higher CFG.

## Tested on

Forge Neo `neo-2.30.1` (branch `neo`, commit `ad60793`, 21 October 2026), with an RTX 5090 (32 GB):
FLUX 3 Image [klein] int8 + Qwen3-VL 4B bf16, 1024 x 1024, 16 steps.

## If Forge Neo adds FLUX 3 itself

Forge Neo has no plan for built-in FLUX 3 today. If a later version adds it:
- **You update Forge to that version:** the extension sees it and switches itself off.
  You can then remove the folder from `extensions/`.
- **You keep an older Forge:** keep the extension. It keeps working; you do not have to update Forge.

## Licence

Code: GPL-3.0 (see `LICENSE` and `NOTICE.md`). The model weights have their own licence from
Black Forest Labs (FLUX Kommunity licence, non-commercial use). This project is not made or endorsed by Black Forest Labs.
