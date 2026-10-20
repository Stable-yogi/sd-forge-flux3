"""
FLUX 3 Image registration - runs on Forge load. Adds the FLUX 3 Image architecture to Forge Neo
at runtime (no core edits):
  * detection fingerprint + a model_list entry (Flux3Base)
  * the loader's transformer builder (Flux3Transformer2DModel -> flux3.dit.Flux3)
  * the diffusion engine (flux3.engine.Flux3Image)
  * cond / uncond of different text lengths never get repeat-padded together
  * a bare DiT auto-loads its text encoder + VAE
  * a 'flux3' UI preset (Euler / Simple, 16 steps, CFG 1, 1024x1024)

Every step is wrapped so a failure never breaks Forge boot. If Forge ever ships FLUX 3 natively,
this extension steps aside.
"""
import json
import os
import struct
import sys
import traceback

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/sd-forge-flux3
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)
CFG_DIR = os.path.join(EXT_ROOT, "hf_config", "Flux3Image")

# Fixed time shift (mu) from the official BFL workflows: ModelSamplingFlux max_shift = base_shift = 2.77259.
FLUX3_SHIFT = 2.77259

PRESET = "flux3"
FLUX3_PRESET_DEFAULTS = {
    "flux3_t2i_sampler": "Euler", "flux3_i2i_sampler": "Euler",
    "flux3_t2i_scheduler": "Simple", "flux3_i2i_scheduler": "Simple",
    "flux3_t2i_step": 16, "flux3_t2i_hr_step": 16, "flux3_i2i_step": 16,
    "flux3_t2i_cfg": 1.0, "flux3_t2i_hr_cfg": 1.0, "flux3_i2i_cfg": 1.0,
    "flux3_t2i_width": 1024, "flux3_t2i_height": 1024,
    "flux3_i2i_width": 1024, "flux3_i2i_height": 1024,
    "flux3_t2i_batch_size": 1, "flux3_i2i_batch_size": 1,
}
# opts.set() needs these in data_labels (KeyError otherwise)
FLUX3_HIDDEN_OPTS = {
    "forge_checkpoint_flux3": None,
    "forge_additional_modules_flux3": [],
    "forge_unet_storage_dtype_flux3": "Automatic",
}

_EXTS = (".safetensors", ".sft", ".gguf")


def log(msg):
    print("[flux3] " + msg)


# ---------------------------------------------------------------- file sniffing
def _st_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)).keys() if k != "__metadata__"]


def _is_flux3_keys(keys):
    return any(k.endswith("content_mode_blocks.image.0.attn_out.weight") for k in keys) and any(k.endswith("single_stream_modulations.txt.lin.weight") for k in keys)


def _bare_flux3_dit(path):
    """True if the file IS a FLUX 3 DiT and carries no text encoder / VAE."""
    try:
        p = str(path)
        low = p.lower()
        if low.endswith((".safetensors", ".sft")):
            keys = _st_keys(p)
            return _is_flux3_keys(keys) and not any(k.startswith("text_encoders.") for k in keys)
        if low.endswith(".gguf"):
            with open(p, "rb") as f:
                head = f.read(16 << 20)
            return b"content_mode_blocks.image" in head and b"single_stream_modulations" in head and b"text_encoders." not in head
    except Exception:
        return False
    return False


def _is_qwen3vl_4b_te(keys):
    return "model.visual.deepstack_merger_list.0.norm.weight" in keys


def _is_flux2_vae(keys):
    return "decoder.post_quant_conv.weight" in keys or "bn.running_mean" in keys


def _find_modules():
    """Find the Qwen3-VL 4B text encoder and the FLUX.2 VAE in Forge's module folders."""
    from modules import shared

    try:
        from modules import paths

        root = paths.models_path
    except Exception:
        root = os.path.join(os.getcwd(), "models")
    co = getattr(shared, "cmd_opts", None)
    te_dirs = [os.path.join(root, "text_encoder")] + list(getattr(co, "text_encoder_dirs", []) or [])
    vae_dirs = [os.path.join(root, "VAE")] + list(getattr(co, "vae_dirs", []) or [])

    def files(dirs):
        out = []
        for d in dirs:
            try:
                out += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith(_EXTS)]
            except Exception:
                continue
        return out

    def pick(cands, prefer):
        for pk in prefer:
            for c in cands:
                if pk in os.path.basename(c).lower():
                    return c
        return cands[0] if cands else None

    te_c = [f for f in files(te_dirs) if "qwen3vl_4b" in os.path.basename(f).lower() or "qwen3-vl-4b" in os.path.basename(f).lower()]
    te = pick(te_c, ("flux3", "bf16"))
    vae_c = [f for f in files(vae_dirs) if "flux2" in os.path.basename(f).lower() and "vae" in os.path.basename(f).lower()]
    vae = pick(vae_c, ("flux2-vae", "flux2_vae"))
    if vae is None:  # renamed file: look inside
        for f in files(vae_dirs):
            try:
                if f.lower().endswith((".safetensors", ".sft")) and _is_flux2_vae(set(_st_keys(f))):
                    vae = f
                    break
            except Exception:
                continue
    return te, vae


def _native_flux3():
    try:
        import backend.loader as _l

        return any(getattr(M, "__name__", "") in ("Flux3", "Flux3Image") and not getattr(M, "_sd_forge_flux3", False) for M in getattr(_l, "possible_models", ()))
    except Exception:
        return False


# ---------------------------------------------------------------- architecture
def _register_arch():
    import backend.loader as loader
    import huggingface_guess.detection as detection
    from huggingface_guess import model_list

    from flux3.engine import Flux3Image

    if getattr(loader, "_sd_forge_flux3", False):
        return

    class Flux3Base(model_list.Flux2K4B):
        huggingface_repo = CFG_DIR
        unet_config = {"image_model": "flux3"}
        sampling_settings = {"shift": FLUX3_SHIFT}
        unet_extra_config = {}
        supported_inference_dtypes = [__import__("torch").bfloat16, __import__("torch").float32]

        def clip_target(self, state_dict={}):
            return {"qwen3vl_4b.transformer": "text_encoder"}

    model_list.Flux3Base = Flux3Base
    model_list.models.insert(0, Flux3Base)

    # 1. detection fingerprint (checked before every native rule)
    _orig_detect = detection.detect_unet_config

    def _detect(state_dict, key_prefix):
        if "{}content_mode_blocks.image.0.attn_out.weight".format(key_prefix) in state_dict and "{}single_stream_modulations.txt.lin.weight".format(key_prefix) in state_dict:
            keys = list(state_dict.keys())
            hidden = int(state_dict["{}content_mode_blocks.image.0.attn_out.weight".format(key_prefix)].shape[0])
            return {
                "image_model": "flux3",
                "hidden_size": hidden,
                "num_heads": hidden // 128,
                "depth": detection.count_blocks(keys, "{}txt_mode_blocks.".format(key_prefix) + "{}."),
                "depth_single_blocks": detection.count_blocks(keys, "{}single_blocks.".format(key_prefix) + "{}."),
                "fused_qkv": "{}single_blocks.0.qkv.weight".format(key_prefix) in state_dict,
                "flow_map": "{}ref_time_in.in_layer.weight".format(key_prefix) in state_dict,
            }
        return _orig_detect(state_dict, key_prefix)

    detection.detect_unet_config = _detect

    # 2. transformer builder: reuse Forge's own DiT build path (quant / fp8 / gguf handling),
    #    with our class swapped in for the duration of the call.
    _orig_lhc = loader.load_huggingface_component

    def _lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if cls_name != "Flux3Transformer2DModel":
            return _orig_lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        import backend.nn.krea as krea_nn

        from flux3.dit import Flux3

        if isinstance(state_dict, dict):
            for k in [k for k in state_dict if k.endswith("comfy_attention.config")]:
                state_dict.pop(k)
        native = krea_nn.SingleStreamDiT
        krea_nn.SingleStreamDiT = Flux3
        try:
            return _orig_lhc(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
        finally:
            krea_nn.SingleStreamDiT = native

    loader.load_huggingface_component = _lhc

    # 3. engine
    Flux3Image.matched_guesses = [Flux3Base]
    Flux3Image._sd_forge_flux3 = True
    loader.possible_models = (Flux3Image,) + tuple(loader.possible_models)

    # 4. cond / uncond with different text lengths run apart (the DiT attends to every token)
    from backend.args import dynamic_args
    from backend.sampling import condition

    _orig_can_concat = condition.ConditionCrossAttn.can_concat

    def _can_concat(self, other):
        if getattr(dynamic_args, "flux3", False) and self.cond.shape != other.cond.shape:
            return False
        return _orig_can_concat(self, other)

    condition.ConditionCrossAttn.can_concat = _can_concat

    loader._sd_forge_flux3 = True
    log("FLUX 3 Image architecture registered (shift %.5f)." % FLUX3_SHIFT)


def _register_loader_wrapper():
    """Reset the flux3 flag per load; a bare FLUX 3 DiT auto-loads its text encoder + VAE."""
    import backend.loader as loader
    import modules.sd_models as sdm
    from backend.args import dynamic_args

    if getattr(sdm.forge_loader, "_flux3", False):
        return
    _orig = sdm.forge_loader

    def _forge_loader(sd, additional_state_dicts=None):
        dynamic_args.flux3 = False
        try:
            if _bare_flux3_dit(sd):
                asd = list(additional_state_dicts or [])
                names = [os.path.basename(str(p)).lower() for p in asd]
                te, vae = _find_modules()
                add = []
                if not any("qwen3vl" in n or "qwen3-vl" in n for n in names):
                    if te:
                        add.append(te)
                    else:
                        log("no Qwen3-VL 4B text encoder found in the text encoder folders.")
                if not any("vae" in n for n in names):
                    if vae:
                        add.append(vae)
                    else:
                        log("no FLUX.2 VAE found in the VAE folders.")
                if add:
                    additional_state_dicts = add + asd
                    log("bare DiT -> auto-loaded: " + ", ".join(os.path.basename(m) for m in add))
        except Exception:
            log("auto module pick failed:\n" + traceback.format_exc())
        return _orig(sd, additional_state_dicts)

    _forge_loader._flux3 = True
    sdm.forge_loader = _forge_loader
    loader.forge_loader = _forge_loader


# ---------------------------------------------------------------- UI preset
def _register_preset():
    import modules_forge.presets as P

    if getattr(P.PresetArch.choices, "_flux3", False) is False:
        _orig_choices = P.PresetArch.choices

        def _choices():
            c = list(_orig_choices())
            if PRESET not in c:
                c.append(PRESET)
            return c

        _choices._flux3 = True
        P.PresetArch.choices = staticmethod(_choices)

    try:
        from modules import shared

        try:
            from modules.options import OptionInfo
        except Exception:
            from modules.shared import OptionInfo
        for k, default in FLUX3_HIDDEN_OPTS.items():
            if k not in shared.opts.data_labels:
                oi = OptionInfo(default)
                oi.section = (None, "Forge Hidden Options")
                shared.opts.data_labels[k] = oi
            shared.opts.data.setdefault(k, default)
        for k, v in FLUX3_PRESET_DEFAULTS.items():
            shared.opts.data.setdefault(k, v)
    except Exception:
        log("preset options step skipped:\n" + traceback.format_exc())

    import modules_forge.main_entry as ME

    if getattr(ME.on_preset_change, "_flux3", False):
        return
    _orig = ME.on_preset_change

    def _wrapped(preset):
        if preset == PRESET:
            try:
                from modules import shared

                for k, v in FLUX3_PRESET_DEFAULTS.items():
                    shared.opts.data[k] = v
                for k, v in FLUX3_HIDDEN_OPTS.items():
                    shared.opts.data.setdefault(k, v)
                if not (shared.opts.data.get("forge_additional_modules_flux3") or []):
                    auto = [m for m in _find_modules() if m]
                    if auto:
                        shared.opts.data["forge_additional_modules_flux3"] = auto
            except Exception:
                log("preset pre-fill error:\n" + traceback.format_exc())
        return _orig(preset)

    _wrapped._flux3 = True
    ME.on_preset_change = _wrapped


if _native_flux3():
    log("this Forge has FLUX 3 built in; the extension steps aside.")
else:
    try:
        _register_arch()
    except Exception:
        log("architecture registration FAILED (Forge boot unaffected):\n" + traceback.format_exc())
    try:
        _register_loader_wrapper()
    except Exception:
        log("loader wrapper skipped:\n" + traceback.format_exc())
    try:
        _register_preset()
    except Exception:
        log("preset registration skipped:\n" + traceback.format_exc())
