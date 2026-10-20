# FLUX 3 Image DiT for Forge Neo.
# Ported from ComfyUI comfy/ldm/flux3/model.py (GPL-3.0). Same math, Forge layer helpers.
#
# Layout: 5 text blocks, 5 image blocks (per "mode"), then 28 joint single blocks over
# [image, text]. The reference-image blocks in the weights stay unused (text-to-image only).
# Every block = parallel attention + SwiGLU MLP under one adaLN and one gate.
# Latent in/out is 128 channels (FLUX.2 VAE, 2x2 packed).
#
# Layers are built with plain torch.nn classes, looked up at build time, so Forge's
# using_forge_operations() swaps in its own Linear / LayerNorm / RMSNorm (fp8, gguf, int8 ...).

import logging

import torch
from torch import nn

from backend.nn.flux import EmbedND, LastLayer, MLPEmbedder, Modulation, attention, timestep_embedding

logger = logging.getLogger("flux3")


def modulated_norm(x, norm, mod):
    return torch.addcmul(mod.shift, norm(x), 1 + mod.scale)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, q, k, v):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, fused_qkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_in = nn.Linear(hidden_size, 2 * mlp_hidden_dim, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, spans, pe, transformer_options={}):
        # spans: (start, end, ModulationOut); the first modulates every row, the others redo their rows
        B, N, _ = x.shape
        x_mod = modulated_norm(x, self.pre_norm, spans[0][2])
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = modulated_norm(x[:, start:end], self.pre_norm, mod)

        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        mlp = self.mlp_in(x_mod)
        del x_mod

        # (B, N, H*D) -> (B, H, N, D) for Forge's attention()
        q = q.view(B, N, self.num_heads, -1).transpose(1, 2)
        k = k.view(B, N, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, N, self.num_heads, -1).transpose(1, 2)
        q, k = self.norm(q, k, v)
        out = self.attn_out(attention(q, k, v, pe=pe, transformer_options=transformer_options))

        gate, up = mlp.chunk(2, dim=-1)
        del mlp
        out = out + self.mlp_out(nn.functional.silu(gate) * up)
        del gate, up

        for start, end, mod in spans:
            x[:, start:end].addcmul_(out[:, start:end], mod.gate)
        return x


class Flux3(nn.Module):
    def __init__(self, hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=28, fused_qkv=False, flow_map=False, **kwargs):
        super().__init__()
        self.in_channels = 128
        self.image_rope_scale = 0.5
        self.flow_map = flow_map
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        content_modes = ("image", "image_cond")

        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=10000, axes_dim=[32, 32, 32, 32])
        self.emb_in = nn.ModuleDict({m: nn.Linear(self.in_channels, hidden_size, bias=False) for m in content_modes})
        self.txt_in = nn.Linear(20480, hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, hidden_size, bias=False)
        if flow_map:
            # step-distilled flow map: embeds the time each step jumps to
            self.ref_time_in = MLPEmbedder(256, hidden_size, bias=False)

        modulation = lambda: Modulation(hidden_size, double=False, bias=False)
        self.early_stream_modulations = nn.ModuleDict({m: modulation() for m in content_modes + ("txt",)})
        self.single_stream_modulations = nn.ModuleDict({m: modulation() for m in content_modes + ("txt",)})

        block = lambda: Block(hidden_size, num_heads, 3.0, fused_qkv=fused_qkv)
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([block() for _ in range(depth)]) for m in content_modes})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(hidden_size, 1, self.in_channels, bias=False)})

        self._warned_no_sigmas = False

    # ---- position ids (t, row, col, l) ----
    def image_ids(self, h, w, index, device):
        ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
        ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * self.image_rope_scale
        ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * self.image_rope_scale
        ids[..., 3] = index
        return ids.flatten(0, 1)

    def text_ids(self, length, device):
        ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
        ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
        return ids

    def rope(self, ids):
        # Forge layout: (1, 1, N, D/2, 2, 2), matches q/k as (B, H, N, D)
        return self.pe_embedder(ids.unsqueeze(0))

    def preprocess_conds(self, context, vec_target=None, transformer_options={}):
        # text only attends to itself at t = 0 in the early blocks
        device = context.device
        vec = self.time_in(timestep_embedding(context.new_zeros(1), 256))
        if vec_target is not None:
            vec = vec + vec_target
        txt = self.txt_in(context)
        streams = [(txt, "txt", self.text_ids(txt.shape[1], device), self.txt_mode_blocks)]
        for tokens, mode, ids, blocks in streams:
            mod = self.early_stream_modulations[mode](vec)[0]
            pe = self.rope(ids)
            for block in blocks:
                block(tokens, [(0, tokens.shape[1], mod)], pe, transformer_options)
        return txt

    def _target_sigma(self, timestep, transformer_options):
        sigmas = transformer_options.get("sampling_sigmas", None)
        if sigmas is None:
            if not self._warned_no_sigmas:
                logger.warning("FLUX 3: no step list from the sampler; the flow-map target falls back to 0.")
                self._warned_no_sigmas = True
            return timestep.new_zeros(1)
        sigmas = sigmas.to(device=timestep.device, dtype=torch.float32)
        i = int((sigmas - timestep[0].float()).abs().argmin())
        return sigmas[min(i + 1, sigmas.shape[0] - 1)].reshape(1)

    def forward(self, x, timestep, context=None, transformer_options={}, **kwargs):
        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device

        if context.shape[0] != B:
            context = context.repeat(B // context.shape[0], 1, 1)

        vec_target = None
        if self.flow_map:
            target = self._target_sigma(timestep, transformer_options)
            vec_target = self.ref_time_in(timestep_embedding(target.to(device=device, dtype=dtype), 256))

        txt_states = self.preprocess_conds(context, vec_target, transformer_options)

        # image_cond and text sit at t = 0
        vec = self.time_in(timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
        if vec_target is not None:
            vec = vec + vec_target
        vec_img, vec_zero = vec[:-1], vec[-1:]
        early = self.early_stream_modulations["image"](vec_img)[0]

        # text-to-image: the joint sequence is [image, text]
        img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
        parts = [(img, "image", self.image_ids(H, W, 0, device))]
        parts.append((txt_states, "txt", self.text_ids(txt_states.shape[1], device)))
        pe = self.rope(torch.cat([ids for _, _, ids in parts]))
        spans, start = [], 0
        for tokens, mode, _ in parts:
            spans.append((start, start + tokens.shape[1], self.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)[0]))
            start += tokens.shape[1]

        hw = H * W
        for block in self.content_mode_blocks["image"]:
            img = block(img, [(0, hw, early)], pe[:, :, :hw], transformer_options)

        x = torch.cat([img] + [tokens for tokens, _, _ in parts[1:]], dim=1)
        del img, parts
        transformer_options = transformer_options.copy()
        transformer_options["total_blocks"] = len(self.single_blocks)
        transformer_options["block_type"] = "single"
        for i, block in enumerate(self.single_blocks):
            transformer_options["block_index"] = i
            x = block(x, spans, pe, transformer_options)

        out = self.final_layer["image"](x[:, :hw], vec_img)
        return out.transpose(1, 2).reshape(B, self.in_channels, H, W)
