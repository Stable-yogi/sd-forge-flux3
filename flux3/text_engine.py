# FLUX 3 text encoding for Forge Neo (Qwen3-VL 4B, 8 hidden-state taps -> 20480 wide).
# Matches ComfyUI comfy/text_encoders/flux.py Flux3* (GPL-3.0):
#   * instruct template once around the whole prompt
#   * right-pad with token 151643 to a multiple of 80; every pad is masked in the encoder
#   * pads sit at rope position 1 (transformers 4.57 behaviour the reference pins)
#   * hidden states before layers 4, 8, ..., 32, no final norm, joined on the feature axis
# The DiT attends to every token, pads included, so the output keeps its full padded length.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from modules.shared import opts

TEMPLATE_HEAD = "<|im_start|>user\n"
TEMPLATE_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
PAD_ID = 151643
PAD_MULTIPLE = 80
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]


def escape_brackets(text: str) -> str:
    # ComfyUI only weights (...); [ ] stay plain text there. Layout JSON uses [ ] a lot.
    return text.replace("[", "\\[").replace("]", "\\]")


class Flux3TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer([text])["input_ids"][0] if text else []

    def tokenize(self, texts):
        return [self._ids(TEMPLATE_HEAD + t + TEMPLATE_TAIL) for t in texts]

    def tokenize_line(self, line: str):
        parsed = parsing.parse_prompt_attention(escape_brackets(line), self.emphasis.name)
        if len(parsed) == 1 or all(w == 1.0 for _, w in parsed):
            text = "".join(t for t, _ in parsed)
            tokens = self._ids(TEMPLATE_HEAD + text + TEMPLATE_TAIL)
            multipliers = [1.0] * len(tokens)
        else:
            tokens, multipliers = [], []
            head = self._ids(TEMPLATE_HEAD)
            tokens += head
            multipliers += [1.0] * len(head)
            for text, weight in parsed:
                ids = self._ids(text)
                tokens += ids
                multipliers += [weight] * len(ids)
            tail = self._ids(TEMPLATE_TAIL)
            tokens += tail
            multipliers += [1.0] * len(tail)

        n_real = len(tokens)
        n_pad = -n_real % PAD_MULTIPLE
        tokens += [PAD_ID] * n_pad
        multipliers += [1.0] * n_pad
        return tokens, multipliers, n_real

    def __call__(self, texts: "SdConditioning"):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                tokens, multipliers, n_real = self.tokenize_line(line)
                cache[line] = self.process_tokens(tokens, multipliers, n_real)
            zs.append(cache[line])
        return zs

    def process_tokens(self, tokens, multipliers, n_real):
        device = memory_management.text_encoder_device()
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        embeds = self.text_encoder.get_input_embeddings()(ids)

        mult = torch.asarray([multipliers]).to(embeds)
        if hasattr(self.emphasis, "after_transformers"):  # older Forge Neo emphasis API
            self.emphasis.tokens = [tokens]
            self.emphasis.multipliers = mult
            self.emphasis.z = embeds
            self.emphasis.after_transformers()
            embeds = self.emphasis.z
        else:  # current upstream: emphasis(z, multipliers) returns z, or None for "None"
            z = self.emphasis(embeds, mult)
            embeds = embeds if z is None else z

        mask = torch.zeros((1, len(tokens)), device=device, dtype=torch.long)
        mask[:, :n_real] = 1
        position_ids = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)

        _, z = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=[n_real],
            intermediate_output=list(LAYERS),
            final_layer_norm_intermediate=False,
            position_ids=position_ids,
        )
        # z: (1, 8, seq, 2560) -> (seq, 8 * 2560), tap-major per token like ComfyUI movedim(1, 2).flatten(2)
        b, n, seq, d = z.shape
        return z.permute(0, 2, 1, 3).reshape(b, seq, n * d)[0]
