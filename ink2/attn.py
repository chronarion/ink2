"""Prefill attention on the GB300 (SM103). The image sends every prefill attention call to the slower upstream cute FA4
fallback on SM103, because tml_fa4's sheared-bias kernel traps there for the full-attention layers' shapes; its own note
says the sliding-window shapes are unaffected. INK2_FA4=window routes the 40 window layers back to the sheared kernel
(full-attention layers stay on the fallback); INK2_FA4=all (default) routes every layer: measured on this GB300 with 32-token KV pages, no trap from 8K to 137K-token
prompts, prompt log-likelihood equal to the fallback's, 137K-token prefill 43.9 -> 21.3 s; full-attention layers with
128-token KV pages (the shape the image's note names) stay on the fallback. INK2_FA4=none keeps the image's choice."""
import os
import sys

MODE = os.environ.get("INK2_FA4", "all")


def install():
    if MODE not in ("window", "all"):
        return
    from vllm.models.inkling2.ops import attention as A
    orig = A.Inkling2FA4RelAttentionKernel.kernel
    state = {"sheared": False}
    image_choice = A._use_sheared_fa4
    A._use_sheared_fa4 = lambda: state["sheared"] or image_choice()

    def kernel(q, key_cache, value_cache, *, window_size, **kw):
        local = tuple(window_size) != (-1, -1)
        state["sheared"] = local or (MODE == "all" and key_cache.shape[1] != 128)
        try:
            return orig(q, key_cache, value_cache, window_size=window_size, **kw)
        finally:
            state["sheared"] = False
    A.Inkling2FA4RelAttentionKernel.kernel = staticmethod(kernel)
    sys.stderr.write(f"INK2_FA4 {MODE}: sheared tml_fa4 kernel for "
                     f"{'all layers' if MODE == 'all' else 'the window layers'}\n")
