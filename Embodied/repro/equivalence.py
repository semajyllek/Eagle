# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Numerical equivalence checks for the batched forward.

The model is deterministic run-to-run, so the only batched-vs-single difference
is that bf16 matmul is not batch-invariant (cuBLAS picks a different GEMM at
batch > 1). These helpers demonstrate that root cause and confirm the model's
batched prefill matches B=1 to within it.
"""
import numpy as np
import torch


def bf16_gemm_demo(device="cuda", dtype=torch.bfloat16, dim=4096, n=64):
    """A single matmul is not batch-invariant in bf16 (but is in fp32).

    Returns ``{'bf16_maxdiff', 'fp32_maxdiff'}`` — the max |row-alone − row-in-batch|.
    """
    torch.manual_seed(0)
    x = torch.randn(n, dim, dtype=dtype, device=device)
    w = torch.randn(dim, dim, dtype=dtype, device=device)
    d_bf16 = ((x[:1] @ w) - (x @ w)[:1]).abs().max().item()
    d_fp32 = ((x[:1].float() @ w.float()) - (x.float() @ w.float())[:1]).abs().max().item()
    return {"bf16_maxdiff": d_bf16, "fp32_maxdiff": d_fp32}


def build_batched_inputs(worker, images, questions):
    """Left-padded batched processor inputs for ``images``/``questions``."""
    worker.tokenizer.padding_side = "left"
    if hasattr(worker.processor, "tokenizer"):
        worker.processor.tokenizer.padding_side = "left"
    texts, imgs = [], []
    for im, q in zip(images, questions):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": im}, {"type": "text", "text": q}]}]
        texts.append(worker.processor.py_apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        ii, _ = worker.processor.process_vision_info(msgs)
        imgs.extend(ii)
    return worker.processor(text=texts, images=imgs, videos=None,
                            return_tensors="pt", padding=True).to(worker.device)


@torch.no_grad()
def prefill_logits(worker, images, questions):
    """Plain causal prefill logits ``[B, S, V]`` (correct positions under left-pad)."""
    inp = build_batched_inputs(worker, images, questions)
    am = inp["attention_mask"]
    pos = (am.long().cumsum(-1) - 1).clamp(min=0)
    ghw = torch.as_tensor(inp.get("image_grid_hws"), dtype=torch.int32, device=worker.device)
    vit = worker.model.extract_feature(inp["pixel_values"].to(worker.dtype), ghw)
    vit = worker.model.mlp1(torch.cat(vit, dim=0))
    out = worker.model.language_model(
        input_ids=inp["input_ids"], attention_mask=am, position_ids=pos,
        visual_features=vit, image_token_index=worker.model.config.image_token_index,
        use_cache=False)
    return out.logits


def logit_equivalence(worker, images, prompts):
    """Compare, on row 0's real positions: the bf16 run-to-run floor, equal-length
    (no-pad) batching, and left-padded batching — each vs B=1.

    Returns a list of dicts ``{comparison, maxD_logit, top1_agree_pct}`` plus the
    logit scale. If the equal-length and left-padded rows match the floor, batching
    and padding add nothing beyond bf16 non-batch-invariance.
    """
    im, q = images[0], prompts[0]
    other_i, other_q = images[1], prompts[1]
    scale = float(prefill_logits(worker, [im], [q]).abs().max())
    S = prefill_logits(worker, [im], [q]).shape[1]

    def cmp(a, b):
        a = a[0, -S:].float().cpu().numpy()
        b = b[0, -S:].float().cpu().numpy()
        return float(np.abs(a - b).max()), float((a.argmax(1) == b.argmax(1)).mean() * 100)

    l1 = prefill_logits(worker, [im], [q])
    fa = prefill_logits(worker, [im, other_i], [q, other_q])
    fb = prefill_logits(worker, [im, other_i], [q, other_q])
    dup = prefill_logits(worker, [im, im], [q, q])
    dif = prefill_logits(worker, [im, other_i], [q, other_q])

    fd, ft = cmp(fa, fb)
    nd, nt = cmp(l1, dup)
    pd_, pt = cmp(l1, dif)
    rows = [
        {"comparison": "bf16 run-to-run floor (same call x2)", "maxD_logit": fd, "top1_agree_pct": ft},
        {"comparison": "batched vs B=1, equal-length (no pad)", "maxD_logit": nd, "top1_agree_pct": nt},
        {"comparison": "batched vs B=1, left-padded", "maxD_logit": pd_, "top1_agree_pct": pt},
    ]
    return {"logit_scale": scale, "rows": rows,
            "padding_clean": pd_ <= nd + max(0.02 * scale, 0.3)}
