# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Equivalence tests for the MoonViT per-image SDPA attention.

Gate for the ``sdpa_attention`` host-sync optimization (replacing the per-image
``int(q_cu_seqlens[i])`` host syncs with one ``.tolist()``): proves the change is
PURELY ADDITIVE -- bit-for-bit identical output to the original per-element
implementation -- and, as a bonus, that ``sdpa_attention`` really is the
block-diagonal (per-image-isolated) attention it claims to be.

Run:
    PYTHONPATH=<repo>/Embodied python <repo>/Embodied/tests/test_modeling_vit.py
"""

import math
import sys

import torch
import torch.nn.functional as F
from eaglevl.utils.locany.modeling_vit import sdpa_attention


def _sdpa_per_element(q, k, v, q_cu_seqlens):
    """The ORIGINAL implementation: per-image slice bounds via ``int(tensor[i])``
    (2 host syncs per image). The optimization must match this bit-for-bit."""
    seq_length = q.shape[0]
    outs = []
    for i in range(1, len(q_cu_seqlens)):
        s, e = int(q_cu_seqlens[i - 1]), int(q_cu_seqlens[i])
        qi = q[s:e].transpose(0, 1)
        ki = k[s:e].transpose(0, 1)
        vi = v[s:e].transpose(0, 1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
        outs.append(oi.transpose(0, 1))
    return torch.cat(outs, dim=0).reshape(seq_length, -1)


def _dense_blockdiag_reference(q, k, v, q_cu_seqlens):
    """Independent reference: one dense attention over the whole packed sequence with
    a -inf block-diagonal mask (image i attends only to image i). Numerically the
    same thing ``sdpa_attention`` computes per slice."""
    s_len = q.shape[0]
    mask = torch.full((s_len, s_len), float("-inf"), dtype=q.dtype)
    for i in range(1, len(q_cu_seqlens)):
        s, e = int(q_cu_seqlens[i - 1]), int(q_cu_seqlens[i])
        mask[s:e, s:e] = 0.0
    qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [H, S, D]
    w = qh @ kh.transpose(-2, -1) / math.sqrt(q.shape[-1]) + mask
    w = torch.softmax(w, dim=-1)
    return (w @ vh).transpose(0, 1).reshape(s_len, -1)


def _make(sizes, num_heads=4, head_dim=8, seed=0):
    torch.manual_seed(seed)
    s_tot = sum(sizes)
    q = torch.randn(s_tot, num_heads, head_dim)
    k = torch.randn(s_tot, num_heads, head_dim)
    v = torch.randn(s_tot, num_heads, head_dim)
    cu = torch.tensor([0] + torch.tensor(sizes).cumsum(0).tolist(), dtype=torch.long)
    return q, k, v, cu


_CASES = ([5], [5, 3], [7, 1, 4, 9], [2, 2, 2], [13, 8])


def test_tolist_is_bit_identical():
    n = 0
    for sizes in _CASES:
        q, k, v, cu = _make(sizes, seed=hash(tuple(sizes)) % 1000)
        out = sdpa_attention(q, k, v, cu, cu)
        ref = _sdpa_per_element(q, k, v, cu)
        assert torch.equal(out, ref), f"sdpa .tolist() != per-element for sizes={sizes}"
        n += 1
    return n


def test_matches_dense_blockdiag():
    n = 0
    for sizes in _CASES:
        q, k, v, cu = _make(sizes, seed=1 + hash(tuple(sizes)) % 1000)
        out = sdpa_attention(q, k, v, cu, cu)
        ref = _dense_blockdiag_reference(q, k, v, cu)
        assert torch.allclose(
            out, ref, atol=1e-5, rtol=1e-4
        ), f"sdpa != dense block-diagonal for sizes={sizes}"
        n += 1
    return n


def main():
    n1 = test_tolist_is_bit_identical()
    n2 = test_matches_dense_blockdiag()
    print(f"OK  sdpa .tolist() bit-identical to per-element: {n1} cases")
    print(f"OK  sdpa == dense block-diagonal mask (allclose): {n2} cases")
    print("All modeling_vit attention equivalence tests passed.")


if __name__ == "__main__":
    sys.exit(main() and 0)
