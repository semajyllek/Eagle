# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Matplotlib plot for the throughput table (``speed_sweep``).

Kept separate so ``import repro`` stays light; matplotlib is imported lazily.
"""


def _finish(fig, as_base64):
    """Either ``plt.show()`` (interactive notebook use, returns ``None``) or
    encode ``fig`` as a base64 PNG string (for embedding in a markdown ``<img>``
    via ``repro.combine``), closing the figure either way."""
    import matplotlib.pyplot as plt

    if not as_base64:
        plt.show()
        return None
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_speed_sweep(
    rows, title="Sequential vs batched (N distinct images)", as_base64=False
):
    """From ``speed_sweep`` rows: wall-clock latency (left) and the end-to-end
    speedup (right) for N distinct images, sequential (N x B=1) vs this branch's
    batched ``predict_batch`` (1 x B=N)."""
    import matplotlib.pyplot as plt

    bs = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(
        bs, [r["sequential_s"] for r in rows], "o-", label="sequential (N x B=1)"
    )
    ax[0].plot(bs, [r["batched_s"] for r in rows], "s-", label="batched (1 x B=N)")
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("wall-clock (s)")
    ax[0].set_title("Latency for N items (end-to-end)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].bar(
        [str(b) for b in bs], [r["endtoend_speedup"] for r in rows], color="C0"
    )
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("batch size")
    ax[1].set_ylabel("speedup x")
    ax[1].set_title("End-to-end speedup")
    ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle(title)
    plt.tight_layout()
    return _finish(fig, as_base64)
