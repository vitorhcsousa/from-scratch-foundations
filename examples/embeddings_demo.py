"""Embeddings block demo — visualise the token → embedding transformation.

Creates a ``TokenAndPositionalEmbedding`` with each ``pos_kind``, runs a
tiny batch through, and prints the concrete input/output values so you can
see exactly what the block does at every step.

Saves a JSON summary to ``artifacts/2026-W08/embeddings_demo.json``.

Usage:
    python examples/embeddings_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from foundations.projects.transformer.embeddings import TokenAndPositionalEmbedding

# ── Config ───────────────────────────────────────────────────────────────────

VOCAB_SIZE = 1_000
D_MODEL = 16          # small so printed vectors are readable
MAX_LEN = 128
BATCH = 2
SEQ_LEN = 6
SHOW_DIMS = 8        # how many embedding dims to print per vector

OUT_DIR = Path("artifacts/2026-W08")

# Fake "sentences" — each row is a sequence of token ids
SAMPLE_IDS = torch.tensor([
    [12, 453, 7, 88, 0, 999],   # sentence A
    [5,  23, 101, 0,  0, 42],   # sentence B  (0 = padding token)
])


def fmt_vec(t: torch.Tensor, n: int = SHOW_DIMS) -> str:
    """Format the first *n* values of a 1-D tensor as a compact string."""
    vals = ", ".join(f"{v:+.4f}" for v in t[:n].tolist())
    suffix = ", …" if t.numel() > n else ""
    return f"[{vals}{suffix}]"


def show_transformation(
    block: TokenAndPositionalEmbedding,
    ids: torch.Tensor,
    label: str,
) -> None:
    """Print a step-by-step view of the token → embedding mapping."""
    block.eval()
    with torch.no_grad():
        # Raw token embeddings (before positional encoding)
        tok_emb = block.token(ids)                     # (B, L, d_model)
        # Full output (token + positional + dropout)
        full_emb = block(ids)                          # (B, L, d_model)

    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")

    for b in range(ids.size(0)):
        print(f"\n  ── Sentence {b} ──")
        print(f"  Token ids: {ids[b].tolist()}\n")
        print(f"  {'pos':>3}  {'id':>5}  {'token embedding':^42}  {'+ positional = final':^42}")
        print(f"  {'---':>3}  {'---':>5}  {'-' * 42}  {'-' * 42}")

        for pos in range(ids.size(1)):
            tid = ids[b, pos].item()
            te = fmt_vec(tok_emb[b, pos])
            fe = fmt_vec(full_emb[b, pos])
            print(f"  {pos:>3}  {tid:>5}  {te:>42}  {fe:>42}")

    print()


def run_variant(pos_kind: str, ids: torch.Tensor) -> dict:
    """Build a block, show the transformation, and return a summary dict."""
    block = TokenAndPositionalEmbedding(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        pos_kind=pos_kind,
        dropout=0.0,
        pad_id=0,
    )

    show_transformation(block, ids, label=f"pos_kind = {pos_kind!r}")

    block.eval()
    with torch.no_grad():
        out = block(ids)

    return {
        "pos_kind": pos_kind,
        "input_shape": list(ids.shape),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "device": str(out.device),
        "num_params": sum(p.numel() for p in block.parameters()),
        "sample_input": ids[0].tolist(),
        "sample_output_first_token": out[0, 0].tolist(),
        "sample_output_last_token": out[0, -1].tolist(),
    }


def main() -> None:
    """Run both variants, print results, and save JSON artifact."""
    torch.manual_seed(42)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              TokenAndPositionalEmbedding — Demo                     ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  vocab_size = {VOCAB_SIZE:<6}  d_model = {D_MODEL:<4}  max_len = {MAX_LEN:<5}            ║")
    print(f"║  batch      = {BATCH:<6}  seq_len = {SEQ_LEN:<4}  pad_id  = 0                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Input token ids (shape {list(SAMPLE_IDS.shape)}):")
    for b in range(SAMPLE_IDS.size(0)):
        print(f"    sentence {b}: {SAMPLE_IDS[b].tolist()}")

    results = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "pad_id": 0,
        },
        "variants": [
            run_variant("sinusoidal", SAMPLE_IDS),
            run_variant("learned", SAMPLE_IDS),
        ],
    }

    # ── Key observations ─────────────────────────────────────────────────
    print("═" * 70)
    print("  Key observations")
    print("═" * 70)
    print()
    print("  • Same token id at different positions → different final vectors")
    print("    (positional encoding breaks the symmetry).")
    print()
    print("  • pad_id=0 → token embedding for id 0 is all zeros,")
    print("    but final output still contains the positional signal.")
    print()
    print("  • Sinusoidal variant: 0 extra params (fixed formula).")
    sin_params = results["variants"][0]["num_params"]
    lrn_params = results["variants"][1]["num_params"]
    print(f"    Learned variant: +{lrn_params - sin_params:,} pos-embedding params.")
    print()

    # ── Save ─────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "embeddings_demo.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  ✓ Saved → {out_path}\n")


if __name__ == "__main__":
    main()