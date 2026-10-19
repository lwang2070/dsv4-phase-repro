# /// script
# dependencies = ["tokenizers>=0.20", "numpy"]
# ///
"""Probe the DeepSeek-V4.1-Flash tokenizer for the NIAH construction.

We need single-token alphabetic keys/values (paper: id >= 10000, <=16 chars),
and we need the 4-token record form `key:value。`.
"""
import json
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

TOK = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"
tok = Tokenizer.from_file(str(TOK))

print("vocab size:", tok.get_vocab_size())

# Special tokens that matter for record stability.
for s in [":", "。", " ", "\n"]:
    ids = tok.encode(s, add_special_tokens=False).ids
    print(f"encode({s!r}) -> {ids}  decoded={[tok.decode([i]) for i in ids]}")

ALPHA = re.compile(r"^[A-Za-z]+$")
cands = []
skipped_roundtrip = 0
for tid in range(tok.get_vocab_size()):
    raw = tok.decode([tid], skip_special_tokens=False)
    if not (1 <= len(raw) <= 16) or not ALPHA.match(raw):
        continue
    if tid < 10000:
        continue
    if tok.encode(raw, add_special_tokens=False).ids != [tid]:
        skipped_roundtrip += 1
        continue
    cands.append((tid, raw))

print("single-token alphabetic candidates (id>=10000):", len(cands))
print("skipped by roundtrip:", skipped_roundtrip)
print("first 20:", cands[:20])
print("last 5:", cands[-5:])

# Verify the 4-token record form for a sample.
colon = tok.encode(":", add_special_tokens=False).ids
period = tok.encode("。", add_special_tokens=False).ids
print("colon ids:", colon, "period ids:", period)
bad = 0
for tid, raw in cands[:2000]:
    other = cands[(tid * 7919) % len(cands)][1]
    rec = f"{raw}:{other}。"
    got = tok.encode(rec, add_special_tokens=False).ids
    want = [tid, colon[0], cands[(tid * 7919) % len(cands)][0], period[0]]
    if got != want:
        bad += 1
        if bad <= 5:
            print("record mismatch:", rec, got, want)
print("record mismatches in 2000 samples:", bad)
