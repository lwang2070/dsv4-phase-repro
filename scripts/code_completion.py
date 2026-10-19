# /// script
# dependencies = ["tokenizers", "httpx"]
# ///
"""Reproduce Figure 1 of arXiv:2609.36322 on the DeepSeek API.

Raw-prefix completion, no chat template, filler length L = 24..39.
The reference continuation is token 8 (id 26); the competitor is 32 (id 2111).
"""
import json
import os
import sys
from pathlib import Path

import httpx
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
URL = "https://api.deepseek.com/beta/completions"


def read_key() -> str:
    """Read DEEPSEEK_API_KEY, falling back to the local credential file."""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".claude" / "credentials" / "deepseek_api_key"
    if path.exists():
        return path.read_text().strip()
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, or put the key in "
        "~/.claude/credentials/deepseek_api_key")


KEY = read_key()


def main():
    src = (ROOT / "code" / "kernel.py").read_text(encoding="utf-8")
    ids = TOK.encode(src, add_special_tokens=False).ids
    print("tokens 980..985:", [(i, ids[i], TOK.decode([ids[i]])) for i in range(980, 986)])
    assert ids[983] == 26 and TOK.decode([26]) == "8", ids[983]
    assert ids[982] == 20199, ids[982]
    prefix_text = TOK.decode(ids[:983])
    assert TOK.encode(prefix_text, add_special_tokens=False).ids == ids[:983]

    rows = []
    with httpx.Client(timeout=120) as client:
        for n_eq in range(9, 25):
            L = 15 + n_eq
            filler = (
                '"""This module contains optimized tensor kernels used by the '
                "model inference runtime.\n"
                + " ".join(["="] * n_eq)
                + '\n"""\n'
            )
            fill_ids = TOK.encode(filler, add_special_tokens=False).ids
            if len(fill_ids) != L:
                print(f"L={L}: filler tokenizes to {len(fill_ids)}, skipping")
                continue
            prompt = filler + prefix_text
            pids = TOK.encode(prompt, add_special_tokens=False).ids
            if pids[:L] != fill_ids or pids[L:] != ids[:983] or pids[-1] != 20199:
                print(f"L={L}: seam changed tokenization, skipping")
                continue
            body = {
                "model": "deepseek-flash",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 5,
            }
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {KEY}"})
            if r.status_code != 200:
                print(f"L={L}: HTTP {r.status_code} {r.text[:300]}")
                rows.append({"L": L, "n_eq": n_eq,
                             "error": f"HTTP {r.status_code}: {r.text[:500]}"})
                continue
            d = r.json()
            ch = d["choices"][0]
            lp = ch.get("logprobs") or {}
            top = (lp.get("top_logprobs") or [{}])[0] if lp else {}
            rows.append({
                "L": L,
                "n_eq": n_eq,
                "text": ch.get("text"),
                "logprobs": top,
                "usage": d.get("usage"),
            })
            print(f"L={L:2d} -> {ch.get('text')!r}  top={top}")
    (ROOT / "data" / "code_completion.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
