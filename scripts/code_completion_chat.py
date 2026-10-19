# /// script
# dependencies = ["tokenizers", "httpx"]
# ///
"""Appendix B.2 control, chat-formatted, same prompt as prob_sweep.py.

Sends the paper's instruction + family-A filler + 974-token code head,
with an assistant prefix of the remaining 9 source tokens, and reads the
logprobs of '8' (id 26) and '32' (id 2111) at the next position.

This early script records the log-odds log P(8) - log P(32). The sign is
the same as the probability difference P(8) - P(32) used by prob_sweep.py,
but the magnitude is not comparable; the field is named log_odds for that
reason. The default output is a new file: the shipped
data/code_completion_chat.json is the pre-fix record and must not be
overwritten by a rerun.
"""
import argparse
import json
import os
from pathlib import Path

import httpx
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
URL = "https://api.deepseek.com/beta/chat/completions"


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

INSTRUCTION = ("Continue the function without changing its block-wise quantization "
               "semantics; avoid a redundant nested cast")
FILLER_HEAD = ("\"\"\"This module contains optimized tensor kernels used by the "
               "model inference runtime.\n")

# Chat-template tokens around the two messages, with thinking disabled.
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_END = "</think>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=8)
    ap.add_argument("--n-max", type=int, default=23)
    ap.add_argument("--out",
                    default=str(ROOT / "data" / "code_completion_chat_rerun.json"))
    args = ap.parse_args()

    src = (ROOT / "code" / "kernel.py").read_text(encoding="utf-8")
    ids = TOK.encode(src, add_special_tokens=False).ids
    assert ids[983] == 26 and TOK.decode([26]) == "8", ids[983]
    assert ids[982] == 20199, ids[982]
    code_head = TOK.decode(ids[:974])
    assistant_prefix = TOK.decode(ids[974:983])
    print("assistant prefix:", repr(assistant_prefix))
    assert assistant_prefix == "compute_dtype, T.Cast(FP", assistant_prefix

    rows = []
    with httpx.Client(timeout=180) as client:
        for n_eq in range(args.n_min, args.n_max + 1):
            filler = FILLER_HEAD + " ".join(["="] * n_eq) + '\n"""\n'
            # The filler already ends with a newline, so one more newline is
            # the blank line between components. Using "\n\n" here adds a
            # second blank line, moves the scored token one position later,
            # and flips the odd/even pattern. See scripts/prob_sweep.py.
            user = INSTRUCTION + "\n\n" + filler + "\n" + code_head
            rendered = (BOS + USER + user + ASSISTANT + THINK_END
                        + assistant_prefix)
            t = len(TOK.encode(rendered, add_special_tokens=False).ids)
            assert t == 1004 + (15 + n_eq), (t, 1004 + 15 + n_eq)
            body = {
                "model": "deepseek-flash",
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant_prefix, "prefix": True},
                ],
                "temperature": 0,
                "max_tokens": 1,
                "thinking": {"type": "disabled"},
                "logprobs": True,
                "top_logprobs": 20,
            }
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {KEY}"})
            if r.status_code != 200:
                print(f"n_eq={n_eq}: HTTP {r.status_code} {r.text[:300]}")
                rows.append({"n_eq": n_eq, "error": f"HTTP {r.status_code}: {r.text[:500]}"})
                continue
            d = r.json()
            ch = d["choices"][0]
            lp = (ch.get("logprobs") or {}).get("content") or []
            top = lp[0].get("top_logprobs", []) if lp else []
            by_tok = {x.get("token"): x.get("logprob") for x in top}
            # The API returns -9999 for tokens it did not report; treat that
            # as missing instead of computing a meaningless difference.
            log_odds = None
            lp8, lp32 = by_tok.get("8"), by_tok.get("32")
            if lp8 is not None and lp32 is not None and lp8 > -9000 and lp32 > -9000:
                log_odds = lp8 - lp32
            row = {
                "n_eq": n_eq,
                "L": 15 + n_eq,
                "text": ch.get("message", {}).get("content"),
                "log_odds": log_odds,
                "top": top[:6],
                "usage": d.get("usage"),
            }
            rows.append(row)
            print(f"L={15+n_eq:2d} top={row['text']!r} log_odds={log_odds} "
                  f"logp8={by_tok.get('8')} logp32={by_tok.get('32')}")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
