# /// script
# dependencies = ["tokenizers", "httpx"]
# ///
"""Sweep the filler length and record P(8) - P(32) at the completion point.

Reproduces Appendix B.2 of arXiv:2609.36322 for three prompt formats:

  chat  the paper's chat-formatted prompt (Appendix B.2): instruction +
        filler + code head as the user message, nine code tokens as the
        assistant prefill.
  raw   the base model's raw prefix (Appendix B.1), no chat template.
  fim   the raw prefix plus a neutral suffix, sent to the fill-in-the-middle
        endpoint. This is NOT the paper's setup (the paper has no suffix) and
        is included as a control.

Why this script exists: our first version separated the filler from the code
head with "\\n\\n". The filler already ends with a newline, so that produced two
blank lines instead of one and moved the scored token one position later.
The parity of the scored position no longer lined up with L, and the old data
looked like period 4 or unstable rather than the paper's period 2. This script
uses one newline and asserts the scored position against the paper's formula
t = 1004 + L.

Run at temperature 1 for the real probability margin, or 0 for top-1 only:

  uv run -q --with tokenizers --with httpx python scripts/prob_sweep.py \
    --format chat --temperature 1 --out data/prob_sweep_chat_t1_run1.json
"""
import argparse
import json
import math
import os
from pathlib import Path

import httpx
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))
CHAT_URL = "https://api.deepseek.com/beta/chat/completions"
RAW_URL = "https://api.deepseek.com/beta/completions"
MODEL = "deepseek-flash"

INSTRUCTION = ("Continue the function without changing its block-wise quantization "
               "semantics; avoid a redundant nested cast")
FILLER_HEAD = ('"""This module contains optimized tensor kernels used by the '
               'model inference runtime.\n')
FIM_SUFFIX = "\n\n# end of file\n"

# Tokens the chat template adds around the messages (thinking disabled).
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_END = "</think>"


def read_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".claude" / "credentials" / "deepseek_api_key"
    if path.exists():
        return path.read_text().strip()
    raise SystemExit(
        "Set DEEPSEEK_API_KEY, or put the key in "
        "~/.claude/credentials/deepseek_api_key")


def filler_text(n_eq: int) -> str:
    return FILLER_HEAD + " ".join(["="] * n_eq) + '\n"""\n'


def probs(top: dict) -> dict:
    """Reported logprobs to probabilities.

    The API writes -9999 for a token it did not report. Return None (not 0)
    for those, so a missing value cannot be mistaken for probability zero.
    """
    return {k: (math.exp(v) if v > -9000 else None) for k, v in top.items()}


def show(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["chat", "raw", "fim"], required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-min", type=int, default=None,
                    help="number of '=' characters (default 8 for chat, 9 for raw/fim)")
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Family A: L = 15 + n_eq tokens. The paper uses L = 23..38 for the chat
    # prompt (n_eq 8..23) and L = 24..39 for the raw prefix (n_eq 9..24).
    n_min = args.n_min if args.n_min is not None else (8 if args.format == "chat" else 9)
    n_max = args.n_max if args.n_max is not None else (23 if args.format == "chat" else 24)

    src = (ROOT / "code" / "kernel.py").read_text(encoding="utf-8")
    ids = TOK.encode(src, add_special_tokens=False).ids
    assert ids[983] == 26 and TOK.decode([26]) == "8", ids[983]
    assert ids[982] == 20199, ids[982]
    code_head = TOK.decode(ids[:974])
    assistant_prefix = TOK.decode(ids[974:983])
    prefix_text = TOK.decode(ids[:983])
    assert TOK.encode(prefix_text, add_special_tokens=False).ids == ids[:983]

    key = read_key()
    rows = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0,
             "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}

    def save():
        """Write what we have. Called after every row so a crash or a bad
        response later in the sweep cannot throw away paid rows."""
        out = {"format": args.format, "temperature": args.temperature,
               "rows": rows, "usage": usage}
        Path(args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    with httpx.Client(timeout=180) as client:
        for n_eq in range(n_min, n_max + 1):
            filler = filler_text(n_eq)
            L = len(TOK.encode(filler, add_special_tokens=False).ids)
            if L != 15 + n_eq:
                raise SystemExit(f"filler tokenizes to {L}, expected {15 + n_eq}")

            if args.format == "chat":
                user = INSTRUCTION + "\n\n" + filler + "\n" + code_head
                # The paper counts the scored position as t = 1004 + L,
                # including chat-template tokens.
                rendered = (BOS + USER + user + ASSISTANT + THINK_END
                            + assistant_prefix)
                t = len(TOK.encode(rendered, add_special_tokens=False).ids)
                assert t == 1004 + L, f"L={L}: scored position {t} != {1004 + L}"
                body = {
                    "model": MODEL,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant_prefix,
                         "prefix": True},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": 1,
                    "thinking": {"type": "disabled"},
                    "logprobs": True,
                    "top_logprobs": 20,
                }
                url = CHAT_URL
                t_pos = t
            else:
                prompt = filler + prefix_text
                pids = TOK.encode(prompt, add_special_tokens=False).ids
                fill_ids = TOK.encode(filler, add_special_tokens=False).ids
                assert pids[:L] == fill_ids and pids[L:] == ids[:983]
                body = {
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": args.temperature,
                    "logprobs": 20,
                }
                if args.format == "fim":
                    body["suffix"] = FIM_SUFFIX
                url = RAW_URL
                # Raw requests get a BOS token the local count does not see,
                # and FIM adds template tokens around the suffix. The position
                # below is filled from usage.prompt_tokens for raw; FIM has no
                # comparable position, so it stays None.
                t_pos = None

            r = client.post(url, json=body,
                            headers={"Authorization": f"Bearer {key}"})
            if r.status_code != 200:
                print(f"L={L:2d}: HTTP {r.status_code} {r.text[:300]}")
                rows.append({"L": L, "t": t_pos,
                             "error": f"HTTP {r.status_code}: {r.text[:500]}"})
                save()
                continue
            try:
                d = r.json()
            except ValueError as e:
                print(f"L={L:2d}: non-JSON response ({e!r}): {r.text[:300]}")
                rows.append({"L": L, "t": t_pos,
                             "error": f"non-JSON response: {e!r}"})
                save()
                continue
            u = d.get("usage") or {}
            if args.format == "chat" and u.get("prompt_tokens") != t:
                # The local chat template must match what the server counts,
                # or the scored position is not the one we think it is.
                raise SystemExit(
                    f"L={L}: server counted {u.get('prompt_tokens')} prompt "
                    f"tokens, local template gives {t}")
            try:
                ch = d["choices"][0]
                if args.format == "chat":
                    top_list = ch["logprobs"]["content"][0]["top_logprobs"]
                    top = {x["token"]: x["logprob"] for x in top_list}
                    text = ch.get("message", {}).get("content")
                else:
                    top = (ch.get("logprobs") or {}).get("top_logprobs", [{}])[0]
                    text = ch.get("text")
            except (KeyError, IndexError, TypeError) as e:
                print(f"L={L:2d}: unreadable response ({e!r}): {r.text[:300]}")
                rows.append({"L": L, "t": t_pos,
                             "error": f"unreadable response: {e!r}"})
                save()
                continue
            p = probs(top)
            p8, p32 = p.get("8"), p.get("32")
            delta = p8 - p32 if p8 is not None and p32 is not None else None
            for k in usage:
                usage[k] += u.get(k, 0) or 0
            if args.format == "raw":
                t_pos = u.get("prompt_tokens")
            rows.append({"L": L, "t": t_pos, "text": text,
                         "p8": p8, "p32": p32,
                         "delta": delta, "top": top})
            save()
            got = "n/a" if delta is None else f"{delta:+.4f}"
            print(f"L={L:2d} t={t_pos} text={text!r:5} "
                  f"p8={show(p8)} p32={show(p32)} delta={got}")

    print("wrote", args.out)


if __name__ == "__main__":
    main()
