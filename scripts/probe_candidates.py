# /// script
# dependencies = ["tokenizers>=0.20", "jinja2"]
# ///
"""Find boundary-stable single-token candidates for NIAH records.

A candidate `s` is usable as a key and a value only if it stays one token
at both record boundaries, so that `key:value。` is exactly 4 tokens.
"""
import re
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))

ALPHA = re.compile(r"^[A-Za-z]+$")
COLON = tok.encode(":", add_special_tokens=False).ids
PERIOD = tok.encode("。", add_special_tokens=False).ids
assert COLON == [28] and PERIOD == [320], (COLON, PERIOD)

base = []
for tid in range(tok.get_vocab_size()):
    raw = tok.decode([tid], skip_special_tokens=False)
    if not (1 <= len(raw) <= 16) or not ALPHA.match(raw) or tid < 10000:
        continue
    if tok.encode(raw, add_special_tokens=False).ids != [tid]:
        continue
    base.append((tid, raw))

stable = []
for tid, raw in base:
    if tok.encode(raw + ":", add_special_tokens=False).ids != [tid, 28]:
        continue
    if tok.encode(":" + raw, add_special_tokens=False).ids != [28, tid]:
        continue
    if tok.encode(raw + "。", add_special_tokens=False).ids != [tid, 320]:
        continue
    if tok.encode(":" + raw + "。", add_special_tokens=False).ids != [28, tid, 320]:
        continue
    if tok.encode(" " + raw, add_special_tokens=False).ids != [223, tid]:
        continue
    stable.append((tid, raw))

print("base single-token alphabetic:", len(base))
print("boundary-stable (key+value):", len(stable))
print("sample:", stable[:15])

# Chat-template alignment: compare with the API's usage.prompt_tokens for
# the exact same user message.
msg = "Say OK only."
print("plain user text tokens:", len(tok.encode(msg, add_special_tokens=False).ids))
print("plain tokens:", tok.encode(msg, add_special_tokens=False).tokens)
for path in [ROOT / "tokenizer" / "chat_template.jinja"]:
    print("template bytes:", path.stat().st_size)
