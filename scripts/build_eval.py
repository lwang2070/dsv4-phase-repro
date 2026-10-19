# /// script
# dependencies = ["tokenizers>=0.20"]
# ///
"""Build the NIAH phase-sensitivity eval set for DeepSeek-V4.1-Flash.

Follows Appendix C.1 of arXiv:2609.36322 as closely as an API-only setup
allows:
  - 128,000-token prompts (measured locally with the official tokenizer)
  - 16,000 key:value records, 4 tokens each
  - target is pair 248; its key starts at 63,976 + ell
  - ell in [8, 40], ell % 8 == residue group, histograms from the paper
  - one binding per (residue, slot), reused across residues

Output: JSONL with one prompt per line.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "tokenizer.json"))

PROMPT_TOKENS = 128_000
N_PAIRS = 16_000
TARGET_IDX = 248
INTRO = (
    "You will see a long text containing many key-value records. "
    "Each record has the form key:value。\n"
    "Here are three examples of the record format:\n"
    "alpha:beta。\n"
    "gamma:delta。\n"
    "epsilon:zeta。\n"
    "In the long text below, many such key-value pairs are hidden among unrelated text.\n"
)
QUERY_SEP = "\n\n"
QUERY = (
    "Now answer the question using the key-value pairs above. "
    "What is the value for key {key}? Answer with only the value. "
    "The value for key {key} is"
)

# Appendix C.1 histograms.
HIST = {
    0: {8: 1, 16: 52, 24: 150, 32: 52, 40: 1},
    1: {9: 2, 17: 68, 25: 146, 33: 40},
    2: {10: 4, 18: 84, 26: 140, 34: 28},
    3: {11: 7, 19: 101, 27: 129, 35: 19},
    4: {12: 12, 20: 116, 28: 116, 36: 12},
}
for _r in (1, 2, 3):
    HIST[8 - _r] = {48 - k: v for k, v in HIST[_r].items()}


def allocate(hist: dict, n: int) -> list[int]:
    """Largest-remainder allocation of n samples over a histogram."""
    total = sum(hist.values())
    exact = {k: v * n / total for k, v in hist.items()}
    out = {k: int(v) for k, v in exact.items()}
    left = n - sum(out.values())
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[:left]:
        out[k] += 1
    vals = []
    for k in sorted(out):
        vals.extend([k] * out[k])
    assert len(vals) == n
    return vals


def load_candidates():
    alpha = re.compile(r"^[A-Za-z]+$")
    base = []
    for tid in range(TOK.get_vocab_size()):
        raw = TOK.decode([tid], skip_special_tokens=False)
        if not (1 <= len(raw) <= 16) or not alpha.match(raw) or tid < 10000:
            continue
        if TOK.encode(raw, add_special_tokens=False).ids != [tid]:
            continue
        base.append((tid, raw))

    keys = base  # key + ":" stays two tokens for every base candidate
    values = [
        (tid, raw) for tid, raw in base
        if TOK.encode(":" + raw + "。", add_special_tokens=False).ids == [28, tid, 320]
    ]
    # Target keys also need " key" to stay two tokens, so the query length is
    # fixed and the query position does not drift.
    query_keys = [
        (tid, raw) for tid, raw in base
        if TOK.encode(" " + raw, add_special_tokens=False).ids == [223, tid]
    ]
    print(f"candidates: keys={len(keys)} values={len(values)} query_keys={len(query_keys)}",
          file=sys.stderr)
    return keys, values, query_keys


def load_corpus_ids(n_lines: int) -> list[int]:
    lines = (ROOT / "data" / "train.txt").read_text(encoding="utf-8").splitlines()[:n_lines]
    return TOK.encode("\n".join(lines), add_special_tokens=False).ids


def build_binding(seed: int, keys, values, query_keys):
    rng = random.Random(seed)
    target_tid, target_key = rng.choice(query_keys)
    distractor_pool = [x for x in keys if x[0] != target_tid]
    distractors = rng.sample(distractor_pool, N_PAIRS - 1)
    klist = distractors[:TARGET_IDX] + [(target_tid, target_key)] + distractors[TARGET_IDX:]
    assert len(klist) == N_PAIRS
    assert len({x[0] for x in klist}) == N_PAIRS
    vlist = rng.sample(values, N_PAIRS)
    pairs = list(zip(klist, vlist))
    text = "".join(f"{k[1]}:{v[1]}。" for k, v in pairs)
    ids = TOK.encode(text, add_special_tokens=False).ids
    assert len(ids) == 4 * N_PAIRS, (len(ids), 4 * N_PAIRS)
    gold = vlist[TARGET_IDX][1]
    gold_tid = vlist[TARGET_IDX][0]
    # The target record must be exactly [key, ':', value, '。'].
    assert ids[4 * TARGET_IDX:4 * TARGET_IDX + 4] == [
        target_tid, 28, gold_tid, 320
    ], ids[4 * TARGET_IDX:4 * TARGET_IDX + 4]
    return {
        "seed": seed,
        "pair_ids": ids,
        "pair_text": text,
        "target_key": target_key,
        "target_tid": target_tid,
        "gold": gold,
        "gold_tid": gold_tid,
    }


def find_subsequence(hay: list[int], needle: list[int]) -> int:
    """Return the first index of needle in hay, or -1."""
    n = len(needle)
    first = needle[0]
    for i in range(len(hay) - n + 1):
        if hay[i] == first and hay[i:i + n] == needle:
            return i
    return -1


def build_prompt(ell: int, binding, corpus_ids, intro_ids, offsets):
    """Assemble one prompt; return ids, text, and actual target position."""
    B0 = 62_984 - len(intro_ids)
    assert B0 > 0, len(intro_ids)
    body_len = B0
    target_desired = 63_976 + ell

    prefill = corpus_ids[offsets[0]:offsets[0] + ell]
    assert len(prefill) == ell
    postfill = corpus_ids[offsets[1]:offsets[1] + 48 - ell]
    assert len(postfill) == 48 - ell
    qsep_ids = TOK.encode(QUERY_SEP, add_special_tokens=False).ids
    query_ids = TOK.encode(QUERY.format(key=binding["target_key"]),
                           add_special_tokens=False).ids
    tail_len = 968 - len(qsep_ids) - len(query_ids)
    assert tail_len > 0, tail_len

    def assemble(body_len_now, tail_len_now):
        body = corpus_ids[offsets[2]:offsets[2] + body_len_now]
        tail = corpus_ids[offsets[3]:offsets[3] + tail_len_now]
        assert len(body) == body_len_now and len(tail) == tail_len_now
        raw = (intro_ids + prefill + body + binding["pair_ids"] + postfill
               + tail + qsep_ids + query_ids)
        text = TOK.decode(raw)
        ids = TOK.encode(text, add_special_tokens=False).ids
        # The seam with the body prefix can retokenize the first record, so
        # anchor on a record a few tokens in.  Then locate the target record
        # itself; it is unique inside the pair body.
        target_rec = [binding["target_tid"], 28, binding["gold_tid"], 320]
        target_pos = -1
        for skip in (4, 8, 12, 16, 20):
            hit = find_subsequence(ids, binding["pair_ids"][skip:skip + 24])
            if hit < 0:
                continue
            guess = hit + 4 * TARGET_IDX - skip
            for j in range(max(0, guess - 16), min(len(ids) - 4, guess + 17)):
                if ids[j:j + 4] == target_rec:
                    target_pos = j
                    break
            if target_pos >= 0:
                break
        assert target_pos >= 0, "target record not found after retokenization"
        return ids, text, target_pos

    # Token seams can shift the target by a token or two.  We measure the
    # actual position and group by it, rather than forcing the nominal one.
    ids, text, target_pos = assemble(body_len, tail_len)
    if len(ids) != PROMPT_TOKENS:
        ids2, text2, target_pos2 = assemble(body_len, tail_len - (len(ids) - PROMPT_TOKENS))
        if (abs(len(ids2) - PROMPT_TOKENS), abs(target_pos2 - target_desired)) <= \
           (abs(len(ids) - PROMPT_TOKENS), abs(target_pos - target_desired)):
            ids, text, target_pos = ids2, text2, target_pos2
    return ids, text, target_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-residue", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_pilot.jsonl"))
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    keys, values, query_keys = load_candidates()
    corpus_ids = load_corpus_ids(200_000)
    print(f"corpus tokens: {len(corpus_ids)}", file=sys.stderr)
    intro_ids = TOK.encode(INTRO, add_special_tokens=False).ids
    n = args.per_residue

    # One binding per slot, reused across the eight residue groups.
    bindings = [
        build_binding(args.seed + s, keys, values, query_keys) for s in range(n)
    ]
    # Patch out the large pair arrays when serialising metadata only.

    rng = random.Random(args.seed + 999)
    rows = []
    for r in range(8):
        ells = allocate(HIST[r], n)
        for slot, ell in enumerate(ells):
            offs = [rng.randrange(0, len(corpus_ids) - 70_000) for _ in range(4)]
            ids, text, target_pos = build_prompt(
                ell, bindings[slot], corpus_ids, intro_ids, offs)
            row = {
                "id": f"r{r}_s{slot}",
                "residue": r,
                "ell": ell,
                "slot": slot,
                "target_pos": target_pos,
                "target_pos_mod8": target_pos % 8,
                "prompt_tokens_local": len(ids),
                "gold": bindings[slot]["gold"],
                "target_key": bindings[slot]["target_key"],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt": text,
            }
            rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {args.out}", file=sys.stderr)
    import collections
    print("per residue:", collections.Counter(r["residue"] for r in rows), file=sys.stderr)
    print("ell means:",
          {r: sum(x["ell"] for x in rows if x["residue"] == r) / n for r in range(8)},
          file=sys.stderr)
    print("target positions mod 8:",
          collections.Counter(r["target_pos_mod8"] for r in rows), file=sys.stderr)


if __name__ == "__main__":
    main()
