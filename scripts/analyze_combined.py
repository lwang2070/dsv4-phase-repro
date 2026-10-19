# /// script
# dependencies = ["numpy"]
# ///
"""Pooled analysis of the two NIAH runs and all API usage.

Joins the two runs by prompt id (not by file order), computes the phase
statistics, and writes data/page_data.json. Run with:

  uv run -q --with numpy python scripts/analyze_combined.py
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def ok(r):
    return (r.get("answer") or "").strip().startswith(r["gold"])


def load(path):
    """Successful rows keyed by id; later success wins."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            continue
        out[r["id"]] = r
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def usage(path, dedupe=True):
    """Count billed rows, cache-hit rows, errors, and the three token fields."""
    n = hit = miss = comp = hit_req = err = 0
    seen = set()
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for r in records:
        if "error" in r:
            err += 1
            continue
        if dedupe:
            key = r.get("id")
            if key in seen:
                continue
            seen.add(key)
        n += 1
        u = r.get("usage") or {}
        h = u.get("prompt_cache_hit_tokens", 0)
        hit += h
        miss += u.get("prompt_cache_miss_tokens", 0)
        comp += u.get("completion_tokens", 0)
        hit_req += h > 0
    return n, hit, miss, comp, hit_req, err


def total(paths, dedupe=True):
    n = hit = miss = comp = hit_req = err = 0
    for fn in paths:
        p = DATA / fn
        if not p.exists():
            continue
        a = usage(p, dedupe)
        n, hit, miss, comp, hit_req, err = (
            n + a[0], hit + a[1], miss + a[2], comp + a[3],
            hit_req + a[4], err + a[5])
    return n, hit, miss, comp, hit_req, err


def main():
    run1 = load(DATA / "results_n64.jsonl")
    run2 = load(DATA / "results_n64_rep.jsonl")
    ids = sorted(set(run1) & set(run2))
    assert len(ids) == 512, len(ids)
    rows = [run1[i] for i in ids] + [run2[i] for i in ids]

    out = {"n_prompts": 512, "n_requests": 1024}
    for p in (0, 1):
        rs = [r for r in rows if r["target_pos"] % 2 == p]
        k = sum(ok(r) for r in rs)
        out[f"phase{p}"] = {"n": len(rs), "k": k,
                            "acc": k / len(rs), "ci": wilson(k, len(rs))}
    out["diff"] = out["phase0"]["acc"] - out["phase1"]["acc"]

    out["residue"] = []
    for g in range(8):
        rs = [r for r in rows if r["target_pos"] % 8 == g]
        k = sum(ok(r) for r in rs)
        out["residue"].append({"r": g, "n": len(rs), "k": k,
                               "acc": k / len(rs), "ci": wilson(k, len(rs))})

    out["runs"] = []
    for name, rmap in (("run1", run1), ("run2", run2)):
        v = list(rmap.values())
        e = [ok(r) for r in v if r["target_pos"] % 2 == 0]
        o = [ok(r) for r in v if r["target_pos"] % 2 == 1]
        out["runs"].append({
            "name": name,
            "acc": sum(ok(r) for r in v) / len(v),
            "even": float(np.mean(e)), "odd": float(np.mean(o)),
            "diff": float(np.mean(e) - np.mean(o)),
        })

    # Cluster bootstrap over the 64 bindings, both runs together.
    slots = sorted({r["slot"] for r in rows})
    by = {s: [r for r in rows if r["slot"] == s] for s in slots}

    # Per-residue cluster bootstrap: same resampling unit as the headline CI,
    # so the error bars do not pretend the 16 prompts per key are independent.
    rng_res = np.random.default_rng(1)
    for item in out["residue"]:
        g = item["r"]
        vals = []
        for _ in range(20_000):
            pick = rng_res.choice(slots, size=len(slots), replace=True)
            xs = [ok(r) for s in pick for r in by[s] if r["target_pos"] % 8 == g]
            vals.append(np.mean(xs))
        item["ci_boot"] = [float(x) for x in np.percentile(vals, [2.5, 97.5])]

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(100_000):
        pick = rng.choice(slots, size=len(slots), replace=True)
        e = [ok(r) for s in pick for r in by[s] if r["target_pos"] % 2 == 0]
        o = [ok(r) for s in pick for r in by[s] if r["target_pos"] % 2 == 1]
        diffs.append(np.mean(e) - np.mean(o))
    diffs = np.array(diffs)
    out["boot"] = {
        "ci": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
        "p": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "resamples": 100_000,
    }

    pos = neg = tie = 0
    for s in slots:
        e = [ok(r) for r in by[s] if r["target_pos"] % 2 == 0]
        o = [ok(r) for r in by[s] if r["target_pos"] % 2 == 1]
        d = float(np.mean(e) - np.mean(o))
        pos += d > 0
        neg += d < 0
        tie += d == 0
    out["bindings"] = {"pos": int(pos), "neg": int(neg),
                       "tie": int(tie), "n": len(slots)}

    # Join the two runs by id before comparing answers.
    a = {i: (run1[i].get("answer") or "").strip() for i in ids}
    b = {i: (run2[i].get("answer") or "").strip() for i in ids}
    out["agreement"] = float(np.mean([a[i] == b[i] for i in ids]))
    both = sum(ok(run1[i]) and ok(run2[i]) for i in ids)
    neither = sum((not ok(run1[i])) and (not ok(run2[i])) for i in ids)
    out["stability"] = {"both": both, "neither": neither,
                        "split": 512 - both - neither}

    # Billed usage across every file that exists.
    main = total(["results_n64.jsonl", "results_n64_rep.jsonl"])
    early = total(["results_smoke.jsonl", "results_n16.jsonl",
                   "results_n16_rep.jsonl"])
    code = total(["code_completion.json", "code_completion_chat_run1.json",
                  "code_completion_chat_run2.json"], dedupe=False)
    n = main[0] + early[0] + code[0]
    hit = main[1] + early[1] + code[1]
    miss = main[2] + early[2] + code[2]
    comp = main[3] + early[3] + code[3]
    hit_req = main[4] + early[4] + code[4]
    err = main[5] + early[5] + code[5]
    cost = hit / 1e6 * 0.006 + miss / 1e6 * 0.30 + comp / 1e6 * 1.20
    out["cost"] = {
        "requests": n, "hit": hit, "miss": miss, "completion": comp,
        "cache_hit_requests": hit_req, "cache_hit_rate": hit_req / n,
        "errors": err,
        "hit_usd": round(hit / 1e6 * 0.006, 2),
        "miss_usd": round(miss / 1e6 * 0.30, 2),
        "out_usd": round(comp / 1e6 * 1.20, 3),
        "peak_usd": round(cost, 2),
    }
    out["experiments"] = {"main": main[0], "early": early[0],
                          "code_completion": code[0]}

    # Prefer the full prompt file when it exists (it is not committed because
    # of its size); otherwise fall back to the 512-row index shipped with the
    # results, which carries the fields this script needs.
    prompts_path = DATA / "eval_n64.jsonl"
    if not prompts_path.exists():
        prompts_path = DATA / "eval_index.jsonl"
    prompts = [json.loads(line) for line in
               open(prompts_path, encoding="utf-8")]
    out["target_shift"] = {
        "shifted": sum(r["target_pos"] != 63_976 + r["ell"] for r in prompts),
        "parity_changed": sum(
            (r["target_pos"] % 2) != ((63_976 + r["ell"]) % 2) for r in prompts),
    }

    (DATA / "page_data.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
