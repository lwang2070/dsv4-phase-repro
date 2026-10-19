# /// script
# dependencies = ["numpy"]
# ///
"""Analyze the NIAH phase-sensitivity results."""
import collections
import json
import math
import sys

import numpy as np


def ok(r):
    return (r.get("answer") or "").strip().startswith(r["gold"])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(path):
    best = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            continue
        best[r["id"]] = r  # later success wins; there is at most one
    return list(best.values())


def main(path):
    rows = load(path)
    print(f"rows: {len(rows)}")
    print(f"overall accuracy: {sum(ok(r) for r in rows)/len(rows):.4f}")

    print("\nby actual target_pos mod 8 (phase = pos mod S; V4.1 S=2):")
    for g in range(8):
        rs = [r for r in rows if r["target_pos_mod8"] == g]
        k = sum(ok(r) for r in rs)
        lo, hi = wilson(k, len(rs))
        print(f"  {g}: n={len(rs):4d} acc={k/len(rs):.3f} "
              f"Wilson95=[{lo:.3f},{hi:.3f}]")

    print("\nby phase mod 2:")
    groups = {}
    for p in (0, 1):
        rs = [r for r in rows if r["target_pos"] % 2 == p]
        k = sum(ok(r) for r in rs)
        groups[p] = (k, len(rs))
        lo, hi = wilson(k, len(rs))
        print(f"  {p}: n={len(rs):4d} acc={k/len(rs):.3f} "
              f"Wilson95=[{lo:.3f},{hi:.3f}]")
    (k0, n0), (k1, n1) = groups[0], groups[1]
    p0, p1 = k0 / n0, k1 / n1
    pp = (k0 + k1) / (n0 + n1)
    se = math.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1))
    z = (p0 - p1) / se
    print(f"  diff(even-odd)={p0-p1:+.4f}  z={z:.2f}  "
          f"p={math.erfc(abs(z)/math.sqrt(2)):.2e}")

    # Cluster bootstrap over bindings: each binding contributes one prompt per
    # residue, so this keeps binding difficulty out of the comparison.
    slots = sorted({r["slot"] for r in rows})
    by = {s: [r for r in rows if r["slot"] == s] for s in slots}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(20000):
        pick = rng.choice(slots, size=len(slots), replace=True)
        e = [ok(r) for s in pick for r in by[s] if r["target_pos"] % 2 == 0]
        o = [ok(r) for s in pick for r in by[s] if r["target_pos"] % 2 == 1]
        if e and o:
            diffs.append(np.mean(e) - np.mean(o))
    diffs = np.array(diffs)
    obs = p0 - p1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    print(f"  cluster bootstrap over {len(slots)} bindings: "
          f"95% CI=[{lo:+.3f},{hi:+.3f}]  p={pval:.2e}")

    # Per-binding even/odd contrast (4 vs 4 prompts).
    print("\nper-binding accuracy (even / odd):")
    pos = neg = tie = 0
    for s in slots:
        e = [ok(r) for r in by[s] if r["target_pos"] % 2 == 0]
        o = [ok(r) for r in by[s] if r["target_pos"] % 2 == 1]
        d = float(np.mean(e) - np.mean(o))
        pos += d > 0
        neg += d < 0
        tie += d == 0
        print(f"  slot {s:2d}: {sum(e)}/{len(e)} vs {sum(o)}/{len(o)}  "
              f"diff={d:+.2f}")
    print(f"  bindings with even>odd: {pos} "
          f"(odd>even: {neg}, tie: {tie})")

    print("\nfailure examples:")
    for r in rows:
        if not ok(r):
            print(f"  {r['id']} mod8={r['target_pos_mod8']} "
                  f"gold={r['gold']!r} ans={r['answer']!r}")


if __name__ == "__main__":
    main(sys.argv[1])
