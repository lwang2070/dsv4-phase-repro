# /// script
# dependencies = []
# ///
"""Summarize the sweep files written by scripts/prob_sweep.py.

The paper's claim for DeepSeek-V4.1-Flash (Appendix B.2, Table 4) is:
8 at odd filler lengths and 32 at even filler lengths for families A, B, and
C (family D answers 8 at L = 23..29, then alternates). For a sweep at
temperature 1 we compare the sign of delta = P(8) - P(32); for a sweep at
temperature 0 the reported top-1 is already a one-hot distribution.

Every sweep file must cover its full length range with no error rows. A
partial file would change the means and the match count, so we stop instead
of summarizing it.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Chat prompt: L = 15 + n_eq with n_eq 8..23. Raw/FIM prefix: n_eq 9..24.
CHAT_LENGTHS = set(range(23, 39))
RAW_LENGTHS = set(range(24, 40))


def load(name, fmt, temperature, lengths):
    """Load one sweep file and check it is the complete sweep we asked for.

    A missing file, a short file, or an error row is an error: silently
    skipping any of them would change the means and the match count without
    saying so.
    """
    path = DATA / name
    if not path.exists():
        raise SystemExit(f"missing sweep file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != fmt or float(data.get("temperature", -1)) != temperature:
        raise SystemExit(
            f"{name}: format={data.get('format')!r} "
            f"temperature={data.get('temperature')!r}, "
            f"expected {fmt!r} / {temperature}")
    rows = data["rows"]
    bad = [row for row in rows if "error" in row]
    if bad:
        raise SystemExit(f"{name}: {len(bad)} error rows, first: {bad[0]}")
    got = [row["L"] for row in rows]
    if len(got) != len(set(got)):
        raise SystemExit(f"{name}: duplicate lengths")
    if set(got) != lengths:
        raise SystemExit(
            f"{name}: lengths {sorted(got)} != expected {sorted(lengths)}")
    return data


def expected(L):
    return "8" if L % 2 == 1 else "32"


def summarize_t1(runs):
    by_L = {}
    for run in runs:
        for row in run["rows"]:
            if row.get("delta") is None:
                raise SystemExit(
                    f"{run.get('format')} t={run.get('temperature')}: "
                    f"L={row['L']} has no delta (8 or 32 was not reported)")
            by_L.setdefault(row["L"], []).append(row["delta"])
    per_L = []
    for L in sorted(by_L):
        vals = by_L[L]
        mean = statistics.mean(vals)
        # A mean of exactly 0 is a tie, which we count against both parities.
        matches = mean > 0 if L % 2 == 1 else mean < 0
        per_L.append({"L": L, "mean_delta": mean,
                      "runs": len(vals),
                      "predicted": expected(L),
                      "matches": matches})
    odd = [x["mean_delta"] for x in per_L if x["L"] % 2 == 1]
    even = [x["mean_delta"] for x in per_L if x["L"] % 2 == 0]
    return {
        "runs": len(runs),
        "per_L": per_L,
        "odd_mean_delta": statistics.mean(odd),
        "even_mean_delta": statistics.mean(even),
        "odd_minus_even": statistics.mean(odd) - statistics.mean(even),
        "sign_matches": sum(x["matches"] for x in per_L),
        "n_lengths": len(per_L),
        "even_below_zero": sum(1 for x in per_L if x["L"] % 2 == 0 and x["mean_delta"] < 0),
        "odd_above_zero": sum(1 for x in per_L if x["L"] % 2 == 1 and x["mean_delta"] > 0),
    }


def summarize_top1(runs):
    out = []
    for i, run in enumerate(runs, 1):
        got = {r["L"]: r["text"] for r in run["rows"]}
        bad = {L: t for L, t in got.items() if t not in ("8", "32")}
        if bad:
            raise SystemExit(f"chat t=0 run{i}: unexpected answer {bad}")
        matches = sum(got[L] == expected(L) for L in sorted(got))
        out.append({"run": i, "answers": got,
                    "matches": matches, "n": len(got)})
    return out


def summarize_flat(runs):
    """Raw and FIM sweeps: delta range and the top-1 answer counts."""
    out = []
    for i, run in enumerate(runs, 1):
        rows = run["rows"]
        deltas = [r["delta"] for r in rows]
        if any(d is None for d in deltas):
            raise SystemExit(f"{run.get('format')}: a row has no delta")
        bad = {r["L"]: r["text"] for r in rows if r["text"] not in ("8", "32")}
        if bad:
            raise SystemExit(f"{run.get('format')} run{i}: unexpected answer {bad}")
        out.append({
            "run": i,
            "n": len(rows),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "top1_8": sum(r["text"] == "8" for r in rows),
            "top1_32": sum(r["text"] == "32" for r in rows),
            "usage": run.get("usage"),
        })
    return out


def main():
    chat_t1 = summarize_t1([load(f"prob_sweep_chat_t1_run{i}.json",
                                 "chat", 1, CHAT_LENGTHS) for i in range(1, 5)])
    chat_t0 = summarize_top1([load(f"prob_sweep_chat_t0_run{i}.json",
                                   "chat", 0, CHAT_LENGTHS) for i in range(1, 3)])
    raw_t1 = summarize_flat([load(f"prob_sweep_raw_t1_run{i}.json",
                                  "raw", 1, RAW_LENGTHS) for i in range(1, 3)])
    fim_t1 = summarize_flat([load("prob_sweep_fim_t1.json",
                                  "fim", 1, RAW_LENGTHS)])

    summary = {"chat_temperature_1": chat_t1,
               "chat_temperature_0": chat_t0,
               "raw_temperature_1": raw_t1,
               "fim_temperature_1": fim_t1}
    (DATA / "prob_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
