# /// script
# dependencies = ["httpx"]
# ///
"""Run the NIAH eval set against the DeepSeek API.

Writes one JSON line per prompt, appending and skipping finished ids so a
crashed run can resume.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

URL = "https://api.deepseek.com/chat/completions"


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


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["id"])
            except Exception:
                pass
    return done


async def one(client, sem, row, key, retries=6):
    body = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 0,
        "max_tokens": 8,
        "thinking": {"type": "disabled"},
        "logprobs": True,
        "top_logprobs": 20,
    }
    async with sem:
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = await client.post(URL, json=body,
                                      headers={"Authorization": f"Bearer {key}"},
                                      timeout=600)
                if r.status_code == 200:
                    d = r.json()
                    ch = d["choices"][0]
                    msg = ch.get("message", {})
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    top = lp[0].get("top_logprobs", []) if lp else []
                    return {
                        "id": row["id"],
                        "residue": row["residue"],
                        "ell": row["ell"],
                        "slot": row["slot"],
                        "target_pos": row["target_pos"],
                        "target_pos_mod8": row["target_pos_mod8"],
                        "gold": row["gold"],
                        "target_key": row["target_key"],
                        "answer": msg.get("content"),
                        "reasoning": msg.get("reasoning_content"),
                        "first_lp": lp[0].get("logprob") if lp else None,
                        "top_logprobs": top,
                        "usage": d.get("usage"),
                        "elapsed": round(time.time() - t0, 1),
                    }
                if r.status_code in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                return {"id": row["id"], "error": f"HTTP {r.status_code}: {r.text[:500]}"}
            except Exception as e:
                if attempt == retries - 1:
                    return {"id": row["id"], "error": repr(e)}
                await asyncio.sleep(min(60, 2 ** attempt))
    return {"id": row["id"], "error": "exhausted retries"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    key = read_key()
    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines()]
    out = Path(args.out)
    done = load_done(out)
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} rows, {len(done)} done, running {len(todo)}", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency,
                          max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [asyncio.create_task(one(client, sem, row, key)) for row in todo]
        n_ok = n_err = 0
        tokens = 0
        with open(out, "a", encoding="utf-8") as f:
            for fut in asyncio.as_completed(tasks):
                res = await fut
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                if "error" in res:
                    n_err += 1
                    print("ERR", res["id"], res["error"][:200], file=sys.stderr)
                else:
                    n_ok += 1
                    u = res.get("usage") or {}
                    tokens += u.get("prompt_tokens", 0)
                    print(f"OK {res['id']} ans={res['answer']!r} "
                          f"gold={res['gold']!r} ptok={u.get('prompt_tokens')} "
                          f"{res['elapsed']}s", file=sys.stderr)
        print(f"done ok={n_ok} err={n_err} input_tokens={tokens}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
