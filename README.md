# Periodic weak spots in DeepSeek-V4.1-Flash: a reproduction attempt

Testing the position-parity effect reported in
[Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression
(arXiv:2609.36322)](https://arxiv.org/abs/2609.36322)
against the DeepSeek API (`deepseek-flash`, reported as
`DeepSeek-V4.1-Flash-1017`).

## Result

| target position mod 2 | n (two runs) | accuracy |
|---|---|---|
| even | 504 | 80.0% |
| odd | 520 | 70.4% |

- Difference: **+9.6 percentage points** (even − odd).
- 95% CI by cluster bootstrap over the 64 bindings: **[+1.8, +17.5] pp**,
  p = 0.016 (100,000 resamples).
- 27 bindings favor even positions, 25 tie, 12 favor odd positions.
- Two independent runs: +12.5 pp and +6.7 pp.

The second experiment in the paper (Figure 1, code completion) was **not**
reproduced. See "Second experiment" below.

## Setup

- 128,000-token prompts, 16,000 `key:value。` records of 4 tokens each.
- Target record #249 (index 248); its key starts at token `63,976 + ell`.
- `ell` in 8..40, histograms follow Appendix C.1; `ell % 8` is the residue.
- One binding per (residue, slot); 512 prompts per run, two runs.

## Repository layout

```
scripts/build_eval.py        build the 512 prompts
scripts/run_eval.py          call the chat API (resumable)
scripts/analyze_combined.py  pooled statistics -> data/page_data.json
scripts/analyze.py           single-run statistics
scripts/code_completion.py   Figure 1 control, raw completion endpoint
scripts/code_completion_chat.py  Figure 1 control, chat format
scripts/probe_*.py           tokenizer probes used while designing the prompt
code/kernel.py               FP8 kernel prefix from the paper (unmodified)
data/results_n64*.jsonl      the 1,024 API responses (shipped)
data/results_n16*.jsonl      early 16-per-group pilot
data/code_completion*.json   code-completion responses
data/eval_index.jsonl        id/ell/target_pos for the 512 prompts
data/page_data.json          every number cited in the report
report.html                  static Chinese report (open in a browser)
code/NOTICE.md               attribution for the paper's kernel excerpt
```

Large assets are **not** committed: `data/train.txt` (197 MB),
`data/eval_n64.jsonl` (271 MB), `data/eval_n16.jsonl`, and the tokenizer.
All of them are regenerable or downloadable (below).

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- A DeepSeek API key in `DEEPSEEK_API_KEY`.

```bash
git clone https://github.com/lwang2070/dsv4-phase-repro
cd dsv4-phase-repro
export DEEPSEEK_API_KEY=...
```

## Assets

Official tokenizer (chat template included; used to measure token counts):

```bash
uv run -q --with huggingface_hub hf download \
  deepseek-ai/DeepSeek-V4.1-Flash \
  tokenizer.json tokenizer_config.json chat_template.jinja \
  --local-dir tokenizer
```

Filler corpus (`agentlans/high-quality-english-sentences`, train split,
one sentence per line):

```bash
uv run -q --with datasets python -c "
from datasets import load_dataset
ds = load_dataset('agentlans/high-quality-english-sentences', split='train')
open('data/train.txt', 'w').write('\n'.join(ds['text']))
"
```

## Reproduce

```bash
# 1. build 512 prompts (~2 minutes)
uv run -q --with tokenizers python scripts/build_eval.py \
  --per-residue 64 --out data/eval_n64.jsonl

# 2. two independent runs (~$10 each at peak prices)
uv run -q --with httpx python scripts/run_eval.py \
  --input data/eval_n64.jsonl --out data/results_n64.jsonl
uv run -q --with httpx python scripts/run_eval.py \
  --input data/eval_n64.jsonl --out data/results_n64_rep.jsonl

# 3. pooled statistics -> data/page_data.json
uv run -q --with numpy python scripts/analyze_combined.py
```

`run_eval.py` appends one JSON line per request and skips finished ids, so an
interrupted run can be restarted with the same command.

The shipped `data/results_n64*.jsonl` let you re-run step 3 without spending
anything. Step 3 falls back to `data/eval_index.jsonl` when the full prompt
file `data/eval_n64.jsonl` is absent.

## Second experiment (Figure 1)

The paper's Figure 1 asks the model to finish a line of FP8 kernel code. The
next token is `8` (the original code, FP8) or `32` (FP32, wrong). Appendix B.2
reports, for the chat format, `8` at odd filler lengths `L` and `32` at even
ones.

We ran `L = 23..38` twice through the chat endpoint:

- Run 1: `32` at L = 25, 29, 33, 37 (period 4).
- Run 2: `32` at L = 35, 37 only.
- Raw completion format (`/beta/completions`), L = 24..39: all `8`.

Neither chat run matches the paper's odd/even pattern, so we do **not** claim
a replication of Figure 1. Differences that may explain it: the paper's chat
experiment uses the Post-trained-0910 checkpoint, the API serves the 1017
version; and Figure 1 itself uses the base model, which the API does not serve.

Run it with:

```bash
uv run -q --with tokenizers --with httpx python scripts/code_completion_chat.py \
  --n-min 8 --n-max 23 --out data/code_completion_chat_run1.json
```

## Cost

1,336 billed requests: 81.9 M cache-hit input tokens, 83.1 M cache-miss input
tokens, 1,823 output tokens. **$25.41 at peak prices**, about $12.7 off-peak
(50% discount). Another 252 requests failed with HTTP 402 and were not billed.

## Caveats

- The paper does not publish its intro text. Our prompt is a faithful
  reconstruction of Appendix C.1, not byte-identical.
- The API is not deterministic: repeated runs agreed on 80% of answers.
  Headline numbers average two runs.
- Overall accuracy is 75.1%, below the paper's 92.4%, so the task is harder
  here.
- Only V4.1-Flash was tested; no V4-Pro and no base model.
- Tokenization can shift the target by a few positions: 25 of 512 prompts
  moved, 22 changed parity.

## License

Our scripts and data are released under the MIT License (see `LICENSE`).
`code/kernel.py` comes from the paper's appendix; the paper is licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and attribution is
in `code/NOTICE.md`. The tokenizer is downloaded from DeepSeek and keeps the
DeepSeek model license.
