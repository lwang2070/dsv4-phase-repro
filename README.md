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

The V4.1-Flash chat version of the second experiment (Appendix B.2, code
completion) was partially reproduced: every odd filler length favors `8`, but
only five of the eight even lengths flip to `32`, so the even side is much
weaker than in the paper. See "Second experiment" below.

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
scripts/code_completion_chat.py  Appendix B.2 chat-format control
scripts/prob_sweep.py        filler sweep: chat / raw / FIM, real probabilities
scripts/analyze_prob_sweep.py  summarize the sweep files
scripts/probe_*.py           tokenizer probes used while designing the prompt
code/kernel.py               FP8 kernel prefix from the paper (unmodified)
data/results_n64*.jsonl      the 1,024 API responses (shipped)
data/results_n16*.jsonl      early 16-per-group pilot
data/code_completion*.json   first-round responses: `code_completion.json`
                             is raw completion (never had the seam bug);
                             `code_completion_chat.json` is chat format with
                             the pre-fix prompt, one token off (kept for the
                             record; reruns write `*_rerun.json`)
data/prob_sweep_*.json       filler sweeps at temperature 0/1 (chat/raw/FIM)
data/prob_sweep_summary.json summary used in the report
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

## Second experiment (Appendix B.2)

The paper asks the model to finish a line of FP8 kernel code. The next token
is `8` (the original code, FP8) or `32` (FP32, wrong). For
DeepSeek-V4.1-Flash with the chat prompt, Appendix B.2 reports, for filler
families A, B, and C, `8` at odd filler lengths `L` and `32` at even ones
(period 2, matching its compression stride); family D answers `8` at
`L = 23..29` and then alternates. We use family A. Figure 1 itself uses the
base model, which the API does not serve; what we can test is the
V4.1-Flash chat result.

We partially reproduce the alternation. Four temperature-1 sweeps of
`L = 23..38` (`scripts/prob_sweep.py --format chat`) agree with the paper's
direction at 13 of 16 lengths. Mean margin `P(8) - P(32)`: **+0.75 at odd L,
-0.06 at even L**; the paper's family A gives +0.35 and -0.57. Every odd
length favors `8`; only five of eight even lengths flip to `32`. The three
disagreements are all even lengths: +0.093, +0.365, and +0.007 (the last
mean is +0.007, essentially a tie; +0.093 is a real gap in the other
direction, not a tie). Two temperature-0 sweeps score 15/16 and 12/16 in top-1,
again with `32` only at even lengths. Repeating the same prompt changes some
answers even at temperature 0; we did not verify why.

Our first attempt missed the pattern because of a one-token reconstruction
error: the filler already ends with a newline, and we added `"\n\n"` after it,
which is two blank lines instead of one. The scored token moved one position
later, so the parity no longer lined up with `L`; the old data looked like
period 4 or unstable. The scripts now assert the paper's scored position,
`t = 1004 + L`.

FIM (`--format fim`, `/beta/completions` with a neutral suffix
`# end of file`, `L = 24..39`) is included as a control, not as the paper's
setting: with a suffix all 16 lengths answer `8`, with
`P(8) - P(32) >= 0.99`. The paper's prompt has no suffix. Raw completion
(`--format raw`) also answers `8` at every length, but it changes more than
the suffix: no instruction, no chat template, and a different endpoint. So
the defensible claim is narrower: on V4.1-Flash-1017, leaving the
"chat format with no suffix" setup made the alternation disappear; we cannot
attribute that to the suffix alone. In the paper's other settings the period
is four, not two: the base model with a raw prefix (Appendix B.1) and the
chat-formatted V4-Flash-0731 (Appendix B.2). We test neither; our raw run is
V4.1-Flash-1017.

Reproduce the numbers:

```bash
for i in 1 2 3 4; do
  uv run -q --with tokenizers --with httpx python scripts/prob_sweep.py \
    --format chat --temperature 1 --out data/prob_sweep_chat_t1_run$i.json
done
uv run -q python scripts/analyze_prob_sweep.py
```

The analyzer needs all nine files (4 chat at temperature 1, 2 chat at
temperature 0, 2 raw, 1 FIM). The shipped files satisfy that; if you delete
one, the analyzer stops instead of summarizing a partial sweep. Regenerate
the others with `--format chat --temperature 0` twice, `--format raw` twice,
and `--format fim` once, matching the file names in `data/`.

The separate chat script (`scripts/code_completion_chat.py`) was fixed the
same way: one newline at the seam, and the same position assertion.

The source prefix comes from `code/kernel.py`; the first token scored is the
`8` in `T.Cast(compute_dtype, T.Cast(FP8...`. Raw completion
(`--format raw`, no chat template, `L = 24..39`) answers `8` at every length
as well.

## Cost

1,336 billed requests: 81.9 M cache-hit input tokens, 83.1 M cache-miss input
tokens, 1,823 output tokens. **$25.41 at peak prices**, about $12.7 off-peak
(50% discount). Another 252 requests failed with HTTP 402 and were not billed.

The later second-experiment sweeps add about 400 requests. Responses that
reached a data file (224 calls, including 80 that a later rerun overwrote)
used 230,192 input tokens and 224 output tokens, about $0.013. The rest were
exploratory calls without a saved usage record; at the same unit price they
add roughly $0.011. **All calls together are about $25.43.**

## Caveats

- The paper does not publish its intro text. Our prompt is a faithful
  reconstruction of Appendix C.1, not byte-identical. Our first chat prompt
  for the second experiment also had one extra newline at the
  filler/code seam; see "Second experiment".
- The API is not deterministic: at temperature 0, repeated runs agreed on
  80% of answers. Headline numbers average two runs.
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
