# HANDOFF: text2sql-lora

Model-agnostic brief. Any engineer or coding assistant should be able to pick this
project up from this file alone. Keep it current: update the status checklist and the
decisions log in the same commit as the change they describe.

Written 2026-10-01. Build freeze 2026-10-01 23:00 PT.

## Rule zero: no unmeasured number is ever written anywhere

Every accuracy, latency, loss, speed-up or size figure in this repo (README, HANDOFF,
commit messages, resume bullets, comments) must come from a run of the code in this
repo, and must be traceable to a file under `results/`. If it was not measured, it is
not written. No estimates, no "expected", no numbers quoted from papers as if they were
ours. Anything not finished is described as roadmap, never as done.

Configuration values (hyperparameters, hardware specs, dataset sizes read from disk)
are not results and may be written, but they must match what the code actually uses.

## Goal

Show, with an execution-based benchmark, how much a small LoRA fine-tune improves a
1.5B-parameter model at text-to-SQL, and where it lands relative to a larger zero-shot
model, all on one consumer GPU.

The deliverable is a three-row results table, measured on the same dev slice with the
same prompt and the same scorer:

| Row | Model | How it runs |
| --- | --- | --- |
| 1 | Qwen2.5-1.5B-Instruct, zero-shot | transformers, fp16, GPU |
| 2 | qwen2.5:7b, zero-shot | Ollama (default 4-bit quant) |
| 3 | Qwen2.5-1.5B-Instruct + LoRA (this repo) | transformers, fp16, GPU |

## Spec

- **Task**: natural-language question + SQLite schema in, one SQLite query out.
- **Data**: Spider (Yu et al., 2018, CC-BY-SA-4.0). Train on `train_spider.json`,
  evaluate on a fixed seeded slice of `dev.json`. Train and dev use disjoint databases,
  so the dev slice measures generalisation to unseen schemas.
- **Metric**: execution accuracy. Predicted and gold SQL are both run against the real
  SQLite database and their result sets are compared. Exact string match is not used.
- **Latency**: p50 and p95 of per-question generation time, plus p95 of SQL execution
  time, measured in the same run that produces the accuracy.
- **Hardware**: one RTX 4060 Laptop GPU (8 GB VRAM), 16 GB RAM, Windows 11. Fully
  local: no hosted inference, no API keys.
- **Tooling**: Python 3.12 managed by uv, PyTorch (CUDA build), transformers, peft,
  Ollama for the 7B baseline, stdlib `sqlite3` for execution.
- **Fallback data plan**: if the Spider databases cannot be obtained, generate TPC-H
  with DuckDB and author a question set against it. The scorer is database-agnostic.

## Architecture

```
t2sql/
  data.py        load Spider examples, read schema from the live SQLite file,
                 build the seeded dev slice and the training subsample
  prompt.py      the single prompt shared by every model; SQL extraction from output
  exec_eval.py   read-only SQL execution with a timeout, result-set comparison,
                 accuracy + latency aggregation (stdlib only)
  backends.py    generators: transformers (base or base+LoRA), Ollama, gold oracle
scripts/
  make_slice.py  write data/dev_slice.json
  evaluate.py    backend x slice -> results/<run>.json (+ per-example predictions)
  train_lora.py  LoRA fine-tune, logs loss per step to results/train_log.jsonl
  plot_loss.py   results/train_log.jsonl -> results/loss_curve.png
  make_table.py  results/*.json -> the README results table
tests/           unit tests for the scorer, run in CI without a GPU
results/         committed run outputs: the only legal source of numbers
data/            downloaded datasets (git-ignored)
```

Data flow: `make_slice` fixes the dev questions once. `evaluate` asks a backend for SQL
for each question, hands prediction and gold to `exec_eval`, and writes one JSON per
run. `make_table` reads those JSON files and renders the table, so the README cannot
drift from the measurements.

## Decisions log

| Date | Decision | Why |
| --- | --- | --- |
| 2026-10-01 | Execution match, not exact match | Many different SQL strings are correct; only results matter. |
| 2026-10-01 | Databases opened read-only, with a per-query timeout | Predicted SQL is untrusted; it must not modify or hang the benchmark. |
| 2026-10-01 | Row order is compared only when the gold query has ORDER BY; column order is ignored | Matches the convention of Spider's execution evaluation. |
| 2026-10-01 | One prompt for all three models | Otherwise the table compares prompts, not models. |
| 2026-10-01 | Schema is read from the SQLite file, not `tables.json` | The file is what the query actually runs against. |
| 2026-10-01 | Greedy decoding everywhere | Deterministic, reproducible scores. |
| 2026-10-01 | Plain fp16 LoRA, no quantisation, for the 1.5B model | It fits in 8 GB without 4-bit loading; fewer moving parts for a one-evening build. |
| 2026-10-01 | Spider obtained from the `HAL-9001/spider-databases` mirror on Hugging Face | `xlangai/spider` ships questions without the SQLite files; this mirror includes them and publishes a SHA256. |
| 2026-10-01 | Dev slice instead of full dev | Time budget for three model runs before the freeze. Slice is seeded and committed. |

## Status checklist

- [ ] Step 0: toolchain (git, uv, ollama, gh), uv env, CUDA PyTorch verified
- [ ] 1: repo on GitHub, Spider downloaded, qwen2.5:7b pulled
- [ ] 2: execution-match harness, smoke-tested on 10 examples
- [ ] 3: zero-shot baselines on the dev slice (1.5B transformers, 7B Ollama)
- [ ] 4: LoRA fine-tune on Spider train, loss curve
- [ ] 5: tuned model re-scored, README opens with the 3-row table
- [ ] 6: pushed, minimal CI green
- [ ] Freeze: 3 resume bullets from measured numbers only

## Roadmap (future work: none of this is started as of 2026-10-01)

Nothing below has been built or measured. Do not describe any of it as done.

1. **GRPO reinforcement learning with execution rewards.** Sample several SQL
   candidates per question, reward by execution match against gold, optimise with
   group-relative policy optimisation on top of the supervised LoRA.
2. **Synthetic self-distillation.** Generate new question/SQL pairs on the training
   databases with a larger model, keep only pairs whose SQL executes and passes
   consistency filters, and add them to the training set.
3. **QLoRA at 7B.** 4-bit quantised base with LoRA adapters so a 7B model can be tuned
   within 8 GB VRAM, scored with the same harness.
4. **Deployed demo.** A small hosted app: pick a database, ask a question, see the SQL
   and the executed result.
5. **Evaluation depth.** Full Spider dev, per-hardness breakdown, and Spider's
   test-suite databases for stricter execution matching.

## How to resume

1. `uv sync` creates the environment (CUDA PyTorch comes from the pinned index in
   `pyproject.toml`).
2. Read the status checklist above for what is done.
3. Every number you may quote lives in `results/`. If it is not there, measure it.
