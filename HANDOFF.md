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
  stats.py       exact McNemar test, paired bootstrap CI (stdlib only)
  backends.py    generators: transformers (base or base+LoRA), Ollama, gold oracle
scripts/
  make_slice.py  write data_slices/dev200.json (seeded) and devfull.json
  evaluate.py    backend x slice -> results/<run>__<slice>.json; caches every model
                 output in predictions/<run>.jsonl and only generates what is missing
  compare.py     paired tests between two runs -> results/compare__<slice>.json
  train_lora.py  LoRA fine-tune, logs loss per step to results/train_log.jsonl;
                 checkpoints every 5 minutes and on stop, --resume continues
  plot_loss.py   results/train_log.jsonl -> results/train_loss_curve.png
  make_table.py  results/*.json -> the README results table
tests/           unit tests for the scorer and stats, run in CI without a GPU
results/         committed run outputs: the only legal source of numbers
predictions/     committed cache of raw model outputs and per-question timings
data_slices/     committed dev slices (question, gold SQL, db_id)
data/            downloaded datasets (git-ignored)
adapters/        trained LoRA weights (git-ignored)
```

Data flow: `make_slice` fixes the dev questions once. `evaluate` asks a backend for SQL
for each question, caches the raw output, hands prediction and gold to `exec_eval`, and
writes one JSON per run and slice. `compare` runs the paired tests. `make_table` reads
those JSON files and renders the table, so the README cannot drift from the
measurements.

Run names used in the table: `qwen1.5b_zeroshot`, `qwen7b_ollama_zeroshot`,
`qwen1.5b_lora`.

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
| 2026-10-01 | Slice is 200 questions, seed 42, stratified over all 20 dev databases (owner-approved) | Proportional coverage of every unseen schema at a size three runs can afford. |
| 2026-10-01 | Every model output is cached per question; scoring reads the cache | Re-scoring is free, and extending to full dev reuses the 200 already generated. |
| 2026-10-01 | Model comparisons use exact McNemar and a paired bootstrap CI | The runs share questions, so paired tests are the right ones; at n=200 unpaired CIs overlap easily. |
| 2026-10-01 | Latency excludes model load and one warm-up call | Load time is a one-off; the metric is steady-state per-question generation time. |
| 2026-10-01 | Ollama client talks to `127.0.0.1` over a persistent session | `localhost` on Windows tries IPv6 first and added about 2 s per request; a first partial 7B run was discarded for this reason and rerun. |
| 2026-10-01 | Latency is end-to-end per serving stack, not a model-only comparison | Ollama runs GGUF Q4_K_M and reuses the cached prompt prefix between consecutive questions on the same schema; transformers runs fp16 with no prefix cache. |
| 2026-10-01 | Gold SQL whitespace is normalised in training targets | Spider gold has irregular spacing; the model should not spend capacity learning it. Execution results are unaffected. |
| 2026-10-01 | Training examples longer than the token cap are dropped, not truncated | A truncated schema would teach the model to guess at tables it cannot see. |
| 2026-10-01 | Adapter is merged into the base weights for evaluation | Same inference path and latency profile as the zero-shot 1.5B row. |
| 2026-10-01 | CI installs only the dev group and tests the stdlib scorer | No GPU in CI; installing CUDA PyTorch there would add minutes and prove nothing. |
| 2026-10-01 | Training loss is computed only at SQL-token positions | Full-sequence vocabulary logits pushed a dry run past 8 GB VRAM into shared memory; slicing hidden states before the LM head removed that. |
| 2026-10-01 | Batches are packed by padded-token count (`--max-tokens 800`), not by example count | VRAM use follows tokens per batch; a token budget keeps the peak predictable across short and long schemas. |
| 2026-10-01 | Training checkpoints every 5 minutes (first one after 1 minute) and whenever the run stops | The first run was lost to a hard shutdown with nothing saved. A crash now costs at most 5 minutes. |
| 2026-10-01 | Checkpoints are written to a temp folder, fsynced, then swapped in; `state.json` is written last | A power cut mid-save must never destroy the previous good checkpoint. |
| 2026-10-01 | Resume rebuilds the batch schedule from the seed and skips completed batches; the checkpoint stores a settings fingerprint | The resumed run is the same run, and a checkpoint cannot be continued with different hyperparameters by accident. |
| 2026-10-01 | A temperature-based pause guard was built, then removed before use (owner instruction) | The owner fixed the cooling problem directly and asked for full speed; the checkpoints remain as the safety net. |
| 2026-10-01 | Priorities after the incident: training and tuned dev200 scoring are the only must-haves; demo app and full dev only if time allows | Recovery cost time before a fixed freeze. |
| 2026-10-01 | Full dev is attempted only because the measured dev200 per-question times projected the remaining 834 questions for all three models to finish before 22:30; order is LoRA, 7B, base | Owner rule: extend only if it fits. A partial full-dev run is not reported, because the three rows must share the same questions. |
| 2026-10-01 | The demo is a local command-line script, not a hosted app | It reuses the evaluated inference path and needed no new dependency; hosting stays on the roadmap. |
| 2026-10-01 | README headline is the full-dev table; the dev200 table stays underneath | Full dev is the standard, tighter measurement. dev200 was the slice fixed in advance, so it is shown rather than dropped. |

## Status checklist

- [x] Step 0: toolchain (git, uv, ollama, gh), uv env, CUDA PyTorch verified
- [x] 1: repo on GitHub (github.com/synnerg/text2sql-lora), Spider downloaded, qwen2.5:7b pulled
- [x] 2: execution-match harness, smoke-tested on 10 examples
- [x] 3: zero-shot baselines on the dev slice (1.5B transformers, 7B Ollama)
- [x] 4: LoRA fine-tune on Spider train, loss curve (run 2 completed one full epoch;
      `results/train_summary.json`, `results/train_loss_curve.png`)
- [x] 5: tuned model re-scored on dev200, paired tests run, README opens with the 3-row
      table (`results/*__dev200.json`, `results/compare__dev200.json`)
- [x] 6: pushed, minimal CI green (ruff + pytest on the scorer and stats)
- [x] Local command-line demo (`scripts/ask.py`); a hosted demo is still roadmap
- [x] Full-dev evaluation (1,034 questions): all three models completed at 22:18, inside
      the 22:30 cutoff (`results/*__devfull.json`, `results/compare__devfull.json`). The
      README headline table is now full dev, with the dev200 table kept below it
- [ ] Freeze: 3 resume bullets from measured numbers only

## Incidents

### 2026-10-01: hard shutdown during the first training run

- The laptop overheated and powered off during the first real training run (the owner
  reports about 6 minutes in). Windows logged Kernel-Power event 41 on reboot at
  20:18:53. The firmware's ACPI thermal zone has a critical trip point of 373 K
  (Kernel-Power event 125).
- That run had no checkpointing, so its weights were lost. Its loss log survived up to
  optimizer step 75 and is kept as `results/train_log_run1_interrupted.jsonl`. It is
  not the run reported in the README.
- Checks after reboot, all passed: `git fsck` clean and local HEAD equal to
  `origin/main`; both baseline prediction caches have 200 rows matching the dev200
  question ids; every JSON under `results/` and `data_slices/` parses and its counts
  are consistent; `spider_data.zip` matches its published SHA256; `PRAGMA quick_check`
  is ok on all 166 Spider databases; the Ollama model blobs match their digests; the
  Qwen2.5-1.5B safetensors file loads (338 tensors); CUDA PyTorch still sees the GPU.
- Changes made: 5-minute checkpoints with resume (see decisions log). A temperature
  pause guard was written and then removed at the owner's instruction once they had
  fixed the cooling problem.
- Run 2 uses the same hyperparameters as run 1:
  `--epochs 1 --max-len 768 --max-tokens 800 --max-batch 8 --grad-accum 4 --lr 2e-4
  --time-budget-min 33`, plus `--ckpt-every-min 5 --resume --wall-deadline 21:25`.

If training is interrupted again, rerun the same command: `--resume` picks up the newest
complete checkpoint in `adapters/qwen1.5b-spider-lora/checkpoint`.

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
   and the executed result. (A local command-line version, `scripts/ask.py`, exists;
   nothing is hosted.)
5. **Evaluation depth.** Per-hardness breakdown and Spider's test-suite databases for
   stricter execution matching. (Full Spider dev was completed on 2026-10-01 and is no
   longer roadmap.)

## How to resume

1. `uv sync` creates the environment (CUDA PyTorch comes from the pinned index in
   `pyproject.toml`).
2. Read the status checklist above for what is done.
3. Every number you may quote lives in `results/`. If it is not there, measure it.
