# Neuralese

Experimental soft-token recurrence with the local Qwen3.5 checkpoint. Each
latent step turns the model's next-token distribution into a weighted average
of input embeddings and feeds that vector back as the next sequence position.

## Setup

```bash
uv venv --system-site-packages
uv pip install --python .venv/bin/python -e .
```

The system site packages option reuses the locally installed CUDA build of
PyTorch.

## GPU placement

Placement uses Accelerate's `device_map="sequential"` with limits of
`--gpu0-max-memory 15GiB` and `--gpu1-max-memory 15GiB`. GPU 0 is filled first;
GPU 1 receives only the overflow but can use as much as the model or runtime
requires. The default policy requires at least two CUDA GPUs and
fails early with an actionable message otherwise. The limits are capacity
caps, not a manually alternating device map.

For a smaller-memory configuration:

```bash
.venv/bin/neuralese --gpu0-max-memory 14GiB --gpu1-max-memory 5GiB \
  --soft-steps 0 'What is 2 plus 2?'
```

Benchmark JSONL result configs record both GPU limits.

## Interactive harness

```bash
.venv/bin/neuralese-tui
```

The Conversation tab keeps multi-turn history and supports soft recurrence,
the no-latent baseline, hard argmax recurrence, and ordinary CoT. Text appears
token by token; soft mode also shows temperature, effective support, retained
mass, and top candidates for every latent step.

The Benchmark tab runs all four modes sequentially from one loaded checkpoint.
Its case matrix updates live with extracted answers, and four output panes keep
each mode's latest generation. The summary shows accuracy, change from
baseline, average latency, and output tokens per second. Its controls expose
latent steps, adaptive versus fixed temperature, target support, top-k,
temperature bounds, entropy stopping, and answer budget. Results are saved
after every condition to `benchmark_tui_results.jsonl` by default.

Support is `exp(entropy)` for the softmax distribution: the number of equally
likely candidates that would have the same uncertainty. Support 1 is
effectively a hard token; support 8 asks adaptive temperature to maintain a
focused mixture with uncertainty equivalent to eight equal candidates. It is
not the number of nonzero probabilities, and it is separate from top-k
truncation.

Latent duration is controlled by **Steps**, not support. Increase Steps to 32
or 64 for a longer recurrent pass. A soft run may finish early when normalized
entropy reaches Entropy stop; use a negative value to disable that guard during
experiments. The live output title reports latent progress, selected
temperature, effective support, and an explicit entropy-stop event. JSONL also
records completed latent steps and time to first visible token.

Use `Esc` to stop after the current token, `Ctrl+L` to clear conversation
history, and `Ctrl+Q` to quit. GPU limits can be overridden at launch:

```bash
.venv/bin/neuralese-tui --gpu0-max-memory 14GiB --gpu1-max-memory 15GiB
```

## Run

```bash
.venv/bin/neuralese 'If a bat and ball cost $1.10 total and the bat costs $1 more, what does the ball cost?'
```

Useful controls:

```text
--soft-steps 8          maximum recurrent positions
--soft-temperature 1.0 fixed distribution temperature (omit for adaptive)
--soft-target-support 8 adaptive effective-support target
--soft-temperature-min 0.1 / --soft-temperature-max 4.0 adaptive bounds
--soft-top-k 64         mixture support; 0 uses the full vocabulary
--entropy-stop 0.75     stop when normalized entropy becomes too high
--max-new-tokens 256    visible answer budget
--no-thinking-scaffold  start recurrence after the template's `<think>\n`
```

By default, `Thinking Process:\n\n` is appended as real tokens after the
template's `<think>\n` prefix before recurrence starts. The closing
`</think>`/visible-answer anchor is added only after latent steps. Adaptive
temperature searches bounded temperatures for effective support near 8;
passing `--soft-temperature` selects fixed behavior. The trace reports the
chosen temperature, effective support, entropy, retained mass, and embedding
diagnostics. Low retained mass emits a warning and entropy stopping remains a
guard against broad mixtures.

Use `--soft-steps 0` as the no-latent-step baseline. For a single-process
comparison of four conditions, use:

```bash
.venv/bin/neuralese --compare --soft-steps 4 --max-new-tokens 256 \
  'If a bat and ball cost $1.10 total and the bat costs $1 more, what does the ball cost?'
```

This prints labeled no-latent, hard-argmax, soft-recurrent, and ordinary
visible-CoT outputs. Hard steps feed selected IDs through the cache; ordinary
CoT lets the model generate its normal thinking tokens. These are comparison
outputs only: evaluate each condition against an answer key; comparison alone
does not establish utility or accuracy.

## Compact benchmark

The bundled `benchmark_cases.jsonl` contains 28 original cases across
arithmetic/unit conversion, state tracking, logic constraints, and symbolic
algorithms: 16 `calibration` cases and 12 longer-chain `challenge` cases. Each
requests an explicit `ANSWER:` line. Run all four modes in one model-loading
process and write JSONL results with:

```bash
.venv/bin/neuralese --benchmark --max-new-tokens 512 \
  --results-file /tmp/neuralese-results.jsonl
```

Use `--benchmark-file PATH` for another JSONL file and repeat `--case ID` to
select cases. Repeat `--difficulty calibration` or `--difficulty challenge` to
filter by tier. A recommended challenge-only run is:

```bash
.venv/bin/neuralese --benchmark --difficulty challenge --soft-steps 8 \
  --max-new-tokens 512 --results-file /tmp/neuralese-challenge.jsonl
```

Results record the mode (`baseline`, `hard_argmax`,
`soft_recurrent`, or `ordinary_cot`), exact configuration, raw output,
normalized extracted answer, correctness, status, and latency. Baseline uses
the same chat/scaffold setup with zero latent positions; recurrent modes add
the requested latent positions, while ordinary CoT uses normal visible-token
thinking. Greedy visible decoding is the default. Give ordinary CoT a generous
budget (at least 512 tokens is recommended); if its explicit marker is
truncated, the result is marked missing and does not receive credit.
