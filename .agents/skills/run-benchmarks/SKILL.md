---
name: run-benchmarks
description: Run ASV benchmarks for the BPE tokenizer. Use when benchmarking is requested, correctness needs verifying, a performance regression is suspected, or changes are being committed.
---

# Running Benchmarks

This project uses [ASV (AirSpeed Velocity)](https://asv.readthedocs.io/) to benchmark the BPE tokenizer implementation. Benchmarks live in `benchmarks/`.

## Benchmark Files

- `benchmarks/bench_pretokenizer.py` — Benchmarks segmentation + pretokenization (`PretokenizerBenchmark.time_segment_and_pretokenize`).
- `benchmarks/bench_tokenizer.py` — Benchmarks BPE merge training with parameterized merge counts (1, 5, 25) (`TokenizerBenchmark.time_train`).

## Commands

Run all commands with `uv run` so the project environment is activated automatically.

### Validate the benchmark suite

```sh
uv run asv check
```

Discovers benchmarks and verifies they load without errors. Does not actually run them. Use this to catch import errors or broken setup methods before committing time to a full run.

### Quick sanity check (results not saved)

```sh
uv run asv run --quick
```

Runs each benchmark once. Fast, but results are **not saved** to disk. Useful for checking that benchmarks run without crashing.

Output looks like:

```
[25.00%] ··· PretokenizerBenchmark.time_segment_and_pretokenize    433±0ms
[50.00%] ··· TokenizerBenchmark.time_train
              param1
             -------- ---------
                1      519±0ms
                5      333±0ms
               25     507±0ms
```

### Full benchmark run (results saved)

```sh
uv run asv run
```

Runs the full benchmark suite with proper repeats and saves results to `.asv/results/`. This builds the package from the latest committed code — uncommitted changes are **not** included.

### Benchmark uncommitted changes

```sh
uv run asv run --environment existing --quick
```

Uses the current Python environment instead of building from a commit, so uncommitted changes are picked up. Use `--quick` for fast iteration (results not saved), or drop `--quick` and add `--set-commit-hash HEAD` to save results.

### Run a specific benchmark

```sh
uv run asv run --bench bench_pretokenizer
```

Or target a specific method:

```sh
uv run asv run --bench "PretokenizerBenchmark.time_segment_and_pretokenize"
```

The `--bench` flag accepts a regex matching the benchmark name.

### Publish results to HTML

```sh
uv run asv publish
```

Collates all saved results into an HTML website written to `.asv/html/` (configured in `asv.conf.json`).

### View stored results from the command line

```sh
uv run asv show
```

Lists commits with benchmark results, the machine name, and environment used.

## When to Run Benchmarks

Run benchmarks when:

- Changes were made to the tokenizer, segmenter, or pretokenizer and correctness/performance needs verifying.
- A performance regression is suspected.
- Code is about to be committed and a sanity check is warranted.
- Completing benchmarks successfully is a basic correctness signal.
- Two approaches are being compared and quantitative data is needed.

## Tips

- Use `--quick` during development to verify benchmarks load and run without errors before committing to a full run.
- Use `--bench` to focus on a single benchmark when iterating on one component.
- Benchmark results are keyed by commit hash. To benchmark uncommitted changes, use `--environment existing`.
- If benchmarks fail with an import error, double-check that the module paths in the benchmark files match your actual source structure.
