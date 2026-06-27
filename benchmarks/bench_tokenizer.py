"""
asv benchmarks for the BPE tokenizer training (merging).

Run with:
    asv run
"""

from pathlib import Path

_SPECIAL_TOKENS = [b"<|endoftext|>"]


class TokenizerBenchmark:
    """Benchmark BPE tokenizer training (merge steps only)."""

    # Number of merges to benchmark
    params = [1, 5, 25]

    def setup_cache(self):
        """Run segment + pretokenize once and return the token counts."""
        from cs336_basics.bpe_tokenizer.api import segment_and_pretokenize

        project_root = Path(__file__).resolve().parent.parent
        input_path = project_root / "tests" / "fixtures" / "tinystories_sample_5M.txt"

        return segment_and_pretokenize(input_path, _SPECIAL_TOKENS)

    def time_train(self, counts, merges):
        from cs336_basics.bpe_tokenizer.api import train

        train(counts, _SPECIAL_TOKENS, merges)
