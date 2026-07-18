"""
asv benchmarks for the BPE tokenizer training (merging).

Run with:
    asv run
"""

from pathlib import Path

_SPECIAL_TOKENS = ["<|endoftext|>"]


class TokenizerBenchmark:
    """Benchmark BPE tokenizer training (merge steps only)."""

    params = [["TinyStoriesV2-GPT4-valid.txt", "owt_valid.txt"], [1, 5, 25]]
    param_names = ["file", "n_merges"]


    def setup_cache(self):
        """Run segment + pretokenize once and return the token counts."""
        from cs336_basics.bpe_tokenizer.api import segment_and_pretokenize

        project_root = Path(__file__).resolve().parent.parent
        input_dir = project_root / "data"

        return {
            file: segment_and_pretokenize(input_dir / file, _SPECIAL_TOKENS, parallel=True)
            for file in self.params[0]
        }

    def time_train(self, counts, file, n_merges):
        from cs336_basics.bpe_tokenizer.api import train

        train(counts[file], _SPECIAL_TOKENS, n_merges)
