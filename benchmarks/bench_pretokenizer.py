"""
asv benchmarks for the segmenter and pretokenizer.

Run with:
    asv run
"""

from pathlib import Path


class PretokenizerBenchmark:
    """Benchmark segmenting + pretokenizing text into token counts."""

    def setup_cache(self):
        """Load the fixture file once to warm the OS page cache."""
        project_root = Path(__file__).resolve().parent.parent
        input_path = project_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
        input_path.read_bytes()  # warm disk cache
        return input_path

    def time_segment_and_pretokenize(self, input_path):
        from cs336_basics.bpe_tokenizer.api import segment_and_pretokenize

        SPECIAL_TOKENS = [b"<|endoftext|>"]

        segment_and_pretokenize(input_path, SPECIAL_TOKENS)
