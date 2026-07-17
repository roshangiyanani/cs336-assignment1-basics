"""Stable public API for the BPE tokenizer pipeline.

Internal implementations may change; these function signatures do not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from cs336_basics.bpe_tokenizer.pretokenizer import GPT_RE

# Type aliases
Tokens = tuple[bytes, ...]


def _segment(
    input_path: Path,
    special_tokens: Sequence[str],
) -> Iterator[str]:
    """Segment a text file into document chunks.

    Returns an iterator of bytes-like objects. The concrete type
    (list, generator, etc.) depends on the underlying implementation.
    """
    from .segmenter import BufferingSegmenter

    return BufferingSegmenter(special_tokens).run(input_path)


def _pretokenize(
    segments: Iterable[str],
) -> Counter[Tokens]:
    """Pretokenize segments into unigram token counts.

    Accepts any iterable of bytes-like objects. The iterable is consumed
    exactly once (generators are safe).
    """
    from .pretokenizer import SimplePretokenizer

    pretokenizer = SimplePretokenizer(re=GPT_RE)
    for seg in segments:
        pretokenizer.process(seg)
    return pretokenizer.finalize()


def segment_and_pretokenize(
    input_path: Path,
    special_tokens: Sequence[str],
) -> Counter[Tokens]:
    """Segment a file then pretokenize the segments.

    Returns unigram token counts. The segments iterator flows directly
    into pretokenization without materialization.
    """
    segments = _segment(input_path, special_tokens)
    return _pretokenize(segments)


def train(
    counts: Counter[Tokens],
    special_tokens: Sequence[str],
    n_merges: int,
) -> None:
    """Train a BPE tokenizer by performing ``n_merges`` merge steps.

    The tokenizer state is internal; this function is meant for
    benchmarking the merge workload.
    """
    from .naive_tokenizer import NaiveTokenizer

    tokenizer = NaiveTokenizer(counts, special_tokens)
    tokenizer.merge_n(n_merges)
