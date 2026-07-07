from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from cs336_basics.bpe_tokenizer.segmenter import BufferingSegmenter, InMemorySegmenter, Segmenter


@pytest.mark.parametrize(
    "content,tokens,expected",
    [
        # --- Basic Segmentation ---
        pytest.param(b"", [b"\n"], [], id="empty input"),
        pytest.param(b"hello world", [b"\n"], [b"hello world"], id="no delimiter"),
        pytest.param(b"doc1\ndoc2", [b"\n"], [b"doc1", b"doc2"], id="two docs"),
        pytest.param(b"doc1\ndoc2\ndoc3", [b"\n"], [b"doc1", b"doc2", b"doc3"], id="three docs"),
        # --- Boundary Cases ---
        pytest.param(b"\ndoc1", [b"\n"], [b"", b"doc1"], id="leading delimiter"),
        pytest.param(b"doc1\n", [b"\n"], [b"doc1"], id="trailing delimiter"),
        pytest.param(b"doc1\n\ndoc2", [b"\n"], [b"doc1", b"", b"doc2"], id="consecutive delimiters"),
        # --- Multiple Tokens ---
        pytest.param(
            b"doc1\ndoc2<|pad|>doc3",
            [b"\n", b"<|pad|>"],
            [b"doc1", b"doc2", b"doc3"],
            id="multiple delimiters",
        ),
        pytest.param(
            b"prefixfoo_suffix",
            [b"foo", b"foobar"],
            [b"prefix", b"_suffix"],
            id="longest match",
        ),
        # --- Binary Content ---
        pytest.param(
            b"\x00\x01\x02\n\xff\xfe\xfd",
            [b"\n"],
            [b"\x00\x01\x02", b"\xff\xfe\xfd"],
            id="binary content",
        ),
    ],
)
@pytest.mark.parametrize(
    "segmenter_factory",
    [
        pytest.param(lambda tokens: InMemorySegmenter(tokens), id="InMemorySegmenter"),
        pytest.param(lambda tokens: BufferingSegmenter(tokens, 1), id="BufferingSegmenter(1)"),
        pytest.param(lambda tokens: BufferingSegmenter(tokens, 100), id="BufferingSegmenter(10)"),
    ],
)
def test_segment(
    tmp_path: Path,
    segmenter_factory: Callable[[Sequence[bytes]], Segmenter],
    content: bytes,
    tokens: list[bytes],
    expected: list[bytes],
):
    file_path = tmp_path / "input.bin"
    file_path.write_bytes(content)
    segmenter = segmenter_factory(tokens)
    assert list(segmenter.run(file_path)) == expected
