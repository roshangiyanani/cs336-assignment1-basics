from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from cs336_basics.bpe_tokenizer.segmenter import BufferingSegmenter, InMemorySegmenter, Segmenter


@pytest.mark.parametrize(
    "content,tokens,expected",
    [
        # --- Basic Segmentation ---
        pytest.param("", ["\n"], [], id="empty input"),
        pytest.param("hello world", ["\n"], ["hello world"], id="no delimiter"),
        pytest.param("doc1\ndoc2", ["\n"], ["doc1", "doc2"], id="two docs"),
        pytest.param("doc1\ndoc2\ndoc3", ["\n"], ["doc1", "doc2", "doc3"], id="three docs"),
        # --- Boundary Cases ---
        pytest.param("\ndoc1", ["\n"], ["", "doc1"], id="leading delimiter"),
        pytest.param("doc1\n", ["\n"], ["doc1"], id="trailing delimiter"),
        pytest.param("doc1\n\ndoc2", ["\n"], ["doc1", "", "doc2"], id="consecutive delimiters"),
        # --- Multiple Tokens ---
        pytest.param(
            "doc1\ndoc2<|pad|>doc3",
            ["\n", "<|pad|>"],
            ["doc1", "doc2", "doc3"],
            id="multiple delimiters",
        ),
        pytest.param(
            "prefixfoo_suffix",
            ["foo", "foobar"],
            ["prefix", "_suffix"],
            id="longest match",
        ),
        # --- Binary Content ---
        pytest.param(
            "\x00\x01\x02\n\xff\xfe\xfd",
            ["\n"],
            ["\x00\x01\x02", "\xff\xfe\xfd"],
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
    segmenter_factory: Callable[[Sequence[str]], Segmenter],
    content: str,
    tokens: list[str],
    expected: list[str],
):
    file_path = tmp_path / "input.bin"
    file_path.write_text(content)
    segmenter = segmenter_factory(tokens)
    assert list(segmenter.run(file_path)) == expected
