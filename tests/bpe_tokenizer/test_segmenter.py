from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from cs336_basics.bpe_tokenizer.segmenter import BufferingSegmenter, InMemorySegmenter, Segmenter

class SegmenterFactoryCase(NamedTuple):
    factory: Callable[[Sequence[str]], Segmenter]
    id: str

SEGMENTER_FACTORY_CASE: list[SegmenterFactoryCase] = [
    SegmenterFactoryCase(lambda tokens: InMemorySegmenter(tokens), id="InMemorySegmenter"),
    SegmenterFactoryCase(lambda tokens: BufferingSegmenter(tokens, 1), id="BufferingSegmenter(1)"),
    SegmenterFactoryCase(lambda tokens: BufferingSegmenter(tokens, 100), id="BufferingSegmenter(100)"),
]


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
@pytest.mark.parametrize("segmenter_case", SEGMENTER_FACTORY_CASE, ids=lambda case: case.id)
def test_segment(
    tmp_path: Path,
    segmenter_case: SegmenterFactoryCase,
    content: str,
    tokens: list[str],
    expected: list[str],
):
    file_path = tmp_path / "input.bin"
    file_path.write_text(content)
    segmenter = segmenter_case.factory(tokens)
    assert list(segmenter.run(file_path)) == expected


@pytest.mark.parametrize(
    "content,start,end,expected",
    [
        # --- Start only ---
        pytest.param("doc1\ndoc2\ndoc3", 4, None, ["", "doc2", "doc3"], id="skip first doc"),
        pytest.param("doc1\ndoc2\ndoc3", 5, None, ["doc2", "doc3"], id="skip mid-token"),
        # --- End only ---
        pytest.param("doc1\ndoc2\ndoc3", 0, 9, ["doc1", "doc2"], id="truncate before delim"),
        pytest.param("doc1\ndoc2\ndoc3", 0, 10, ["doc1", "doc2"], id="truncate after delim"),
        # --- Both start and end ---
        pytest.param("doc1\ndoc2\ndoc3", 5, 9, ["doc2"], id="full segment"),
        pytest.param("doc1\ndoc2\ndoc3", 4, 10, ["", "doc2"], id="empty and full segment"),
        pytest.param("doc1\ndoc2\ndoc3", 4, 11, ["", "doc2", "d"], id="empty and full and partial segment"),
        # --- Edge cases ---
        pytest.param("doc1\ndoc2", 0, None, ["doc1", "doc2"], id="defaults same as no args"),
        pytest.param("doc1", 100, None, [], id="start beyond file length"),
        pytest.param("doc1\ndoc2", 4, 4, [], id="zero length slice"),
    ],
)
@pytest.mark.parametrize("segmenter_case", SEGMENTER_FACTORY_CASE, ids=lambda case: case.id)
def test_segment_with_start_end(
    tmp_path: Path,
    segmenter_case: SegmenterFactoryCase,
    content: str,
    start: int,
    end: int | None,
    expected: list[str],
):
    file_path = tmp_path / "input.bin"
    file_path.write_text(content)
    segmenter = segmenter_case.factory(["\n"])
    assert list(segmenter.run(file_path, start=start, end=end)) == expected


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("\u00e9", id="latin e-acute"),
        pytest.param("\u00f1", id="latin n-tilde"),
        pytest.param("\u4e00", id="cjk character"),
        pytest.param("\u2603", id="snowman emoji"),
        pytest.param("\U0001f600", id="grinning face"),
    ],
)
@pytest.mark.parametrize("segmenter_case", SEGMENTER_FACTORY_CASE, ids=lambda case: case.id)
def test_reject_non_ascii_token(segmenter_case: SegmenterFactoryCase, token: str):
    with pytest.raises(ValueError, match="must be ascii"):
        segmenter_case.factory([token])
