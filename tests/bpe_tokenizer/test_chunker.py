# Folders:
from __future__ import annotations

from pathlib import Path

import pytest

from cs336_basics.bpe_tokenizer.chunker import Chunker


class TestValidation:
    def test_reject_no_tokens(self):
        with pytest.raises(ValueError, match="at least one special_token"):
            Chunker([])

    def test_reject_empty_token(self):
        with pytest.raises(ValueError, match="len > 0"):
            Chunker([""])


    @pytest.mark.parametrize(
        "tokens",
        [
            pytest.param(["\u00e9"], id="latin e-acute"),
            pytest.param(["\u4e00"], id="cjk character"),
            pytest.param(["\U0001f600"], id="grinning face"),
        ],
    )
    def test_reject_non_ascii_token(self, tokens: list[str]):
        with pytest.raises(ValueError, match="must be ascii"):
            Chunker(tokens)

    def test_reject_nonexclusive_tokens(self):
        with pytest.raises(ValueError, match="must be non-exclusive"):
            Chunker(["|", "<|end|>"])

    @pytest.mark.parametrize(
        "min_size,num_chunks,match",
        [
            pytest.param(0, 0, "at least min_size or num_chunks must be given", id="neither given"),
            pytest.param(-1, 2, "min_size, if given, must be at least 1", id="negative min_size"),
            pytest.param(1, -1, "must have at least 1 chunk", id="negative num_chunks"),
        ],
    )
    def test_reject_invalid_chunk_args(self, tmp_path: Path, min_size: int, num_chunks: int, match: str):
        file_path = tmp_path / "input.txt"
        file_path.write_text("a|b|c|d")
        chunker = Chunker(["|"])
        with pytest.raises(ValueError, match=match):
            list(chunker.chunk(file_path, min_size=min_size, num_chunks=num_chunks))


# --- Basic chunking ---

@pytest.mark.parametrize(
    "content,num_chunks,expected",
    [
        # --- Empty / trivial ---
        pytest.param("", 1, [], id="empty file"),
        pytest.param("", 5, [], id="empty file many chunks"),
        # --- Single chunk ---
        pytest.param("hello", 1, [(0, None)], id="single chunk"),
        pytest.param("hello|world", 1, [(0, None)], id="single chunk with delimiter"),
        # --- Two chunks ---
        pytest.param(
            "doc1|doc2|doc3|doc4",
            2,
            # 19 chars
            # chunk_size = 10
            # chunk_1 is [0, 10)
            # adjust 10: "d" in "doc3|" -> 15
            [(0, 15), (15, None)],
            id="two chunks (split is after delim)"
        ),
        pytest.param(
            "doc1|doc22|doc3|doc4",
            2,
            # 20 chars
            # chunk_size = 10
            # chunk_1 is [0, 10)
            # adjust 10: "|" in "doc22|" -> 11
            [(0, 11), (11, None)],
            id="two chunks (split is at/before delim)"
        ),
        # --- Three chunks ---
        pytest.param(
            "a|b|c|d|e|f|g",
            3,
            # 13 chars
            # chunk_size = 5
            # chunk_1 is [0, 5)
            # adjust 5: "|" of "c|" -> 6
            # chunk_2 is [6, 11)
            # adjust 11: "|" of "f|" -> 12
            [(0, 6), (6, 12), (12, None)],
            id="three chunks",
        ),
        pytest.param(
            "a|b|c|d|e|f",
            3,
            # 11 chars
            # chunk_size = 4
            # chunk_1 is [0, 4)
            # adjust 4: "c" of "c|" -> 6
            # chunk_2 is [6, 10)
            # adjust 10: "f" -> EOF
            [(0, 6), (6, None)],
            id="three chunks - no third",
        ),
        # --- No delimiters in content (falls back to EOF) ---
        pytest.param("helloworld", 2, [(0, None)], id="no delimiters"),
        pytest.param("helloworld", 3, [(0, None)], id="no delimiters three chunks"),
    ],
)
def test_chunk(
    tmp_path: Path,
    content: str,
    num_chunks: int,
    expected: list[tuple[int, int | None]],
):
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(["|"])
    assert list(chunker.chunk(file_path, num_chunks=num_chunks, min_size=0)) == expected


# --- min_size adjustment ---

@pytest.mark.parametrize(
    "content,num_chunks,min_size,expected",
    [
        # Target chunk size < min_size, so fewer chunks
        pytest.param(
            "a|b|c|d",
            10,
            100,
            [(0, None)],
            id="min_size larger than file",
        ),
        pytest.param(
            "a|b|c|d|e|f|g|h",
            10,
            3,
            # 15 chars
            # chunk_size = 2
            # chunk_1 is [0, 2)
            # adjust 2: "b" of "b|" -> 4
            # chunk_2 is [4, 6)
            # adjust 6: "d" of "d|" -> 8
            [(0, 4), (4, 8), (8, 12), (12, None)],
            id="min_size reduces chunks",
        ),
        pytest.param(
            "a|b|c|d",
            2,
            2,
            [(0, 6), (6, None)],
            id="target greater than min_size",
        ),
    ],
)
def test_min_size_adjustment(
    tmp_path: Path,
    content: str,
    num_chunks: int,
    min_size: int,
    expected: list[tuple[int, int | None]],
):
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(["|"])
    assert list(chunker.chunk(file_path, num_chunks=num_chunks, min_size=min_size)) == expected


# --- min_size only (num_chunks=0) ---

@pytest.mark.parametrize(
    "content,min_size,expected",
    [
        pytest.param(
            "a|b|c|d",
            4,
            # 8 chars
            # chunk_size = 4
            # chunk_1 is [0, 4)
            # adjust 4: "|" of "c|" -> 6
            [(0, 6), (6, None)],
            id="min_size only fits twice",
        ),
        pytest.param(
            "a|b|c|d",
            10,
            # 8 chars
            # chunk_size = 10, larger than file
            [(0, None)],
            id="min_size larger than file",
        ),
        pytest.param(
            "a|b|c|d|e|f",
            3,
            # 12 chars
            # chunk_size = 3
            # chunk_1 is [0, 3)
            # adjust 3: "|" of "|c|" -> 4
            # chunk_2 is [4, 7)
            # adjust 7: "|" of "|e|" -> 8
            # chunk_3 is [8, 11)
            # adjust 11: no "|" found -> EOF
            [(0, 4), (4, 8), (8, None)],
            id="min_size only many chunks",
        ),
    ],
)
def test_min_size_only(tmp_path: Path, content: str, min_size: int, expected: list[tuple[int, int | None]]):
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(["|"])
    assert list(chunker.chunk(file_path, min_size=min_size)) == expected


# --- Invariants ---

def test_chunks_non_overlapping(tmp_path: Path):
    content = "doc1|doc2|doc3|doc4|doc5|doc6|doc7|doc8"
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(["|"])
    chunks = list(chunker.chunk(file_path, num_chunks=4, min_size=0))

    assert chunks[0][0] == 0, "first chunk starts at 0"
    assert chunks[-1][1] is None, "last chunk end is None"

    for i in range(len(chunks) - 1):
        assert chunks[i][1] == chunks[i + 1][0], "chunks should not overlap or have gaps"



def test_chunk_count(tmp_path: Path):
    content = "doc1|doc2|doc3|doc4|doc5|doc6|doc7|doc8"
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(["|"])

    for num in range(1, 10):
        chunks = list(chunker.chunk(file_path, num_chunks=num, min_size=0))
        assert len(chunks) <= num, f"expected <={num} chunks, got {len(chunks)}"



@pytest.mark.parametrize(
    "content, tokens, num_chunks, expected",
    [
        pytest.param(
            "doc1-doc2<|sep|>doc3-doc4",
            ["-", "<|sep|>"],
            2,
            # 25 chars
            # chunk_size = 13
            # chunk_1 is [0, 13)
            # adjust 13: "p" of "<|sep|>" -> need to see whole special token -> "d" of "doc4" -> 21
            # chunk_2 is [21, None)
            [(0, 21), (21, None)],
            id="mixed delimiters",
        ),
        pytest.param(
            "a<|end|>b<|end|>c<|end|>d",
            ["<|end|>"],
            2,
            [(0, 24), (24, None)],
            id="long special token",
        ),
    ],
)
def test_special_tokens(
    tmp_path: Path,
    content: str,
    tokens: list[str],
    num_chunks: int,
    expected: list[tuple[int, int | None]],
):
    file_path = tmp_path / "input.txt"
    file_path.write_text(content)
    chunker = Chunker(tokens)
    assert list(chunker.chunk(file_path, num_chunks=num_chunks, min_size=0)) == expected
