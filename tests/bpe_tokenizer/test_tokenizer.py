from __future__ import annotations

from collections import Counter

import pytest

from cs336_basics.bpe_tokenizer.naive_tokenizer import NaiveTokenizer
from cs336_basics.bpe_tokenizer.tokenizer import Tokenizer, TokenPair, Tokens

# ---------------------------------------------------------------------------
# Vocab Initialization
# ---------------------------------------------------------------------------


def test_initialize_vocab_no_special_tokens():
    vocab = Tokenizer._initialize_vocab([])
    assert len(vocab) == 256
    assert vocab[0] == b"\x00"
    assert vocab[65] == b"A"  # chr(65) == 'A'


def test_initialize_vocab_with_special_tokens():
    special = [b"<|pad|>", b"<|eos|>"]
    vocab = Tokenizer._initialize_vocab(special)
    assert len(vocab) == 258
    assert vocab[256] == b"<|pad|>"
    assert vocab[257] == b"<|eos|>"


# ---------------------------------------------------------------------------
# Pair Counting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pretokenized,expected_pairs",
    [
        pytest.param(
            [((b"a", b"b", b"c"), 1)],
            Counter({(b"a", b"b"): 1, (b"b", b"c"): 1}),
            id="single word count 1",
        ),
        pytest.param(
            [((b"a", b"b"), 2)],
            Counter({(b"a", b"b"): 2}),
            id="pair weighted by count",
        ),
        pytest.param(
            [((b"a", b"b"), 1), ((b"b", b"c"), 1)],
            Counter({(b"a", b"b"): 1, (b"b", b"c"): 1}),
            id="two separate words",
        ),
        pytest.param(
            [((b"a", b"b"), 1), ((b"a", b"b"), 2)],
            Counter({(b"a", b"b"): 3}),
            id="same pair from two words",
        ),
        pytest.param(
            [],
            Counter(),
            id="empty pretokenized",
        ),
    ],
)
def test_count_token_pairs(pretokenized: list[tuple[Tokens, int]], expected_pairs: Counter[TokenPair]):
    result = NaiveTokenizer._count_token_pairs(pretokenized)
    assert result == expected_pairs


# ---------------------------------------------------------------------------
# Merge Index Finding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,merge_pair,expected_indexes",
    [
        pytest.param((b"a", b"b", b"c"), (b"a", b"b"), [0], id="pair at start"),
        pytest.param((b"a", b"b", b"c"), (b"b", b"c"), [1], id="pair at end"),
        pytest.param((b"a", b"b", b"a", b"b"), (b"a", b"b"), [0, 2], id="two non-overlapping pairs"),
        pytest.param(
            (b"a", b"a", b"a"),
            (b"a", b"a"),
            [0],
            id="overlapping pair skips second occurrence",
        ),
        pytest.param((b"a", b"c"), (b"a", b"b"), [], id="pair not in word"),
        pytest.param((b"a",), (b"a", b"b"), [], id="single token word"),
    ],
)
def test_indexes_to_merge(word: Tokens, merge_pair: TokenPair, expected_indexes: list[int]):
    assert NaiveTokenizer._indexes_to_merge(word, merge_pair) == expected_indexes


# ---------------------------------------------------------------------------
# Merge Application
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,merge_indexes,replacement,expected_word,expected_inc,expected_dec",
    [
        pytest.param(
            (b"a", b"b", b"c"),
            [0],
            b"ab",
            (b"ab", b"c"),
            [(b"ab", b"c")],
            [(b"b", b"c")],
            id="merge first pair",
        ),
        pytest.param(
            (b"a", b"b", b"c"),
            [1],
            b"bc",
            (b"a", b"bc"),
            [(b"a", b"bc")],
            [(b"a", b"b")],
            id="merge last pair",
        ),
        pytest.param(
            (b"a", b"b", b"a", b"b"),
            [0, 2],
            b"ab",
            (b"ab", b"ab"),
            [(b"ab", b"a"), (b"b", b"ab")],
            [(b"b", b"a"), (b"b", b"a")],
            id="merge two non-overlapping pairs",
        ),
        pytest.param(
            (b"a", b"b", b"c"),
            [],
            b"ab",
            (b"a", b"b", b"c"),
            [],
            [],
            id="no merge indexes",
        ),
    ],
)
def test_apply_merge(
    word: Tokens,
    merge_indexes: list[int],
    replacement: bytes,
    expected_word: Tokens,
    expected_inc: list[TokenPair],
    expected_dec: list[TokenPair],
):
    result = NaiveTokenizer._apply_merge_and_get_count_diff(word, merge_indexes, replacement)
    assert result.new_word == expected_word
    assert result.incremented_pairs == expected_inc
    assert result.decremented_pairs == expected_dec


# ---------------------------------------------------------------------------
# Integration: merge_once
# ---------------------------------------------------------------------------


def test_merge_once():
    pretokenized = Counter({(b"a", b"b", b"c"): 10, (b"a", b"b"): 5})
    tokenizer = NaiveTokenizer(pretokenized, [])
    tokenizer.merge_once()

    # "a"+"b" is the most common pair (10+5=15)
    assert tokenizer.merges == [(b"a", b"b")]
    assert len(tokenizer.vocab) == 257  # 256 + 1 merge
    assert tokenizer.vocab[256] == b"ab"


def test_merge_n():
    pretokenized = Counter({(b"a", b"b", b"c", b"d"): 3})
    tokenizer = NaiveTokenizer(pretokenized, [])
    tokenizer.merge_n(2)

    assert len(tokenizer.merges) == 2
    assert len(tokenizer.vocab) == 258  # 256 + 2 merges


def test_merge_until():
    pretokenized = Counter({(b"a", b"b", b"c"): 1})
    tokenizer = NaiveTokenizer(pretokenized, [])
    # vocab starts at 256, target is 258 → 2 merges needed
    tokenizer.merge_until(258)
    assert len(tokenizer.vocab) == 258


def test_merge_until_already_exceeded():
    pretokenized = Counter({(b"a", b"b"): 1})
    tokenizer = NaiveTokenizer(pretokenized, [])
    with pytest.raises(ValueError, match="too large"):
        tokenizer.merge_until(250)


# ---------------------------------------------------------------------------
# as_output
# ---------------------------------------------------------------------------


def test_as_output():
    pretokenized = Counter({(b"a", b"b"): 5})
    tokenizer = NaiveTokenizer(pretokenized, [b"<|pad|>"])
    tokenizer.merge_once()

    vocab_dict, merges = tokenizer.as_output()
    assert isinstance(vocab_dict, dict)
    assert vocab_dict[256] == b"<|pad|>"
    assert vocab_dict[257] == b"ab"
    assert merges == [(b"a", b"b")]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


def test_empty_pretokenized():
    tokenizer = NaiveTokenizer(Counter(), [])
    assert len(tokenizer.token_pair_counts) == 0


def test_single_byte_words_no_pairs():
    pretokenized = Counter({(b"a",): 5, (b"b",): 3})
    tokenizer = NaiveTokenizer(pretokenized, [])
    assert len(tokenizer.token_pair_counts) == 0


# ---------------------------------------------------------------------------
# Abstract Class
# ---------------------------------------------------------------------------


def test_abstract_tokenizer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Tokenizer([])
