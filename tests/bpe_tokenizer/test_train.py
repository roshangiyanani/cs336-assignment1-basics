from __future__ import annotations

from collections import Counter

import pytest

from cs336_basics.bpe_tokenizer.train import TokenizeTrainer, TokenPair, Tokens

# ---------------------------------------------------------------------------
# Vocab Initialization
# ---------------------------------------------------------------------------


def test_initialize_vocab_no_special_tokens():
    vocab = TokenizeTrainer._initialize_vocab([])
    assert len(vocab) == 256
    assert vocab[0] == b"\x00"
    assert vocab[65] == b"A"  # chr(65) == 'A'


def test_initialize_vocab_with_special_tokens():
    special = ["<|pad|>", "<|eos|>"]
    vocab = TokenizeTrainer._initialize_vocab(special)
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
    result = TokenizeTrainer._count_token_pairs(pretokenized)[0]
    assert result == expected_pairs


# ---------------------------------------------------------------------------
# Merge Application (_merge_and_get_count_diff)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,merge_pair,expected_word,expected_inc,expected_dec",
    [
        pytest.param(
            (b"a", b"b", b"c"),
            (b"a", b"b"),
            (b"ab", b"c"),
            [(b"ab", b"c")],
            [(b"b", b"c")],
            id="merge first pair",
        ),
        pytest.param(
            (b"a", b"b", b"c"),
            (b"b", b"c"),
            (b"a", b"bc"),
            [(b"a", b"bc")],
            [(b"a", b"b")],
            id="merge last pair",
        ),
        pytest.param(
            (b"a", b"b", b"a", b"b"),
            (b"a", b"b"),
            (b"ab", b"ab"),
            [(b"ab", b"a"), (b"ab", b"ab")],
            [(b"b", b"a"), (b"ab", b"a")],
            id="merge two non-overlapping pairs",
        ),
        pytest.param(
            (b"a", b"a", b"a"),
            (b"a", b"a"),
            (b"aa", b"a"),
            [(b"aa", b"a")],
            [(b"a", b"a")],
            id="overlapping pair skips second occurrence",
        ),
        pytest.param(
            (b"a", b"c"),
            (b"a", b"b"),
            (b"a", b"c"),
            [],
            [],
            id="pair not in word",
        ),
        pytest.param(
            (b"a",),
            (b"a", b"b"),
            (b"a",),
            [],
            [],
            id="single token word",
        ),
        pytest.param(
            (b"x", b"a", b"b", b"a", b"b", b"y"),
            (b"a", b"b"),
            (b"x", b"ab", b"ab", b"y"),
            [(b"x", b"ab"), (b"ab", b"a"), (b"ab", b"ab"), (b"ab", b"y")],
            [(b"x", b"a"), (b"b", b"a"), (b"ab", b"a"), (b"b", b"y")],
            id="merge two overlapping-separated pairs in xababy",
        ),
    ],
)
def test_merge_and_get_count_diff(
    word: Tokens,
    merge_pair: TokenPair,
    expected_word: Tokens,
    expected_inc: list[TokenPair],
    expected_dec: list[TokenPair],
):
    replacement = b"".join(merge_pair)
    result = TokenizeTrainer._merge_and_get_count_diff(word, merge_pair, replacement)
    assert result.new_word == expected_word
    assert result.incremented_pairs == expected_inc
    assert result.decremented_pairs == expected_dec


# ---------------------------------------------------------------------------
# Integration: merge_once
# ---------------------------------------------------------------------------


def test_merge_once():
    pretokenized = Counter({(b"a", b"b", b"c"): 10, (b"a", b"b"): 5})
    tokenizer = TokenizeTrainer([], pretokenized)
    tokenizer.merge_once()

    # "a"+"b" is the most common pair (10+5=15)
    assert tokenizer.merges == [(b"a", b"b")]
    assert len(tokenizer.vocab) == 257  # 256 + 1 merge
    assert tokenizer.vocab[256] == b"ab"


def test_merge_n():
    pretokenized = Counter({(b"a", b"b", b"c", b"d"): 3})
    tokenizer = TokenizeTrainer([], pretokenized)
    tokenizer.merge_n(2)

    assert len(tokenizer.merges) == 2
    assert len(tokenizer.vocab) == 258  # 256 + 2 merges


def test_merge_until():
    pretokenized = Counter({(b"a", b"b", b"c"): 1})
    tokenizer = TokenizeTrainer([], pretokenized)
    # vocab starts at 256, target is 258 → 2 merges needed
    tokenizer.merge_until(258)
    assert len(tokenizer.vocab) == 258


def test_merge_until_already_exceeded():
    pretokenized = Counter({(b"a", b"b"): 1})
    tokenizer = TokenizeTrainer([], pretokenized)
    with pytest.raises(ValueError, match="too large"):
        tokenizer.merge_until(250)


# ---------------------------------------------------------------------------
# as_output
# ---------------------------------------------------------------------------


def test_as_output():
    pretokenized = Counter({(b"a", b"b"): 5})
    tokenizer = TokenizeTrainer(["<|pad|>"], pretokenized)
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
    tokenizer = TokenizeTrainer([], Counter())
    assert len(tokenizer.token_pair_count_max_heap) == 0


def test_single_byte_words_no_pairs():
    pretokenized = Counter({(b"a",): 5, (b"b",): 3})
    tokenizer = TokenizeTrainer([], pretokenized)
    assert len(tokenizer.token_pair_count_max_heap) == 0
