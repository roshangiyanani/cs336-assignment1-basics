from __future__ import annotations

from collections import Counter
from typing import assert_never

import pytest

from cs336_basics.bpe_tokenizer.pretokenizer import WORD_RE, Pretokenizer, SimplePretokenizer


@pytest.mark.parametrize(
    "input_bytes,expected_counter",
    [
        # --- Basic Cases ---
        pytest.param(b"", Counter(), id="empty input"),
        pytest.param(b"hi", Counter({(b"h", b"i"): 1}), id="single word"),
        pytest.param(
            b"hello world",
            Counter(
                {
                    (b"h", b"e", b"l", b"l", b"o"): 1,
                    (b"w", b"o", b"r", b"l", b"d"): 1,
                }
            ),
            id="two words",
        ),
        # --- Whitespace Handling ---
        pytest.param(
            b"a b c",
            Counter({(b"a",): 1, (b"b",): 1, (b"c",): 1}),
            id="multiple single-byte words",
        ),
        pytest.param(
            b"hi   hi",
            Counter({(b"h", b"i"): 2}),
            id="multiple spaces",
        ),
        pytest.param(
            b"hello\nworld",
            Counter(
                {
                    (b"h", b"e", b"l", b"l", b"o"): 1,
                    (b"w", b"o", b"r", b"l", b"d"): 1,
                }
            ),
            id="newline separator",
        ),
        # --- Repeated Words ---
        pytest.param(
            b"hi hi hi",
            Counter({(b"h", b"i"): 3}),
            id="repeated word",
        ),
        pytest.param(
            b"hi hello hi",
            Counter({(b"h", b"i"): 2, (b"h", b"e", b"l", b"l", b"o"): 1}),
            id="mixed repeated words",
        ),
        # --- Binary / Non-ASCII ---
        pytest.param(
            b"\xff\xfe",
            Counter({(b"\xff", b"\xfe"): 1}),
            id="non-ascii bytes",
        ),
        pytest.param(
            b"\x00\x01 \x02\x03",
            Counter({(b"\x00", b"\x01"): 1, (b"\x02", b"\x03"): 1}),
            id="binary words",
        ),
    ],
)
@pytest.mark.parametrize("input_type", [bytes, bytearray, memoryview])
def test_pretokenize(input_bytes: bytes, input_type: type, expected_counter: Counter):
    if input_type is bytes:
        input = input_bytes
    elif input_type is bytearray:
        input = bytearray(input_bytes)
    elif input_type is memoryview:
        input = memoryview(bytearray(input_bytes))
    else:
        assert_never(input_type)

    pretokenizer = SimplePretokenizer(re=WORD_RE)
    pretokenizer.process(input)
    result = pretokenizer.finalize()
    assert result == expected_counter


def test_accumulation_across_calls():
    pretokenizer = SimplePretokenizer(re=WORD_RE)
    pretokenizer.process(b"hi")
    pretokenizer.process(b"hello")
    pretokenizer.process(b"hi")
    result = pretokenizer.finalize()
    assert result == Counter({(b"h", b"i"): 2, (b"h", b"e", b"l", b"l", b"o"): 1})


def test_abstract_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Pretokenizer()
